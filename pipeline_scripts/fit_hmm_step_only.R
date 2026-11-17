# Fit a 3-state momentuHMM model using step length only (no turning angle).
#
# Same pipeline as fit_hmm.R, with the turning-angle stream removed, so the two fits
# show what turning angle adds to the state classification.
#
# Input:  regularised_tracks.csv (from interpolation_script.R), 5-minute positions in UTM 31N metres
# Output: hmm_fit_step_only.rds, hmm_states_step_only.csv, hmm_pimin_sweep_step_only.csv,
#         and hmm_pimin_sweep.png redrawn with both models
#
# States are ranked by mean step length:
#   state 1 = near-stationary / drifting, state 2 = fishing (the middle state),
#   state 3 = transit (fastest).
# Fishing is labelled at a posterior threshold Pi_min, swept from 0.5 to 0.9 (1.0 is only hit by underflow).

suppressPackageStartupMessages({
  library(momentuHMM)
})

# ─────────────────────────────────────────────
#  PATHS AND SETTINGS
# ─────────────────────────────────────────────

project_root <- normalizePath(".")      # run from the project folder: Rscript pipeline_scripts/<name>.R
input_csv    <- file.path(project_root, "pipeline_outputs/gps_tracks/regularised_tracks.csv")
fit_rds      <- file.path(project_root, "pipeline_outputs/hmm_results/hmm_fit_step_only.rds")
states_csv   <- file.path(project_root, "pipeline_outputs/hmm_results/hmm_states_step_only.csv")
sweep_csv    <- file.path(project_root, "pipeline_outputs/hmm_results/hmm_pimin_sweep_step_only.csv")

MIN_STEPS    <- 20                          # segments need at least this many steps
PIMIN_VALUES <- seq(0.5, 0.9, by = 0.1)
STEP_MINUTES <- 5
ZERO_STEP_KM <- 0.001                       # 1 m, replaces exact zero steps
N_STARTS     <- 3                           # all 10 starts agreed in fit_hmm.R, so 3 is enough

# ─────────────────────────────────────────────
#  LOAD AND PREPARE
# ─────────────────────────────────────────────

regular <- read.csv(input_csv, stringsAsFactors = FALSE)
regular$time <- as.POSIXct(regular$time, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
regular <- regular[order(regular$ID, regular$time), ]

# Keep segments with enough points (segment = 30 min split; must have min 20 points)
n_rows <- table(regular$ID)
keep   <- names(n_rows)[n_rows >= MIN_STEPS + 1]
regular <- regular[regular$ID %in% keep, ]
cat(sprintf("Segments kept: %d of %d (>= %d steps)\n",
            length(keep), length(n_rows), MIN_STEPS))

# Metres to km, so step lengths are km per 5-minute step
regular$x <- regular$x / 1000
regular$y <- regular$y / 1000

# prepData adds step (km between consecutive positions) and angle (change in heading, radians, -pi to pi)
# angle is NA at each segment's first and last position and wherever a step has zero length
data <- prepData(regular, type = "UTM", coordNames = c("x", "y"))

# Gamma step distribution can't take zero steps; replace each with 1 m (0.001 km)
n_zero <- sum(data$step == 0, na.rm = TRUE)
data$step[which(data$step == 0)] <- ZERO_STEP_KM
cat(sprintf("Replaced %d zero-length steps with %.3f km\n", n_zero, ZERO_STEP_KM))

cat(sprintf("Steps: %d, segments: %d\n", sum(!is.na(data$step)), length(keep)))

# ─────────────────────────────────────────────
#  STARTING VALUES FROM STEP QUANTILES
# ─────────────────────────────────────────────

# step length is defined as the distance between consecutive locations (in km per 5-minute step)

step_q <- quantile(data$step, probs = c(0.2, 0.5, 0.9), na.rm = TRUE)
cat("Base step quantiles (km per 5 min):", round(step_q, 3), "\n")

# ─────────────────────────────────────────────
#  FIT FROM SEVERAL STARTING VALUES, KEEP THE BEST VALID FIT
# ─────────────────────────────────────────────

# Random starts around the step quantiles, sorted so state 1 is the slowest
make_par0 <- function() {
  means <- sort(step_q * exp(rnorm(3, 0, 0.4)))
  list(step = c(means, means * runif(3, 0.5, 1.5)))     # gamma means, then sds
}

# Each state has a gamma step-length distribution (mean, sd); turning angle is not used
# Transition probabilities: chance of switching from one state to another between consecutive 5-min steps
# fitHMM (nlm) adjusts all parameters to maximise the log-likelihood, summed over all state sequences
set.seed(42)
candidates <- list()
for (k in seq_len(N_STARTS)) {
  Par0 <- make_par0()
  cat(sprintf("Start %d of %d\n", k, N_STARTS))
  fit_k <- tryCatch(
    suppressWarnings(fitHMM(
      data,
      nbStates     = 3,
      dist         = list(step = "gamma"),
      Par0         = Par0,
      retryFits    = 0
    )),
    error = function(e) {
      cat("  start failed:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (is.null(fit_k)) next

  # A valid fit has a converged optimiser, finite likelihood and computable state probabilities
  valid <- isTRUE(fit_k$mod$code %in% c(1, 2)) && is.finite(fit_k$mod$minimum) &&
    !is.null(tryCatch(stateProbs(fit_k), error = function(e) NULL))
  cat(sprintf("  log-likelihood %.1f, valid: %s\n", -fit_k$mod$minimum, valid))
  if (valid) candidates[[length(candidates) + 1]] <- fit_k
}

if (length(candidates) == 0) {
  stop("No start gave a valid fit. Revisit the step distribution or the number of states.")
}

# Best fit = highest log-likelihood (mod$minimum is the negative log-likelihood)
best_index <- which.min(sapply(candidates, function(m) m$mod$minimum))
mod <- candidates[[best_index]]
cat(sprintf("Valid fits: %d of %d; best log-likelihood %.1f\n",
            length(candidates), N_STARTS, -mod$mod$minimum))

saveRDS(mod, fit_rds)
cat("Saved", fit_rds, "\n")
cat("Optimiser code:", mod$mod$code, "(1 or 2 means converged)\n")
cat("Log-likelihood:", -mod$mod$minimum, "\n")

pars <- getPar(mod)
print(pars)
# Transition matrix: row = current state, column = next state; each row sums to 1
cat("Transition matrix:\n")
print(round(mod$mle$gamma, 4))

# ─────────────────────────────────────────────
#  DECODE AND LABEL STATES
# ─────────────────────────────────────────────

# Viterbi: the single most likely sequence of states for each segment
viterbi_state <- viterbi(mod)
# stateProbs: probability of each state at each step, using data before and after it (forward-backward)
probs <- stateProbs(mod)
colnames(probs) <- paste0("prob_state", 1:3)

state_mean <- as.numeric(pars$Par$step[1:3])      # gamma means come first in Par$step
fishing_state <- order(state_mean)[2]             # middle mean step length

# No turning-angle check here: this model has no angle stream
cat(sprintf("State mean step (km): %s\n", paste(round(state_mean, 3), collapse = ", ")))

states_out <- data.frame(
  ID = data$ID,
  time = data$time,
  x = data$x,
  y = data$y,
  step_km = data$step,
  angle = data$angle,
  viterbi_state = viterbi_state,
  probs
)
write.csv(states_out, states_csv, row.names = FALSE)
cat("Saved", states_csv, "\n")

# ─────────────────────────────────────────────
#  PI_MIN SWEEP
# ─────────────────────────────────────────────

count_runs <- function(label, id) {
  # number of contiguous TRUE blocks within each segment
  sum(tapply(label, id, function(v) sum(diff(c(FALSE, v)) == 1)))   # a run starts at each FALSE -> TRUE switch
}

n_points <- nrow(states_out)
# A step is labelled fishing when its fishing-state probability is >= Pi_min
sweep <- do.call(rbind, lapply(PIMIN_VALUES, function(p) {
  fishing <- states_out[[paste0("prob_state", fishing_state)]] >= p
  data.frame(
    pi_min          = p,
    fishing_points  = sum(fishing),
    fishing_share   = mean(fishing),
    fishing_hours   = sum(fishing) * STEP_MINUTES / 60,   # each step covers 5 minutes
    fishing_runs    = count_runs(fishing, states_out$ID)
  )
}))
write.csv(sweep, sweep_csv, row.names = FALSE)
print(sweep)
cat("Saved", sweep_csv, "\n")

# Redraw the sweep plot with both models side by side
source(file.path(project_root, "pipeline_scripts/plot_state_sweep.R"))
