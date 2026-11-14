# Turn per-fix HMM fishing labels into fishing events with a minimum length.
#
# Input:  hmm_fit.rds, hmm_states.csv (from fit_hmm.R)
# Output: hmm_states_events.csv, hmm_events_sweep.csv
#
# Rules, applied within each segment, in this order:
#   1. Gap fill: a single non-fishing fix between two fishing fixes becomes fishing,
#      so fishing-X-fishing is one event.
#   2. Minimum event length: a fishing event must contain at least MIN_EVENT_FIXES
#      consecutive fishing fixes. Shorter runs are relabelled non-fishing.
# Applied to the Viterbi fishing state and to every Pi_min posterior threshold.

suppressPackageStartupMessages({
  library(momentuHMM)
})

# ─────────────────────────────────────────────
#  PATHS AND SETTINGS
# ─────────────────────────────────────────────

project_root <- normalizePath(".")      # run from the project folder: Rscript pipeline_scripts/<name>.R
fit_rds      <- file.path(project_root, "pipeline_outputs/hmm_results/hmm_fit.rds")
states_csv   <- file.path(project_root, "pipeline_outputs/hmm_results/hmm_states.csv")
events_csv   <- file.path(project_root, "pipeline_outputs/hmm_results/hmm_states_events.csv")
sweep_csv    <- file.path(project_root, "pipeline_outputs/hmm_results/hmm_events_sweep.csv")

MIN_EVENT_FIXES <- 2                        # at least 2 consecutive fishing fixes per event
MAX_GAP_FIXES   <- 1                        # fill non-fishing gaps of this many fixes or fewer
PIMIN_VALUES    <- seq(0.5, 1.0, by = 0.1)
STEP_MINUTES    <- 5

# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────

mod <- readRDS(fit_rds)
states <- read.csv(states_csv, stringsAsFactors = FALSE)
states$time <- as.POSIXct(states$time, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
states <- states[order(states$ID, states$time), ]

# Same fishing-state rule as fit_hmm.R: the middle mean step length
state_mean <- as.numeric(getPar(mod)$Par$step[1:3])
fishing_state <- order(state_mean)[2]
cat(sprintf("Fishing state: %d (mean step %.3f km)\n", fishing_state, state_mean[fishing_state]))

# ─────────────────────────────────────────────
#  EVENT RULES
# ─────────────────────────────────────────────

# Fill interior non-fishing runs of length <= max_gap that have fishing on both sides
fill_gaps <- function(v, max_gap) {
  r <- rle(v)
  n <- length(r$values)
  if (n < 3) return(v)
  interior <- seq(2, n - 1)
  fill <- interior[!r$values[interior] & r$lengths[interior] <= max_gap]
  r$values[fill] <- TRUE
  inverse.rle(r)
}

# Relabel fishing runs shorter than min_fixes as non-fishing
drop_short <- function(v, min_fixes) {
  r <- rle(v)
  r$values[r$values & r$lengths < min_fixes] <- FALSE
  inverse.rle(r)
}

apply_rules <- function(label, id) {
  out <- label
  for (rows in split(seq_along(label), id)) {
    v <- fill_gaps(label[rows], MAX_GAP_FIXES)
    out[rows] <- drop_short(v, MIN_EVENT_FIXES)
  }
  out
}

# Number each event within the whole table (0 = not fishing)
event_ids <- function(label, id) {
  starts <- label & !c(FALSE, head(label, -1) & head(id, -1) == tail(id, -1))
  ifelse(label, cumsum(starts), 0L)
}

count_runs <- function(label, id) {
  sum(tapply(label, id, function(v) sum(diff(c(FALSE, v)) == 1)))
}

summarise <- function(name, pi_min, raw, event, id) {
  n_events <- count_runs(event, id)
  data.frame(
    label              = name,
    pi_min             = pi_min,
    raw_points         = sum(raw),
    raw_runs           = count_runs(raw, id),
    gap_fixes_filled   = sum(event & !raw),
    short_fixes_dropped = sum(raw & !event),
    event_points       = sum(event),
    event_share        = mean(event),
    event_hours        = sum(event) * STEP_MINUTES / 60,
    n_events           = n_events,
    mean_event_minutes = if (n_events > 0) sum(event) * STEP_MINUTES / n_events else NA
  )
}

# ─────────────────────────────────────────────
#  APPLY TO VITERBI AND EACH PI_MIN
# ─────────────────────────────────────────────

raw_vit <- states$viterbi_state == fishing_state
states$fishing_viterbi_event <- apply_rules(raw_vit, states$ID)
states$event_id_viterbi <- event_ids(states$fishing_viterbi_event, states$ID)
rows <- list(summarise("viterbi", NA, raw_vit, states$fishing_viterbi_event, states$ID))

prob_fishing <- states[[paste0("prob_state", fishing_state)]]
for (p in PIMIN_VALUES) {
  raw <- prob_fishing >= p
  event <- apply_rules(raw, states$ID)
  states[[sprintf("fishing_pimin%.1f_event", p)]] <- event
  rows[[length(rows) + 1]] <- summarise("pi_min", p, raw, event, states$ID)
}

write.csv(states, events_csv, row.names = FALSE)
cat("Saved", events_csv, "\n")

sweep <- do.call(rbind, rows)
write.csv(sweep, sweep_csv, row.names = FALSE)
print(sweep, digits = 4)
cat("Saved", sweep_csv, "\n")
