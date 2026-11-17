# Plot each HMM state's share of steps against the posterior threshold Pi_min.
#
# Input:  hmm_states.csv (from fit_hmm.R, step + turning angle, interpolated)
#         hmm_states_step_only.csv (from fit_hmm_step_only.R), if it exists
#         hmm_states_angle_only.csv (from fit_hmm_angle_only.R), if it exists
#         hmm_states_no_interp.csv (from fit_hmm_no_interp.R), if it exists
# Output: hmm_pimin_sweep.png         step only vs turning angle only
#         hmm_pimin_sweep_interp.png  interpolated vs no interpolation
# One panel per model, one line per state, with each state's fitted values in a table below.
# Models whose file is missing are left out.
#
# Run on its own (Rscript pipeline_scripts/plot_state_sweep.R) or sourced at the end of any fit script.

project_root <- normalizePath(".")      # run from the project folder: Rscript pipeline_scripts/<name>.R

PIMIN_VALUES <- seq(0.5, 0.9, by = 0.1)
STATE_COLORS <- c("#2a78d6", "#eb6834", "#1baf7a")   # colorblind-checked categorical order
STATE_PCH    <- c(19, 17, 15)                         # circle, triangle, square
KM_STEP_TO_KNOTS <- 12 / 1.852                        # 12 steps per hour, 1.852 km per nautical mile

# Share of positions (%) where each state's posterior probability is >= Pi_min
state_shares <- function(states_out) {
  sapply(1:3, function(s) {
    sapply(PIMIN_VALUES, function(p) mean(states_out[[paste0("prob_state", s)]] >= p) * 100)
  })
}

# Fitted values per state, from the model saved alongside each states file
state_table <- function(states_file) {
  mle <- readRDS(file.path(project_root, sub("hmm_states(.*)\\.csv", "hmm_fit\\1.rds", states_file)))$mle
  has_step  <- !is.null(mle$step)
  has_angle <- !is.null(mle$angle)
  mean_step <- if (has_step) mle$step["mean", ] else rep(NA, 3)
  sd_step   <- if (has_step) mle$step["sd", ] else rep(NA, 3)
  kappa     <- if (has_angle) mle$angle["concentration", ] else rep(NA, 3)

  # Same role rules as the fit scripts: rank by step length, or by turning with no step stream
  role <- character(3)
  if (has_step) {
    role[order(mean_step)] <- c("Near-stationary", "Fishing", "Transit")
  } else {
    role[order(kappa)] <- c("Fishing", "Some turning", "Very straight")
  }

  shown <- function(x, digits) ifelse(is.na(x), "-", formatC(x, format = "f", digits = digits))
  data.frame(role = role,
             speed = shown(mean_step * KM_STEP_TO_KNOTS, 2),
             spread = shown(sd_step * KM_STEP_TO_KNOTS, 2),
             turning = shown(kappa, 1))
}

# Draw a state table in the current plot region; values describe each state, they are not cutoffs
draw_table <- function(tab) {
  par(mar = c(0.5, 5, 0.5, 2))
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 5))
  col_x <- c(0.04, 0.16, 0.45, 0.64, 0.83)
  headers <- c("State", "Role", "Mean speed (kn)", "Speed SD (kn)", "Turning conc.")
  text(col_x, 4.5, headers, adj = 0, font = 2, col = "grey20", cex = 0.9)
  for (s in 1:3) {
    y <- 4.5 - s
    points(0, y, col = STATE_COLORS[s], pch = STATE_PCH[s], cex = 1.2)
    text(col_x, y, c(s, tab$role[s], tab$speed[s], tab$spread[s], tab$turning[s]),
         adj = 0, col = "grey20", cex = 0.9)
  }
  text(0, 0.4, "Fitted state values, not hard cutoffs. Turning conc.: high = straight, low = lots of turning.",
       adj = 0, col = "grey40", cex = 0.8)
}

plot_sweep <- function(models, out_png) {
  models <- models[file.exists(file.path(project_root, models))]
  if (length(models) == 0) return(invisible(NULL))

  panels <- lapply(models, function(f) {
    states_out <- read.csv(file.path(project_root, f), stringsAsFactors = FALSE)
    list(share = state_shares(states_out), n_steps = sum(!is.na(states_out$step_km)),
         table = state_table(f))
  })
  y_max <- max(sapply(panels, function(p) max(p$share))) * 1.05   # shared y-axis

  png(out_png, width = 1000 * length(panels), height = 860, res = 120)
  layout(matrix(seq_len(2 * length(panels)), nrow = 2), heights = c(3, 1))   # plot above table
  for (name in names(panels)) {
    share <- panels[[name]]$share
    par(mar = c(5, 5, 4, 2), las = 1)
    plot(NA, xlim = range(PIMIN_VALUES), ylim = c(0, y_max),
         xlab = "Pi_min (posterior threshold)",
         ylab = "Share of positions at or above threshold (%)",
         main = sprintf("%s (%s steps used)", name,
                        format(panels[[name]]$n_steps, big.mark = ",")),
         bty = "l", col.axis = "grey30")
    abline(h = pretty(c(0, y_max)), col = "grey90")
    for (s in 1:3) {
      lines(PIMIN_VALUES, share[, s], col = STATE_COLORS[s], lwd = 2)
      points(PIMIN_VALUES, share[, s], col = STATE_COLORS[s], pch = STATE_PCH[s], cex = 1.2)
    }
    legend("topright", legend = paste("State", 1:3), col = STATE_COLORS, pch = STATE_PCH,
           lwd = 2, bty = "n", text.col = "grey20")
    draw_table(panels[[name]]$table)
  }
  dev.off()
  cat("Saved", out_png, "\n")
}

plot_sweep(c("Step only"          = "pipeline_outputs/hmm_results/hmm_states_step_only.csv",
             "Turning angle only" = "pipeline_outputs/hmm_results/hmm_states_angle_only.csv"),
           file.path(project_root, "result_figures/hmm_pimin_sweep.png"))

if (file.exists(file.path(project_root, "pipeline_outputs/hmm_results/hmm_states_no_interp.csv"))) {
  plot_sweep(c("Interpolated (crawl)" = "pipeline_outputs/hmm_results/hmm_states.csv",
               "No interpolation"     = "pipeline_outputs/hmm_results/hmm_states_no_interp.csv"),
             file.path(project_root, "result_figures/hmm_pimin_sweep_interp.png"))
}
