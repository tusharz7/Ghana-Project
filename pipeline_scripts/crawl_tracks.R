# Regularise every boat-trip segment to 5-minute steps with crawl (crwMLE + crwPredict).
#
# Input:  trip_tracks.csv (from build_trip_tracks.py), one track per trip segment (track_id)
# Output: regularised_tracks.csv, plus a printed check that every step is 5 minutes.
#
# Each segment is fitted on its own so that separate trips are never joined together.
# crwMLE is called directly, not through crawlWrap, because crawlWrap hides the real
# error and fails segments that fit fine on their own.

suppressPackageStartupMessages({
  library(crawl)
})

# ─────────────────────────────────────────────
#  PATHS AND SETTINGS
# ─────────────────────────────────────────────

project_root <- normalizePath(".")      # run from the project folder: Rscript pipeline_scripts/<name>.R
input_csv    <- file.path(project_root, "pipeline_outputs/gps_tracks/trip_tracks.csv")
output_csv   <- file.path(project_root, "pipeline_outputs/gps_tracks/regularised_tracks.csv")

TIME_STEP        <- "5 min"  # regular step for predictions
MIN_POINTS       <- 10       # segments with fewer pings cannot be fitted reliably
STATIONARY_SD_M  <- 10       # segments whose x and y spread is below this (metres) are skipped
MAX_RETRY_PASSES <- 3        # extra passes over segments that failed to fit

# ─────────────────────────────────────────────
#  LOAD AND PREPARE
# ─────────────────────────────────────────────

tracks <- read.csv(input_csv, stringsAsFactors = FALSE)
tracks$timestamp <- as.POSIXct(tracks$timestamp, tz = "UTC")
tracks$ID <- tracks$track_id

# Drop duplicate timestamps inside a segment (crwMLE cannot use them)
tracks <- tracks[!duplicated(tracks[, c("ID", "timestamp")]), ]

track_sizes <- table(tracks$ID)
enough_ids  <- names(track_sizes)[track_sizes >= MIN_POINTS]
cat(sprintf("Segments: %d total, %d with >= %d points, %d too short (skipped)\n",
            length(track_sizes), length(enough_ids), MIN_POINTS,
            length(track_sizes) - length(enough_ids)))

tracks <- tracks[tracks$ID %in% enough_ids, c("ID", "boat_id", "trip_index", "returned_to_tema",
                                               "timestamp", "x", "y")]
tracks <- tracks[order(tracks$ID, tracks$timestamp), ]

# Near-stationary segments have no movement for the model to explain
spread_m <- sapply(split(tracks, tracks$ID), function(d) max(sd(d$x), sd(d$y)))
stationary_ids <- names(spread_m)[spread_m < STATIONARY_SD_M]
if (length(stationary_ids) > 0) {
  cat(sprintf("Skipping %d near-stationary segment(s): %s\n",
              length(stationary_ids), paste(stationary_ids, collapse = ", ")))
}
candidate_ids <- setdiff(enough_ids, stationary_ids)

# ─────────────────────────────────────────────
#  FIT AND PREDICT
# ─────────────────────────────────────────────

# Returns the 5-minute predicted positions for one segment, or NULL if the fit fails
fit_segment <- function(id) {
  track <- tracks[tracks$ID == id, c("ID", "timestamp", "x", "y")]
  names(track)[names(track) == "timestamp"] <- "time"

  # attempts = 10 retries from perturbed starting values, which fixes most optimiser failures
  fit <- tryCatch(
    suppressWarnings(suppressMessages(crwMLE(
      mov.model  = ~1,
      data       = track,
      coord      = c("x", "y"),
      Time.name  = "time",
      time.scale = "hours",
      attempts   = 10,
      need.hess  = TRUE
    ))),
    error = function(e) NULL
  )

  if (!inherits(fit, "crwFit") || !isTRUE(fit$convergence == 0) || any(is.na(fit$se))) {
    return(NULL)
  }

  grid <- seq(min(track$time), max(track$time), by = TIME_STEP)
  pred <- as.data.frame(crwPredict(fit, predTime = grid, return.type = "flat"))
  # Keep only the regular 5-minute predictions ("p"), not the original observed rows ("o")
  pred <- pred[pred$locType == "p", c("ID", "time", "mu.x", "mu.y", "speed")]
  names(pred)[names(pred) == "mu.x"] <- "x"
  names(pred)[names(pred) == "mu.y"] <- "y"
  pred
}

results <- list()
pending <- candidate_ids

for (pass in 0:MAX_RETRY_PASSES) {
  if (length(pending) == 0) break
  cat(sprintf("Pass %d: fitting %d segment(s)\n", pass, length(pending)))
  still_failing <- character(0)
  for (i in seq_along(pending)) {
    id   <- pending[i]
    pred <- fit_segment(id)
    if (is.null(pred)) {
      still_failing <- c(still_failing, id)
    } else {
      results[[id]] <- pred
    }
    if (i %% 100 == 0 || i == length(pending)) {
      cat(sprintf("  %d/%d done\n", i, length(pending)))
    }
  }
  pending <- still_failing
}
failed <- pending

# ─────────────────────────────────────────────
#  COMBINE, CHECK INTERVALS, SAVE
# ─────────────────────────────────────────────

regular <- do.call(rbind, results)
rownames(regular) <- NULL
regular$boat_id <- sub("_t[0-9]+_s[0-9]+$", "", regular$ID)
regular$trip_index <- as.integer(sub(".*_t([0-9]+)_s[0-9]+$", "\\1", regular$ID))
regular$segment <- as.integer(sub(".*_s([0-9]+)$", "\\1", regular$ID))
regular$returned_to_tema <- tracks$returned_to_tema[match(regular$ID, tracks$ID)]

write.csv(regular, output_csv, row.names = FALSE)
cat("Saved", output_csv, "\n")
cat("Rows:", nrow(regular), " Segments fitted:", length(unique(regular$ID)),
    " Stationary skipped:", length(stationary_ids),
    " Failed after retries:", length(failed), "\n")
if (length(failed) > 0) cat("Failed segments:\n", paste(" -", failed, collapse = "\n"), "\n", sep = "")

# Every step within a segment should be exactly 5 minutes
step_check <- do.call(rbind, lapply(split(regular, regular$ID), function(d) {
  steps <- as.numeric(difftime(d$time[-1], d$time[-nrow(d)], units = "mins"))
  data.frame(ID = d$ID[1], n_steps = length(steps),
             all_5min = all(abs(steps - 5) < 1e-6))
}))
cat("Segments where every step is 5 min:", sum(step_check$all_5min), "of", nrow(step_check), "\n")
