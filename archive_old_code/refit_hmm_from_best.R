# Refit the saved 3-state HMM from its own estimates, to reach a true optimum.
#
# Why: the saved fit stopped with a large gradient and a negative-curvature Hessian
# entry for state 3's mean, so its standard errors could not be inverted.
#
# Output: hmm_fit_refit.rds and a printed check of the gradient, Hessian and CIbeta.

suppressPackageStartupMessages({
  library(momentuHMM)
})

project_root <- normalizePath(".")
old_fit  <- readRDS(file.path(project_root, "hmm_fit.rds"))
new_rds  <- file.path(project_root, "hmm_fit_refit.rds")

# Reuse the same prepared data as the original fit
regular <- read.csv(file.path(project_root, "regularised_tracks.csv"), stringsAsFactors = FALSE)
regular$time <- as.POSIXct(regular$time, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
regular <- regular[order(regular$ID, regular$time), ]
n_rows <- table(regular$ID)
regular <- regular[regular$ID %in% names(n_rows)[n_rows >= 21], ]
regular$x <- regular$x / 1000
regular$y <- regular$y / 1000
data <- prepData(regular, type = "UTM", coordNames = c("x", "y"))
data$step[which(data$step == 0)] <- 0.001

# Start from the saved estimates on the natural scale (step, angle) and working scale (transitions)
pars <- getPar(old_fit)
est  <- old_fit$mod$estimate              # working scale: step 1-6, angle 7-9, transitions 10-15, delta 16-17
beta0  <- matrix(est[10:15], nrow = 1)    # one row, six transition coefficients (order as in CIbeta)
delta0 <- as.numeric(old_fit$mle$delta[1, ])  # natural scale, shared by all segments
cat("Starting from saved estimates\n")

refit <- fitHMM(
  data,
  nbStates     = 3,
  dist         = list(step = "gamma", angle = "vm"),
  Par0         = pars$Par,
  beta0        = beta0,
  delta0       = delta0,
  estAngleMean = list(angle = FALSE),
  retryFits    = 0,
  nlmPar       = list(iterlim = 2000)
)

saveRDS(refit, new_rds)
cat("Saved", new_rds, "\n")
cat("Optimiser code:", refit$mod$code, " iterations:", refit$mod$iterations, "\n")
cat("Negative log-likelihood:", refit$mod$minimum, " (old:", old_fit$mod$minimum, ")\n")
cat("Max |gradient|:", max(abs(refit$mod$gradient)), " (old:", max(abs(old_fit$mod$gradient)), ")\n")

H <- refit$mod$hessian
cat("Hessian NA:", sum(is.na(H)), " Inf:", sum(is.infinite(H)), "\n")
if (!anyNA(H) && all(is.finite(H))) {
  ev <- eigen(H, symmetric = TRUE, only.values = TRUE)$values
  cat("Hessian eigenvalues min/max:", signif(min(ev), 4), signif(max(ev), 4), "\n")
}
ci <- CIbeta(refit)
cat("CIbeta standard errors missing:", sum(is.na(unlist(lapply(ci, function(e) e$se)))), "\n")
