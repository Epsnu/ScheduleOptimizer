source("R/schedule_model.R")

default_hyper_bounds <- function() {
  list(
    log_lambda_lower = rep(-6, 4),
    log_lambda_upper = rep(6, 4),
    log_sigma_lower = rep(log(0.02), 3),
    log_sigma_upper = rep(log(3.0), 3)
  )
}

unpack_eta <- function(eta) {
  stopifnot(length(eta) == 7)
  list(
    lambda = exp(eta[1:4]),
    sigma = exp(eta[5:7])
  )
}

log_prior_eta <- function(eta, bounds = default_hyper_bounds()) {
  lambda_eta <- eta[1:4]
  sigma_eta <- eta[5:7]

  lambda_ok <- all(lambda_eta >= bounds$log_lambda_lower) &&
    all(lambda_eta <= bounds$log_lambda_upper)
  sigma_ok <- all(sigma_eta >= bounds$log_sigma_lower) &&
    all(sigma_eta <= bounds$log_sigma_upper)

  if (!lambda_ok || !sigma_ok) {
    return(-Inf)
  }
  0
}

pair_log_likelihood <- function(model, preferred_theta, other_theta, eta, sharpness = 1) {
  hyper <- unpack_eta(eta)
  preferred_score <- evaluate_schedule(
    model,
    preferred_theta,
    lambda = hyper$lambda,
    sigma = hyper$sigma
  )
  other_score <- evaluate_schedule(
    model,
    other_theta,
    lambda = hyper$lambda,
    sigma = hyper$sigma
  )

  log(plogis(sharpness * (other_score - preferred_score)))
}

log_posterior_eta <- function(pairs, eta, sharpness = 1, bounds = default_hyper_bounds()) {
  lp <- log_prior_eta(eta, bounds = bounds)
  if (!is.finite(lp)) {
    return(-Inf)
  }

  total <- lp
  for (pair in pairs) {
    total <- total + pair_log_likelihood(
      model = pair$model,
      preferred_theta = pair$preferred_theta,
      other_theta = pair$other_theta,
      eta = eta,
      sharpness = sharpness
    )
  }
  total
}

metropolis_calibrate <- function(
  pairs,
  n_iter = 12000,
  burn_in = 2000,
  thin = 5,
  init_eta = c(rep(0, 4), log(c(0.5, 0.5, 0.5))),
  proposal_sd = c(rep(0.25, 4), rep(0.15, 3)),
  sharpness = 1,
  bounds = default_hyper_bounds(),
  seed = NULL
) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  eta <- init_eta
  current_lp <- log_posterior_eta(pairs, eta, sharpness = sharpness, bounds = bounds)
  draws <- matrix(NA_real_, nrow = n_iter, ncol = length(init_eta))
  accepted <- logical(n_iter)

  for (iter in seq_len(n_iter)) {
    proposal <- eta + rnorm(length(eta), mean = 0, sd = proposal_sd)
    proposal_lp <- log_posterior_eta(pairs, proposal, sharpness = sharpness, bounds = bounds)

    log_alpha <- proposal_lp - current_lp
    if (is.finite(proposal_lp) && log(runif(1)) < log_alpha) {
      eta <- proposal
      current_lp <- proposal_lp
      accepted[iter] <- TRUE
    }

    draws[iter, ] <- eta
  }

  kept_idx <- seq.int(from = burn_in + 1, to = n_iter, by = thin)
  kept <- draws[kept_idx, , drop = FALSE]
  colnames(kept) <- c(
    "log_lambda1", "log_lambda2", "log_lambda3", "log_lambda4",
    "log_sigma1", "log_sigma2", "log_sigma3"
  )

  lambda_draws <- exp(kept[, 1:4, drop = FALSE])
  sigma_draws <- exp(kept[, 5:7, drop = FALSE])
  colnames(lambda_draws) <- paste0("lambda", 1:4)
  colnames(sigma_draws) <- paste0("sigma", 1:3)

  list(
    draws = kept,
    lambda_draws = lambda_draws,
    sigma_draws = sigma_draws,
    acceptance_rate = mean(accepted),
    posterior_means = list(
      lambda = colMeans(lambda_draws),
      sigma = colMeans(sigma_draws)
    )
  )
}

make_preference_pair <- function(model, preferred_theta, other_theta) {
  list(
    model = model,
    preferred_theta = preferred_theta,
    other_theta = other_theta
  )
}
