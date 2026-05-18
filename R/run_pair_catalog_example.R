source("R/calibration_pairs.R")

catalog <- build_paper_example_pair_catalog()

print_pair_catalog_summary(
  catalog,
  lambda = c(10, 10, 10, 10),
  sigma = c(0.1, 0.1, 0.1)
)

samples <- metropolis_calibrate(
  pairs = catalog$pairs,
  n_iter = 8000,
  burn_in = 2000,
  thin = 5,
  seed = 123
)

cat("\nAcceptance rate:\n")
print(samples$acceptance_rate)

cat("\nPosterior means:\n")
print(samples$posterior_means)

best_fit <- optimize_theta(
  model = catalog$model,
  lambda = samples$posterior_means$lambda,
  sigma = samples$posterior_means$sigma,
  n_restarts = 40,
  maxit = 4000,
  seed = 123
)

cat("\nOptimized rounded schedule:\n")
print(best_fit$theta_round)

cat("\nConstraint check:\n")
print(best_fit$constraints)
