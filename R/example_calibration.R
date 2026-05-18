source("R/calibration_pairs.R")

# Example workflow:
# 1. Build a labeled calibration catalog of preferred and inferior schedules.
# 2. Sample the posterior for lambda and sigma using flat bounded priors on the
#    log scale.
# 3. Use posterior summaries inside the inner schedule optimization.

catalog <- build_paper_example_pair_catalog()

samples <- metropolis_calibrate(
  pairs = catalog$pairs,
  n_iter = 6000,
  burn_in = 1000,
  thin = 5,
  seed = 123
)

print(samples$acceptance_rate)
print(samples$posterior_means)

# Once you settle on posterior means or medians for lambda and sigma, use them
# in the inner schedule optimization:
best_fit <- optimize_theta(
  model = catalog$model,
  lambda = samples$posterior_means$lambda,
  sigma = samples$posterior_means$sigma,
  n_restarts = 40,
  maxit = 4000,
  seed = 123
)

print(best_fit$value)
print(best_fit$theta_round)
print(best_fit$constraints)
