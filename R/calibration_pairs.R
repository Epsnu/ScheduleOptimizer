source("R/schedule_model.R")
source("R/bayesian_calibration.R")

theta_from_values <- function(model, values) {
  matrix(values, nrow = model$n, ncol = model$x, byrow = TRUE)
}

describe_theta <- function(model, theta, lambda = rep(1, 4), sigma = rep(0.1, 3)) {
  rounded <- round_theta(theta)
  list(
    theta = theta,
    theta_round = rounded,
    objective = objective_f(model, theta),
    penalty_C1 = penalty_C1(model, theta, sigma[1]),
    penalty_C2 = penalty_C2(model, theta, sigma[2]),
    penalty_C3 = penalty_C3(model, theta),
    penalty_C4 = penalty_C4(model, theta, sigma[3]),
    total = evaluate_schedule(model, theta, lambda = lambda, sigma = sigma),
    constraints = check_discrete_constraints(model, rounded)
  )
}

build_paper_example_pair_catalog <- function() {
  model <- load_schedule_data(
    "tests/fixtures/prefs_paper_example.csv",
    "tests/fixtures/shifts_paper_example.csv"
  )

  preferred <- theta_from_values(
    model,
    c(
      4, 4,
      0, 0,
      2, 0,
      3, 0,
      1, 1
    )
  )

  split_shift <- theta_from_values(
    model,
    c(
      4, 0,
      4, 0,
      2, 0,
      3, 0,
      1, 1
    )
  )

  unavailable_assignment <- theta_from_values(
    model,
    c(
      1, 1,
      0, 0,
      2, 0,
      3, 0,
      4, 4
    )
  )

  undercovered <- theta_from_values(
    model,
    c(
      4, 4,
      0, 0,
      2, 0,
      0, 0,
      1, 1
    )
  )

  overcovered <- theta_from_values(
    model,
    c(
      4, 4,
      3, 0,
      2, 0,
      3, 0,
      1, 1
    )
  )

  fractional_near_valid <- theta_from_values(
    model,
    c(
      3.92, 4.08,
      0.00, 0.02,
      1.96, 0.04,
      3.04, 0.00,
      1.10, 0.94
    )
  )

  list(
    model = model,
    schedules = list(
      preferred = preferred,
      split_shift = split_shift,
      unavailable_assignment = unavailable_assignment,
      undercovered = undercovered,
      overcovered = overcovered,
      fractional_near_valid = fractional_near_valid
    ),
    pairs = list(
      make_preference_pair(model, preferred, split_shift),
      make_preference_pair(model, preferred, unavailable_assignment),
      make_preference_pair(model, preferred, undercovered),
      make_preference_pair(model, preferred, overcovered),
      make_preference_pair(model, preferred, fractional_near_valid)
    )
  )
}

print_pair_catalog_summary <- function(catalog, lambda = rep(1, 4), sigma = rep(0.1, 3)) {
  for (name in names(catalog$schedules)) {
    cat("\n===", name, "===\n")
    summary_row <- describe_theta(catalog$model, catalog$schedules[[name]], lambda = lambda, sigma = sigma)
    print(summary_row$theta_round)
    print(summary_row$constraints)
    cat(
      sprintf(
        "objective=%.4f, C1=%.4f, C2=%.4f, C3=%.4f, C4=%.4f, total=%.4f\n",
        summary_row$objective,
        summary_row$penalty_C1,
        summary_row$penalty_C2,
        summary_row$penalty_C3,
        summary_row$penalty_C4,
        summary_row$total
      )
    )
  }
}
