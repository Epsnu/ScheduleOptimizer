load_schedule_data <- function(pref_file, shift_file) {
  prefs <- read.csv(pref_file, stringsAsFactors = FALSE, check.names = FALSE)
  shifts <- read.csv(shift_file, stringsAsFactors = FALSE, check.names = FALSE)

  shift_names <- colnames(prefs)[-1]
  shift_lookup <- shifts[!duplicated(shifts$name), c("name", "hours", "num_workers")]
  rownames(shift_lookup) <- shift_lookup$name

  missing <- setdiff(shift_names, shift_lookup$name)
  if (length(missing) > 0) {
    warning(
      sprintf("Missing shift metadata for: %s", paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }

  S <- matrix(0, nrow = length(shift_names), ncol = 2)
  colnames(S) <- c("hours", "num_workers")
  for (idx in seq_along(shift_names)) {
    shift_name <- shift_names[idx]
    if (shift_name %in% shift_lookup$name) {
      S[idx, ] <- as.numeric(shift_lookup[shift_name, c("hours", "num_workers")])
    }
  }

  P <- as.matrix(prefs[, -1, drop = FALSE])
  storage.mode(P) <- "double"

  total_required_hours <- sum(S[, 1] * S[, 2])
  max_shift_hours <- if (nrow(S) > 0) max(S[, 1]) else 0
  x <- as.integer(max(ceiling(total_required_hours / nrow(P)), max_shift_hours))

  list(
    member_names = prefs[[1]],
    shift_names = shift_names,
    n = nrow(P),
    k = ncol(P),
    P = P,
    S = S,
    x = x
  )
}

clip_shift_value <- function(model, s) {
  min(max(as.numeric(s), 0), model$k)
}

P_hat_i <- function(model, i, s) {
  s <- clip_shift_value(model, s)
  left <- floor(s)
  right <- ceiling(s)
  alpha <- s - left

  left_pref <- if (left == 0) 0 else model$P[i, left]
  right_pref <- if (right == 0) 0 else model$P[i, right]

  (1 - alpha) * left_pref + alpha * right_pref
}

objective_f <- function(model, theta) {
  total <- 0
  for (i in seq_len(model$n)) {
    for (j in seq_len(model$x)) {
      total <- total - P_hat_i(model, i, theta[i, j])
    }
  }
  total
}

kernel_K <- function(sigma, phi) {
  if (sigma <= 0) {
    stop("sigma must be positive.")
  }
  exp(-(phi ^ 2) / (2 * sigma * sigma))
}

C_im <- function(model, theta, m, i, sigma) {
  coverage <- 0
  for (j in seq_len(model$x)) {
    coverage <- coverage + kernel_K(sigma, theta[i, j] - m)
  }
  coverage
}

C_m <- function(model, theta, m, sigma) {
  coverage <- 0
  for (i in seq_len(model$n)) {
    coverage <- coverage + C_im(model, theta, m, i, sigma)
  }
  coverage
}

penalty_C1 <- function(model, theta, sigma) {
  penalty <- 0
  for (m in seq_len(model$k)) {
    required_coverage <- model$S[m, 1] * model$S[m, 2]
    penalty <- penalty + (C_m(model, theta, m, sigma) - required_coverage) ^ 2
  }
  penalty
}

penalty_C2 <- function(model, theta, sigma) {
  penalty <- 0
  for (m in seq_len(model$k)) {
    shift_hours <- model$S[m, 1]
    for (i in seq_len(model$n)) {
      penalty <- penalty + max(0, C_im(model, theta, m, i, sigma) - shift_hours) ^ 2
    }
  }
  penalty
}

penalty_C3 <- function(model, theta, tau = 1) {
  penalty <- 0
  for (i in seq_len(model$n)) {
    for (j in seq_len(model$x)) {
      shift_value <- clip_shift_value(model, theta[i, j])
      if (shift_value == 0) {
        next
      }
      penalty <- penalty + max(0, tau - P_hat_i(model, i, shift_value)) ^ 2
    }
  }
  penalty
}

penalty_C4 <- function(model, theta, sigma) {
  penalty <- 0
  for (m in seq_len(model$k)) {
    shift_hours <- model$S[m, 1]
    for (i in seq_len(model$n)) {
      coverage <- C_im(model, theta, m, i, sigma)
      penalty <- penalty + coverage ^ 2 * (coverage - shift_hours) ^ 2
    }
  }
  penalty
}

evaluate_schedule <- function(model, theta, lambda, sigma, tau = 1) {
  stopifnot(length(lambda) == 4)
  stopifnot(length(sigma) == 3)
  stopifnot(all(dim(theta) == c(model$n, model$x)))

  objective_f(model, theta) +
    lambda[1] * penalty_C1(model, theta, sigma[1]) +
    lambda[2] * penalty_C2(model, theta, sigma[2]) +
    lambda[3] * penalty_C3(model, theta, tau = tau) +
    lambda[4] * penalty_C4(model, theta, sigma[3])
}

round_theta <- function(theta) {
  round(theta)
}

make_empty_theta <- function(model) {
  matrix(0, nrow = model$n, ncol = model$x)
}

count_empty_slots <- function(theta_row) {
  sum(theta_row == 0)
}

initial_theta_greedy <- function(model, randomize = TRUE) {
  theta <- make_empty_theta(model)
  remaining_capacity <- rep(model$x, model$n)

  shift_order <- seq_len(model$k)
  if (randomize && length(shift_order) > 1) {
    shift_order <- sample(shift_order)
  }

  for (m in shift_order) {
    shift_hours <- as.integer(model$S[m, 1])
    workers_needed <- as.integer(model$S[m, 2])

    if (shift_hours <= 0 || workers_needed <= 0) {
      next
    }

    pref_scores <- model$P[, m]
    worker_order <- order(
      pref_scores > 0,
      pref_scores,
      remaining_capacity,
      runif(model$n),
      decreasing = TRUE
    )

    chosen_workers <- integer(0)
    for (worker_idx in worker_order) {
      if (remaining_capacity[worker_idx] >= shift_hours) {
        chosen_workers <- c(chosen_workers, worker_idx)
      }
      if (length(chosen_workers) == workers_needed) {
        break
      }
    }

    if (length(chosen_workers) < workers_needed) {
      fallback_order <- order(remaining_capacity, runif(model$n), decreasing = TRUE)
      for (worker_idx in fallback_order) {
        if (!(worker_idx %in% chosen_workers) && remaining_capacity[worker_idx] >= shift_hours) {
          chosen_workers <- c(chosen_workers, worker_idx)
        }
        if (length(chosen_workers) == workers_needed) {
          break
        }
      }
    }

    for (worker_idx in chosen_workers) {
      empty_slots <- which(theta[worker_idx, ] == 0)
      slots_to_fill <- head(empty_slots, shift_hours)
      if (length(slots_to_fill) == shift_hours) {
        theta[worker_idx, slots_to_fill] <- m
        remaining_capacity[worker_idx] <- remaining_capacity[worker_idx] - shift_hours
      }
    }
  }

  theta
}

random_theta <- function(model) {
  theta <- make_empty_theta(model)
  for (i in seq_len(model$n)) {
    available <- which(model$P[i, ] > 0)
    pool <- c(0, available)
    theta[i, ] <- sample(pool, size = model$x, replace = TRUE)
  }
  theta
}

check_discrete_constraints <- function(model, theta_round) {
  stopifnot(all(dim(theta_round) == c(model$n, model$x)))

  shift_counts <- numeric(model$k)
  unique_worker_counts <- numeric(model$k)
  atomic_ok <- logical(model$k)
  availability_ok <- TRUE

  for (m in seq_len(model$k)) {
    shift_counts[m] <- sum(theta_round == m)
    unique_worker_counts[m] <- sum(apply(theta_round == m, 1, any))
    worker_hours <- apply(theta_round == m, 1, sum)
    atomic_ok[m] <- any(worker_hours == model$S[m, 1] | model$S[m, 1] == 0)
  }

  for (i in seq_len(model$n)) {
    for (j in seq_len(model$x)) {
      assigned <- theta_round[i, j]
      if (assigned == 0) {
        next
      }
      if (model$P[i, assigned] <= 0) {
        availability_ok <- FALSE
      }
    }
  }

  list(
    c1_ok = all(shift_counts == model$S[, 1] * model$S[, 2]),
    c2_ok = all(unique_worker_counts == model$S[, 2]),
    c3_ok = availability_ok,
    c4_ok = all(atomic_ok)
  )
}

is_discrete_valid <- function(model, theta_round) {
  constraints <- check_discrete_constraints(model, theta_round)
  all(unlist(constraints))
}

is_better_solution <- function(model, candidate_theta, candidate_value, incumbent_theta, incumbent_value) {
  if (is.null(incumbent_theta)) {
    return(TRUE)
  }

  candidate_valid <- is_discrete_valid(model, candidate_theta)
  incumbent_valid <- is_discrete_valid(model, incumbent_theta)

  if (candidate_valid && !incumbent_valid) {
    return(TRUE)
  }
  if (!candidate_valid && incumbent_valid) {
    return(FALSE)
  }

  candidate_value < incumbent_value
}

propose_theta_neighbor <- function(model, theta) {
  proposal <- theta
  move_type <- sample(c("swap", "reassign", "relocate"), size = 1)

  if (move_type == "swap") {
    idx_a <- c(sample.int(model$n, 1), sample.int(model$x, 1))
    idx_b <- c(sample.int(model$n, 1), sample.int(model$x, 1))
    temp <- proposal[idx_a[1], idx_a[2]]
    proposal[idx_a[1], idx_a[2]] <- proposal[idx_b[1], idx_b[2]]
    proposal[idx_b[1], idx_b[2]] <- temp
    return(proposal)
  }

  if (move_type == "reassign") {
    worker_idx <- sample.int(model$n, 1)
    slot_idx <- sample.int(model$x, 1)
    available <- which(model$P[worker_idx, ] > 0)
    pool <- unique(c(0, available, seq_len(model$k)))
    proposal[worker_idx, slot_idx] <- sample(pool, 1)
    return(proposal)
  }

  nonzero_locations <- which(proposal != 0, arr.ind = TRUE)
  zero_locations <- which(proposal == 0, arr.ind = TRUE)
  if (nrow(nonzero_locations) > 0 && nrow(zero_locations) > 0) {
    src_idx <- nonzero_locations[sample.int(nrow(nonzero_locations), 1), ]
    dst_idx <- zero_locations[sample.int(nrow(zero_locations), 1), ]
    proposal[dst_idx[1], dst_idx[2]] <- proposal[src_idx[1], src_idx[2]]
    proposal[src_idx[1], src_idx[2]] <- 0
  }
  proposal
}

optimize_theta <- function(
  model,
  lambda,
  sigma,
  tau = 1,
  n_restarts = 10,
  maxit = 5000,
  temp_start = 5,
  temp_end = 0.01,
  seed = NULL
) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  score_theta <- function(theta) {
    evaluate_schedule(model, theta, lambda = lambda, sigma = sigma, tau = tau)
  }

  best <- list(value = Inf, theta = NULL, theta_round = NULL, constraints = NULL)
  total_iterations <- max(1L, as.integer(maxit))

  for (restart_idx in seq_len(n_restarts)) {
    current_theta <- if (restart_idx %% 2 == 1) {
      initial_theta_greedy(model, randomize = TRUE)
    } else {
      random_theta(model)
    }
    current_value <- score_theta(current_theta)

    if (is_better_solution(model, current_theta, current_value, best$theta, best$value)) {
      best$theta <- current_theta
      best$value <- current_value
      best$theta_round <- current_theta
      best$constraints <- check_discrete_constraints(model, current_theta)
    }

    for (iter_idx in seq_len(total_iterations)) {
      temp_fraction <- if (total_iterations == 1) 0 else (iter_idx - 1) / (total_iterations - 1)
      temperature <- temp_start * ((temp_end / temp_start) ^ temp_fraction)

      candidate_theta <- propose_theta_neighbor(model, current_theta)
      candidate_value <- score_theta(candidate_theta)
      delta <- candidate_value - current_value

      accept <- FALSE
      if (delta <= 0) {
        accept <- TRUE
      } else if (temperature > 0 && runif(1) < exp(-delta / temperature)) {
        accept <- TRUE
      }

      if (accept) {
        current_theta <- candidate_theta
        current_value <- candidate_value
      }

      if (is_better_solution(model, current_theta, current_value, best$theta, best$value)) {
        best$theta <- current_theta
        best$value <- current_value
        best$theta_round <- current_theta
        best$constraints <- check_discrete_constraints(model, current_theta)
      }

      if (is_discrete_valid(model, current_theta)) {
        best$theta <- current_theta
        best$value <- current_value
        best$theta_round <- current_theta
        best$constraints <- check_discrete_constraints(model, current_theta)
        break
      }
    }
  }

  best$method <- "discrete_simulated_annealing"
  best
}
