import numpy as np
import pandas as pd
from io import StringIO
from contextlib import redirect_stdout


class Schedule:
    def __init__(self, pref_file, shift_file):
        # Load CSV files.
        df_prefs = pd.read_csv(pref_file)
        df_shifts = pd.read_csv(shift_file)

        self.member_names = df_prefs["name"].to_numpy()
        self.shift_names = df_prefs.columns[1:].to_numpy()

        self.n = df_prefs.shape[0]
        self.k = len(self.shift_names)

        self.P = df_prefs.iloc[:, 1:].to_numpy(dtype=float)

        shift_lookup = (
            df_shifts.drop_duplicates(subset="name", keep="first")
            .set_index("name")[["hours", "num_workers"]]
        )
        missing_shifts = [name for name in self.shift_names if name not in shift_lookup.index]
        shift_rows = []
        if missing_shifts:
            missing = ", ".join(missing_shifts)
            print(f"Warning: missing shift metadata for: {missing}\n")
        for name in self.shift_names:
            if name in shift_lookup.index:
                shift_rows.append(shift_lookup.loc[name].to_numpy(dtype=float))
            else:
                shift_rows.append(np.array([0.0, 0.0]))

        self.S = np.vstack(shift_rows) if shift_rows else np.zeros((0, 2), dtype=float)

        total_required_hours = float(np.sum(self.S[:, 0] * self.S[:, 1]))
        max_shift_hours = float(np.max(self.S[:, 0])) if self.k else 0.0
        self.x = int(max(np.ceil(total_required_hours / self.n), max_shift_hours))

    def evaluate(self, theta, lambda_arr, sigma_arr):
        theta = self._validate_theta(theta)
        if len(lambda_arr) != 4:
            raise ValueError("lambda_arr must contain 4 penalty weights.")
        if len(sigma_arr) != 3:
            raise ValueError("sigma_arr must contain 3 sigma values.")

        objective = self.f(theta)
        objective += lambda_arr[0] * self.Penalty_C1(theta, sigma_arr[0])
        objective += lambda_arr[1] * self.Penalty_C2(theta, sigma_arr[1])
        objective += lambda_arr[2] * self.Penalty_C3(theta)
        objective += lambda_arr[3] * self.Penalty_C4(theta, sigma_arr[2])
        return float(objective)

    def _validate_theta(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n, self.x):
            raise ValueError(f"theta must have shape {(self.n, self.x)}.")
        return theta

    def _clip_shift_value(self, s):
        return float(np.clip(s, 0.0, float(self.k)))

    # helper P_hat function
    def P_hat_i(self, i, s):
        s = self._clip_shift_value(s)
        left = int(np.floor(s))
        right = int(np.ceil(s))
        alpha = s - left

        left_pref = 0.0 if left == 0 else self.P[i, left - 1]
        right_pref = 0.0 if right == 0 else self.P[i, right - 1]
        return float((1.0 - alpha) * left_pref + alpha * right_pref)

    # cost function
    def f(self, theta):
        total = 0.0
        for i in range(self.n):
            for j in range(self.x):
                total -= self.P_hat_i(i, theta[i, j])
        return float(total)

    @staticmethod
    def K(sigma, phi):
        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        return float(np.exp(-(phi ** 2) / (2.0 * sigma * sigma)))

    # Global shift coverage score C_m(theta).
    def C_m(self, theta, m, sigma):
        coverage = 0.0
        for i in range(self.n):
            coverage += self.C_im(theta, m, i, sigma)
        return float(coverage)

    # Per-worker shift coverage score C_{i,m}(theta).
    def C_im(self, theta, m, i, sigma):
        coverage = 0.0
        for j in range(self.x):
            coverage += self.K(sigma, theta[i, j] - m)
        return float(coverage)

    # penalty C1
    def Penalty_C1(self, theta, sigma):
        penalty = 0.0
        for m in range(1, self.k + 1):
            required_coverage = self.S[m - 1, 0] * self.S[m - 1, 1]
            penalty += (self.C_m(theta, m, sigma) - required_coverage) ** 2
        return float(penalty)

    # penalty C2
    def Penalty_C2(self, theta, sigma):
        penalty = 0.0
        for m in range(1, self.k + 1):
            shift_hours = self.S[m - 1, 0]
            for i in range(self.n):
                penalty += max(0.0, self.C_im(theta, m, i, sigma) - shift_hours) ** 2
        return float(penalty)

    # penalty C3
    def Penalty_C3(self, theta, tau=1):
        penalty = 0.0
        for i in range(self.n):
            for j in range(self.x):
                shift_value = self._clip_shift_value(theta[i, j])
                if shift_value == 0.0:
                    continue
                penalty += max(0.0, tau - self.P_hat_i(i, shift_value)) ** 2
        return float(penalty)

    # penalty C4
    def Penalty_C4(self, theta, sigma):
        penalty = 0.0
        for m in range(1, self.k + 1):
            shift_hours = self.S[m - 1, 0]
            for i in range(self.n):
                coverage = self.C_im(theta, m, i, sigma)
                penalty += coverage ** 2 * (coverage - shift_hours) ** 2
        return float(penalty)

    # Normalize rows by mean of row.
    def normalizePrefs(self):
        row_means = self.P.mean(axis=1, keepdims=True)
        if np.any(row_means == 0):
            raise ValueError("Cannot normalize preferences for a worker with all-zero preferences.")
        self.P = self.P / row_means
