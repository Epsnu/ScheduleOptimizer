import unittest
from pathlib import Path

import numpy as np

from schedule import Schedule


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(
            FIXTURES / "prefs_paper_example.csv",
            FIXTURES / "shifts_paper_example.csv",
        )

    def test_constructor_builds_expected_shapes(self):
        self.assertEqual(self.schedule.n, 5)
        self.assertEqual(self.schedule.k, 4)
        self.assertEqual(self.schedule.x, 2)
        self.assertEqual(self.schedule.P.shape, (5, 4))
        self.assertEqual(self.schedule.S.shape, (4, 2))

    def test_missing_shift_metadata_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Missing shift metadata for: s3"):
            Schedule(
                FIXTURES / "prefs_missing_shift.csv",
                FIXTURES / "shifts_missing_shift.csv",
            )

    def test_p_hat_supports_zero_sentinel_and_interpolation(self):
        self.assertEqual(self.schedule.P_hat_i(0, 0.0), 0.0)
        self.assertEqual(self.schedule.P_hat_i(0, 1.0), 0.0)
        self.assertEqual(self.schedule.P_hat_i(0, 4.0), 10.0)
        self.assertEqual(self.schedule.P_hat_i(1, 1.5), 1.0)

    def test_objective_matches_paper_example_schedule(self):
        theta = np.array(
            [
                [4.0, 4.0],
                [0.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [1.0, 1.0],
            ]
        )
        self.assertEqual(self.schedule.f(theta), -48.0)

    def test_valid_integer_schedule_has_tiny_penalties_with_small_sigma(self):
        theta = np.array(
            [
                [4.0, 4.0],
                [0.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [1.0, 1.0],
            ]
        )
        sigma = 0.05

        self.assertLess(self.schedule.Penalty_C1(theta, sigma), 1e-8)
        self.assertLess(self.schedule.Penalty_C2(theta, sigma), 1e-8)
        self.assertEqual(self.schedule.Penalty_C3(theta), 0.0)
        self.assertLess(self.schedule.Penalty_C4(theta, sigma), 1e-8)

    def test_split_shift_is_penalized(self):
        theta = np.array(
            [
                [4.0, 0.0],
                [4.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [1.0, 1.0],
            ]
        )
        sigma = 0.05

        self.assertGreater(self.schedule.Penalty_C4(theta, sigma), 1.0)
        self.assertLess(self.schedule.Penalty_C2(theta, sigma), 1e-8)

    def test_unavailable_shift_is_penalized_by_c3(self):
        theta = np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        self.assertGreater(self.schedule.Penalty_C3(theta), 0.0)

    def test_evaluate_rejects_bad_theta_shape(self):
        with self.assertRaisesRegex(ValueError, "theta must have shape"):
            self.schedule.evaluate(
                np.zeros((self.schedule.n, self.schedule.x + 1)),
                lambda_arr=[1, 1, 1, 1],
                sigma_arr=[0.1, 0.1, 0.1],
            )


if __name__ == "__main__":
    unittest.main()
