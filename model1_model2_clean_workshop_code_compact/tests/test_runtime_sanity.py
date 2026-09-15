"""The Stage-1 runtime derivative check: relative error only away from zero."""
from __future__ import annotations

import numpy as np

from model1.runtime import AmortizedScoreRuntime

errors = AmortizedScoreRuntime.finite_difference_errors


def test_small_absolute_error_near_zero_does_not_inflate_relative_error():
    # The failing case: an absolute error of 1e-4 on a derivative near 0.006.
    absolute, relative = errors(np.array([0.0061, 2.0]), np.array([0.0060, 2.0]))
    assert np.isclose(absolute, 1e-4)
    assert relative == 0.0


def test_relative_error_is_still_checked_away_from_zero():
    absolute, relative = errors(np.array([0.51, -3.0]), np.array([0.50, -3.0]))
    assert np.isclose(absolute, 0.01)
    assert np.isclose(relative, 0.02)


def test_all_derivatives_near_zero_leave_only_the_absolute_error():
    absolute, relative = errors(np.array([0.01, -0.02]), np.array([0.0, 0.0]))
    assert np.isclose(absolute, 0.02)
    assert relative == 0.0
