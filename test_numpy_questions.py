# ##################################################
# YOU SHOULD NOT TOUCH THIS FILE !
# ##################################################

import math as m
import numpy as np

import pytest

from numpy_questions import wallis_product, max_index


def test_max_index():
    X = np.array([[0, 1], [2, 0]])
    assert max_index(X) == (1, 0)

    X = np.random.randn(100, 100)
    i, j = max_index(X)
    assert np.all(X[i, j] >= X)

    with pytest.raises(ValueError):
        max_index(None)

    with pytest.raises(ValueError):
        max_index([[0, 1], [2, 0]])

    with pytest.raises(ValueError):
        max_index(np.array([0, 1]))


def test_wallis_product():
    pi_approx = wallis_product(0)
    assert pi_approx == 2.

    pi_approx = wallis_product(1)
    assert pi_approx == 8 / 3

    pi_approx = wallis_product(100000)
    assert abs(pi_approx - m.pi) < 1e-4
