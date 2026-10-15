"""Gruppe: Gamma-Zufallszahlen, Dirichlet, Gewichte der Mitglieder, Nutzenformen, Skala und Macht."""

import numpy as np
import pytest

import gru_group as gp
from entscheidungsmatrix_generator import SplitMix64, make_matrix


def test_gamma_variates_have_mean_and_variance_equal_to_the_shape():
    for shape in (0.3, 1.0, 2.5):
        rng = SplitMix64(7)
        x = np.array([gp.gamma_variate(shape, rng) for _ in range(20000)])
        assert x.mean() == pytest.approx(shape, rel=0.05) and x.var() == pytest.approx(shape, rel=0.1)


def test_dirichlet_sums_to_one_is_positive_and_deterministic():
    a = gp.dirichlet(np.array([0.5, 1.0, 2.0]), SplitMix64(3))
    assert a.sum() == pytest.approx(1.0) and (a >= gp.WEIGHT_FLOOR / 2).all()
    assert (a == gp.dirichlet(np.array([0.5, 1.0, 2.0]), SplitMix64(3))).all()
    tiny = gp.dirichlet(np.array([1e-6, 1.0]), SplitMix64(1))
    assert tiny.min() > 0


def test_members_scatter_more_for_a_small_kappa():
    def spread(kappa):
        out = []
        for seed in range(60):
            center, W = gp.member_weights(5, 4, kappa, SplitMix64(seed))
            out.append(np.abs(W - center).sum(axis=1).mean() / 2)
        return float(np.mean(out))

    assert spread(3.0) > spread(10.0) > spread(100.0)
    assert spread(100.0) < 0.1


def test_member_weights_shapes_and_normalisation():
    center, W = gp.member_weights(7, 5, 10.0, SplitMix64(2))
    assert center.shape == (5,) and W.shape == (7, 5)
    assert center.sum() == pytest.approx(1.0) and W.sum(axis=1) == pytest.approx(np.ones(7))


def test_normalized_values_use_the_nominal_range():
    x = make_matrix(5, 5, 0.0, SplitMix64(1))
    V = gp.normalized_values(x.signed)
    assert V.shape == (5, 5) and V.min() >= -1e-9 and V.max() <= 1 + 1e-9


def test_equal_form_utilities_are_linear_in_the_normalised_values():
    x = make_matrix(8, 5, 0.0, SplitMix64(4))
    V = gp.normalized_values(x.signed)
    _, W = gp.member_weights(4, 5, 10.0, SplitMix64(5))
    U = gp.member_utilities(V, np.asarray(x.goodness), W, "gleich", SplitMix64(6))
    assert U == pytest.approx(W @ V.T)


def test_mixed_forms_give_every_kind_and_stay_in_zero_one():
    kinds = set()
    for seed in range(20):
        kinds |= set(gp.member_forms(5, "gemischt", SplitMix64(seed)))
    assert kinds == {0, 1, 2}
    assert gp.member_forms(4, "gleich", SplitMix64(1)) == [0, 0, 0, 0]
    x = make_matrix(8, 5, 0.0, SplitMix64(4))
    V = gp.normalized_values(x.signed)
    _, W = gp.member_weights(6, 5, 10.0, SplitMix64(5))
    U = gp.member_utilities(V, np.asarray(x.goodness), W, "gemischt", SplitMix64(9))
    assert U.shape == (6, 8) and U.min() > 0 and U.max() <= 1.0 + 1e-9


def test_scale_changes_only_member_zero_and_power_normalises():
    U = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert gp.apply_scale(U, 10.0).tolist() == [[10.0, 20.0], [3.0, 4.0]]
    assert U.tolist() == [[1.0, 2.0], [3.0, 4.0]]                         # das Original bleibt
    p = gp.member_power(4, 3.0)
    assert p.sum() == pytest.approx(1.0) and p[0] / p[1] == pytest.approx(3.0) and p[1] == p[2] == p[3]
