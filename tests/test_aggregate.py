"""Aggregationen, Regret, Ränge und tau gegen die Handrechnung der Mini-Instanz (tests/conftest.py)."""

import numpy as np
import pytest

import gru_aggregate as ag
from conftest import MINI_U, MINI_V, MINI_W


def test_the_mini_instance_matches_the_member_utilities():
    assert MINI_V @ MINI_W.T == pytest.approx(MINI_U.T)


def test_mean_of_the_utilities_and_mean_of_the_weights_pick_the_compromise():
    assert ag.aggregate(MINI_U, MINI_V, MINI_W, "mean_u").tolist() == pytest.approx([0.5, 0.6, 0.5])
    assert ag.aggregate(MINI_U, MINI_V, MINI_W, "mean_w").tolist() == pytest.approx([0.5, 0.6, 0.5])
    assert ag.winner(ag.aggregate(MINI_U, MINI_V, MINI_W, "mean_u")) == 1
    assert set(ag.member_bests(MINI_U).tolist()) == {0, 2}                      # B ist der Beste von niemandem


def test_geometric_mean_minimum_median_and_normalised_mean():
    assert ag.aggregate(MINI_U, MINI_V, MINI_W, "gm_u").tolist() == pytest.approx([0.0, 0.6, 0.0], abs=1e-4)
    assert ag.aggregate(MINI_U, MINI_V, MINI_W, "min_u").tolist() == pytest.approx([0.0, 0.6, 0.0])
    assert ag.aggregate(MINI_U, MINI_V, MINI_W, "median_u").tolist() == pytest.approx([0.5, 0.6, 0.5])
    assert ag.aggregate(MINI_U, MINI_V, MINI_W, "mean_u_norm").tolist() == pytest.approx([0.3125, 0.375, 0.3125])
    for method in ("gm_u", "min_u", "median_u", "mean_u_norm"):
        assert ag.winner(ag.aggregate(MINI_U, MINI_V, MINI_W, method)) == 1, method


def test_borda_ties_go_to_the_smallest_number():
    assert ag.borda_points(MINI_U).tolist() == [2.0, 2.0, 2.0]
    assert ag.winner(ag.aggregate(MINI_U, MINI_V, MINI_W, "borda")) == 0


def test_a_member_scale_flips_the_mean_but_not_gm_normalised_mean_or_borda():
    U2 = MINI_U.copy()
    U2[1] *= 10
    assert ag.aggregate(U2, MINI_V, MINI_W, "mean_u").tolist() == pytest.approx([0.5, 3.3, 5.0])
    assert ag.winner(ag.aggregate(U2, MINI_V, MINI_W, "mean_u")) == 2
    assert ag.aggregate(U2, MINI_V, MINI_W, "gm_u")[1] == pytest.approx(np.sqrt(0.6 * 6.0))
    for method in ("gm_u", "mean_u_norm", "borda", "min_u"):
        assert ag.winner(ag.aggregate(U2, MINI_V, MINI_W, method)) == ag.winner(ag.aggregate(MINI_U, MINI_V, MINI_W, method)), method


def test_power_weights_the_arithmetic_and_geometric_means_but_not_median_min_or_borda():
    pi = np.array([0.75, 0.25])
    assert ag.aggregate(MINI_U, MINI_V, MINI_W, "mean_u", pi).tolist() == pytest.approx([0.75, 0.6, 0.25])
    assert ag.aggregate(MINI_U, MINI_V, MINI_W, "mean_w", pi).tolist() == pytest.approx([0.75, 0.6, 0.25])
    assert ag.aggregate(MINI_U, MINI_V, MINI_W, "gm_u", pi)[1] == pytest.approx(0.6)
    for method in ("median_u", "min_u", "borda"):
        assert ag.aggregate(MINI_U, MINI_V, MINI_W, method, pi).tolist() == ag.aggregate(MINI_U, MINI_V, MINI_W, method).tolist()


def test_unknown_aggregation_raises():
    with pytest.raises(ValueError):
        ag.aggregate(MINI_U, MINI_V, MINI_W, "diktator")


def test_regret_by_hand():
    assert ag.regret(MINI_U, 1).tolist() == pytest.approx([0.4, 0.4])
    assert ag.regret(MINI_U, 0).tolist() == pytest.approx([0.0, 1.0])
    assert ag.regret(MINI_U, 2).tolist() == pytest.approx([1.0, 0.0])
    assert ag.regret(np.array([[0.5, 0.5, 0.5]]), 1).tolist() == [0.0]           # alles gleich: kein Regret, keine Division durch null


def test_ranks_ties_and_rounding():
    assert ag.ranks_of(np.array([0.3, 0.9, 0.9, 0.1])).tolist() == [2, 0, 1, 3]
    assert ag.ranks_of(np.array([1.0, 1.0000000000000007])).tolist() == [0, 1]    # LP-Rauschen zählt als Gleichstand
    assert ag.winner(np.array([0.5, 0.5, 0.2])) == 0


def test_member_bests_and_kendall_tau_b():
    assert ag.member_bests(MINI_U).tolist() == [0, 2]
    assert ag.kendall_tau_b(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(1.0)
    assert ag.kendall_tau_b(np.array([1, 2, 3]), np.array([3, 2, 1])) == pytest.approx(-1.0)
    assert ag.kendall_tau_b(np.array([1, 1, 1]), np.array([1, 2, 3])) == 0.0
