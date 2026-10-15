"""Randfälle der Kernroutinen (DEMO-PLAYBOOK §12): ein Mitglied, zwei Alternativen, Nutzen null, Gleichstände, Skala und Macht an den Grenzen, kleinste und größte Gruppe."""

import numpy as np
import pytest

import gru_aggregate as ag
import gru_ahp as ahp
import gru_constants as C
import gru_evaluation as ev
from entscheidungsmatrix_generator import SplitMix64


def test_a_single_member_is_every_aggregation():
    U = np.array([[0.2, 0.9, 0.5]])
    V = np.array([[0.2, 0.0], [0.9, 0.1], [0.5, 0.3]])
    W = np.array([[1.0, 0.0]])
    for method in C.AGGREGATIONS:
        assert ag.winner(ag.aggregate(U, V, W, method)) == 1, method
    assert ag.regret(U, 1).tolist() == [0.0]


def test_two_alternatives_and_two_members():
    U = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert ag.winner(ag.aggregate(U, np.eye(2), np.eye(2), "mean_u")) == 0      # 0,55 gegen 0,45
    assert ag.borda_points(U).tolist() == [1.0, 1.0]                            # Gleichstand
    assert ag.kendall_tau_b(ag.aggregate(U, np.eye(2), np.eye(2), "mean_u"), U[0]) == pytest.approx(1.0)


def test_zero_utilities_in_the_geometric_mean_stay_finite_and_tie_at_the_floor():
    U = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    gm = ag.aggregate(U, np.eye(3)[:, :2], np.eye(2), "gm_u")
    assert np.isfinite(gm).all() and gm[0] == pytest.approx(gm[2]) and gm[1] == pytest.approx(C.UTILITY_FLOOR)


def test_equal_utilities_have_no_regret_and_the_smallest_winner():
    U = np.full((3, 4), 0.5)
    assert ag.regret(U, 2).tolist() == [0.0, 0.0, 0.0]
    assert ag.winner(ag.aggregate(U, np.eye(4)[:, :2], np.eye(3)[:, :2], "mean_u")) == 0


def test_analysis_with_the_smallest_and_the_largest_group():
    a = ev.analyse(ev.Settings(n=6, m=3, k=3, kappa=3.0, agg="gm_u", seed=2))
    assert a.score.shape == (6,) and a.tau_matrix.shape == (4, 4) and sorted(a.rank.tolist()) == list(range(6))
    b = ev.analyse(ev.Settings(n=20, m=8, k=9, kappa=100.0, agg="borda", form="gemischt", scale=0.1, power=10.0, seed=9))
    assert b.score.shape == (20,) and b.tau_matrix.shape == (10, 10) and b.regret.shape == (9,)


def test_the_scale_changes_the_reported_utilities_but_not_the_true_regret():
    a = ev.analyse(ev.Settings(scale=10.0, seed=6))
    b = ev.analyse(ev.Settings(scale=1.0, seed=6))
    assert (a.U[0] == 10 * b.U[0]).all() and (a.U_true == b.U_true).all()
    assert a.regret_true.shape == a.regret.shape


def test_a_pairwise_matrix_for_a_single_pair_of_criteria():
    A = ahp.pairwise_matrix(np.array([0.7, 0.3]), 0.0, SplitMix64(1))
    assert A.ravel() == pytest.approx([1.0, 7 / 3, 3 / 7, 1.0])
    assert ahp.eigen_priority(A) == pytest.approx([0.7, 0.3])
