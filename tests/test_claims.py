"""Jede Zahl der README, gemessen mit den eigenen Experimenten der Demo (200 feste Matrizen, SWEEP_SEEDS). Die Werte sind deterministisch (SplitMix64)."""

import pytest

import gru_constants as C
import gru_evaluation as ev

TOL = 0.0006


@pytest.fixture(scope="module")
def exp():
    return {"fit": ev.experiment_fit(), "agg": ev.experiment_aggregations(), "jud": ev.experiment_judgments(), "scale": ev.experiment_scale_power()}


def _cell(rows, k, kappa):
    return next(r for r in rows if r["k"] == k and r["kappa"] == kappa)


def test_the_winner_fits_somebody_but_the_order_rarely_fits_a_member(exp):
    rows = exp["fit"]
    cells = [(k, kappa) for k in C.EXP_KS for kappa in C.KAPPA_CHOICES]
    assert [_cell(rows, k, ka)["nobody"] for k, ka in cells] == pytest.approx([0.055, 0.03, 0.015, 0.0, 0.025, 0.02, 0.005, 0.0, 0.01, 0.005, 0.0, 0.0], abs=TOL)
    assert [_cell(rows, k, ka)["same_order"] for k, ka in cells] == pytest.approx([0.03, 0.115, 0.16, 0.415, 0.07, 0.105, 0.21, 0.48, 0.07, 0.15, 0.285, 0.63], abs=TOL)
    assert [_cell(rows, k, ka)["unanimous"] for k, ka in cells] == pytest.approx([0.325, 0.43, 0.605, 0.73, 0.155, 0.335, 0.51, 0.65, 0.07, 0.23, 0.435, 0.615], abs=TOL)
    assert [_cell(rows, k, ka)["distinct"] for k, ka in cells[4:8]] == pytest.approx([2.39, 1.94, 1.635, 1.4], abs=TOL)


def test_the_group_order_is_the_middle_of_the_members(exp):
    rows = exp["fit"]
    five = [_cell(rows, 5, ka) for ka in C.KAPPA_CHOICES]
    assert [r["tau_group"] for r in five] == pytest.approx([0.685, 0.789, 0.868, 0.925], abs=TOL)
    assert [r["tau_members"] for r in five] == pytest.approx([0.516, 0.671, 0.797, 0.883], abs=TOL)
    assert all(r["tau_group"] > r["tau_members"] for r in rows)
    assert [r["regret_mean"] for r in five] == pytest.approx([0.054, 0.028, 0.012, 0.004], abs=TOL)
    assert [r["regret_max"] for r in five] == pytest.approx([0.18, 0.097, 0.044, 0.017], abs=TOL)
    assert (_cell(rows, 9, 3.0)["regret_mean"], _cell(rows, 9, 3.0)["regret_max"]) == pytest.approx((0.062, 0.272), abs=TOL)


def test_aggregations_differ_in_one_case_of_ten_or_more(exp):
    rows = {r["agg"]: r for r in exp["agg"]["rows"]}
    assert [rows[a]["same"] for a in C.AGGREGATIONS] == pytest.approx([1.0, 1.0, 0.985, 0.85, 0.83, 0.9, 0.99], abs=TOL)
    assert [rows[a]["somebody"] for a in C.AGGREGATIONS] == pytest.approx([0.98, 0.98, 0.975, 0.97, 0.94, 0.985, 0.98], abs=TOL)
    assert [rows[a]["regret_mean"] for a in C.AGGREGATIONS] == pytest.approx([0.028, 0.028, 0.029, 0.033, 0.04, 0.031, 0.028], abs=TOL)
    assert [rows[a]["regret_max"] for a in C.AGGREGATIONS] == pytest.approx([0.097, 0.097, 0.095, 0.109, 0.101, 0.105, 0.097], abs=TOL)
    assert [rows[a]["tau_mean"] for a in C.AGGREGATIONS] == pytest.approx([0.789, 0.789, 0.787, 0.787, 0.747, 0.796, 0.789], abs=TOL)
    assert (exp["agg"]["diff_same_form"], exp["agg"]["diff_mixed_forms"]) == pytest.approx((0.0, 0.185), abs=TOL)


def test_geometric_against_arithmetic_matters_more_than_judgments_against_priorities(exp):
    rows = {r["sigma"]: r for r in exp["jud"]}
    assert [rows[s]["procedures"]["aij_eigen"]["tv_am"] for s in C.SIGMA_CHOICES] == pytest.approx([0.0608, 0.0627, 0.0706], abs=TOL)
    assert [rows[s]["procedures"]["aij_eigen"]["win"] for s in C.SIGMA_CHOICES] == pytest.approx([0.885, 0.89, 0.885], abs=TOL)
    assert [rows[s]["procedures"]["aip_arith_eigen"]["tv_am"] for s in C.SIGMA_CHOICES] == pytest.approx([0.0, 0.0185, 0.0377], abs=TOL)
    assert [rows[s]["procedures"]["aip_arith_eigen"]["win"] for s in C.SIGMA_CHOICES] == pytest.approx([1.0, 0.955, 0.94], abs=TOL)
    assert [rows[s]["aij_aip_tv"] for s in C.SIGMA_CHOICES] == pytest.approx([0.0, 0.0015, 0.0061], abs=TOL)
    assert [rows[s]["aij_aip_same"] for s in C.SIGMA_CHOICES] == pytest.approx([1.0, 0.995, 0.985], abs=TOL)
    assert all(rows[s]["procedures"]["aij_rgmm"]["tv_am"] == pytest.approx(rows[s]["procedures"]["aip_geo_rgmm"]["tv_am"], abs=1e-9) for s in C.SIGMA_CHOICES)


def test_the_scale_of_one_member_flips_the_mean_but_not_the_geometric_mean(exp):
    rows = {r["agg"]: r for r in exp["scale"]["scale"]}
    assert [rows[a]["change"] for a in C.AGGREGATIONS] == pytest.approx([0.0, 0.077, 0.0, 0.13, 0.182, 0.0, 0.0], abs=TOL)
    assert rows["mean_u"]["dictator"] == pytest.approx(0.8397, abs=TOL) and rows["mean_w"]["dictator"] == pytest.approx(0.7214, abs=TOL)


def test_power_buys_the_winner_and_costs_the_others(exp):
    rows = {r["power"]: r for r in exp["scale"]["power"]}
    assert [rows[f]["own_best"] for f in C.POWER_CHOICES] == pytest.approx([0.77, 0.815, 0.845, 0.925], abs=TOL)
    assert [rows[f]["other_best"] for f in C.POWER_CHOICES] == pytest.approx([0.714, 0.713, 0.701, 0.666], abs=TOL)
    assert [rows[f]["regret_own"] for f in C.POWER_CHOICES] == pytest.approx([0.024, 0.016, 0.012, 0.003], abs=TOL)
    assert [rows[f]["regret_others"] for f in C.POWER_CHOICES] == pytest.approx([0.029, 0.033, 0.036, 0.048], abs=TOL)
