"""Auswertung: Fall der App, Grenzfälle und Experimente auf wenigen Seeds (Aussagen über die festen Matrizen stehen in test_claims.py)."""

import numpy as np
import pytest

import gru_aggregate as ag
import gru_constants as C
import gru_evaluation as ev


def test_de_uses_a_decimal_comma():
    assert ev.de(3.14159, 2) == "3,14"
    assert ev.de(2.0, 0) == "2"


def test_analysis_of_the_default_case_is_consistent():
    a = ev.analyse(ev.Settings())
    s = a.settings
    assert a.W.shape == (s.k, s.m) and a.U.shape == (s.k, s.n) and a.score.shape == (s.n,)
    assert a.W.sum(axis=1) == pytest.approx(np.ones(s.k)) and a.power.sum() == pytest.approx(1.0)
    assert a.win == int(np.flatnonzero(a.rank == 0)[0]) and a.nobody == (a.win not in set(a.bests.tolist()))
    assert a.tau_matrix.shape == (s.k + 1, s.k + 1) and np.diag(a.tau_matrix).tolist() == pytest.approx([1.0] * (s.k + 1))
    assert a.taus.tolist() == pytest.approx(a.tau_matrix[s.k, : s.k].tolist())
    assert set(a.winners) == set(C.AGGREGATIONS) and a.winners[s.agg] == a.win


def test_the_same_settings_give_the_same_analysis():
    a, b = ev.analyse(ev.Settings(seed=7)), ev.analyse(ev.Settings(seed=7))
    assert (a.U == b.U).all() and a.win == b.win and (a.W == b.W).all()


def test_scale_and_power_act_only_where_documented():
    base = ev.analyse(ev.Settings(seed=6))
    scaled = ev.analyse(ev.Settings(seed=6, scale=10.0))
    assert scaled.winners["mean_u"] != base.winners["mean_u"]
    for method in ("mean_w", "gm_u", "borda", "mean_u_norm"):
        assert scaled.winners[method] == base.winners[method], method
    strong = ev.analyse(ev.Settings(seed=6, power=10.0))
    for method in ("median_u", "min_u", "borda"):
        assert strong.winners[method] == base.winners[method], method
    assert strong.winners["mean_u"] == strong.bests[0]


def test_regret_true_ignores_the_scale_but_not_the_choice():
    a = ev.analyse(ev.Settings(seed=6, scale=10.0, agg="mean_u"))
    assert a.regret_true.tolist() == pytest.approx(ag.regret(a.U_true, a.win).tolist())
    assert a.regret.tolist() == pytest.approx(ag.regret(a.U, a.win).tolist())


def test_equal_form_makes_mean_weights_and_mean_utilities_the_same_winner():
    for seed in range(1, 15):
        a = ev.analyse(ev.Settings(seed=seed))
        assert a.winners["mean_w"] == a.winners["mean_u"]


def test_experiments_accept_a_small_set_of_seeds():
    seeds = C.SWEEP_SEEDS[:6]
    fit = ev.experiment_fit(ks=(3, 5), kappas=(3.0, 100.0), seeds=seeds)
    assert len(fit) == 4 and all(0.0 <= r["unanimous"] <= 1.0 and r["regret_max"] >= r["regret_mean"] for r in fit)
    agg = ev.experiment_aggregations(seeds=seeds)
    assert [r["agg"] for r in agg["rows"]] == list(C.AGGREGATIONS) and agg["diff_same_form"] == 0.0
    jud = ev.experiment_judgments(sigmas=(0.0, 0.3), seeds=seeds)
    assert len(jud) == 2 and jud[0]["aij_aip_tv"] < 1e-9 and jud[0]["procedures"]["aip_arith_eigen"]["win"] == 1.0
    sc = ev.experiment_scale_power(powers=(1.0, 10.0), seeds=seeds)
    assert [r["agg"] for r in sc["scale"]] == list(C.AGGREGATIONS) and len(sc["power"]) == 2
    assert all(r["change"] == 0.0 for r in sc["scale"] if r["agg"] in ("mean_w", "gm_u", "borda", "mean_u_norm"))


def test_a_more_diverse_group_is_less_often_unanimous():
    rows = ev.experiment_fit(ks=(5,), kappas=(3.0, 100.0), seeds=C.SWEEP_SEEDS[:40])
    assert rows[0]["unanimous"] < rows[1]["unanimous"] and rows[0]["regret_mean"] > rows[1]["regret_mean"]
