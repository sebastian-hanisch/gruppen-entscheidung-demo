"""Orakel: Gruppenaggregation gegen unabhängige Rechenwege und Eigenschaften (DEMO-PLAYBOOK §12).

1. Mittlere Gewichte = Mittel der Nutzen bei gleicher (linearer) Form; bei gemischten Formen nicht.
2. Skala: geometrisches Mittel, normiertes Mittel und Borda ändern ihre Rangfolge bei Umskalierung eines Mitglieds nie; das Mittel der Nutzen kippt in einem Teil der Fälle.
3. AIJ = AIP beim geometrischen Mittel mit Zeilen-GM-Prioritäten (auch bei verrauschten Matrizen); beim Eigenvektor nur bei konsistenten Matrizen exakt.
4. Einstimmigkeit und Pareto: sind alle Mitglieder gleich, ist jede Aggregation ihre Rangfolge; bevorzugen alle a vor b, steht a bei jeder Aggregation vor b.
5. Macht: f -> unendlich macht den Sieger des Mittels zum Besten von Mitglied 0; f = 1 ist das einfache Mittel.
"""

import numpy as np
import pytest

import gru_aggregate as ag
import gru_ahp as ahp
import gru_constants as C
import gru_evaluation as ev
import gru_group as gp
from entscheidungsmatrix_generator import SplitMix64

SEEDS = tuple(range(300000, 300060))


def group(seed, k=5, n=8, m=4):
    rng = np.random.default_rng(seed)
    V = rng.random((n, m))
    W = rng.dirichlet(np.ones(m), size=k)
    return V, W, W @ V.T


def test_mean_weights_equal_the_mean_of_the_utilities_for_the_same_linear_form():
    for seed in SEEDS:
        V, W, U = group(seed)
        assert ag.aggregate(U, V, W, "mean_w") == pytest.approx(ag.aggregate(U, V, W, "mean_u"), abs=1e-12)


def test_mixed_forms_break_that_identity_in_some_cases():
    diff = 0
    for seed in range(1, 41):
        a = ev.analyse(ev.Settings(form="gemischt", seed=seed))
        diff += int(a.winners["mean_w"] != a.winners["mean_u"])
        b = ev.analyse(ev.Settings(form="gleich", seed=seed))
        assert b.winners["mean_w"] == b.winners["mean_u"]
    assert diff > 0


def test_geometric_normalised_and_borda_ignore_a_member_scale_but_the_mean_does_not():
    flips = 0
    for seed in SEEDS:
        V, W, U = group(seed)
        rng = np.random.default_rng(seed + 1)
        U2 = U.copy()
        U2[int(rng.integers(len(U)))] *= float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        for method in ("gm_u", "mean_u_norm", "borda"):
            assert (ag.ranks_of(ag.aggregate(U2, V, W, method)) == ag.ranks_of(ag.aggregate(U, V, W, method))).all(), method
        flips += int(ag.winner(ag.aggregate(U2, V, W, "mean_u")) != ag.winner(ag.aggregate(U, V, W, "mean_u")))
    assert flips > 0


def test_aij_equals_the_geometric_aip_for_the_row_geometric_mean():
    for seed in SEEDS[:30]:
        rng = SplitMix64(seed)
        W = np.random.default_rng(seed).dirichlet(np.ones(4), size=5)
        As = [ahp.pairwise_matrix(w, 0.5, rng) for w in W]
        assert ahp.aij(As, "rgmm") == pytest.approx(ahp.aip(As, "geometric", "rgmm"), abs=1e-12)


def test_for_the_eigenvector_aij_and_aip_differ_a_little_only_when_matrices_are_inconsistent():
    gaps = []
    for seed in SEEDS[:30]:
        rng = SplitMix64(seed)
        W = np.random.default_rng(seed).dirichlet(np.ones(4), size=5)
        exact = [ahp.pairwise_matrix(w, 0.0, rng) for w in W]
        assert ahp.aij(exact, "eigen") == pytest.approx(ahp.aip(exact, "geometric", "eigen"), abs=1e-9)
        noisy = [ahp.pairwise_matrix(w, 0.6, rng) for w in W]
        gaps.append(ahp.tv_distance(ahp.aij(noisy, "eigen"), ahp.aip(noisy, "geometric", "eigen")))
    assert max(gaps) > 1e-6 and float(np.mean(gaps)) < 0.03


def test_identical_members_and_pareto():
    for seed in SEEDS[:30]:
        V, W, U = group(seed, k=1)
        Uc, Wc = np.tile(U, (4, 1)), np.tile(W, (4, 1))
        for method in C.AGGREGATIONS:
            assert (ag.ranks_of(ag.aggregate(Uc, V, Wc, method)) == ag.ranks_of(U[0])).all(), method
        V, W, U = group(seed)
        a, b = int(np.argmax(U.min(axis=0))), int(np.argmin(U.max(axis=0)))
        if (U[:, a] > U[:, b]).all():
            for method in C.AGGREGATIONS:
                s = ag.aggregate(U, V, W, method)
                assert s[a] >= s[b] - 1e-12, method


def test_a_large_power_makes_the_winner_the_best_of_member_zero():
    for seed in SEEDS[:30]:
        V, W, U = group(seed)
        pi = gp.member_power(len(U), 1e6)
        assert ag.winner(ag.aggregate(U, V, W, "mean_u", pi)) == ag.winner(U[0])
        assert ag.aggregate(U, V, W, "mean_u", gp.member_power(len(U), 1.0)) == pytest.approx(U.mean(axis=0))
