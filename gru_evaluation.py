"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte (reine Funktionen). Nur numpy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import gru_aggregate as ag
import gru_ahp as ahp
import gru_constants as C
import gru_group as gp
from entscheidungsmatrix_generator import SplitMix64, make_matrix


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    k: int = C.DEFAULT_K
    kappa: float = C.DEFAULT_KAPPA
    agg: str = C.DEFAULT_AGG
    form: str = C.DEFAULT_FORM
    scale: float = C.DEFAULT_SCALE
    power: float = C.DEFAULT_POWER
    seed: int = C.DEFAULT_SEED


def build_group(n: int, m: int, k: int, kappa: float, form: str, seed: int):
    """(Matrix, normierte Werte V, Zentrum, Gewichte W (k x m), Nutzen U (k x n))."""
    x = make_matrix(n, m, 0.0, SplitMix64(seed))
    V = gp.normalized_values(x.signed)
    center, W = gp.member_weights(k, m, kappa, SplitMix64(seed + C.OFFSET_GROUP))
    U = gp.member_utilities(V, np.asarray(x.goodness, dtype=float), W, form, SplitMix64(seed + C.OFFSET_FORMS))
    return x, V, center, W, U


@dataclass
class Analysis:
    settings: Settings
    V: np.ndarray
    center: np.ndarray
    W: np.ndarray
    U: np.ndarray               # Nutzen, wie die Mitglieder sie melden (Mitglied 0 mit dem Faktor der Skala)
    U_true: np.ndarray          # Nutzen ohne die Skala
    power: np.ndarray           # Gewichte der Mitglieder im gewichteten Mittel
    score: np.ndarray           # Gruppenwertung der gewählten Aggregation
    rank: np.ndarray            # 0 = bester
    win: int
    bests: np.ndarray           # der Beste jedes Mitglieds
    nobody: bool                # der Sieger ist der Beste von keinem Mitglied
    same_order: bool            # die Gruppenordnung ist genau die Ordnung eines Mitglieds
    regret: np.ndarray          # je Mitglied (gemessen an den gemeldeten Nutzen)
    regret_true: np.ndarray     # je Mitglied (gemessen an den wahren Nutzen ohne Skala)
    taus: np.ndarray            # Kendalls tau-b Gruppe gegen Mitglied
    tau_matrix: np.ndarray      # (k + 1) x (k + 1): Mitglieder, zuletzt die Gruppe
    member_tau: float           # mittleres tau zwischen je zwei Mitgliedern
    winners: dict               # Sieger jeder Aggregation (mit derselben Skala und Macht)


def analyse(settings: Settings) -> Analysis:
    s = settings
    _, V, center, W, U_true = build_group(s.n, s.m, s.k, s.kappa, s.form, s.seed)
    U = gp.apply_scale(U_true, s.scale)
    pi = gp.member_power(s.k, s.power)
    score = ag.aggregate(U, V, W, s.agg, pi)
    win = ag.winner(score)
    bests = ag.member_bests(U)
    gr = ag.ranks_of(score)
    rows = np.vstack([U, score[None, :]])
    k = s.k
    tau_matrix = np.array([[ag.kendall_tau_b(rows[a], rows[b]) for b in range(k + 1)] for a in range(k + 1)])
    off = [tau_matrix[a, b] for a in range(k) for b in range(a + 1, k)]
    return Analysis(s, V, center, W, U, U_true, pi, score, gr, win, bests, win not in set(bests.tolist()), any((ag.ranks_of(u) == gr).all() for u in U),
                    ag.regret(U, win), ag.regret(U_true, win), tau_matrix[k, :k].copy(), tau_matrix, float(np.mean(off)) if off else 1.0,
                    {a: ag.winner(ag.aggregate(U, V, W, a, pi)) for a in C.AGGREGATIONS})


# ----------------------------------------------------------------------------- Experimente

def experiment_fit(ks=C.EXP_KS, kappas=C.KAPPA_CHOICES, seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M) -> list[dict]:
    """Experiment 1: Passt die Gruppenentscheidung (Mittel der Nutzen) zu jemandem? Je Mitgliederzahl und Verschiedenheit."""
    out = []
    for k in ks:
        for kappa in kappas:
            distinct, unan, nobody, same, tg, tm, rm, rx = [], [], [], [], [], [], [], []
            for seed in seeds:
                a = analyse(Settings(n=n, m=m, k=k, kappa=kappa, agg="mean_u", seed=seed))
                distinct.append(len(set(a.bests.tolist())))
                unan.append(len(set(a.bests.tolist())) == 1)
                nobody.append(a.nobody)
                same.append(a.same_order)
                tg.append(float(a.taus.mean()))
                tm.append(a.member_tau)
                rm.append(float(a.regret.mean()))
                rx.append(float(a.regret.max()))
            out.append({"k": k, "kappa": kappa, "distinct": float(np.mean(distinct)), "unanimous": float(np.mean(unan)), "nobody": float(np.mean(nobody)),
                        "same_order": float(np.mean(same)), "tau_group": float(np.mean(tg)), "tau_members": float(np.mean(tm)), "regret_mean": float(np.mean(rm)),
                        "regret_max": float(np.mean(rx)), "matrices": len(seeds)})
    return out


def experiment_aggregations(k=C.EXP_K, kappa=C.EXP_KAPPA, seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M) -> dict:
    """Experiment 2: Aggregationen im Vergleich (gleiche Form) und der Bruch mittlere Gewichte gegen Mittel der Nutzen bei gemischten Formen."""
    acc = {a: {"same": [], "nobody": [], "rm": [], "rx": [], "tm": [], "tn": []} for a in C.AGGREGATIONS}
    diff = {"gleich": [], "gemischt": []}
    for seed in seeds:
        base = analyse(Settings(n=n, m=m, k=k, kappa=kappa, agg="mean_u", seed=seed))
        for a in C.AGGREGATIONS:
            r = analyse(Settings(n=n, m=m, k=k, kappa=kappa, agg=a, seed=seed))
            acc[a]["same"].append(r.win == base.win)
            acc[a]["nobody"].append(r.nobody)
            acc[a]["rm"].append(float(r.regret.mean()))
            acc[a]["rx"].append(float(r.regret.max()))
            acc[a]["tm"].append(float(r.taus.mean()))
            acc[a]["tn"].append(float(r.taus.min()))
        for form in ("gleich", "gemischt"):
            r = analyse(Settings(n=n, m=m, k=k, kappa=kappa, form=form, seed=seed))
            diff[form].append(r.winners["mean_w"] != r.winners["mean_u"])
    rows = [{"agg": a, "same": float(np.mean(v["same"])), "somebody": 1.0 - float(np.mean(v["nobody"])), "regret_mean": float(np.mean(v["rm"])), "regret_max": float(np.mean(v["rx"])),
             "tau_mean": float(np.mean(v["tm"])), "tau_min": float(np.mean(v["tn"]))} for a, v in acc.items()]
    return {"rows": rows, "diff_same_form": float(np.mean(diff["gleich"])), "diff_mixed_forms": float(np.mean(diff["gemischt"])), "k": k, "kappa": kappa, "matrices": len(seeds)}


PROCEDURES = ("aij_eigen", "aij_rgmm", "aip_geo_eigen", "aip_geo_rgmm", "aip_arith_eigen")
PROCEDURE_LABELS = {"aij_eigen": "Urteile aggregieren (geometrisch), Eigenvektor", "aij_rgmm": "Urteile aggregieren (geometrisch), Zeilen-GM",
                    "aip_geo_eigen": "Prioritäten aggregieren (geometrisch), Eigenvektoren", "aip_geo_rgmm": "Prioritäten aggregieren (geometrisch), Zeilen-GM",
                    "aip_arith_eigen": "Prioritäten aggregieren (arithmetisch), Eigenvektoren"}


def experiment_judgments(sigmas=C.SIGMA_CHOICES, k=C.EXP_K, kappa=C.EXP_KAPPA, seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M) -> list[dict]:
    """Experiment 3: Urteile oder Prioritäten aggregieren (AHP-Paarvergleiche der Kriteriengewichte), geometrisch gegen arithmetisch, mit Rauschen sigma."""
    out = []
    for sigma in sigmas:
        acc = {p: {"tv_am": [], "tv_gm": [], "win": []} for p in PROCEDURES}
        d_tv, d_same = [], []
        for seed in seeds:
            x, V, _, W, _ = build_group(n, m, k, kappa, "gleich", seed)
            rng = SplitMix64(seed + C.OFFSET_AHP)
            As = [ahp.pairwise_matrix(w, sigma, rng) for w in W]
            tgt_am, tgt_gm = W.mean(axis=0), ahp.geometric_mean_vector(W)
            base = ag.winner(V @ tgt_am)
            res = {"aij_eigen": ahp.aij(As, "eigen"), "aij_rgmm": ahp.aij(As, "rgmm"), "aip_geo_eigen": ahp.aip(As, "geometric", "eigen"),
                   "aip_geo_rgmm": ahp.aip(As, "geometric", "rgmm"), "aip_arith_eigen": ahp.aip(As, "arithmetic", "eigen")}
            for p, w in res.items():
                acc[p]["tv_am"].append(ahp.tv_distance(w, tgt_am))
                acc[p]["tv_gm"].append(ahp.tv_distance(w, tgt_gm))
                acc[p]["win"].append(ag.winner(V @ w) == base)
            d_tv.append(ahp.tv_distance(res["aij_eigen"], res["aip_geo_eigen"]))
            d_same.append(ag.winner(V @ res["aij_eigen"]) == ag.winner(V @ res["aip_geo_eigen"]))
        out.append({"sigma": sigma, "procedures": {p: {k_: float(np.mean(v)) for k_, v in acc[p].items()} for p in PROCEDURES},
                    "aij_aip_tv": float(np.mean(d_tv)), "aij_aip_same": float(np.mean(d_same)), "matrices": len(seeds)})
    return out


def experiment_scale_power(powers=C.POWER_CHOICES, k=C.EXP_K, kappa=C.EXP_KAPPA, seeds=C.SWEEP_SEEDS, n=C.EXP_N, m=C.EXP_M) -> dict:
    """Experiment 4: Skala (ein Mitglied meldet mit dem Faktor c in [0,1; 10]) und Macht (Mitglied 1 zählt f-fach)."""
    aggs = C.AGGREGATIONS
    change = {a: [] for a in aggs}
    dictator = {a: [] for a in aggs}
    for seed in seeds:
        _, V, _, W, U = build_group(n, m, k, kappa, "gleich", seed)
        rng = SplitMix64(seed + C.OFFSET_SCALE)
        for who in range(k):
            c = float(np.exp(np.log(0.1) + rng.uniform(1)[0] * (np.log(10.0) - np.log(0.1))))
            U2 = U.copy()
            U2[who] *= c
            bests2 = ag.member_bests(U2)
            for a in aggs:
                w0 = ag.winner(ag.aggregate(U, V, W, a))
                w1 = ag.winner(ag.aggregate(U2, V, W, a))
                change[a].append(w0 != w1)
                if c >= 3.0:
                    dictator[a].append(w1 == bests2[who])
    scale_rows = [{"agg": a, "change": float(np.mean(change[a])), "dictator": float(np.mean(dictator[a])) if dictator[a] else None} for a in aggs]
    power_rows = []
    for f in powers:
        own, other, r0, rn = [], [], [], []
        for seed in seeds:
            _, V, _, W, U = build_group(n, m, k, kappa, "gleich", seed)
            pi = gp.member_power(k, f)
            win = ag.winner(ag.aggregate(U, V, W, "mean_u", pi))
            bests = ag.member_bests(U)
            reg = ag.regret(U, win)
            own.append(win == bests[0])
            other.append(float(np.mean([win == bests[j] for j in range(1, k)])))
            r0.append(float(reg[0]))
            rn.append(float(reg[1:].mean()))
        power_rows.append({"power": f, "own_best": float(np.mean(own)), "other_best": float(np.mean(other)), "regret_own": float(np.mean(r0)), "regret_others": float(np.mean(rn))})
    return {"scale": scale_rows, "power": power_rows, "k": k, "kappa": kappa, "matrices": len(seeds)}


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def _fit_cell(rows: list[dict], k: int, kappa: float) -> dict:
    return next(r for r in rows if r["k"] == k and r["kappa"] == kappa)


def verdict_fit(rows: list[dict]) -> tuple[str, str]:
    pct = lambda x: de(100 * x, 1)  # noqa: E731
    c = _fit_cell(rows, 5, 10.0)
    z = _fit_cell(rows, 9, 3.0)
    text = (f"Bei 5 Mitgliedern und mäßiger Verschiedenheit (κ = 10) ist der Sieger in {pct(1 - c['nobody'])} % der Matrizen der Beste von jemandem, die Gruppenordnung aber nur in "
            f"{pct(c['same_order'])} % genau die Ordnung eines Mitglieds. Sie ist die Mitte: Kendalls τ zwischen Gruppe und Mitglied {de(c['tau_group'], 2)}, zwischen je zwei Mitgliedern "
            f"{de(c['tau_members'], 2)}; der Regret ist im Mittel {pct(c['regret_mean'])} % der eigenen Nutzenspannweite, im schlimmsten Mitglied {pct(c['regret_max'])} %. Bei 9 sehr "
            f"verschiedenen Mitgliedern (κ = 3) sind es {pct(z['regret_mean'])} % und {pct(z['regret_max'])} %; nur {pct(z['unanimous'])} % der Gruppen sind einstimmig.")
    return "info", text


def verdict_aggregations(res: dict) -> tuple[str, str]:
    pct = lambda x: de(100 * x, 1)  # noqa: E731
    rows = {r["agg"]: r for r in res["rows"]}
    text = (f"Bei {res['k']} Mitgliedern (κ = {de(res['kappa'], 0)}): Mittlere Gewichte und Mittel der Nutzen haben immer denselben Sieger, geometrisches Mittel in "
            f"{pct(rows['gm_u']['same'])} %, Median in {pct(rows['median_u']['same'])} %, Minimum in {pct(rows['min_u']['same'])} %, Borda in {pct(rows['borda']['same'])} %. Der mittlere Regret "
            f"liegt zwischen {pct(min(r['regret_mean'] for r in rows.values()))} und {pct(max(r['regret_mean'] for r in rows.values()))} %. Haben die Mitglieder verschiedene Nutzenformen, "
            f"unterscheiden sich mittlere Gewichte und Mittel der Nutzen in {pct(res['diff_mixed_forms'])} % der Matrizen (gleiche Form: {pct(res['diff_same_form'])} %).")
    return "info", text


def verdict_judgments(rows: list[dict]) -> tuple[str, str]:
    pct = lambda x: de(100 * x, 1)  # noqa: E731
    lo = min(rows, key=lambda r: r["sigma"])
    hi = max(rows, key=lambda r: r["sigma"])
    geo, ari = hi["procedures"]["aip_geo_eigen"], hi["procedures"]["aip_arith_eigen"]
    text = (f"Urteile oder Prioritäten aggregieren macht kaum einen Unterschied: Bei Rauschen {de(100 * hi['sigma'], 0)} % haben beide denselben Sieger in {pct(hi['aij_aip_same'])} % der "
            f"Matrizen (Abstand der Gewichte {de(hi['aij_aip_tv'], 3)}; ohne Rauschen {de(lo['aij_aip_tv'], 3)}). Geometrisch gegen arithmetisch dagegen: Der Abstand der Gewichte zum "
            f"arithmetischen Mittel der verborgenen Gewichte ist {de(geo['tv_am'], 3)} gegen {de(ari['tv_am'], 3)}, und der Sieger ist der mit dem arithmetischen Mittel in {pct(geo['win'])} gegen "
            f"{pct(ari['win'])} %.")
    return "warning", text


def verdict_scale(res: dict) -> tuple[str, str]:
    pct = lambda x: de(100 * x, 1)  # noqa: E731
    rows = {r["agg"]: r for r in res["scale"]}
    inv = [a for a in C.AGGREGATIONS if rows[a]["change"] == 0.0]
    p1 = next(r for r in res["power"] if r["power"] == 1.0)
    p10 = max(res["power"], key=lambda r: r["power"])
    text = (f"Meldet ein Mitglied seine Nutzen mit einem Faktor zwischen 0,1 und 10, wechselt der Sieger beim Mittel der Nutzen in {pct(rows['mean_u']['change'])} %, beim Median in "
            f"{pct(rows['median_u']['change'])} %, beim Minimum in {pct(rows['min_u']['change'])} % der Fälle; bei {len(inv)} der {len(C.AGGREGATIONS)} Aggregationen nie. Zählt ein Mitglied "
            f"{de(p10['power'], 0)}-fach, ist der Sieger in {pct(p10['own_best'])} % sein Bester (einfach: {pct(p1['own_best'])} %), der Regret der anderen steigt von "
            f"{pct(p1['regret_others'])} auf {pct(p10['regret_others'])} %.")
    return "warning", text
