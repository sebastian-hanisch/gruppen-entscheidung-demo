"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import gru_constants as C
from gru_evaluation import Analysis, PROCEDURE_LABELS, PROCEDURES, de


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def member_names(k: int) -> list[str]:
    return [f"M{i + 1}" for i in range(k)]


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def members_table(a: Analysis) -> pd.DataFrame:
    names = alt_names(a.settings.n)
    mem = member_names(a.settings.k)
    rows = []
    for i in range(a.settings.k):
        rows.append({"Mitglied": mem[i] + (" (Skala ×" + de(a.settings.scale, 1) + ")" if i == 0 and a.settings.scale != 1.0 else ""),
                     "Bester": names[a.bests[i]], "Bester = Sieger": "ja" if a.bests[i] == a.win else "nein", "Regret der Wahl": float(a.regret_true[i]),
                     "τ zur Gruppenordnung": float(a.taus[i]), "größtes Gewicht": f"K{int(np.argmax(a.W[i])) + 1} ({de(100 * float(a.W[i].max()), 0)} %)",
                     "Macht": de(float(a.power[i] / a.power.min()), 0) + "-fach"})
    return pd.DataFrame(rows)


def style_members(df: pd.DataFrame):
    return df.style.format({"Regret der Wahl": lambda v: f"{de(100 * v, 1)} %", "τ zur Gruppenordnung": lambda v: de(v, 2)})


def build_group_score(a: Analysis) -> go.Figure:
    """Gruppenwertung je Alternative (sortiert), die Besten der Mitglieder markiert."""
    n = a.settings.n
    names = alt_names(n)
    order = np.argsort(a.rank)
    best_set = set(a.bests.tolist())
    colors = [C.COLOR_GROUP if i == a.win else (C.COLOR_BEST if i in best_set else "#8A96A6") for i in order]
    fig = go.Figure(go.Bar(x=[names[i] for i in order], y=[a.score[i] for i in order], marker_color=colors,
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title=C.AGG_SHORT[a.settings.agg] + " (Wertung)")
    fig.update_xaxes(title="Alternative (grün: Sieger, orange: Bester eines Mitglieds)")
    return _fixed(fig, 340)


def build_tau_matrix(a: Analysis) -> go.Figure:
    labels = member_names(a.settings.k) + ["Gruppe"]
    z = a.tau_matrix
    text = [[de(v, 2) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, text=text, texttemplate="%{text}", colorscale="Blues", zmin=-0.2, zmax=1.0, colorbar=dict(title="τ"),
                               hovertemplate="%{y} gegen %{x}: τ = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _fixed(fig, 340)


def build_regret(a: Analysis) -> go.Figure:
    mem = member_names(a.settings.k)
    fig = go.Figure(go.Bar(x=mem, y=[100 * r for r in a.regret_true], marker_color=[C.COLOR_MEMBER[i % len(C.COLOR_MEMBER)] for i in range(a.settings.k)]))
    fig.update_yaxes(title="Regret der Wahl (% der eigenen Nutzenspannweite)", rangemode="tozero")
    return _fixed(fig, 300)


def winners_table(a: Analysis) -> pd.DataFrame:
    names = alt_names(a.settings.n)
    best_set = set(a.bests.tolist())
    return pd.DataFrame([{"Aggregation": C.AGG_LABELS[g], "Sieger": names[w], "Bester eines Mitglieds": "ja" if w in best_set else "nein",
                          "wie die gewählte": "ja" if w == a.win else "nein"} for g, w in a.winners.items()])


# ----------------------------------------------------------------------------- Experimente

def fit_table(rows: list[dict]) -> pd.DataFrame:
    p = lambda x: f"{de(100 * x, 1)} %"  # noqa: E731
    return pd.DataFrame([{"Mitglieder": r["k"], "κ": de(r["kappa"], 0), "verschiedene Beste": de(r["distinct"], 2), "einstimmig": p(r["unanimous"]),
                          "Sieger = Bester von jemandem": p(1 - r["nobody"]), "Ordnung = die eines Mitglieds": p(r["same_order"]), "τ Gruppe–Mitglied": de(r["tau_group"], 2),
                          "τ Mitglied–Mitglied": de(r["tau_members"], 2), "Regret (Mittel / schlimmstes)": f"{p(r['regret_mean'])} / {p(r['regret_max'])}"} for r in rows])


def build_fit(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for idx, k in enumerate(sorted({r["k"] for r in rows})):
        sel = sorted((r for r in rows if r["k"] == k), key=lambda r: r["kappa"])
        fig.add_trace(go.Scatter(x=[de(r["kappa"], 0) for r in sel], y=[100 * r["same_order"] for r in sel], mode="lines+markers", name=f"{k} Mitglieder: Ordnung = die eines Mitglieds",
                                 line=dict(color=C.COLOR_MEMBER[idx], width=3)))
        fig.add_trace(go.Scatter(x=[de(r["kappa"], 0) for r in sel], y=[100 * r["unanimous"] for r in sel], mode="lines", name=f"{k} Mitglieder: einstimmig",
                                 line=dict(color=C.COLOR_MEMBER[idx], dash="dot")))
    fig.update_xaxes(title="Verschiedenheit κ (klein = sehr verschieden)", type="category")
    fig.update_yaxes(title="Anteil der Matrizen (%)", range=[0, 102])
    return _fixed(fig, 380)


def aggregations_table(res: dict) -> pd.DataFrame:
    p = lambda x: f"{de(100 * x, 1)} %"  # noqa: E731
    return pd.DataFrame([{"Aggregation": C.AGG_SHORT[r["agg"]], "gleicher Sieger wie Mittel der Nutzen": p(r["same"]), "Bester von jemandem": p(r["somebody"]),
                          "Regret (Mittel)": p(r["regret_mean"]), "Regret (schlimmstes Mitglied)": p(r["regret_max"]), "τ Mittel": de(r["tau_mean"], 2),
                          "τ schlechtestes Mitglied": de(r["tau_min"], 2)} for r in res["rows"]])


def judgments_table(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        for p in PROCEDURES:
            d = r["procedures"][p]
            out.append({"Rauschen": f"{de(100 * r['sigma'], 0)} %", "Verfahren": PROCEDURE_LABELS[p], "Abstand zum arithmetischen Mittel (TV)": de(d["tv_am"], 3),
                        "zum geometrischen Mittel (TV)": de(d["tv_gm"], 3), "Sieger = Sieger mit mittleren Gewichten": f"{de(100 * d['win'], 1)} %"})
    return pd.DataFrame(out)


def scale_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Aggregation": C.AGG_SHORT[r["agg"]], "Sieger wechselt bei Faktor 0,1 bis 10": f"{de(100 * r['change'], 1)} %",
                          "Sieger = Bester des skalierten Mitglieds (Faktor ≥ 3)": "-" if r["dictator"] is None else f"{de(100 * r['dictator'], 1)} %"} for r in res["scale"]])


def power_table(res: dict) -> pd.DataFrame:
    p = lambda x: f"{de(100 * x, 1)} %"  # noqa: E731
    return pd.DataFrame([{"Macht von Mitglied 1": f"{de(r['power'], 0)}-fach", "Sieger = Bester von Mitglied 1": p(r["own_best"]),
                          "Sieger = Bester eines anderen (je Mitglied)": p(r["other_best"]), "Regret Mitglied 1": p(r["regret_own"]), "Regret der anderen": p(r["regret_others"])}
                         for r in res["power"]])
