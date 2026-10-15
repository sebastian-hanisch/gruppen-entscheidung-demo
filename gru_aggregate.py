"""Aggregation der Mitgliedsurteile, Regret, Ränge und Kendalls tau-b. Nur numpy."""

from __future__ import annotations

import numpy as np

import gru_constants as C


def ranks_of(u: np.ndarray) -> np.ndarray:
    """Rang 0..n-1 (0 = bester); Gleichstand (auf neun Stellen gerundet): kleinere Nummer zuerst."""
    u = np.round(np.asarray(u, dtype=float), 9)
    order = np.lexsort((np.arange(len(u)), -u))
    r = np.empty(len(u), dtype=int)
    r[order] = np.arange(len(u))
    return r


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendalls tau-b (Gleichstände berücksichtigt); 0, wenn eine Seite keine Unterschiede hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.sign(a[:, None] - a[None, :])
    sb = np.sign(b[:, None] - b[None, :])
    denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return 0.0 if denom == 0 else float((sa * sb).sum() / denom)


def borda_points(U: np.ndarray) -> np.ndarray:
    """Borda-Punkte je Alternative: die Alternative auf Rang r eines Mitglieds bekommt n - 1 - r Punkte, Summe über die Mitglieder."""
    n = U.shape[1]
    return np.sum([n - 1 - ranks_of(u) for u in U], axis=0).astype(float)


def aggregate(U: np.ndarray, V: np.ndarray, W: np.ndarray, method: str, power: np.ndarray | None = None) -> np.ndarray:
    """Gruppenwertung je Alternative. power: Gewichte der Mitglieder (Summe 1) bei den gewichteten Mitteln; Median, Minimum und Borda kennen keine Macht."""
    U = np.asarray(U, dtype=float)
    k = U.shape[0]
    pi = np.full(k, 1.0 / k) if power is None else np.asarray(power, dtype=float)
    if method == "mean_w":
        return np.asarray(V, dtype=float) @ (pi @ np.asarray(W, dtype=float))
    if method == "mean_u":
        return pi @ U
    if method == "gm_u":
        return np.exp(pi @ np.log(np.maximum(U, C.UTILITY_FLOOR)))
    if method == "median_u":
        return np.median(U, axis=0)
    if method == "min_u":
        return U.min(axis=0)
    if method == "borda":
        return borda_points(U)
    if method == "mean_u_norm":
        return pi @ (U / U.sum(axis=1, keepdims=True))
    raise ValueError(f"unbekannte Aggregation: {method}")


def winner(score: np.ndarray) -> int:
    """Alternative mit der größten Wertung; Gleichstand (auf neun Stellen gerundet): kleinste Nummer."""
    return int(np.argmin(ranks_of(score)))


def regret(U: np.ndarray, choice: int) -> np.ndarray:
    """Regret je Mitglied: (bester eigener Nutzen - Nutzen der Wahl) / eigene Nutzenspannweite; 0, wenn das Mitglied alle gleich bewertet."""
    U = np.asarray(U, dtype=float)
    span = U.max(axis=1) - U.min(axis=1)
    safe = np.where(span > 0, span, 1.0)
    return (U.max(axis=1) - U[:, choice]) / safe


def member_bests(U: np.ndarray) -> np.ndarray:
    """Der Beste jedes Mitglieds (bei Gleichstand die kleinste Nummer)."""
    return np.array([winner(u) for u in np.asarray(U, dtype=float)])
