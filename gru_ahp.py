"""Paarvergleiche der Kriteriengewichte (AHP): Matrix, Eigenvektor, Zeilen-Geometrisches-Mittel, AIJ und AIP. Nur numpy."""

from __future__ import annotations

import numpy as np

from entscheidungsmatrix_generator import SplitMix64


def pairwise_matrix(w: np.ndarray, sigma: float, rng: SplitMix64) -> np.ndarray:
    """Reziproke Matrix a_ij = w_i / w_j * exp(sigma Z_ij) für i < j, a_ji = 1 / a_ij, Diagonale 1. sigma = 0: vollständig konsistent."""
    w = np.asarray(w, dtype=float)
    m = len(w)
    A = np.ones((m, m))
    z = rng.normal(m * (m - 1) // 2)
    t = 0
    for i in range(m):
        for j in range(i + 1, m):
            a = w[i] / w[j] * np.exp(sigma * z[t])
            t += 1
            A[i, j], A[j, i] = a, 1.0 / a
    return A


def eigen_priority(A: np.ndarray) -> np.ndarray:
    """Prioritäten als normierter Hauptdominanzeigenvektor (Summe 1)."""
    vals, vecs = np.linalg.eig(np.asarray(A, dtype=float))
    v = np.abs(np.real(vecs[:, int(np.argmax(np.real(vals)))]))
    return v / v.sum()


def rgmm_priority(A: np.ndarray) -> np.ndarray:
    """Prioritäten als normiertes geometrisches Mittel der Zeilen (Summe 1)."""
    g = np.exp(np.log(np.asarray(A, dtype=float)).mean(axis=1))
    return g / g.sum()


def geometric_mean_vector(P: np.ndarray) -> np.ndarray:
    """Normiertes elementweises geometrisches Mittel von Vektoren (Zeilen von P); Summe 1."""
    g = np.exp(np.log(np.maximum(np.asarray(P, dtype=float), 1e-12)).mean(axis=0))
    return g / g.sum()


def aij(As: list[np.ndarray], method: str = "eigen") -> np.ndarray:
    """Urteile aggregieren: elementweises geometrisches Mittel der Matrizen, danach die Prioritäten (eigen oder rgmm)."""
    AG = np.exp(np.mean([np.log(A) for A in As], axis=0))
    return eigen_priority(AG) if method == "eigen" else rgmm_priority(AG)


def aip(As: list[np.ndarray], how: str = "geometric", method: str = "eigen") -> np.ndarray:
    """Prioritäten aggregieren: erst je Mitglied die Prioritäten, danach das geometrische oder arithmetische Mittel."""
    prio = eigen_priority if method == "eigen" else rgmm_priority
    P = np.array([prio(A) for A in As])
    if how == "geometric":
        return geometric_mean_vector(P)
    if how == "arithmetic":
        return P.mean(axis=0)
    raise ValueError(f"unbekanntes Mittel: {how}")


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Totalvariationsabstand (halbe L1-Norm) zweier Gewichtsvektoren."""
    return 0.5 * float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum())
