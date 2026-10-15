"""Die Gruppe: Gewichte um ein gemeinsames Zentrum, Nutzen der Mitglieder, Skala, Macht. Zufall aus SplitMix64 (plattformunabhängig). Nur numpy."""

from __future__ import annotations

import math

import numpy as np

import gru_constants as C
from entscheidungsmatrix_generator import CRITERIA, SplitMix64, utility


def gamma_variate(shape: float, rng: SplitMix64) -> float:
    """Eine Gamma(shape, 1)-Zufallszahl (Marsaglia und Tsang; für shape < 1 über shape + 1 und u^(1/shape))."""
    if shape < 1.0:
        return gamma_variate(shape + 1.0, rng) * float(rng.uniform(1)[0]) ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = float(rng.normal(1)[0])
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue
        u = float(rng.uniform(1)[0])
        if math.log(max(u, 1e-300)) < 0.5 * x * x + d - d * v + d * math.log(v):
            return d * v


WEIGHT_FLOOR = 1e-4                           # kein Gewicht ist ganz null (Paarvergleiche bilden Quotienten)


def dirichlet(alpha: np.ndarray, rng: SplitMix64) -> np.ndarray:
    """Dirichlet(alpha) über normierte Gamma-Zufallszahlen; jedes Gewicht mindestens WEIGHT_FLOOR (danach neu normiert)."""
    g = np.array([gamma_variate(float(max(a, 1e-3)), rng) for a in alpha])
    w = np.maximum(g / g.sum(), WEIGHT_FLOOR)
    return w / w.sum()


def member_weights(k: int, m: int, kappa: float, rng: SplitMix64) -> tuple[np.ndarray, np.ndarray]:
    """(Zentrum, Gewichte der k Mitglieder): Zentrum ~ Dirichlet(1), Mitglied ~ Dirichlet(kappa * Zentrum). Kleines kappa: sehr verschieden; großes: fast gleich."""
    center = dirichlet(np.ones(m), rng)
    return center, np.array([dirichlet(kappa * center, rng) for _ in range(k)])


def normalized_values(signed: np.ndarray) -> np.ndarray:
    """Werte auf die feste Nenn-Spannweite je Kriterium (höher = besser)."""
    signed = np.asarray(signed, dtype=float)
    m = signed.shape[1]
    lo = np.array([CRITERIA[j][3] if CRITERIA[j][1] > 0 else -CRITERIA[j][4] for j in range(m)], dtype=float)
    hi = np.array([CRITERIA[j][4] if CRITERIA[j][1] > 0 else -CRITERIA[j][3] for j in range(m)], dtype=float)
    return (signed - lo) / (hi - lo)


def member_utilities(V: np.ndarray, goodness: np.ndarray, W: np.ndarray, form: str, rng: SplitMix64) -> np.ndarray:
    """k x n Nutzen. gleich: alle linear in den Rohwerten (V @ w). gemischt: je Mitglied linear (Rohwerte), konkav (Wurzel der Güte) oder Cobb-Douglas (exp des gewichteten Logarithmus);
    alle drei liegen in (0, 1]."""
    k = len(W)
    U = np.empty((k, len(V)))
    kinds = member_forms(k, form, rng)
    for i, w in enumerate(W):
        if kinds[i] == 0:
            U[i] = V @ w
        elif kinds[i] == 1:
            U[i] = utility(goodness, w, "konkav")
        else:
            U[i] = np.exp(utility(goodness, w, "cobb"))
    return U


def member_forms(k: int, form: str, rng: SplitMix64) -> list[int]:
    """Die Form je Mitglied (0 linear, 1 konkav, 2 Cobb-Douglas), aus demselben Strom wie in member_utilities gezogen."""
    return [0] * k if form == "gleich" else [int(rng.uniform(1)[0] * 3) for _ in range(k)]


def apply_scale(U: np.ndarray, factor: float) -> np.ndarray:
    """Mitglied 0 meldet seine Nutzen mit dem Faktor `factor` (nur die Skala, nicht die Ordnung)."""
    out = np.array(U, dtype=float, copy=True)
    out[0] *= factor
    return out


def member_power(k: int, factor: float) -> np.ndarray:
    """Gewichte der Mitglieder im gewichteten Mittel: Mitglied 0 zählt `factor`-fach, die anderen einfach; Summe 1."""
    p = np.ones(k)
    p[0] = factor
    return p / p.sum()
