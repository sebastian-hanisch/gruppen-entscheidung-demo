"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Zwei Mitglieder, drei Alternativen A, B, C, zwei Kriterien: V = [[1; 0], [0,6; 0,6], [0; 1]], Gewichte w1 = (1; 0), w2 = (0; 1); Nutzen M1 = (1; 0,6; 0), M2 = (0; 0,6; 1).
Mittel der Nutzen (0,5; 0,6; 0,5): Sieger B, der Beste von niemandem (M1: A, M2: C). Geometrisch (0; 0,6; 0): B. Minimum (0; 0,6; 0): B. Median = Mittel bei K = 2: B.
Borda (Punkte 2, 1, 0): A 2, B 2, C 2: Gleichstand, kleinste Nummer A.
Skala: M2 mit dem Faktor 10 (0; 6; 10): Mittel (0,5; 3,3; 5): Sieger C (kippt); geometrisch (0; 1,897; 0): B; normiert (M1/1,6, M2/16): (0,3125; 0,375; 0,3125): B.
Regret bei der Wahl B: jedes Mitglied (1 - 0,6) / 1 = 0,4; bei A: M1 0, M2 1; bei C: M1 1, M2 0.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MINI_V = np.array([[1.0, 0.0], [0.6, 0.6], [0.0, 1.0]])
MINI_W = np.array([[1.0, 0.0], [0.0, 1.0]])
MINI_U = np.array([[1.0, 0.6, 0.0], [0.0, 0.6, 1.0]])


@pytest.fixture
def mini():
    return MINI_U.copy(), MINI_V.copy(), MINI_W.copy()
