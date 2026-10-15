"""AHP-Paarvergleiche: Matrix, Eigenvektor, Zeilen-GM, AIJ und AIP gegen die Handrechnung."""

import numpy as np
import pytest

import gru_ahp as ahp
from entscheidungsmatrix_generator import SplitMix64

W1 = np.array([0.5, 0.3, 0.2])
W2 = np.array([0.1, 0.3, 0.6])
GEO = np.array([0.2570, 0.3448, 0.3982])                                 # normiertes geometrisches Mittel von W1 und W2 (Handrechnung: sqrt(0,05), sqrt(0,09), sqrt(0,12))


def test_a_consistent_matrix_has_the_weights_as_priority():
    A = ahp.pairwise_matrix(W1, 0.0, SplitMix64(1))
    assert A[0, 1] == pytest.approx(0.5 / 0.3) and A[2, 0] == pytest.approx(0.2 / 0.5) and np.diag(A).tolist() == [1.0] * 3
    assert A * A.T == pytest.approx(np.ones((3, 3)))                        # reziprok
    assert ahp.eigen_priority(A) == pytest.approx(W1, abs=1e-9)
    assert ahp.rgmm_priority(A) == pytest.approx(W1, abs=1e-9)


def test_noise_keeps_the_matrix_reciprocal_and_moves_the_priorities():
    A = ahp.pairwise_matrix(W1, 0.5, SplitMix64(2))
    assert A * A.T == pytest.approx(np.ones((3, 3)))
    assert not np.allclose(ahp.eigen_priority(A), W1, atol=1e-3)


def test_geometric_mean_vector_and_aip_by_hand():
    assert ahp.geometric_mean_vector(np.array([W1, W2])).tolist() == pytest.approx(GEO.tolist(), abs=1e-4)
    As = [ahp.pairwise_matrix(W1, 0.0, SplitMix64(1)), ahp.pairwise_matrix(W2, 0.0, SplitMix64(2))]
    assert ahp.aip(As, "geometric", "eigen").tolist() == pytest.approx(GEO.tolist(), abs=1e-4)
    assert ahp.aip(As, "arithmetic", "eigen").tolist() == pytest.approx([0.3, 0.3, 0.4], abs=1e-9)
    with pytest.raises(ValueError):
        ahp.aip(As, "harmonic")


def test_aij_of_consistent_matrices_equals_the_geometric_aip():
    As = [ahp.pairwise_matrix(W1, 0.0, SplitMix64(1)), ahp.pairwise_matrix(W2, 0.0, SplitMix64(2))]
    assert ahp.aij(As, "eigen").tolist() == pytest.approx(GEO.tolist(), abs=1e-4)
    assert ahp.aij(As, "rgmm").tolist() == pytest.approx(GEO.tolist(), abs=1e-4)


def test_tv_distance_by_hand():
    assert ahp.tv_distance(np.array([0.5, 0.5]), np.array([0.2, 0.8])) == pytest.approx(0.3)
    assert ahp.tv_distance(W1, W1) == 0.0


def test_the_hand_example_has_the_documented_distance_between_means():
    am = np.array([0.3, 0.3, 0.4])
    assert ahp.tv_distance(am, GEO) == pytest.approx(0.5 * (0.0430 + 0.0448 + 0.0018), abs=0.0005)     # |0,3 - 0,2570| + |0,3 - 0,3448| + |0,4 - 0,3982|, halbiert
