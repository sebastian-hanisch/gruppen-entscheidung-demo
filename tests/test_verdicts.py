"""Verdikte an von Hand gesetzten Zeilen (DEMO-PLAYBOOK §12): jede Zahl im Text und die Auswahl der Zellen steht hier."""

import gru_constants as C
import gru_evaluation as ev


def _fit(k, kappa, nobody=0.02, same=0.105, tg=0.789, tm=0.671, rm=0.028, rx=0.097, unan=0.335):
    return {"k": k, "kappa": kappa, "distinct": 1.94, "unanimous": unan, "nobody": nobody, "same_order": same, "tau_group": tg, "tau_members": tm, "regret_mean": rm,
            "regret_max": rx, "matrices": 200}


def test_verdict_fit_picks_the_standard_and_the_quarrelsome_group():
    rows = [_fit(3, 3.0), _fit(5, 10.0), _fit(9, 3.0, rm=0.062, rx=0.272, unan=0.07), _fit(9, 100.0, rm=0.004, rx=0.022)]
    kind, text = ev.verdict_fit(rows)
    assert kind == "info"
    assert "ist der Sieger in 98,0 % der Matrizen der Beste von jemandem" in text
    assert "genau die Ordnung eines Mitglieds" in text and "nur in 10,5 %" in text
    assert "zwischen Gruppe und Mitglied 0,79, zwischen je zwei Mitgliedern 0,67" in text
    assert "im Mittel 2,8 % der eigenen Nutzenspannweite, im schlimmsten Mitglied 9,7 %" in text
    assert "(κ = 3) sind es 6,2 % und 27,2 %; nur 7,0 % der Gruppen sind einstimmig" in text


def test_verdict_aggregations_names_same_winner_regret_range_and_mixed_forms():
    rows = [{"agg": a, "same": s, "somebody": 0.98, "regret_mean": r, "regret_max": 0.1, "tau_mean": 0.78, "tau_min": 0.6}
            for a, s, r in (("mean_w", 1.0, 0.028), ("mean_u", 1.0, 0.028), ("gm_u", 0.985, 0.029), ("median_u", 0.85, 0.033), ("min_u", 0.83, 0.04), ("borda", 0.9, 0.031),
                            ("mean_u_norm", 0.99, 0.028))]
    kind, text = ev.verdict_aggregations({"rows": rows, "diff_same_form": 0.0, "diff_mixed_forms": 0.185, "k": 5, "kappa": 10.0, "matrices": 200})
    assert kind == "info"
    assert "geometrisches Mittel in 98,5 %, Median in 85,0 %, Minimum in 83,0 %, Borda in 90,0 %" in text
    assert "zwischen 2,8 und 4,0 %" in text
    assert "in 18,5 % der Matrizen (gleiche Form: 0,0 %)" in text


def _jud(sigma, tv_same, same, geo_tv, ari_tv, geo_win, ari_win):
    procs = {p: {"tv_am": geo_tv, "tv_gm": 0.02, "win": geo_win} for p in ev.PROCEDURES}
    procs["aip_arith_eigen"] = {"tv_am": ari_tv, "tv_gm": 0.07, "win": ari_win}
    return {"sigma": sigma, "procedures": procs, "aij_aip_tv": tv_same, "aij_aip_same": same, "matrices": 200}


def test_verdict_judgments_uses_the_lowest_and_the_highest_noise():
    rows = [_jud(0.0, 0.0, 1.0, 0.0608, 0.0, 0.885, 1.0), _jud(0.3, 0.0015, 0.995, 0.0627, 0.0185, 0.89, 0.955), _jud(0.6, 0.0061, 0.985, 0.0709, 0.0377, 0.88, 0.94)]
    kind, text = ev.verdict_judgments(rows)
    assert kind == "warning"
    assert "Bei Rauschen 60 % haben beide denselben Sieger in 98,5 % der Matrizen (Abstand der Gewichte 0,006; ohne Rauschen 0,000)" in text
    assert "zum arithmetischen Mittel der verborgenen Gewichte ist 0,071 gegen 0,038" in text
    assert "mit dem arithmetischen Mittel in 88,0 gegen 94,0 %" in text


def test_verdict_scale_counts_invariant_aggregations_and_names_the_power_effect():
    scale = [{"agg": a, "change": c, "dictator": 0.7} for a, c in (("mean_w", 0.0), ("mean_u", 0.077), ("gm_u", 0.0), ("median_u", 0.13), ("min_u", 0.182), ("borda", 0.0),
                                                                 ("mean_u_norm", 0.0))]
    power = [{"power": 1.0, "own_best": 0.77, "other_best": 0.714, "regret_own": 0.024, "regret_others": 0.029},
             {"power": 10.0, "own_best": 0.925, "other_best": 0.666, "regret_own": 0.003, "regret_others": 0.048}]
    kind, text = ev.verdict_scale({"scale": scale, "power": power, "k": 5, "kappa": 10.0, "matrices": 200})
    assert kind == "warning"
    assert "wechselt der Sieger beim Mittel der Nutzen in 7,7 %, beim Median in 13,0 %, beim Minimum in 18,2 % der Fälle; bei 4 der 7 Aggregationen nie" in text
    assert "Zählt ein Mitglied 10-fach, ist der Sieger in 92,5 % sein Bester (einfach: 77,0 %)" in text
    assert "von 2,9 auf 4,8 %" in text
    assert len(C.AGGREGATIONS) == 7
