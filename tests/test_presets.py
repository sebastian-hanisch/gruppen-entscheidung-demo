"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import gru_constants as C
import gru_evaluation as ev
import gru_presets as P

STANDARD = "Standardfall (5 Mitglieder, mäßig verschieden)"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_and_choices():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and C.K_MIN <= p["k"] <= C.K_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["kappa"] in C.KAPPA_CHOICES and p["agg"] in C.AGGREGATIONS and p["form"] in C.FORMS and p["scale"] in C.SCALE_CHOICES and p["power"] in C.POWER_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[STANDARD]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice():
    assert set(C.AGG_LABELS) == set(C.AGGREGATIONS) == set(C.AGG_SHORT)
    assert set(C.FORM_LABELS) == set(C.FORMS)
    assert set(C.POWER_AGGS) <= set(C.AGGREGATIONS)


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200 and C.SWEEP_SEEDS[0] == 200000


def _analysis(name, **override):
    p = dict(C.PRESETS[name])
    p.update(override)
    return ev.analyse(ev.Settings(**p))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis(STANDARD)
    assert len(set(a.bests.tolist())) >= 3 and not a.nobody and not a.same_order and float(a.regret_true.max()) >= 0.2

    one = _analysis("Einstimmig")
    assert len(set(one.bests.tolist())) == 1 and one.same_order and float(one.regret_true.max()) <= 0.01

    quarrel = _analysis("Zerstritten")
    assert len(set(quarrel.bests.tolist())) >= 5 and float(quarrel.regret_true.max()) >= 0.3

    scale = _analysis("Skala")
    plain = _analysis("Skala", scale=1.0)
    assert scale.win != plain.win and scale.winners["gm_u"] == plain.winners["gm_u"] and scale.winners["mean_u_norm"] == plain.winners["mean_u_norm"]

    expert = _analysis("Experte")
    base = _analysis("Experte", power=1.0)
    assert expert.win == expert.bests[0] and base.win != base.bests[0]
    assert float(expert.regret_true[1:].mean()) > float(base.regret_true[1:].mean())
