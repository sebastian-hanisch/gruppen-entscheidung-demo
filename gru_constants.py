"""Konstanten: Regler-Grenzen, Aggregationen, Verschiedenheit, feste Annahmen der Experimente und Presets."""

N_MIN, N_MAX, DEFAULT_N = 6, 20, 12
M_MIN, M_MAX, DEFAULT_M = 3, 8, 5
K_MIN, K_MAX, DEFAULT_K = 3, 9, 5
SEED_MAX, DEFAULT_SEED = 999999, 6            # je Preset geprüft (tests/test_presets.py)

KAPPA_CHOICES = (3.0, 10.0, 30.0, 100.0)      # Verschiedenheit: w_k ~ Dirichlet(kappa * Zentrum); klein = sehr verschieden, groß = fast gleich
DEFAULT_KAPPA = 10.0

AGGREGATIONS = ("mean_w", "mean_u", "gm_u", "median_u", "min_u", "borda", "mean_u_norm")
AGG_LABELS = {
    "mean_w": "Mittlere Gewichte (ein gemeinsames gewichtetes Summenmodell)",
    "mean_u": "Arithmetisches Mittel der Nutzen",
    "gm_u": "Geometrisches Mittel der Nutzen",
    "median_u": "Median der Nutzen (egalitärer Gegenpol zur Mitte)",
    "min_u": "Minimum der Nutzen (Maximin: der am schlechtesten gestellte zählt)",
    "borda": "Borda über die Ränge (nur als Gegenstück: blind für Abstände)",
    "mean_u_norm": "Mittel der auf Summe 1 normierten Nutzen",
}
AGG_SHORT = {"mean_w": "mittlere Gewichte", "mean_u": "Mittel der Nutzen", "gm_u": "geometrisch", "median_u": "Median", "min_u": "Minimum", "borda": "Borda",
             "mean_u_norm": "normiertes Mittel"}
DEFAULT_AGG = "mean_u"
POWER_AGGS = ("mean_w", "mean_u", "gm_u", "mean_u_norm")     # nur diese Aggregationen kennen eine Macht (gewichtetes Mittel)

FORMS = ("gleich", "gemischt")                # gleiche Nutzenform für alle (linear in den Rohwerten) oder je Mitglied eine von drei
FORM_LABELS = {"gleich": "gleich: alle linear in den Rohwerten", "gemischt": "gemischt: linear, konkav, Cobb-Douglas je Mitglied"}
DEFAULT_FORM = "gleich"
SCALE_CHOICES = (0.1, 1.0, 10.0)              # Faktor, mit dem Mitglied 1 seine Nutzen meldet
DEFAULT_SCALE = 1.0
POWER_CHOICES = (1.0, 2.0, 3.0, 10.0)         # Macht von Mitglied 1 (zählt f-fach im gewichteten Mittel)
DEFAULT_POWER = 1.0

SIGMA_CHOICES = (0.0, 0.3, 0.6)               # Rauschen der Paarvergleiche im AHP-Experiment
UTILITY_FLOOR = 1e-9                          # Untergrenze der Nutzen im geometrischen Mittel
TOL_TIE = 1e-9

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 12
EXP_M = 5
EXP_K = 5
EXP_KAPPA = 10.0
EXP_KS = (3, 5, 9)
OFFSET_GROUP = 104729                         # Strom der Gruppe: SplitMix64(seed + OFFSET_GROUP)
OFFSET_FORMS = 7919                           # Strom der Nutzenformen
OFFSET_AHP = 15485863                         # Strom des Urteilsrauschens
OFFSET_SCALE = 32452843                       # Strom der Skalenfaktoren im Experiment

COLOR_GROUP = "#2F6B65"
COLOR_BEST = "#D68A2E"
COLOR_LINE = "rgba(80,80,80,0.7)"
COLOR_MEMBER = ("#4c78a8", "#D68A2E", "#7a5195", "#8A96A6", "#c0392b", "#59a14f", "#b07aa1", "#9c755f", "#76b7b2")

PRESETS = {
    "Standardfall (5 Mitglieder, mäßig verschieden)": {"n": DEFAULT_N, "m": DEFAULT_M, "k": 5, "kappa": 10.0, "agg": "mean_u", "form": "gleich", "scale": 1.0, "power": 1.0,
                                                       "seed": DEFAULT_SEED},
    "Einstimmig": {"n": 12, "m": 5, "k": 5, "kappa": 100.0, "agg": "mean_u", "form": "gleich", "scale": 1.0, "power": 1.0, "seed": 1},
    "Zerstritten": {"n": 12, "m": 5, "k": 9, "kappa": 3.0, "agg": "mean_u", "form": "gleich", "scale": 1.0, "power": 1.0, "seed": 6},
    "Skala": {"n": 12, "m": 5, "k": 5, "kappa": 10.0, "agg": "mean_u", "form": "gleich", "scale": 10.0, "power": 1.0, "seed": 6},
    "Experte": {"n": 12, "m": 5, "k": 5, "kappa": 10.0, "agg": "mean_u", "form": "gleich", "scale": 1.0, "power": 10.0, "seed": 6},
}
PRESET_HELP = {
    "Standardfall (5 Mitglieder, mäßig verschieden)": "Fünf Mitglieder mit verschiedenen Gewichten: Der Sieger ist der Beste von jemandem, die Gruppenordnung ist die Ordnung von niemandem.",
    "Einstimmig": "Fast gleiche Gewichte (κ = 100): alle haben denselben Besten, die Gruppenordnung gleicht fast jeder Mitgliedsordnung.",
    "Zerstritten": "Neun Mitglieder mit sehr verschiedenen Gewichten (κ = 3): viele verschiedene Beste, großer Regret für das am schlechtesten gestellte Mitglied.",
    "Skala": "Mitglied 1 meldet seine Nutzen mit dem Faktor 10: Das arithmetische Mittel der Nutzen folgt ihm, das geometrische Mittel und das normierte Mittel nicht.",
    "Experte": "Mitglied 1 zählt zehnfach: Der Sieger wird sein Bester, der Regret der anderen steigt.",
}
