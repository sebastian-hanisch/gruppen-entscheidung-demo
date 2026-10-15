# Gruppe von Entscheidern: wessen Gewichte, welche Aggregation? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-gruppen-entscheidung-demo.streamlit.app/)**

Vierzehntes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Kind von
[AHP](https://github.com/sebastian-hanisch/ahp-demo). Mehrere simulierte **Entscheider mit verschiedenen verborgenen Gewichten** bewerten dieselben Alternativen; ihre Urteile (oder
Prioritäten) werden auf sieben Arten aggregiert. Gemessen wird, was die Wahl jedem Mitglied kostet (**Regret**), wie nah die Gruppenordnung an den Mitgliedsordnungen liegt
(**Kendalls τ**) und woran die Aggregation hängt (Mittel, Skala, Macht).

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich). Die Mitglieder sind Orakel, keine
Menschen; es gibt keine richtige Gruppenwahl, nur Abstände zu den Mitgliedern.

## Warum dieses Problem

**Die Gruppenentscheidung passt fast immer zu jemandem, die Gruppenordnung fast nie, und die Skala der Urteile entscheidet mit.** Die Kernfrage der Demo: Wen entscheidet die Gruppe, wie
viel hängt an der Art des Mittels, und was kostet eine Skala oder eine Macht?

## Befunde und Korrekturen gegenüber dem Plan

- **Die Hypothese „die Rangfolge kann keiner wollen“ gilt für die Ordnung, nicht für den Sieger.** Bei 5 Mitgliedern und mäßiger Verschiedenheit ist der Sieger in 98,0 % der Matrizen
  der Beste von jemandem, die Gruppenordnung aber nur in 10,5 % genau die Ordnung eines Mitglieds.
- **Sie ist die Mitte:** Kendalls τ zwischen Gruppe und Mitglied ist 0,79, zwischen je zwei Mitgliedern 0,67; der mittlere Regret ist 2,8 % der eigenen Nutzenspannweite, der des am
  schlechtesten Gestellten 9,7 %. Bei 9 sehr verschiedenen Mitgliedern sind es 6,2 % und 27,2 %.
- **Die Art des Mittels entscheidet in jedem zehnten Fall:** Das geometrische Mittel der Gewichte statt des arithmetischen ändert den Sieger in 11,5 % der Matrizen. Urteile oder
  Prioritäten aggregieren fast nie (0,0 / 0,5 / 1,5 % bei Rauschen 0 / 30 / 60 %).
- **Die Skala ist die eigentliche Falle:** Meldet ein Mitglied seine Nutzen mit einem Faktor zwischen 0,1 und 10, wechselt der Sieger beim Mittel der Nutzen in 7,7 %, beim Median in
  13,0 %, beim Minimum in 18,2 % der Fälle; beim geometrischen Mittel, beim normierten Mittel, bei Borda und bei den mittleren Gewichten nie.
- **Mittlere Gewichte und Mittel der Nutzen sind dasselbe, solange alle Mitglieder dieselbe lineare Form haben;** bei gemischten Formen unterscheiden sie sich in 18,5 % der Matrizen.
- **Jedes Experiment startet auf eigenen Klick** (einige Sekunden bis rund 15 Sekunden).

## Modell

- **Matrix:** n Alternativen (6 bis 20, Standard 12), m Kriterien (3 bis 8), Werte auf die feste Nenn-Spannweite normiert.
- **Gruppe:** K Mitglieder (3 bis 9, Standard 5) mit Gewichten w_k ~ Dirichlet(κ · c) um ein gemeinsames Zentrum c ~ Dirichlet(1); κ = 3 / 10 / 30 / 100 (sehr verschieden bis fast
  gleich). Nutzen linear in den Rohwerten oder gemischt (linear, konkav, Cobb-Douglas je Mitglied).
- **Aggregationen:** mittlere Gewichte, Mittel der Nutzen, geometrisches Mittel, Median, Minimum, Borda (nur als Gegenstück), Mittel der auf Summe 1 normierten Nutzen; die gewichteten
  Mittel kennen eine Macht (Mitglied 1 zählt 1-, 2-, 3- oder 10-fach).
- **Maße:** Regret = (bester eigener Nutzen − Nutzen der Wahl) / eigene Nutzenspannweite; τ zwischen Gruppenwertung und Mitgliedsnutzen.
- **Urteile oder Prioritäten:** AHP-Paarvergleiche der Kriteriengewichte je Mitglied mit Rauschen σ (0 / 0,3 / 0,6); AIJ (geometrisch gemittelte Matrix, Eigenvektor oder Zeilen-GM) gegen
  AIP (geometrisches oder arithmetisches Mittel der Prioritäten).
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), 12 Alternativen, 5 Kriterien, 5 Mitglieder (κ = 10, soweit nicht variiert), unabhängig von den Reglern.

Formeln im 📐-Expander der App.

## Methodik

1. **Dieselbe Matrix, dieselbe Gruppe, alle Aggregationen:** Vergleiche sind gepaart.
2. **Messgröße ist der Abstand zu den Mitgliedern**, nicht eine Wahrheit: Beste, Ordnung, τ, Regret.
3. **Gegenproben:** gleiche gegen gemischte Nutzenformen, Skala eines Mitglieds, Macht eines Mitglieds.
4. **Urteile gegen Prioritäten:** geometrisch gegen arithmetisch, Eigenvektor gegen Zeilen-GM, mit und ohne Rauschen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: `tests/test_claims.py`. 200 feste Matrizen, 12 Alternativen, 5 Kriterien. **Der Standardfehler eines Anteils ist 1,5 bis 3,5 Punkte; Unterschiede unter etwa 5 Punkten sind
nicht belastbar und stehen nicht als Aussage.**

| Frage | Befund | Test |
|---|---|---|
| Passt der Sieger zu jemandem? 5 Mitglieder, κ = 3 / 10 / 30 / 100 | Sieger ist Bester von jemandem in **97,5 / 98,0 / 99,5 / 100 %**; einstimmig sind 15,5 / 33,5 / 51,0 / 65,0 %; verschiedene Beste im Mittel 2,39 / 1,94 / 1,64 / 1,40. 9 Mitglieder, κ = 3: 99,0 %, einstimmig 7,0 %. | `test_the_winner_fits_somebody_but_the_order_rarely_fits_a_member` |
| Ist die Gruppenordnung die eines Mitglieds? (dieselben Zellen) | **7,0 / 10,5 / 21,0 / 48,0 %**; 9 Mitglieder: 7,0 / 15,0 / 28,5 / 63,0 %; 3 Mitglieder: 3,0 / 11,5 / 16,0 / 41,5 %. | dasselbe |
| Die Mitte: τ Gruppe–Mitglied gegen Mitglied–Mitglied; Regret (Mittel / schlimmstes) | κ = 3: 0,69 gegen 0,52, 5,4 % / 18,0 %; **κ = 10: 0,79 gegen 0,67, 2,8 % / 9,7 %**; κ = 30: 0,87 gegen 0,80, 1,2 % / 4,4 %; κ = 100: 0,93 gegen 0,88, 0,4 % / 1,7 %; 9 Mitglieder, κ = 3: 6,2 % / 27,2 %. | `test_the_group_order_is_the_middle_of_the_members` |
| Aggregationen im Vergleich (5 Mitglieder, κ = 10): gleicher Sieger wie das Mittel der Nutzen | mittlere Gewichte **100 %**, geometrisch **98,5 %**, Median **85,0 %**, Minimum **83,0 %**, Borda **90,0 %**, normiertes Mittel 99,0 %; Regret im Mittel 2,8 bis 4,0 %, schlimmster 9,5 bis 10,9 %; τ 0,75 bis 0,80. | `test_aggregations_differ_in_one_case_of_ten_or_more` |
| Gemischte Nutzenformen: mittlere Gewichte gegen Mittel der Nutzen, Sieger verschieden | gleiche Form **0,0 %**, gemischte Formen **18,5 %**. | dasselbe |
| Urteile oder Prioritäten (Rauschen 0 / 30 / 60 %): Sieger gleich bei AIJ-Eigenvektor und AIP-geometrisch; Abstand der Gewichte (TV) | **100 / 99,5 / 98,5 %**; TV 0,000 / 0,0015 / 0,0061. | `test_geometric_against_arithmetic_matters_more_than_judgments_against_priorities` |
| Geometrisch gegen arithmetisch: Abstand zum arithmetischen Mittel der verborgenen Gewichte (TV); Sieger wie mit dem arithmetischen Mittel | AIJ geometrisch **0,061 / 0,063 / 0,071**, 88,5 / 89,0 / 88,5 %; AIP arithmetisch **0,000 / 0,019 / 0,038**, 100 / 95,5 / 94,0 %. | dasselbe |
| Skala: ein Mitglied meldet mit c in [0,1; 10], Sieger wechselt (mittlere Gewichte, Mittel der Nutzen, geometrisch, Median, Minimum, Borda, normiertes Mittel) | **0,0 / 7,7 / 0,0 / 13,0 / 18,2 / 0,0 / 0,0 %**; Sieger = Bester des skalierten Mitglieds (Faktor ≥ 3): Mittel der Nutzen 84,0 % gegen 72,1 % bei mittleren Gewichten. | `test_the_scale_of_one_member_flips_the_mean_but_not_the_geometric_mean` |
| Macht von Mitglied 1 (1 / 2 / 3 / 10-fach, Mittel der Nutzen): Sieger = sein Bester; Regret seiner / der anderen | **77,0 / 81,5 / 84,5 / 92,5 %**; 2,4 / 1,6 / 1,2 / 0,3 % gegen 2,9 / 3,3 / 3,6 / 4,8 %. | `test_power_buys_the_winner_and_costs_the_others` |

## Ehrliche Grenzen

- **Es gibt keine richtige Gruppenwahl.** Die Demo misst Abstände zu den Mitgliedern; jede Aggregation ist eine Annahme darüber, wie die Mitglieder zählen.
- **Ehrliche Mitglieder.** Keine Manipulation, keine Strategie, keine Koalitionen; die Skala ist ein Faktor, kein Nullpunkt.
- **Verschiedenheit nur in den Gewichten.** Die Mitglieder sehen dieselbe Matrix und dieselben Kriterien.
- **Der Regret ist normiert** (eigene Nutzenspannweite); andere Normierungen verschieben die Zahlen.
- **Die Mitglieder sind Orakel.** Die Originalaufsätze zu AIJ und AIP und zur Reziprozität des geometrischen Mittels sind hier nicht gelesen; die Definitionen sind die einfachen, durch
  Orakel gesichert, deshalb stehen im Text keine Autoren.
- **Kleine Unterschiede sind Rauschen.** Geometrisch gegen normiertes Mittel (0,5 Punkte Siegerwechsel) und die Spalten der Aggregationen mit unter 5 Punkten Unterschied.

## Verifikation

- **Kern:** Aggregationen, Regret, Ränge, τ, Paarvergleiche, Eigenvektor und Zeilen-GM gegen die Handrechnung der Mini-Instanz (`tests/conftest.py`: zwei Mitglieder, drei Alternativen;
  B ist der Kompromiss, den niemand an erster Stelle hat; ein Faktor 10 kippt das Mittel auf C, das geometrische und das normierte Mittel bleiben bei B).
- **Orakel** (`tests/test_oracle_group.py`): mittlere Gewichte gleich Mittel der Nutzen bei gleicher Form (Gegenprobe gemischt); Skala ohne Wirkung auf geometrisch, normiert und Borda;
  AIJ gleich AIP geometrisch mit Zeilen-GM; Einstimmigkeit und Pareto; Macht gegen unendlich.
- **Randfälle** (`tests/test_edge_cases.py`): ein Mitglied, zwei Alternativen, Nutzen null im geometrischen Mittel, Gleichstände, kleinste und größte Gruppe, Skala und Macht an den Grenzen.
- **Gruppe** (`tests/test_group.py`): Gamma-Zufallszahlen gegen Mittelwert und Varianz, Dirichlet, Verschiedenheit nimmt mit κ ab, Nutzenformen.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** jede Zahl dieser README in `tests/test_claims.py`; Verdikte an von Hand gesetzten Zeilen in `test_verdicts.py`; Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Mutationstest** vor dem ersten Commit (`tools/mutate_demo.py` der Website, mit `--budget` und `--timeout`); Stand in `tests/mutation_state.json`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, jede Auswahl), alle Presets, Permalink, PDF und jedes Experiment erst nach eigenem Klick, Footer,
  Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `gru_constants.py` | Regler-Grenzen, Aggregationen, Verschiedenheit, feste Annahmen der Experimente, Presets |
| `gru_group.py` | Gamma- und Dirichlet-Ziehung, Gewichte und Nutzen der Mitglieder, Skala, Macht |
| `gru_aggregate.py` | sieben Aggregationen, Regret, Ränge, Kendalls τ-b |
| `gru_ahp.py` | Paarvergleiche, Eigenvektor, Zeilen-GM, AIJ und AIP |
| `gru_evaluation.py` | Fall der App, vier Experimente, Verdikte (reine Funktionen) |
| `gru_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `gru_visualization.py` | Mitglieder, Gruppenwertung, τ-Matrix, Regret, Tabellen und Abbildungen der Experimente (Achsen gesperrt) |
| `gru_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Mini-Instanz, Gruppe, Aggregation, AHP, Orakel, Randfälle, Generator, Auswertung, Verdikte, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Manipulation und Strategie der Mitglieder:** die Mitglieder melden ehrlich.
- **Koalitionen, Verhandlung und Diskussion:** die Gruppe ist eine Punktwolke von Gewichten, kein Prozess.
- **Verschiedene Kriteriensätze oder Alternativen je Mitglied:** alle sehen dieselbe Matrix.
- **Weitere Rangaggregationen (Copeland, Kemeny):** die Rangaggregation steht in der Turnier-Linie; hier nur Borda als Gegenstück.
- **Einen „wahren“ Gruppennutzen:** die Mitglieder haben keinen gemeinsamen, die Demo misst nur Abstände.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html).
