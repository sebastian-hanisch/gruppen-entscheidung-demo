"""Gruppe von Entscheidern: wessen Gewichte, welche Aggregation? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vierzehntes Stück der Linie Mehrkriterien-Ranking (Kind von AHP). Dieselbe neutrale Entscheidungsmatrix; mehrere simulierte Entscheider mit verschiedenen verborgenen Gewichten.
Sieben Aggregationen (Mittel der Gewichte, Mittel der Nutzen, geometrisches Mittel, Median, Minimum, Borda, normiertes Mittel), Regret und τ, Urteile oder Prioritäten (AHP), Skala, Macht.
Kern: gru_group.py, gru_aggregate.py, gru_ahp.py; Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import gru_constants as C
from gru_evaluation import (
    Settings,
    analyse,
    de,
    experiment_aggregations,
    experiment_fit,
    experiment_judgments,
    experiment_scale_power,
    verdict_aggregations,
    verdict_fit,
    verdict_judgments,
    verdict_scale,
)
from gru_pdf_export import build_pdf
from gru_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from gru_visualization import (
    aggregations_table,
    alt_names,
    build_fit,
    build_group_score,
    build_regret,
    build_tau_matrix,
    fit_table,
    judgments_table,
    members_table,
    power_table,
    scale_table,
    style_members,
    winners_table,
)

st.set_page_config(page_title="Gruppe von Entscheidern – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Experiment 1 rechnet …")
def _exp_fit():
    return experiment_fit()


@st.cache_data(show_spinner="Experiment 2 rechnet …")
def _exp_aggregations():
    return experiment_aggregations()


@st.cache_data(show_spinner="Experiment 3 rechnet …")
def _exp_judgments():
    return experiment_judgments()


@st.cache_data(show_spinner="Experiment 4 rechnet …")
def _exp_scale():
    return experiment_scale_power()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("👥 Gruppe von Entscheidern: wessen Gewichte, welche Aggregation?")
st.markdown(
    """
**Die Gruppenentscheidung passt fast immer zu jemandem, die Gruppenordnung fast nie, und die Skala der Urteile entscheidet mit.** Dies ist das vierzehnte Stück der Linie
**Mehrkriterien-Ranking**: Mehrere simulierte Entscheider haben **verschiedene verborgene Gewichte** für dieselben Alternativen; die Demo aggregiert ihre Urteile (oder Prioritäten) auf
sieben Arten und misst, was die Wahl jedem Mitglied kostet (**Regret**) und wie nah die Gruppenordnung an den Mitgliedsordnungen liegt (**Kendalls τ**). Es gibt keine richtige
Gruppenwahl: Jede Aggregation ist eine Annahme darüber, wie die Mitglieder zählen.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Gruppenaggregation", expanded=True):
    st.markdown(
        """
1. **Mitglieder:** Jedes hat eigene Gewichte um ein gemeinsames Zentrum; κ steuert die Verschiedenheit (klein = sehr verschieden, groß = fast gleich). Daraus entsteht je Mitglied
   ein Nutzen je Alternative.
2. **Aggregation:** *mittlere Gewichte* (ein gemeinsames Summenmodell), *Mittel der Nutzen*, *geometrisches Mittel*, *Median*, *Minimum* (der am schlechtesten Gestellte zählt),
   *Borda* (nur die Ränge), *Mittel der normierten Nutzen*.
3. **Regret** eines Mitglieds: (bester eigener Nutzen − Nutzen der Gruppenwahl) / eigene Nutzenspannweite. **τ:** wie ähnlich die Gruppenordnung der Ordnung des Mitglieds ist.
4. **Skala:** Meldet ein Mitglied seine Nutzen mit einem anderen Faktor, ändert sich das Mittel der Nutzen, das geometrische Mittel nicht.
5. **Macht:** Ein Mitglied kann mehrfach zählen (nur bei den gewichteten Mitteln).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Gruppe**")
    k = st.slider("Mitglieder", *bounds("k_slider"), key="k_slider", help="Wie viele Entscheider die Alternativen bewerten.")
    kappa = st.selectbox("Verschiedenheit κ", C.KAPPA_CHOICES, key="kappa_select", format_func=lambda v: de(v, 0),
                         help="Wie weit die Gewichte der Mitglieder um ihr gemeinsames Zentrum streuen: 3 = sehr verschieden, 100 = fast gleich.")
    form = st.selectbox("Nutzenform der Mitglieder", C.FORMS, key="form_select", format_func=lambda x: C.FORM_LABELS[x],
                        help="Gleich: alle linear in den Rohwerten. Gemischt: jedes Mitglied hat eine von drei Formen; dann sind mittlere Gewichte und Mittel der Nutzen nicht mehr dasselbe.")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen bewertet werden.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider", help="Nach wie vielen Kriterien jede Alternative gemessen ist.")
    st.markdown("**Aggregation**")
    agg = st.selectbox("Aggregation", C.AGGREGATIONS, key="agg_select", format_func=lambda x: C.AGG_LABELS[x],
                       help="Wie die Urteile der Mitglieder zu einer Gruppenwertung werden.")
    scale = st.selectbox("Skala von Mitglied 1", C.SCALE_CHOICES, key="scale_select", format_func=lambda v: f"Faktor {de(v, 1)}",
                         help="Mit welchem Faktor Mitglied 1 seine Nutzen meldet: dieselbe Ordnung, andere Skala.")
    power = st.selectbox("Macht von Mitglied 1", C.POWER_CHOICES, key="power_select", format_func=lambda v: f"{de(v, 0)}-fach",
                         disabled=agg not in C.POWER_AGGS, help="Wie oft Mitglied 1 im gewichteten Mittel zählt. Median, Minimum und Borda kennen keine Macht.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "k_slider": int(k), "kappa_select": float(kappa), "agg_select": agg, "form_select": form, "scale_select": float(scale),
                   "power_select": float(power), "seed_input": int(seed)})

settings = Settings(int(n), int(m), int(k), float(kappa), agg, form, float(scale), float(power), int(seed))
a = _analysis(settings)
names = alt_names(settings.n)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sieger der Gruppe", names[a.win], delta="Bester von keinem Mitglied" if a.nobody else f"Bester von {int((a.bests == a.win).sum())} von {settings.k} Mitgliedern",
          delta_color="off", help="Die Alternative mit der größten Gruppenwertung der gewählten Aggregation.")
c2.metric("Verschiedene Beste", f"{len(set(a.bests.tolist()))} von {settings.k}", delta="einstimmig" if len(set(a.bests.tolist())) == 1 else "uneinig", delta_color="off",
          help="Wie viele verschiedene Alternativen die Mitglieder als ihren Besten haben.")
c3.metric("Regret (Mittel / schlimmstes)", f"{de(100 * float(a.regret_true.mean()), 1)} % / {de(100 * float(a.regret_true.max()), 1)} %",
          delta=f"τ Gruppe–Mitglied {de(float(a.taus.mean()), 2)}", delta_color="off",
          help="Was die Wahl die Mitglieder kostet, in Anteilen ihrer eigenen Nutzenspannweite; darunter, wie ähnlich die Gruppenordnung den Mitgliedsordnungen ist.")
c4.metric("Ordnung wie ein Mitglied", "ja" if a.same_order else "nein", delta=f"τ zwischen Mitgliedern {de(a.member_tau, 2)}", delta_color="off",
          help="Ob die Gruppenordnung genau die Ordnung eines Mitglieds ist.")

left, right_col = st.columns([3, 2])
with left:
    st.markdown("**Gruppenwertung** (grün: Sieger; orange: Bester eines Mitglieds)")
    st.plotly_chart(build_group_score(a), width="stretch", key="score_chart")
with right_col:
    st.markdown("**Regret je Mitglied**")
    st.plotly_chart(build_regret(a), width="stretch", key="regret_chart")

st.dataframe(style_members(members_table(a)), hide_index=True, width="stretch")
st.caption("Regret und τ beziehen sich auf die wahren Nutzen der Mitglieder (ohne die Skala von Mitglied 1). Größtes Gewicht: das Kriterium, das dem Mitglied am wichtigsten ist.")

tab_t, tab_w = st.tabs(["τ zwischen Mitgliedern und Gruppe", "Sieger aller Aggregationen"])
with tab_t:
    st.plotly_chart(build_tau_matrix(a), width="stretch", key="tau_chart")
with tab_w:
    st.dataframe(winners_table(a), hide_index=True, width="stretch")
    st.caption("Mit derselben Skala und Macht wie oben. Mittlere Gewichte und Mittel der Nutzen stimmen bei gleicher Nutzenform immer überein.")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="gruppenentscheidung.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wen entscheidet die Gruppe?")
if a.nobody:
    st.warning(f"Der Sieger {names[a.win]} ist der Beste von keinem Mitglied: ein Kompromiss, den niemand an erster Stelle hat. Das ist bei {settings.k} Mitgliedern selten, aber es kommt vor.")
else:
    st.info(f"Der Sieger {names[a.win]} ist der Beste von {int((a.bests == a.win).sum())} von {settings.k} Mitgliedern; die Gruppenordnung "
            f"{'ist' if a.same_order else 'ist nicht'} die Ordnung eines Mitglieds. Der schlimmste Regret ist {de(100 * float(a.regret_true.max()), 1)} % der Nutzenspannweite.")
if settings.scale != 1.0 and agg in ("mean_u", "median_u", "min_u"):
    st.caption(f"Mitglied 1 meldet mit dem Faktor {de(settings.scale, 1)}: Diese Aggregation hängt an der Skala der Urteile (das geometrische Mittel und das normierte Mittel nicht).")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption(f"{len(C.SWEEP_SEEDS)} feste Matrizen (SWEEP_SEEDS), {C.EXP_N} Alternativen, {C.EXP_M} Kriterien, unabhängig von den Reglern; jedes Experiment rechnet einige Sekunden bis rund "
           "15 Sekunden und startet erst auf Klick. Standardfehler der Anteile 1,5 bis 3,5 Punkte: Unterschiede unter etwa 5 Punkten sind nicht belastbar.")

st.markdown("### Experiment 1: Passt die Gruppenentscheidung zu jemandem?")
st.caption("Mittel der Nutzen, gegen Mitgliederzahl und Verschiedenheit: Sieger, Ordnung, τ und Regret.")
if st.button("▶ Experiment 1 rechnen", key="exp_fit_start"):
    st.session_state["exp_fit"] = _exp_fit()
if st.session_state.get("exp_fit"):
    rows = st.session_state["exp_fit"]
    st.plotly_chart(build_fit(rows), width="stretch", key="fit_chart")
    verdict(verdict_fit(rows))
    st.dataframe(fit_table(rows), hide_index=True, width="stretch")

st.markdown("### Experiment 2: Welche Aggregation?")
st.caption("Sieben Aggregationen bei 5 Mitgliedern (κ = 10), und der Bruch mittlere Gewichte gegen Mittel der Nutzen bei gemischten Nutzenformen.")
if st.button("▶ Experiment 2 rechnen", key="exp_agg_start"):
    st.session_state["exp_agg"] = _exp_aggregations()
if st.session_state.get("exp_agg"):
    res = st.session_state["exp_agg"]
    verdict(verdict_aggregations(res))
    st.dataframe(aggregations_table(res), hide_index=True, width="stretch")

st.markdown("### Experiment 3: Urteile oder Prioritäten?")
st.caption("Paarvergleiche der Kriteriengewichte je Mitglied (5 Mitglieder, κ = 10) mit Rauschen: Urteile aggregieren (AIJ) gegen Prioritäten aggregieren (AIP), geometrisch gegen arithmetisch.")
if st.button("▶ Experiment 3 rechnen", key="exp_jud_start"):
    st.session_state["exp_jud"] = _exp_judgments()
if st.session_state.get("exp_jud"):
    rows = st.session_state["exp_jud"]
    verdict(verdict_judgments(rows))
    st.dataframe(judgments_table(rows), hide_index=True, width="stretch")

st.markdown("### Experiment 4: Skala und Macht")
st.caption("Ein Mitglied meldet seine Nutzen mit einem Faktor zwischen 0,1 und 10; ein Mitglied zählt 1-, 2-, 3- oder 10-fach.")
if st.button("▶ Experiment 4 rechnen", key="exp_scale_start"):
    st.session_state["exp_scale"] = _exp_scale()
if st.session_state.get("exp_scale"):
    res = st.session_state["exp_scale"]
    verdict(verdict_scale(res))
    st.dataframe(scale_table(res), hide_index=True, width="stretch")
    st.dataframe(power_table(res), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es gibt keine richtige Gruppenwahl** | Die Demo misst Abstände zu den Mitgliedern, nicht eine Wahrheit; jede Aggregation ist eine Annahme darüber, wie die Mitglieder zählen. | – |
| **Ehrliche Mitglieder** | Keine Manipulation, keine Strategie, keine Koalitionen; eine übertriebene Meldung wirkt beim Mittel der Nutzen (Experiment 4). | – |
| **Verschiedenheit nur in den Gewichten** | Die Mitglieder sehen dieselbe Matrix und dieselben Kriterien; verschiedene Kriteriensätze oder Wertungen sind nicht Teil. | Verfahrensvergleich |
| **Der Regret ist normiert** | Er misst an der eigenen Nutzenspannweite; andere Normierungen verschieben die Zahlen. | – |
| **Die Skala ist hier ein Faktor** | Echte Mitglieder unterscheiden sich in Nullpunkt und Spannweite, nicht nur im Faktor. | – |
| **Die Mitglieder sind Orakel** | Keine Menschen, die diskutieren und ihre Meinung ändern. Die Originalaufsätze zu AIJ und AIP sind hier nicht geprüft. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gruppe.** $K$ Mitglieder mit Gewichten $w_k \sim \mathrm{Dirichlet}(\kappa\, c)$ um ein Zentrum $c$; Nutzen $u_k(a)$ der Alternative $a$ (hier $u_k(a) = w_k^\top x(a)$ mit den auf die
Nenn-Spannweite normierten Werten $x$).

**Aggregationen** mit Mitgliedsgewichten $\pi_k$ ($\sum \pi_k = 1$):
$U^{\mathrm{AM}}(a) = \sum_k \pi_k u_k(a)$ (bei gleicher linearer Form gleich $x(a)^\top \sum_k \pi_k w_k$, den mittleren Gewichten),
$U^{\mathrm{GM}}(a) = \prod_k u_k(a)^{\pi_k}$, $U^{\min}(a) = \min_k u_k(a)$, Median, Borda $\sum_k (n - 1 - \mathrm{Rang}_k(a))$ und das Mittel der normierten Nutzen
$\sum_k \pi_k\, u_k(a) / \sum_b u_k(b)$.

**Regret** von $k$ bei der Wahl $a^\ast$: $\dfrac{\max_a u_k(a) - u_k(a^\ast)}{\max_a u_k(a) - \min_a u_k(a)}$.

**Skala.** Ersetzt ein Mitglied $u_k$ durch $c\, u_k$, ändert sich $U^{\mathrm{AM}}$ um $\pi_k (c - 1) u_k$, $U^{\mathrm{GM}}$ um den gemeinsamen Faktor $c^{\pi_k}$ (die Ordnung bleibt), Borda und
das normierte Mittel gar nicht.

**Urteile oder Prioritäten (AHP).** Mitglied $k$ gibt eine reziproke Matrix $A_k$ mit $a_{ij} \approx w_{k,i} / w_{k,j}$. *AIJ:* $A^G_{ij} = \prod_k (A_k)_{ij}^{1/K}$, danach Priorität
(Eigenvektor oder Zeilen-GM). *AIP:* erst $p_k$ je Mitglied, danach das geometrische oder arithmetische Mittel. Mit der Zeilen-GM-Priorität sind AIJ und AIP-geometrisch dasselbe.

**Literatur:** AIJ und AIP sowie die Reziprozität des geometrischen Mittels sind hier nicht im Original geprüft und deshalb nicht mit Autoren angegeben.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
