"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

import numpy as np
from fpdf import FPDF

import gru_constants as C
from gru_evaluation import Analysis, de

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "↔": "<->", "–": "-", "—": "-", "τ": "tau", "κ": "kappa", "×": "x", "≥": ">=", "≤": "<=", "≠": "!=", "−": "-", "€": "EUR",
            "·": "*", "★": "*"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    names = [f"A{i + 1}" for i in range(s.n)]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Gruppe von Entscheidern: wessen Gewichte, welche Aggregation?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{s.n} Alternativen, {s.m} Kriterien, {s.k} Mitglieder (Verschiedenheit kappa = {de(s.kappa, 0)}), Aggregation: {C.AGG_LABELS[s.agg]}. Nutzenform: "
                            f"{C.FORM_LABELS[s.form]}. Skala von Mitglied 1: {de(s.scale, 1)}, Macht von Mitglied 1: {de(s.power, 0)}. Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(f"Sieger: {names[a.win]}"
                            f"{' (der Beste von keinem Mitglied)' if a.nobody else ' (der Beste von mindestens einem Mitglied)'}. Gruppenordnung "
                            f"{'gleich' if a.same_order else 'nicht gleich'} der Ordnung eines Mitglieds; mittleres tau Gruppe-Mitglied {de(float(a.taus.mean()), 2)}, "
                            f"zwischen Mitgliedern {de(a.member_tau, 2)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Mitglieder (Bester, Regret der Wahl, tau zur Gruppenordnung)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for i in range(s.k):
        pdf.cell(0, 5, _s(f"M{i + 1}: Bester {names[a.bests[i]]}, Regret {de(100 * float(a.regret_true[i]), 1)} %, tau {de(float(a.taus[i]), 2)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Alternativen nach der Gruppenwertung"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for i in np.argsort(a.rank)[:15]:
        pdf.cell(0, 5, _s(f"Rang {int(a.rank[i]) + 1}: {names[i]}, Wertung {de(float(a.score[i]), 3)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Es gibt keine richtige Gruppenwahl: Die Demo misst, was die Wahl den Mitgliedern kostet. Die Mitglieder sind simulierte Orakel, keine Menschen. "
                            "Quelle: gruppen-entscheidung-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
