# ==========================================
# 🧪 TEST ANALYSIS COMPOUND (SHORT VERSION)
# ==========================================

import pytest

from coordchempy import analyse_compound

TESTS = [
    (
        "[Co(NH3)5(Cl)]2+",
        ['Pentaamminechlorocobalt(III)', 'Metal: Co', 'Formal oxidation state: +3', '[Ar] 4s0 3d6', '18', 'Octahedral']
    ),
    (
        "[Pt(NH3)2(Cl)2]",
        ['Diamminedichloroplatinum(II)', 'Metal: Pt', 'Formal oxidation state: +2', '6s0 5d8', '16', 'Square planar'],
    ),
]


@pytest.mark.parametrize("formula, expected_lines", TESTS)
def test_analyse_compound(formula, expected_lines):
    """Vérifie le texte renvoyé par le return de la fonction analyse_compound."""
    # 1. On récupère la grande chaîne de texte renvoyée par ta fonction
    result_text = analyse_compound(formula)

    # Si jamais la fonction renvoie None ou autre chose, on le convertit proprement en chaîne
    output_text = str(result_text)

    # 2. On vérifie que chaque élément clé est bien présent dans le texte (sans casser à cause des majuscules)
    for line in expected_lines:
        assert line.lower() in output_text.lower(), (
            f"Élément manquant pour {formula}.\n"
            f"Attendu : '{line}'\n"
            f"Texte reçu : \n{output_text}"
        )
