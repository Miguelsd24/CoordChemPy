# ==========================================
# 🧪 TEST ANALYSIS COMPOUND (SHORT VERSION)
# ==========================================

import pytest

from coordchempy import analyse_compound

# Liste de test minimaliste : ( "Formule", [ "Lignes attendues" ] )
TESTS = [
    (
        "[Co(NH3)5(Cl)]2+",
        [
            "IUPAC: pentaamminechlorocobalt(III)",
            "Oxidation: Co (3+)",
            "Config: [Ar] 4s0 3d6",
            "Electrons: 18",
            "Geometry: Octahedral",
        ],
    ),
    (
        "[Pt(NH3)2(Cl)2]",
        [
            "IUPAC: diamminedichloroplatinum(II)",
            "Oxidation: Pt (2+)",
            "Config: [Xe] 6s0 5d8",
            "Electrons: 16",
            "Geometry: Square planar",
        ],
    ),
    (
        "[Co(en)3]3+",
        [
            "IUPAC: tris(ethane-1,2-diamine)cobalt(III)",
            "Oxidation: Co (3+)",
            "Geometry: Octahedral",
        ],
    ),
]


@pytest.mark.parametrize("formula, expected_lines", TESTS)
def test_analyse_compound(formula, expected_lines, capsys):
    """Vérifie les prints de la fonction via capsys."""
    analyse_compound(formula)

    captured = capsys.readouterr()
    output_text = captured.out

    for line in expected_lines:
        assert line in output_text, f"Ligne manquante pour {formula} : '{line}'"
