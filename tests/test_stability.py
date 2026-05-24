import pytest

from coordchempy import StabilityEngine

TESTS = [
    (
        "[Mn(CO)6]+",
        ["[Cr(en)3]3+", "[Ni(NH3)4]2+"],
    ),  # Mn > Cr and Ni
    ("[Cr(en)3]3+", ["[Ni(NH3)4]2+"]),  # Cr > Ni
]


@pytest.mark.parametrize("formula, target_formulas", TESTS)
def test_stability_qualitative_order(formula, target_formulas):
    """
    Verifies, based on specified parameters, that each complex has a higher score
    than the less stable complexes in the list.
    """

    engine = StabilityEngine(formula)
    result = engine.final_score()

    assert result is not None, f"The engine return None for {formula}"
    current_score = result.total

    for target in target_formulas:
        target_engine = StabilityEngine(target)
        target_result = target_engine.final_score()

        assert target_result is not None, (
            f"The engine return None for the target {target}"
        )
        target_score = target_result.total

        assert current_score > target_score, (
            f"Qualitative error for {formula}.\n"
            f"The score of {formula} ({current_score}) should be larger"
            f"than {target} ({target_score})."
        )
