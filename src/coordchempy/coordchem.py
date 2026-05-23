# ==========================================
# IMPORTS
# ==========================================
"""
Core imports required for:
- Data loading
- Numerical calculations
- 3D visualisation
- Rendering
- Coordination chemistry analysis
"""

import io
import json
import math
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import py3Dmol
import roman
from ase import Atoms
from ase.io import write
from IPython import get_ipython
from IPython.display import Markdown, display


# ==========================================
# DATABASE LOADING
# ==========================================
"""
Load all chemistry databases used by the engine:
- Metals
- Ligands
- Counter ions
"""

BASE_DIR = Path(__file__).resolve().parent.parent.parent

with open(BASE_DIR / "data" / "metals.json", encoding="utf-8") as file:
    data_metals = json.load(file)

with open(BASE_DIR / "data" / "ligands.json", encoding="utf-8") as file:
    data_ligands = json.load(file)

with open(BASE_DIR / "data" / "counter_ions.json", encoding="utf-8") as file:
    data_counter_ions = json.load(file)


# ==========================================
# GENERAL UTILITIES
# ==========================================
"""
Utility functions used throughout the project for:
- Database searches
- Charge conversion
- Common validations
"""


def find_ligand(ligand_input):
    """
    Return the canonical ligand key from the database.

    The search is performed using:
    1. Direct formula lookup
    2. Ligand abbreviation lookup
    """

    if ligand_input in data_ligands:
        return ligand_input

    for ligand_key, properties in data_ligands.items():
        if properties.get("abbr") == ligand_input:
            return ligand_key

    return False


def find_counter_ion(counter_ion_input):
    """
    Return the canonical counter-ion key from the database.

    The search is performed using:
    1. Direct key lookup
    2. Formula lookup
    3. Abbreviation lookup
    """

    if counter_ion_input in data_counter_ions:
        return counter_ion_input

    for ion_key, properties in data_counter_ions.items():
        if properties.get("formula") == counter_ion_input:
            return ion_key

    for ion_key, properties in data_counter_ions.items():
        if properties.get("abbr") == counter_ion_input:
            return ion_key

    return False


def transform_charge(charge):
    """
    Convert a charge string into an integer.

    Supported formats:
    '+', '-', '2+', '+2', '3-', '-3', '0'
    """

    if charge is None or charge == "":
        return 0

    charge = charge.strip()

    try:
        return int(charge)

    except ValueError:
        pass

    if charge == "+":
        return 1

    if charge == "-":
        return -1

    match = re.match(r"(\d+)([+-])", charge)

    if match:
        value = int(match.group(1))
        sign = match.group(2)

        return value if sign == "+" else -value

    match = re.match(r"([+-])(\d+)", charge)

    if match:
        sign = match.group(1)
        value = int(match.group(2))

        return value if sign == "+" else -value

    raise ValueError(f"Invalid charge format: {charge}")


# ==========================================
# FORMULA VALIDATION
# ==========================================
"""
Validation utilities used before parsing.

These functions ensure:
- Correct coordination sphere syntax
- Correct counter-ion syntax
- Safe parsing throughout the project
"""


def formula_verification(formula):
    """
    Validate the coordination sphere syntax.

    Expected format example:
    [Fe(NH3)6]3+
    [PtCl2(NH3)2]
    """

    if not isinstance(formula, str):
        raise ValueError(
            "Coordination sphere formula must be a string."
        )

    clean_formula = formula.replace(" ", "")

    if clean_formula in {"", "[]"}:
        raise ValueError(
            "Coordination sphere formula cannot be empty."
        )

    pattern = (
        r"\[([sdtq])?"
        r"([A-Z][a-z]?)"
        r"([1-9]\d*)?"
        r"(\((?:.+)\)(?:[1-9]\d*)?)*"
        r"\]"
        r"([0-9+-]+)?$"
    )

    match = re.match(pattern, clean_formula)

    if not match:
        raise ValueError(
            "Invalid coordination sphere format."
        )

    return match


def counter_ions_verification(formula_counter_ions):
    """
    Validate counter-ion formula syntax.

    Expected format example:
    (PF6)2(ClO4)
    """

    if not isinstance(formula_counter_ions, str):
        raise ValueError(
            "Counter-ion formula must be a string."
        )

    clean_formula = formula_counter_ions.replace(" ", "")

    if clean_formula in {"", "()"}:
        raise ValueError(
            "Counter-ion formula cannot be empty."
        )

    pattern = r"(\((.*?)\)([1-9]\d*))*"

    if not re.match(pattern, clean_formula):
        raise ValueError(
            "Invalid counter-ion formula format."
        )

    return clean_formula


# ==========================================
# FORMULA PARSING
# ==========================================
"""
Main parsing functions extracting:
- Metals
- Ligands
- Counter ions
- Bond order
- Stoichiometry
"""


def parse_counter_ions(formula_counter_ions):
    """
    Parse counter ions and their stoichiometric coefficients.
    """

    clean_formula = counter_ions_verification(
        formula_counter_ions
    )

    matches = re.findall(
        r"\((.*?)\)(\d*)",
        clean_formula,
    )

    expanded_ions = []
    unique_ions = []
    coefficients = []

    for counter_ion, coefficient in matches:

        coefficient = int(coefficient) if coefficient else 1

        if coefficient > 12:
            raise ValueError(
                "Counter-ion coefficient cannot exceed 12."
            )

        ion_key = find_counter_ion(counter_ion)

        if ion_key is False:
            raise ValueError(
                f"Unknown counter ion: {counter_ion}"
            )

        expanded_ions.extend([ion_key] * coefficient)

        unique_ions.append(ion_key)

        coefficients.append(coefficient)

    return expanded_ions, unique_ions, coefficients


def parse_metal(formula):
    """
    Extract metal center(s) from the coordination sphere.
    """

    match = formula_verification(formula)

    metal = match.group(2)

    coefficient = (
        int(match.group(3))
        if match.group(3)
        else 1
    )

    if metal not in data_metals:
        raise ValueError(
            f"Unknown metal: {metal}"
        )

    if coefficient not in {1, 2}:
        raise ValueError(
            "Only mono- and dinuclear complexes are supported."
        )

    return [metal] * coefficient


def bond_order(formula):
    """
    Return the metal-metal bond order.

    Prefix mapping:
    s = single
    d = double
    t = triple
    q = quadruple
    """

    match = formula_verification(formula)

    order_map = {
        "s": 1,
        "d": 2,
        "t": 3,
        "q": 4,
    }

    order = order_map.get(match.group(1), 0)

    metal_count = (
        int(match.group(3))
        if match.group(3)
        else 1
    )

    if metal_count == 1 and order != 0:
        raise ValueError(
            "Metal-metal bonds require two metal centers."
        )

    return order


def parse_ligands(formula):
    """
    Extract ligands and stoichiometric coefficients.

    Bridging ligands are internally identified
    using negative coefficients.
    """

    match = formula_verification(formula)

    ligands_block = match.group(4)

    matches = re.findall(
        r"\((.*?)\)(\d*)",
        ligands_block,
    )

    expanded_ligands = []
    unique_ligands = []
    coefficients = []

    for ligand, coefficient in matches:

        coefficient = int(coefficient) if coefficient else 1

        if coefficient > 12:
            raise ValueError(
                "Ligand coefficient cannot exceed 12."
            )

        is_bridging = ligand.startswith("m-")

        if is_bridging:
            ligand = ligand[2:]
            coefficient *= -1

        ligand_key = find_ligand(ligand)

        if ligand_key is False:
            raise ValueError(
                f"Unknown ligand: {ligand}"
            )

        coefficients.append(coefficient)

        unique_ligands.append(ligand_key)

        expanded_ligands.extend(
            [ligand_key] * abs(coefficient)
        )

    return (
        expanded_ligands,
        unique_ligands,
        coefficients,
    )


def count_bridging_ligands(formula):
    """
    Count the number of bridging ligands.
    """

    _, _, coefficients = parse_ligands(formula)

    return sum(
        1
        for coefficient in coefficients
        if coefficient < 0
    )


def parse_elements(formula):
    """
    Return all chemical components present in the complex.
    """

    elements = []

    elements.extend(parse_ligands(formula)[0])

    elements.extend(parse_metal(formula))

    return elements

# ==========================================
# CHARGE ANALYSIS
# ==========================================
"""
Charge-related calculations:
- Coordination sphere charge
- Ligand contribution
- Metal oxidation state
"""


def counter_ions_charge(formula_counter_ions):
    """
    Calculate the total charge of all counter ions.
    """

    counter_ions = parse_counter_ions(
        formula_counter_ions
    )[0]

    total_charge = 0

    for counter_ion in counter_ions:
        total_charge += data_counter_ions[
            counter_ion
        ]["charge"]

    return total_charge


def complex_charge(
    formula,
    formula_counter_ions=None,
):
    """
    Return the total charge of the coordination sphere.
    """

    match = formula_verification(formula)

    sphere_charge = transform_charge(
        match.group(5)
    )

    if formula_counter_ions is None:
        return sphere_charge

    counter_charge = counter_ions_charge(
        formula_counter_ions
    )

    if counter_charge == 0:
        raise ValueError(
            "Counter ions cannot be globally neutral."
        )

    if sphere_charge != 0 and sphere_charge != -counter_charge:
        raise ValueError(
            "Counter-ion charge does not match sphere charge."
        )

    return -counter_charge


def ligands_charge(formula):
    """
    Calculate the total ligand charge.
    """

    ligands = parse_ligands(formula)[0]

    total_charge = 0

    for ligand in ligands:

        ligand = ligand.replace("m-", "")

        total_charge += data_ligands[
            ligand
        ]["charge"]

    return total_charge


def metal_charge(
    formula,
    formula_counter_ions=None,
):
    """
    Calculate the formal metal charge.
    """

    total_charge = complex_charge(
        formula,
        formula_counter_ions,
    )

    ligand_charge = ligands_charge(formula)

    metal_count = len(parse_metal(formula))

    return (
        total_charge - ligand_charge
    ) // metal_count


def oxidation_state(formula):
    """
    Calculate the d-electron configuration
    of the metal center.
    """

    metal = parse_metal(formula)[0]

    oxidation = (
        data_metals[metal]["group"]
        - metal_charge(formula)
    )

    possible_states = data_metals[
        metal
    ]["possible_ox_state"]

    if metal_charge(formula) not in possible_states:

        sign = (
            "+"
            if metal_charge(formula) > 0
            else ""
        )

        remark = (
            "This oxidation state is chemically unlikely "
            f"for {metal} ({sign}{metal_charge(formula)})."
        )

    else:
        remark = ""

    if oxidation < 0 or oxidation > 12:
        raise ValueError(
            "Impossible oxidation state detected."
        )

    return oxidation, remark


# ==========================================
# ELECTRON COUNTING
# ==========================================
"""
Electronic structure calculations:
- Electron counting
- Stability rules
- Valence electron analysis
"""


def electron_count(formula):
    """
    Perform electron counting using
    the ionic counting method.
    """

    oxidation = oxidation_state(formula)[0]

    metal_count = len(parse_metal(formula))

    electrons = oxidation * metal_count

    for ligand in parse_ligands(formula)[0]:

        if ligand.startswith("m-"):

            ligand = ligand[2:]

            electrons += data_ligands[
                ligand
            ]["bridging_e"]

        else:
            electrons += data_ligands[
                ligand
            ]["donor_e"]

    electrons += 2 * bond_order(formula)

    if (
        metal_count == 2
        and electrons % 2 != 0
    ):
        raise ValueError(
            "Dinuclear electron count must be even."
        )

    if metal_count == 2:
        return electrons // 2

    return electrons


def electrons_probability(formula):
    """
    Estimate complex stability according
    to the 16/18-electron rule.
    """

    electrons = electron_count(formula)

    if electrons in {16, 18}:
        return ""

    if electrons > 22:
        return (
            "This coordination complex is "
            "highly unstable and structurally unrealistic."
        )

    return (
        "This coordination complex does not follow "
        "the 16/18-electron rule and may be unstable."
    )


def electronic_structure(formula):
    """
    Return the simplified electronic structure
    of the metal center.
    """

    metal = parse_metal(formula)[0]

    period = data_metals[metal]["period"]

    oxidation = oxidation_state(formula)[0]

    metal_charge_value = metal_charge(formula)

    noble_gases = {
        4: "Ar",
        5: "Kr",
        6: "Xe",
    }

    if metal_charge_value in {0, 2}:

        s_electrons = 2 - metal_charge_value

        d_electrons = oxidation - s_electrons

    elif metal_charge_value >= 3 or metal_charge_value == 1:

        s_electrons = 0

        d_electrons = oxidation

    else:

        s_electrons = 2

        d_electrons = oxidation - 2

    return [
        noble_gases.get(period),
        period,
        s_electrons,
        d_electrons,
    ]

# ==========================================
# ELECTRONIC PROPERTIES
# ==========================================
"""
Advanced electronic structure analysis:
- Spin state
- Orbital filling
- Magnetic behaviour
- Jahn-Teller distortion
"""


def low_spin_configuration(d_electrons):
    """
    Calculate low-spin orbital occupation.

    Electrons preferentially pair
    in lower-energy orbitals.
    """

    paired_electrons = d_electrons % 2

    unpaired_electrons = (
        d_electrons
        - paired_electrons * 2
    )

    return (
        paired_electrons,
        unpaired_electrons,
    )


def high_spin_configuration(d_electrons):
    """
    Calculate high-spin orbital occupation.

    Electrons occupy degenerate orbitals
    before pairing.
    """

    paired_electrons = 0
    unpaired_electrons = 0

    if d_electrons <= 5:

        unpaired_electrons = d_electrons

    else:

        remaining = d_electrons - 5

        unpaired_electrons = 5 - remaining

        paired_electrons = remaining

    return (
        paired_electrons,
        unpaired_electrons,
    )


def determine_spin_state(formula):
    """
    Estimate the preferred spin state.

    Current implementation:
    - Strong-field ligands -> low spin
    - Weak-field ligands -> high spin
    """

    ligands = parse_ligands(formula)[0]

    strong_field_score = 0

    for ligand in ligands:

        ligand = ligand.replace("m-", "")

        strong_field_score += data_ligands[
            ligand
        ].get("field", 1)

    average_field = (
        strong_field_score / len(ligands)
    )

    d_electrons = electronic_structure(
        formula
    )[3]

    if average_field >= 3:
        return (
            "low spin",
            low_spin_configuration(
                d_electrons
            ),
        )

    return (
        "high spin",
        high_spin_configuration(
            d_electrons
        ),
    )


def fill_d_orbitals(
    paired_electrons,
    unpaired_electrons,
):
    """
    Fill the five d orbitals.

    Orbital ordering:
    t2g → eg
    """

    orbitals = [0, 0, 0, 0, 0]

    for index in range(len(orbitals)):

        if paired_electrons > 0:

            orbitals[index] += 2

            paired_electrons -= 1

        elif unpaired_electrons > 0:

            orbitals[index] += 1

            unpaired_electrons -= 1

    return orbitals


def magnetic_moment(formula):
    """
    Estimate the magnetic moment
    using the spin-only formula.
    """

    spin_state = determine_spin_state(
        formula
    )

    unpaired = spin_state[1][1]

    return round(
        math.sqrt(
            unpaired * (unpaired + 2)
        ),
        2,
    )


def magnetic_behavior(formula):
    """
    Determine if the complex is:
    - Diamagnetic
    - Paramagnetic
    """

    spin_state = determine_spin_state(
        formula
    )

    unpaired = spin_state[1][1]

    if unpaired == 0:
        return "Diamagnetic"

    return "Paramagnetic"


def crystal_field_stabilization_energy(formula):
    """
    Estimate the Crystal Field
    Stabilization Energy (CFSE).

    Approximation based on octahedral fields.
    """

    d_electrons = electronic_structure(
        formula
    )[3]

    t2g = min(d_electrons, 6)

    eg = max(0, d_electrons - 6)

    cfse = (
        -0.4 * t2g
        + 0.6 * eg
    )

    return round(cfse, 2)


def jahn_teller_distortion(formula):
    """
    Predict Jahn-Teller distortion.

    Distortion occurs when degenerate
    orbitals are asymmetrically occupied.
    """

    spin_state = determine_spin_state(
        formula
    )

    orbitals = fill_d_orbitals(
        spin_state[1][0],
        spin_state[1][1],
    )

    for index in range(2):

        if orbitals[index] != orbitals[index + 1]:
            return "Weak Jahn-Teller distortion"

    for index in range(3, 4):

        if orbitals[index] != orbitals[index + 1]:
            return "Strong Jahn-Teller distortion"

    return "No Jahn-Teller distortion"

# ==========================================
# COORDINATION COMPOUND NOMENCLATURE
# ==========================================
"""
IUPAC-inspired nomenclature generation:
- Ligand prefixes
- Bridging ligands
- Counter ions
- Oxidation state notation
"""


# ==========================================
# NOMENCLATURE PREFIXES
# ==========================================

STANDARD_PREFIXES = {
    1: "",
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
    7: "hepta",
    8: "octa",
    9: "nona",
    10: "deca",
    11: "undeca",
    12: "dodeca",
    13: "trideca",
    14: "tetradeca",
    15: "pentadeca",
    16: "hexadeca",
}

SPECIAL_PREFIXES = {
    1: "",
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
    6: "hexakis",
    7: "heptakis",
    8: "octakis",
    9: "nonakis",
    10: "decakis",
    11: "undecakis",
    12: "dodecakis",
    13: "tridecakis",
    14: "tetradecakis",
    15: "pentadecakis",
    16: "hexadecakis",
}


# ==========================================
# LIGAND NAMING UTILITIES
# ==========================================


def ligand_nomenclature_name(ligand):
    """
    Return the nomenclature-compatible ligand name.
    """

    offset = 2 if ligand.startswith("m-") else 0

    ligand_key = ligand[offset:]

    ligand_data = data_ligands[ligand_key]

    return ligand_data.get(
        "nomenclature",
        ligand_data["name"],
    )


def use_special_prefix(ligand_name):
    """
    Determine whether special multiplicative
    prefixes must be used.
    """

    for ligand in data_ligands.values():

        if (
            ligand["name"] == ligand_name
            and ligand.get("coeff") == "yes"
        ):
            return True

    return False


# ==========================================
# COUNTER-ION NOMENCLATURE
# ==========================================


def naming_counter_ions(
    formula_counter_ions,
):
    """
    Generate the counter-ion name sequence.
    """

    if formula_counter_ions is None:
        return ""

    ions = parse_counter_ions(
        formula_counter_ions
    )[1]

    ion_names = []

    for ion in ions:

        ion_names.append(
            data_counter_ions[ion]["name"]
        )

    ion_names.sort(key=str.lower)

    return " ".join(ion_names)


# ==========================================
# MAIN NOMENCLATURE ENGINE
# ==========================================


def naming_compound(
    formula,
    formula_counter_ions=None,
):
    """
    Generate the complete compound name.
    """

    ligands = parse_ligands(formula)[1]

    coefficients = parse_ligands(formula)[2]

    metals = parse_metal(formula)

    complex_total_charge = complex_charge(
        formula,
        formula_counter_ions,
    )

    ligands_with_coefficients = []

    for index in range(len(ligands)):

        ligand_name = ligand_nomenclature_name(
            ligands[index]
        )

        ligands_with_coefficients.append(
            (
                ligand_name,
                coefficients[index],
            )
        )

    ligands_with_coefficients.sort(
        key=lambda item: item[0].lower()
    )

    # ==========================================
    # METAL NAME
    # ==========================================

    if complex_total_charge < 0:

        metal_name = data_metals[
            metals[0]
        ]["secondary_name"]

        compound_name = naming_counter_ions(
            formula_counter_ions
        )

    else:

        metal_name = data_metals[
            metals[0]
        ]["name"]

        compound_name = ""

    # ==========================================
    # BRIDGING LIGANDS
    # ==========================================

    mu_symbol = "-μ-"

    for ligand_name, coefficient in ligands_with_coefficients:

        if coefficient < 0:

            coefficient = abs(coefficient)

            if use_special_prefix(
                ligand_name
            ):

                prefix = SPECIAL_PREFIXES[
                    coefficient
                ]

                compound_name += (
                    f"{mu_symbol}"
                    f"{prefix}"
                    f"({ligand_name})"
                )

            else:

                prefix = STANDARD_PREFIXES[
                    coefficient
                ]

                compound_name += (
                    f"{mu_symbol}"
                    f"{prefix}"
                    f"{ligand_name}"
                )

    # ==========================================
    # DINUCLEAR PREFIX
    # ==========================================

    coefficient_divisor = 1

    add_parenthesis = False

    non_bridging_count = (
        len(parse_ligands(formula)[0])
        - count_bridging_ligands(formula)
    )

    if (
        len(metals) == 2
        and non_bridging_count > 0
    ):

        compound_name += (
            SPECIAL_PREFIXES[2] + "("
        )

        coefficient_divisor = 2

        add_parenthesis = True

    elif (
        len(metals) == 2
        and non_bridging_count == 0
    ):

        compound_name += STANDARD_PREFIXES[2]

    # ==========================================
    # TERMINAL LIGANDS
    # ==========================================

    for ligand_name, coefficient in ligands_with_coefficients:

        if coefficient > 0:

            corrected_coefficient = (
                coefficient
                // coefficient_divisor
            )

            if use_special_prefix(
                ligand_name
            ):

                prefix = SPECIAL_PREFIXES[
                    corrected_coefficient
                ]

                compound_name += (
                    f"{prefix}"
                    f"({ligand_name})"
                )

            else:

                prefix = STANDARD_PREFIXES[
                    corrected_coefficient
                ]

                compound_name += (
                    f"{prefix}"
                    f"{ligand_name}"
                )

    # ==========================================
    # METAL + OXIDATION STATE
    # ==========================================

    compound_name += metal_name

    oxidation = metal_charge(formula)

    if oxidation == 0:

        oxidation_roman = "0"

    elif oxidation < 0:

        oxidation_roman = (
            "-"
            + roman.toRoman(abs(oxidation))
        )

    else:

        oxidation_roman = roman.toRoman(
            oxidation
        )

    compound_name += (
        f"({oxidation_roman})"
    )

    # ==========================================
    # FINAL FORMATTING
    # ==========================================

    if add_parenthesis:
        compound_name += ")"

    if compound_name.startswith("-"):
        compound_name = compound_name[1:]

    compound_name = re.sub(
        r" -μ-",
        " μ-",
        compound_name,
    )

    if complex_total_charge > 0:

        counter_ions_name = (
            naming_counter_ions(
                formula_counter_ions
            )
        )

        if counter_ions_name:

            compound_name += (
                " " + counter_ions_name
            )

    compound_name = (
        compound_name[:1].upper()
        + compound_name[1:]
    )

    return compound_name

# ============================================================
# STABILITY ANALYSIS
# ============================================================

@dataclass
class StabilityResult:
    """
    Store all partial stability scores and the final score.
    """

    total: float

    electron: float
    hsab: float
    chelate: float
    field: float
    charge: float
    geometry: float
    oxidation: float
    backbonding: float
    steric: float


# ============================================================
# STABILITY ENGINE
# ============================================================

class StabilityEngine:
    """
    Estimate the thermodynamic and electronic stability
    of a coordination complex using empirical scoring models.
    """

    def __init__(self, formula):

        self.formula = formula.replace(" ", "")

        metals = parse_metal(self.formula)

        if not metals:
            raise ValueError(
                "Error: No metal center detected."
            )

        self.metal = metals[0]

        self.ligands = parse_ligands(
            self.formula
        )[0]

        self.cn = len(self.ligands)

        self.electrons = electron_count(
            self.formula
        )

        self.charge = metal_charge(
            self.formula
        )

        self.ox, _ = oxidation_state(
            self.formula
        )

        self.m_data = data_metals.get(
            self.metal,
            {
                "hardness": 5,
                "group": 10,
            },
        )

        self.series_bonus = self._series_bonus()

    # ========================================================
    # PERIODIC SERIES BONUS
    # ========================================================

    def _series_bonus(self):
        """
        Add stability bonuses for transition metal series
        known to form particularly stable complexes.

        Returns:
            int
        """

        fifth_row = {
            "Pd",
            "Ag",
            "Cd",
            "Pt",
            "Au",
            "Hg",
        }

        fourth_row = {
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
        }

        if self.metal in fifth_row:
            return 10

        if self.metal in fourth_row:
            return 5

        return 0

    # ========================================================
    # ELECTRON COUNT SCORE
    # ========================================================

    def electron_score(self):
        """
        Evaluate electronic stability based on the
        16e / 18e rule.

        Returns:
            float
        """

        preferred_18e = {
            "Ni",
            "Pd",
            "Pt",
        }

        target = (
            18
            if self.metal in preferred_18e
            else 16
        )

        gap = abs(
            self.electrons - target
        )

        score = 100 - (gap * 10)

        # Strong π-acceptor bonus
        if "CO" in self.formula:
            score += 8

        score += self.series_bonus

        return max(
            0,
            min(100, score),
        )

    # ========================================================
    # HSAB SCORE
    # ========================================================

    def hsab_score(self):
        """
        Evaluate hard-soft acid-base compatibility.

        Returns:
            float
        """

        metal_hardness = self.m_data.get(
            "hardness",
            5,
        )

        scores = []

        for ligand in self.ligands:

            ligand_data = data_ligands.get(
                ligand.replace("m-", ""),
                {},
            )

            ligand_hardness = (
                ligand_data.get("HSAB", {})
                .get("hardness", 5)
            )

            ligand_field = ligand_data.get(
                "field",
                1,
            )

            compatibility = math.exp(
                -0.45
                * abs(
                    metal_hardness
                    - ligand_hardness
                )
            )

            scores.append(
                compatibility
                * (1 + ligand_field / 4)
            )

        if not scores:
            return 50

        return (
            sum(scores)
            / len(scores)
            * 100
        )

    # ========================================================
    # CHELATION SCORE
    # ========================================================

    def chelate_score(self):
        """
        Estimate the chelate stabilization effect.

        Returns:
            float
        """

        total_denticity = 0

        for ligand in self.ligands:

            ligand_data = data_ligands.get(
                ligand.replace("m-", ""),
                {},
            )

            total_denticity += ligand_data.get(
                "denticity",
                1,
            )

        chelation_bonus = (
            total_denticity - self.cn
        )

        score = (
            60 + chelation_bonus * 25
        )

        return max(
            0,
            min(100, score),
        )

    # ========================================================
    # LIGAND FIELD SCORE
    # ========================================================

    def field_score(self):
        """
        Evaluate ligand field stabilization.

        Returns:
            float
        """

        total_field = 0

        for ligand in self.ligands:

            ligand_data = data_ligands.get(
                ligand.replace("m-", ""),
                {},
            )

            total_field += ligand_data.get(
                "field",
                1,
            )

        normalized = (
            total_field / max(1, self.cn)
        )

        return min(
            100,
            normalized * 25,
        )

    # ========================================================
    # CHARGE SCORE
    # ========================================================

    def charge_score(self):
        """
        Penalize highly charged complexes.

        Returns:
            float
        """

        return max(
            0,
            100 - abs(self.charge) * 8,
        )

    # ========================================================
    # GEOMETRY SCORE
    # ========================================================

    def geometry_score(self):
        """
        Estimate geometrical stability.

        Returns:
            float
        """

        if self.cn == 6:
            return 95

        if self.cn == 4:

            if self.metal in {
                "Ni",
                "Pd",
                "Pt",
            }:
                return 98

            return 80

        if self.cn == 5:
            return 78

        if self.cn == 2:
            return 85

        return 60

    # ========================================================
    # OXIDATION STATE SCORE
    # ========================================================

    def oxidation_score(self):
        """
        Compare oxidation state with common
        experimentally observed values.

        Returns:
            float
        """

        preferred_states = (
            data_metals.get(
                self.metal,
                {},
            ).get(
                "possible_ox_state",
                [self.ox],
            )
        )

        difference = min(
            abs(self.ox - state)
            for state in preferred_states
        )

        return max(
            30,
            100 - difference * 18,
        )

    # ========================================================
    # π-BACKBONDING SCORE
    # ========================================================

    def backbonding_score(self):
        """
        Evaluate π-backbonding stabilization.

        Returns:
            float
        """

        score = 0

        for ligand in self.ligands:

            ligand_name = ligand.replace(
                "m-",
                "",
            )

            ligand_data = data_ligands.get(
                ligand_name,
                {},
            )

            if ligand_data.get(
                "pi_acceptor"
            ):
                score += 35

            if ligand_name == "CN":
                score += 10

            if ligand_name == "CO":
                score += 15

        return min(
            100,
            score + self.series_bonus,
        )

    # ========================================================
    # STERIC SCORE
    # ========================================================

    def steric_score(self):
        """
        Estimate steric congestion around the metal.

        Returns:
            float
        """

        steric_bulk = 0

        for ligand in self.ligands:

            ligand_data = data_ligands.get(
                ligand.replace("m-", ""),
                {},
            )

            steric_bulk += ligand_data.get(
                "steric_bulk",
                1,
            )

        return max(
            0,
            100 - steric_bulk * 3,
        )

    # ========================================================
    # FINAL STABILITY SCORE
    # ========================================================

    def final_score(self):
        """
        Combine all stability contributions into
        a weighted final stability index.

        Returns:
            StabilityResult
        """

        scores = {
            "electron": self.electron_score(),
            "hsab": self.hsab_score(),
            "chelate": self.chelate_score(),
            "field": self.field_score(),
            "charge": self.charge_score(),
            "geometry": self.geometry_score(),
            "oxidation": self.oxidation_score(),
            "backbonding": self.backbonding_score(),
            "steric": self.steric_score(),
        }

        weights = {
            "electron": 0.22,
            "hsab": 0.17,
            "chelate": 0.16,
            "field": 0.12,
            "charge": 0.08,
            "geometry": 0.12,
            "oxidation": 0.09,
            "backbonding": 0.03,
            "steric": 0.01,
        }

        total_score = sum(
            scores[key] * weights[key]
            for key in scores
        )

        total_score = round(
            max(0, min(100, total_score)),
            2,
        )

        return StabilityResult(
            total=total_score,
            **scores,
        )
    
# ============================================================
# ISOMERS ANALYSIS
# ============================================================

# Stereoisomer dictionary
stereoisomers_dico = {
    "Ma6": 1,
    "Ma5b1": 1,
    "Ma4b2": 2,
    "Ma3b3": 2,
    "Ma4b1c1": 2,
    "Ma3b1c1d1": 5,
    "Ma2b1c1d1e1": 15,
    "Ma1b1c1d1e1f1": 30,
    "Ma2b2c2": 6,
    "Ma2b2c1d1": 8,
    "Ma3b2c1": 3,
    "Ma2b2": 2,
}

# Enantiomer dictionary
enantiomers_dico = {
    "Ma6": 0,
    "Ma5b1": 0,
    "Ma4b2": 0,
    "Ma3b3": 0,
    "Ma4b1c1": 0,
    "Ma3b1c1d1": 1,
    "Ma2b1c1d1e1": 6,
    "Ma1b1c1d1e1f1": 15,
    "Ma2b2c2": 1,
    "Ma2b2c1d1": 2,
    "Ma3b2c1": 0,
    "Ma2b2": 0,
}


# ============================================================
# ISOMERS CALCULATION
# ============================================================

def isomers(formula):
    """
    Calculate the number of stereoisomers
    and enantiomeric pairs.

    Returns:
        tuple:
            - stereoisomers
            - enantiomer pairs
    """

    key = ""

    ligand_numbers = []

    alphabet = string.ascii_lowercase

    ligands_data = parse_ligands(formula)

    # --------------------------------------------------------
    # Metal nuclearity
    # --------------------------------------------------------

    if len(parse_metal(formula)) == 1:
        key += "M"

    else:
        key += "M2"

    # --------------------------------------------------------
    # Ligand coefficients
    # --------------------------------------------------------

    for coeff in ligands_data[2]:
        ligand_numbers.append(int(abs(coeff)))

    ligand_numbers.sort(reverse=True)

    # --------------------------------------------------------
    # Build isomer key
    # --------------------------------------------------------

    for i in range(len(ligand_numbers)):

        key += (
            alphabet[i]
            + str(ligand_numbers[i])
        )

    # --------------------------------------------------------
    # Square planar special case
    # --------------------------------------------------------

    geometry = get_geometry(formula)[1]

    if (
        key == "Ma2b2"
        and geometry == "square planar"
    ):
        return 2, 0

    # --------------------------------------------------------
    # General lookup
    # --------------------------------------------------------

    stereoisomers = stereoisomers_dico.get(key)

    enantiomers = enantiomers_dico.get(key)

    return stereoisomers, enantiomers


# ============================================================
# FORMULA DISPLAY UTILITIES
# ============================================================

def get_clean_formula(
    formula,
    formula_counter_ions=None,
):
    """
    Convert the raw formula into a cleaner
    LaTeX-compatible representation.

    Returns:
        str
    """

    clean_formula = formula.replace(" ", "")

    # Replace bridging notation
    clean_formula = re.sub(
        r"m-",
        r"μ-",
        clean_formula,
    )

    # --------------------------------------------------------
    # Coordination sphere formatting
    # --------------------------------------------------------

    end = clean_formula.find("]")

    clean_sphere = re.sub(
        r"(\d+)",
        r"_{\1}",
        clean_formula[: end + 1],
    )

    clean_sphere = re.sub(
        r"\(([A-Z][a-z]?)\)",
        r"\1",
        clean_sphere,
    )

    # --------------------------------------------------------
    # Counter ions formatting
    # --------------------------------------------------------

    clean_counter_ions = ""

    if formula_counter_ions is not None:

        clean_counter_ions = re.sub(
            r"(\d+)",
            r"_{\1}",
            formula_counter_ions,
        )

        clean_counter_ions = re.sub(
            r"\(([A-Z][a-z]?)\)",
            r"\1",
            clean_counter_ions,
        )

    # --------------------------------------------------------
    # Charge formatting
    # --------------------------------------------------------

    charge = complexe_charge(
        formula,
        formula_counter_ions,
    )

    if charge > 0:

        return (
            "$"
            + clean_sphere
            + "^{+"
            + str(charge)
            + "}"
            + clean_counter_ions
            + "$"
        )

    if charge < 0:

        return (
            "$"
            + clean_counter_ions
            + clean_sphere
            + "^{"
            + str(charge)
            + "}$"
        )

    return "$" + clean_sphere + "$"


# ============================================================
# CHEMICAL CONSISTENCY RULES
# ============================================================

def chemical_rules(formula):
    """
    Verify global chemical consistency rules.

    Raises:
        ValueError
    """

    if (
        len(parse_metal(formula)) == 1
        and count_bridging_ligands(formula) > 0
    ):
        raise ValueError(
            "Error: Mononuclear complexes cannot contain bridging ligands."
        )


# ============================================================
# ANALYSIS RENDERING
# ============================================================

def render_analysis(lines):
    """
    Render analysis depending on the execution environment:
    - Jupyter Notebook
    - Streamlit
    - Standard terminal

    Returns:
        str | Markdown
    """

    is_streamlit = False
    is_notebook = False

    # --------------------------------------------------------
    # Streamlit detection
    # --------------------------------------------------------

    if "streamlit" in sys.modules:

        from streamlit.runtime.scriptrunner import (
            get_script_run_ctx,
        )

        if get_script_run_ctx() is not None:
            is_streamlit = True

    # --------------------------------------------------------
    # Notebook detection
    # --------------------------------------------------------

    try:

        shell = get_ipython().__class__.__name__

        if shell == "ZMQInteractiveShell":
            is_notebook = True

    except NameError:
        pass

    # --------------------------------------------------------
    # Notebook rendering
    # --------------------------------------------------------

    if is_notebook:

        markdown_text = "\n".join(
            f"* {line}"
            for line in lines
        )

        return display(
            Markdown(markdown_text)
        )

    # --------------------------------------------------------
    # Streamlit rendering
    # --------------------------------------------------------

    if is_streamlit:

        return "\n\n".join(lines)

    # --------------------------------------------------------
    # Standard terminal rendering
    # --------------------------------------------------------

    return (
        "\n".join(lines)
        .replace("**", "")
    )


# ==========================================
# 3D VISUALIZATION SECTION
# ==========================================
# Functions used to generate molecular
# geometries and interactive 3D renderings.


# ==========================================
# MAIN COORDINATION GEOMETRIES
# ==========================================


def linear(r):
    """
    Generate coordinates for a linear geometry.
    """
    return [
        (r, 0, 0),
        (-r, 0, 0),
    ]


def trigonal_planar(r):
    """
    Generate coordinates for a trigonal planar
    geometry.
    """
    return [
        (r, 0, 0),
        (
            -r / 2,
            r * np.sqrt(3) / 2,
            0,
        ),
        (
            -r / 2,
            -r * np.sqrt(3) / 2,
            0,
        ),
    ]


def tetrahedral(r):
    """
    Generate coordinates for a tetrahedral
    geometry.
    """

    base = np.array(
        [
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
        ]
    )

    base = (
        base
        / np.linalg.norm(base[0])
    )

    base = r * base

    return [
        tuple(vector)
        for vector in base
    ]


def square_planar(r):
    """
    Generate coordinates for a square planar
    geometry.
    """
    return [
        (r, 0, 0),
        (-r, 0, 0),
        (0, r, 0),
        (0, -r, 0),
    ]


def trigonal_bipyramidal(r):
    """
    Generate coordinates for a trigonal
    bipyramidal geometry.
    """
    return [
        (0, 0, r),
        (0, 0, -r),

        (r, 0, 0),

        (
            r * np.cos(np.radians(120)),
            r * np.sin(np.radians(120)),
            0,
        ),

        (
            r * np.cos(np.radians(240)),
            r * np.sin(np.radians(240)),
            0,
        ),
    ]


def octahedral(r):
    """
    Generate coordinates for an octahedral
    geometry.
    """
    return [
        (r, 0, 0),
        (-r, 0, 0),

        (0, r, 0),
        (0, -r, 0),

        (0, 0, r),
        (0, 0, -r),
    ]

def get_geometry(formula, r=0):
    """
    Determine the most probable coordination
    geometry based on the coordination number
    and electronic configuration.
    """

    coordination_number = len(
        parse_ligands(formula)[0]
    )

    if coordination_number == 1:

        return (
            [(r, 0, 0)],
            "linear",
        )

    elif coordination_number == 2:

        return (
            linear(r),
            "linear",
        )

    elif coordination_number == 3:

        return (
            trigonal_planar(r),
            "trigonal planar",
        )

    elif (
        coordination_number == 4
        and oxidation_state(formula)[0] == 8
    ):

        return (
            square_planar(r),
            "square planar",
        )

    elif (
        coordination_number == 4
        and oxidation_state(formula)[0] != 8
    ):

        return (
            tetrahedral(r),
            "tetrahedral",
        )

    elif coordination_number == 5:

        return (
            trigonal_bipyramidal(r),
            "trigonal bipyramidal",
        )

    elif coordination_number == 6:

        return (
            octahedral(r),
            "octahedral",
        )

    raise ValueError(
        "Error: 3D visualization is not "
        "available for coordination numbers "
        "greater than 6."
    )


# ==========================================
# LIGAND GEOMETRY SECTION
# ==========================================
# Internal ligand geometries used to generate
# realistic 3D molecular structures.


def ligand_linear(
    ligand,
    ligand_coord,
    r,
):
    """
    Generate coordinates for a linear ligand.
    """

    ligand_position = np.array(
        ligand_coord
    )

    direction = (
        ligand_position
        / np.linalg.norm(ligand_position)
    )

    inter_distance = (
        data_ligands[ligand]
        ["inter_distance"]
    )

    position = (
        direction
        * (inter_distance + r)
    )

    return [
        tuple(
            float(value)
            for value in position
        )
    ]


def ligand_dlinear(
    ligand,
    ligand_coord,
    r,
):
    """
    Generate coordinates for a double-linear
    ligand geometry.
    """

    ligand_position = np.array(
        ligand_coord
    )

    direction = (
        ligand_position
        / np.linalg.norm(ligand_position)
    )

    inter_distance_1 = (
        data_ligands[ligand]
        ["inter_distance"]
    )

    inter_distance_2 = (
        data_ligands[ligand]
        ["inter_distance2"]
    )

    position_1 = (
        direction
        * (inter_distance_1 + r)
    )

    position_2 = (
        direction
        * (
            inter_distance_1
            + inter_distance_2
            + r
        )
    )

    return [
        tuple(
            float(value)
            for value in position_1
        ),
        tuple(
            float(value)
            for value in position_2
        ),
    ]


def ligand_bent(
    ligand,
    ligand_coord,
    r,
):
    """
    Generate coordinates for a bent ligand
    geometry.
    """

    ligand_position = np.array(
        ligand_coord
    )

    direction = (
        ligand_position
        / np.linalg.norm(ligand_position)
    )

    inter_distance_1 = (
        data_ligands[ligand]
        ["inter_distance"]
    )

    inter_distance_2 = (
        data_ligands[ligand]
        ["inter_distance2"]
    )

    if abs(direction[0]) > 0.1:
        temp_vector = np.array([0, 1, 0])

    else:
        temp_vector = np.array([1, 0, 0])

    perpendicular = np.cross(
        direction,
        temp_vector,
    )

    perpendicular /= np.linalg.norm(
        perpendicular
    )

    theta = np.deg2rad(60)

    position_1 = (
        (
            np.cos(theta)
            * inter_distance_1
            + r
        )
        * direction
        + (
            np.sin(theta)
            * inter_distance_1
        )
        * perpendicular
    )

    position_2 = (
        (
            np.cos(theta)
            * inter_distance_2
            + r
        )
        * direction
        + (
            -np.sin(theta)
            * inter_distance_2
        )
        * perpendicular
    )

    return [
        tuple(
            float(value)
            for value in position_1
        ),
        tuple(
            float(value)
            for value in position_2
        ),
    ]

def ligand_tetrahedral(
    ligand,
    ligand_coord,
    r,
):
    """
    Generate coordinates for a tetrahedral
    ligand geometry.
    """

    ligand_position = np.array(
        ligand_coord
    )

    direction = (
        ligand_position
        / np.linalg.norm(ligand_position)
    )

    inter_distance = (
        data_ligands[ligand]
        ["inter_distance"]
    )

    if abs(direction[0]) > 0.1:
        temp_vector = np.array([0, 1, 0])

    else:
        temp_vector = np.array([1, 0, 0])

    u = np.cross(
        direction,
        temp_vector,
    )

    u /= np.linalg.norm(u)

    w = np.cross(direction, u)

    positions = []

    theta = np.deg2rad(-54.75)

    for index in range(3):

        phi = np.deg2rad(index * 120)

        position = (
            (
                np.cos(theta)
                * inter_distance
                + r
            )
            * direction
            + (
                np.sin(theta)
                * np.cos(phi)
                * inter_distance
            )
            * u
            + (
                np.cos(theta)
                * np.sin(phi)
                * inter_distance
            )
            * w
        )

        positions.append(
            tuple(
                float(value)
                for value in position
            )
        )

    return positions


def get_geometry_ligand(ligand_input):
    """
    Return the internal geometry type of a
    ligand from the database.
    """

    geometry = (
        data_ligands[ligand_input]
        .get("geometry")
    )

    if geometry is not None:
        return geometry

    return False


# ==========================================
# ATOMIC POSITION GENERATION
# ==========================================
# Functions used to generate the complete
# atomic coordinates of the coordination
# complex.


def atoms_position(
    formula,
    r=1.7,
):
    """
    Generate the 3D coordinates of all atoms
    of the coordination compound.
    """

    positions = [(0, 0, 0)]

    ligand_positions = (
        get_geometry(formula, r)[0]
    )

    ligands = parse_ligands(formula)[0]

    for index, ligand in enumerate(ligands):

        geometry = (
            get_geometry_ligand(ligand)
        )

        # ==================================
        # SINGLE SPHERE LIGAND
        # ==================================

        if geometry == "sphere":

            positions += [
                ligand_positions[index]
            ]

        # ==================================
        # LINEAR LIGAND
        # ==================================

        elif geometry == "linear":

            positions += [
                ligand_positions[index]
            ]

            positions += ligand_linear(
                ligand,
                ligand_positions[index],
                r,
            )

        # ==================================
        # DOUBLE-LINEAR LIGAND
        # ==================================

        elif geometry == "dlinear":

            positions += [
                ligand_positions[index]
            ]

            positions += ligand_dlinear(
                ligand,
                ligand_positions[index],
                r,
            )

        # ==================================
        # BENT LIGAND
        # ==================================

        elif geometry == "bent":

            positions += [
                ligand_positions[index]
            ]

            positions += ligand_bent(
                ligand,
                ligand_positions[index],
                r,
            )

        # ==================================
        # TETRAHEDRAL LIGAND
        # ==================================

        elif geometry == "tetrahedral":

            positions += [
                ligand_positions[index]
            ]

            positions += ligand_tetrahedral(
                ligand,
                ligand_positions[index],
                r,
            )

        # ==================================
        # UNSUPPORTED GEOMETRY
        # ==================================

        else:

            raise ValueError(
                "Error: Ligand geometry "
                "not available in 3D."
            )

    return positions


def get_atoms(ligand_input):
    """
    Extract all atom symbols from a ligand
    formula.
    """

    ligand_info = data_ligands.get(
        ligand_input
    )

    donor_atoms = ligand_info.get(
        "donor_atoms"
    )

    result = (
        [donor_atoms[0]]
        if donor_atoms[0]
        else []
    )

    atoms = re.findall(
        r"[A-Z][a-z]?\d*",
        ligand_input,
    )

    for atom in atoms:

        match = re.match(
            r"([A-Z][a-z]?)(\d*)",
            atom,
        )

        symbol = match.group(1)

        count = (
            int(match.group(2))
            if match.group(2)
            else 1
        )

        if symbol == donor_atoms[0]:
            continue

        result.extend([symbol] * count)

    return result

def atom_symbols(formula):
    """
    Generate the complete ordered list of
    atomic symbols composing the complex.
    """

    metals = parse_metal(formula)

    atoms_list = metals.copy()

    ligands = parse_ligands(formula)[0]

    for ligand in ligands:

        atoms_list += get_atoms(ligand)

    return atoms_list


def create_compound_render(formula):
    """
    Create the ASE Atoms object used for
    3D visualization.
    """

    compound = Atoms(
        atom_symbols(formula),
        positions=atoms_position(formula),
    )

    return compound


# ==========================================
# 3D RENDERING SECTION
# ==========================================
# Interactive molecular visualization using
# py3Dmol.


def render_complex(
    compound,
    atoms_size=0.4,
    render_type="Ball and Stick",
):
    """
    Render a coordination complex in 3D.
    """

    # ======================================
    # ASE -> XYZ CONVERSION
    # ======================================

    xyz_string = io.StringIO()

    write(
        xyz_string,
        compound,
        format="xyz",
    )

    xyz_content = xyz_string.getvalue()

    # ======================================
    # PY3DMOL VIEW CREATION
    # ======================================

    view = py3Dmol.view(
        width=400,
        height=400,
    )

    view.addModel(
        xyz_content,
        "xyz",
    )

    # ======================================
    # RENDER STYLE SELECTION
    # ======================================

    if render_type == "Ball and Stick":

        view.setStyle(
            {
                "stick": {},
                "sphere": {
                    "scale": atoms_size
                },
            }
        )

    elif render_type == "Stick":

        view.setStyle(
            {
                "stick": {}
            }
        )

    elif render_type == "Sphere":

        view.setStyle(
            {
                "sphere": {
                    "scale": atoms_size
                }
            }
        )

    elif render_type == "Lines":

        view.setStyle(
            {
                "line": {}
            }
        )

    elif render_type == "VDW":

        view.addSurface(py3Dmol.VDW)

    view.zoomTo()

    # ======================================
    # STREAMLIT DETECTION
    # ======================================

    is_streamlit = False

    if "streamlit" in sys.modules:

        from streamlit.runtime.scriptrunner import (
            get_script_run_ctx,
        )

        if get_script_run_ctx() is not None:
            is_streamlit = True

    # ======================================
    # STREAMLIT OUTPUT
    # ======================================

    if is_streamlit:

        html_content = view._make_html()

        return html_content

    # ======================================
    # NOTEBOOK OUTPUT
    # ======================================

    return view.show()


