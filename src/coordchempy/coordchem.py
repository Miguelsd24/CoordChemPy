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

Style rules used across the project:
- Long-form scientific comments
- Strict section separation
- Readable vertical spacing
- Stable and explicit logic
- No hidden side effects
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
Load all chemistry databases used by the engine.

Loaded databases:
- Metals
- Ligands
- Counter ions

The databases are stored as JSON files and are
loaded once during module initialisation.
"""

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------
# Metals database
# --------------------------------------------------

with open(
    BASE_DIR / "data" / "metals.json",
    encoding="utf-8",
) as file:

    data_metals = json.load(file)


# --------------------------------------------------
# Ligands database
# --------------------------------------------------

with open(
    BASE_DIR / "data" / "ligands.json",
    encoding="utf-8",
) as file:

    data_ligands = json.load(file)


# --------------------------------------------------
# Counter ions database
# --------------------------------------------------

with open(
    BASE_DIR / "data" / "counter_ions.json",
    encoding="utf-8",
) as file:

    data_counter_ions = json.load(file)


# ==========================================
# GENERAL UTILITIES
# ==========================================
"""
Shared utility functions used throughout the
coordination chemistry engine.

Responsibilities:
- Database lookup
- Charge conversion
- Common validation helpers
- Parsing support
"""


def find_ligand(ligand_input):
    """
    Return the canonical ligand key from the database.

    Search priority:
    1. Direct formula lookup
    2. Ligand abbreviation lookup

    Parameters
    ----------
    ligand_input : str
        Ligand formula or abbreviation.

    Returns
    -------
    str | bool
        Canonical ligand key if found,
        otherwise False.
    """

    # --------------------------------------
    # Direct lookup
    # --------------------------------------

    if ligand_input in data_ligands:
        return ligand_input

    # --------------------------------------
    # Abbreviation lookup
    # --------------------------------------

    for ligand_key, properties in data_ligands.items():

        if properties.get("abbr") == ligand_input:
            return ligand_key

    return False



def find_counter_ion(counter_ion_input):
    """
    Return the canonical counter-ion key.

    Search priority:
    1. Direct key lookup
    2. Formula lookup
    3. Abbreviation lookup

    Parameters
    ----------
    counter_ion_input : str
        Counter-ion formula or abbreviation.

    Returns
    -------
    str | bool
        Canonical counter-ion key if found,
        otherwise False.
    """

    # --------------------------------------
    # Direct lookup
    # --------------------------------------

    if counter_ion_input in data_counter_ions:
        return counter_ion_input

    # --------------------------------------
    # Formula lookup
    # --------------------------------------

    for ion_key, properties in data_counter_ions.items():

        if properties.get("formula") == counter_ion_input:
            return ion_key

    # --------------------------------------
    # Abbreviation lookup
    # --------------------------------------

    for ion_key, properties in data_counter_ions.items():

        if properties.get("abbr") == counter_ion_input:
            return ion_key

    return False



def transform_charge(charge):
    """
    Convert a charge string into an integer.

    Supported formats:
    - '+'
    - '-'
    - '2+'
    - '+2'
    - '3-'
    - '-3'
    - '0'

    Parameters
    ----------
    charge : str | int | None

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If the format is invalid.
    """

    # --------------------------------------
    # Empty charge handling
    # --------------------------------------

    if charge is None or charge == "":
        return 0

    charge = charge.strip()

    # --------------------------------------
    # Direct integer conversion
    # --------------------------------------

    try:
        return int(charge)

    except ValueError:
        pass

    # --------------------------------------
    # Simple + / - notation
    # --------------------------------------

    if charge == "+":
        return 1

    if charge == "-":
        return -1

    # --------------------------------------
    # Format: 2+ / 3-
    # --------------------------------------

    match = re.match(r"(\d+)([+-])", charge)

    if match:

        value = int(match.group(1))
        sign = match.group(2)

        return value if sign == "+" else -value

    # --------------------------------------
    # Format: +2 / -3
    # --------------------------------------

    match = re.match(r"([+-])(\d+)", charge)

    if match:

        sign = match.group(1)
        value = int(match.group(2))

        return value if sign == "+" else -value

    # --------------------------------------
    # Invalid format
    # --------------------------------------

    raise ValueError(
        f"Invalid charge format: {charge}"
    )

# ==========================================
# FORMULA VALIDATION
# ==========================================
"""
Validation utilities used before parsing.

These functions ensure:
- Correct coordination sphere syntax
- Correct counter-ion syntax
- Safe parsing throughout the project
- Early detection of malformed formulas
"""


def formula_verification(formula):
    """
    Validate the coordination sphere syntax.

    Supported examples:
    ------------------
    [Fe(NH3)6]3+
    [PtCl2(NH3)2]
    [dRe2(Cl)8]2-

    Validation rules:
    -----------------
    - Formula must be a string
    - Empty formulas are rejected
    - Metal symbol must be valid
    - Coordination syntax must follow
      the internal parser specification

    Parameters
    ----------
    formula : str

    Returns
    -------
    re.Match

    Raises
    ------
    ValueError
        If the formula format is invalid.
    """

    # --------------------------------------
    # Type validation
    # --------------------------------------

    if not isinstance(formula, str):

        raise ValueError(
            "Coordination sphere formula "
            "must be a string."
        )

    # --------------------------------------
    # Remove optional spaces
    # --------------------------------------

    clean_formula = formula.replace(" ", "")

    # --------------------------------------
    # Empty formula validation
    # --------------------------------------

    if clean_formula in {"", "[]"}:

        raise ValueError(
            "Coordination sphere formula "
            "cannot be empty."
        )

    # --------------------------------------
    # Coordination sphere pattern
    # --------------------------------------
    #
    # Groups:
    # 1 → Metal-metal bond prefix
    # 2 → Metal symbol
    # 3 → Metal coefficient
    # 4 → Ligand block
    # 5 → Global charge
    #
    # Supported bond prefixes:
    # s → single
    # d → double
    # t → triple
    # q → quadruple
    #
    # Example:
    # [dRe2(Cl)8]2-
    #
    # 1 = d
    # 2 = Re
    # 3 = 2
    # 4 = (Cl)8
    # 5 = 2-
    #
    # --------------------------------------

    pattern = (
        r"\[([sdtq])?"
        r"([A-Z][a-z]?)"
        r"([1-9]\d*)?"
        r"(\((?:.+)\)(?:[1-9]\d*)?)*"
        r"\]"
        r"([0-9+-]+)?$"
    )

    match = re.match(
        pattern,
        clean_formula,
    )

    # --------------------------------------
    # Invalid syntax
    # --------------------------------------

    if not match:

        raise ValueError(
            "Invalid coordination "
            "sphere format."
        )

    return match



def counter_ions_verification(
    formula_counter_ions,
):
    """
    Validate counter-ion formula syntax.

    Supported examples:
    ------------------
    (PF6)2
    (ClO4)
    (PF6)2(ClO4)

    Validation rules:
    -----------------
    - Input must be a string
    - Empty expressions are rejected
    - Parentheses syntax is mandatory
    - Stoichiometric coefficients must
      remain positive integers

    Parameters
    ----------
    formula_counter_ions : str

    Returns
    -------
    str
        Sanitized counter-ion formula.

    Raises
    ------
    ValueError
        If the syntax is invalid.
    """

    # --------------------------------------
    # Type validation
    # --------------------------------------

    if not isinstance(
        formula_counter_ions,
        str,
    ):

        raise ValueError(
            "Counter-ion formula "
            "must be a string."
        )

    # --------------------------------------
    # Remove spaces
    # --------------------------------------

    clean_formula = (
        formula_counter_ions
        .replace(" ", "")
    )

    # --------------------------------------
    # Empty validation
    # --------------------------------------

    if clean_formula in {"", "()"}:

        raise ValueError(
            "Counter-ion formula "
            "cannot be empty."
        )

    # --------------------------------------
    # Counter-ion pattern
    # --------------------------------------

    pattern = r"(\((.*?)\)([1-9]\d*))*"

    # --------------------------------------
    # Syntax validation
    # --------------------------------------

    if not re.match(
        pattern,
        clean_formula,
    ):

        raise ValueError(
            "Invalid counter-ion "
            "formula format."
        )

    return clean_formula


# ==========================================
# FORMULA PARSING
# ==========================================
"""
Main parsing utilities used to extract
all chemically relevant information from
the coordination sphere.

Extracted information:
- Metal centres
- Ligands
- Counter ions
- Bond order
- Stoichiometry
- Bridging ligands
"""


def parse_counter_ions(
    formula_counter_ions,
):
    """
    Parse counter ions and their
    stoichiometric coefficients.

    Parameters
    ----------
    formula_counter_ions : str

    Returns
    -------
    tuple
        (
            expanded_ions,
            unique_ions,
            coefficients,
        )

    Notes
    -----
    expanded_ions:
        Fully expanded list containing
        duplicated ions.

    unique_ions:
        Unique ordered ion list.

    coefficients:
        Stoichiometric coefficients.
    """

    # --------------------------------------
    # Formula validation
    # --------------------------------------

    clean_formula = (
        counter_ions_verification(
            formula_counter_ions
        )
    )

    # --------------------------------------
    # Extract ion blocks
    # --------------------------------------

    matches = re.findall(
        r"\((.*?)\)(\d*)",
        clean_formula,
    )

    expanded_ions = []
    unique_ions = []
    coefficients = []

    # --------------------------------------
    # Parse each ion
    # --------------------------------------

    for counter_ion, coefficient in matches:

        # ----------------------------------
        # Default coefficient
        # ----------------------------------

        coefficient = (
            int(coefficient)
            if coefficient
            else 1
        )

        # ----------------------------------
        # Safety limit
        # ----------------------------------

        if coefficient > 12:

            raise ValueError(
                "Counter-ion coefficient "
                "cannot exceed 12."
            )

        # ----------------------------------
        # Database lookup
        # ----------------------------------

        ion_key = find_counter_ion(
            counter_ion
        )

        if ion_key is False:

            raise ValueError(
                f"Unknown counter ion: "
                f"{counter_ion}"
            )

        # ----------------------------------
        # Expanded storage
        # ----------------------------------

        expanded_ions.extend(
            [ion_key] * coefficient
        )

        unique_ions.append(ion_key)

        coefficients.append(coefficient)

    return (
        expanded_ions,
        unique_ions,
        coefficients,
    )



def parse_metal(formula):
    """
    Extract metal centre(s) from the
    coordination sphere.

    Supported nuclearities:
    -----------------------
    - Mononuclear
    - Dinuclear

    Parameters
    ----------
    formula : str

    Returns
    -------
    list[str]

    Raises
    ------
    ValueError
        If the metal is unknown or if
        unsupported nuclearity is detected.
    """

    # --------------------------------------
    # Validate formula
    # --------------------------------------

    match = formula_verification(
        formula
    )

    # --------------------------------------
    # Extract metal symbol
    # --------------------------------------

    metal = match.group(2)

    # --------------------------------------
    # Extract coefficient
    # --------------------------------------

    coefficient = (
        int(match.group(3))
        if match.group(3)
        else 1
    )

    # --------------------------------------
    # Database validation
    # --------------------------------------

    if metal not in data_metals:

        raise ValueError(
            f"Unknown metal: {metal}"
        )

    # --------------------------------------
    # Nuclearity validation
    # --------------------------------------

    if coefficient not in {1, 2}:

        raise ValueError(
            "Only mono- and dinuclear "
            "complexes are supported."
        )

    return [metal] * coefficient



def bond_order(formula):
    """
    Return the metal-metal bond order.

    Prefix mapping:
    ----------------
    s → single bond
    d → double bond
    t → triple bond
    q → quadruple bond

    Parameters
    ----------
    formula : str

    Returns
    -------
    int
        Bond order value.
    """

    # --------------------------------------
    # Validate formula
    # --------------------------------------

    match = formula_verification(
        formula
    )

    # --------------------------------------
    # Prefix → order mapping
    # --------------------------------------

    order_map = {
        "s": 1,
        "d": 2,
        "t": 3,
        "q": 4,
    }

    order = order_map.get(
        match.group(1),
        0,
    )

    # --------------------------------------
    # Metal count
    # --------------------------------------

    metal_count = (
        int(match.group(3))
        if match.group(3)
        else 1
    )

    # --------------------------------------
    # Consistency validation
    # --------------------------------------

    if (
        metal_count == 1
        and order != 0
    ):

        raise ValueError(
            "Metal-metal bonds require "
            "two metal centres."
        )

    return order

def parse_ligands(formula):
    """
    Extract ligands and stoichiometric
    coefficients from the coordination sphere.

    Bridging ligands are internally encoded
    using negative coefficients.

    Example:
    --------
    (NH3)4(m-Cl)2

    Returns:
    --------
    expanded_ligands :
        Fully expanded ligand list.

    unique_ligands :
        Unique ordered ligand list.

    coefficients :
        Stoichiometric coefficients.

        Negative values indicate
        bridging ligands.
    """

    # --------------------------------------
    # Validate formula
    # --------------------------------------

    match = formula_verification(
        formula
    )

    # --------------------------------------
    # Extract ligand block
    # --------------------------------------

    ligands_block = match.group(4)

    # --------------------------------------
    # Extract all ligand groups
    # --------------------------------------

    matches = re.findall(
        r"\((.*?)\)(\d*)",
        ligands_block,
    )

    expanded_ligands = []
    unique_ligands = []
    coefficients = []

    # --------------------------------------
    # Parse ligands
    # --------------------------------------

    for ligand, coefficient in matches:

        # ----------------------------------
        # Default coefficient
        # ----------------------------------

        coefficient = (
            int(coefficient)
            if coefficient
            else 1
        )

        # ----------------------------------
        # Safety limit
        # ----------------------------------

        if coefficient > 12:

            raise ValueError(
                "Ligand coefficient "
                "cannot exceed 12."
            )

        # ----------------------------------
        # Bridging ligand detection
        # ----------------------------------

        is_bridging = ligand.startswith(
            "m-"
        )

        if is_bridging:

            ligand = ligand[2:]

            coefficient *= -1

        # ----------------------------------
        # Database lookup
        # ----------------------------------

        ligand_key = find_ligand(
            ligand
        )

        if ligand_key is False:

            raise ValueError(
                f"Unknown ligand: "
                f"{ligand}"
            )

        # ----------------------------------
        # Storage
        # ----------------------------------

        coefficients.append(
            coefficient
        )

        unique_ligands.append(
            ligand_key
        )

        expanded_ligands.extend(
            [ligand_key]
            * abs(coefficient)
        )

    return (
        expanded_ligands,
        unique_ligands,
        coefficients,
    )



def count_bridging_ligands(formula):
    """
    Count the number of bridging ligands.

    Bridging ligands are identified using
    negative stoichiometric coefficients.

    Parameters
    ----------
    formula : str

    Returns
    -------
    int
    """

    _, _, coefficients = parse_ligands(
        formula
    )

    return sum(
        1
        for coefficient in coefficients
        if coefficient < 0
    )



def parse_elements(formula):
    """
    Return all chemical components present
    in the coordination compound.

    Included components:
    --------------------
    - Ligands
    - Metal centres

    Parameters
    ----------
    formula : str

    Returns
    -------
    list[str]
    """

    elements = []

    # --------------------------------------
    # Ligands
    # --------------------------------------

    elements.extend(
        parse_ligands(formula)[0]
    )

    # --------------------------------------
    # Metals
    # --------------------------------------

    elements.extend(
        parse_metal(formula)
    )

    return elements


# ==========================================
# CHARGE ANALYSIS
# ==========================================
"""
Charge-related calculations.

This section handles:
- Coordination sphere charge
- Ligand contribution
- Counter-ion contribution
- Metal oxidation states
- Formal charge consistency
"""


def counter_ions_charge(
    formula_counter_ions,
):
    """
    Calculate the total charge of all
    counter ions.

    Parameters
    ----------
    formula_counter_ions : str

    Returns
    -------
    int
    """

    # --------------------------------------
    # Expand ions
    # --------------------------------------

    counter_ions = (
        parse_counter_ions(
            formula_counter_ions
        )[0]
    )

    total_charge = 0

    # --------------------------------------
    # Sum charges
    # --------------------------------------

    for counter_ion in counter_ions:

        total_charge += (
            data_counter_ions[
                counter_ion
            ]["charge"]
        )

    return total_charge



def complex_charge(
    formula,
    formula_counter_ions=None,
):
    """
    Return the global charge of the
    coordination sphere.

    Parameters
    ----------
    formula : str

    formula_counter_ions : str | None

    Returns
    -------
    int

    Notes
    -----
    If counter ions are supplied,
    electroneutrality is verified.
    """

    # --------------------------------------
    # Validate formula
    # --------------------------------------

    match = formula_verification(
        formula
    )

    # --------------------------------------
    # Sphere charge
    # --------------------------------------

    sphere_charge = transform_charge(
        match.group(5)
    )

    # --------------------------------------
    # No counter ions
    # --------------------------------------

    if formula_counter_ions is None:
        return sphere_charge

    # --------------------------------------
    # Counter-ion charge
    # --------------------------------------

    counter_charge = (
        counter_ions_charge(
            formula_counter_ions
        )
    )

    # --------------------------------------
    # Neutrality validation
    # --------------------------------------

    if counter_charge == 0:

        raise ValueError(
            "Counter ions cannot be "
            "globally neutral."
        )

    # --------------------------------------
    # Charge consistency validation
    # --------------------------------------

    if (
        sphere_charge != 0
        and sphere_charge != -counter_charge
    ):

        raise ValueError(
            "Counter-ion charge does "
            "not match sphere charge."
        )

    return -counter_charge



def ligands_charge(formula):
    """
    Calculate the total ligand charge.

    Parameters
    ----------
    formula : str

    Returns
    -------
    int
    """

    ligands = parse_ligands(
        formula
    )[0]

    total_charge = 0

    # --------------------------------------
    # Sum ligand charges
    # --------------------------------------

    for ligand in ligands:

        ligand = ligand.replace(
            "m-",
            "",
        )

        total_charge += (
            data_ligands[
                ligand
            ]["charge"]
        )

    return total_charge



def metal_charge(
    formula,
    formula_counter_ions=None,
):
    """
    Calculate the formal metal oxidation
    state using charge balance.

    Parameters
    ----------
    formula : str

    formula_counter_ions : str | None

    Returns
    -------
    int
    """

    # --------------------------------------
    # Global complex charge
    # --------------------------------------

    total_charge = complex_charge(
        formula,
        formula_counter_ions,
    )

    # --------------------------------------
    # Ligand contribution
    # --------------------------------------

    ligand_charge = ligands_charge(
        formula
    )

    # --------------------------------------
    # Number of metal centres
    # --------------------------------------

    metal_count = len(
        parse_metal(formula)
    )

    # --------------------------------------
    # Formal oxidation state
    # --------------------------------------

    return (
        total_charge
        - ligand_charge
    ) // metal_count



def oxidation_state(formula):
    """
    Calculate the d-electron configuration
    of the metal centre.

    Parameters
    ----------
    formula : str

    Returns
    -------
    tuple
        (
            d_electron_count,
            chemical_warning,
        )

    Notes
    -----
    The returned oxidation value corresponds
    to the remaining d electrons after
    oxidation.
    """

    # --------------------------------------
    # Metal extraction
    # --------------------------------------

    metal = parse_metal(
        formula
    )[0]

    # --------------------------------------
    # d-electron count
    # --------------------------------------

    oxidation = (
        data_metals[metal]["group"]
        - metal_charge(formula)
    )

    # --------------------------------------
    # Known accessible oxidation states
    # --------------------------------------

    possible_states = (
        data_metals[metal]
        ["possible_ox_state"]
    )

    # --------------------------------------
    # Chemical plausibility check
    # --------------------------------------

    if (
        metal_charge(formula)
        not in possible_states
    ):

        sign = (
            "+"
            if metal_charge(formula) > 0
            else ""
        )

        remark = (
            "This oxidation state is "
            "chemically unlikely for "
            f"{metal} "
            f"({sign}"
            f"{metal_charge(formula)})."
        )

    else:

        remark = ""

    # --------------------------------------
    # Impossible configurations
    # --------------------------------------

    if oxidation < 0 or oxidation > 12:

        raise ValueError(
            "Impossible oxidation "
            "state detected."
        )

    return oxidation, remark

# ============================================================
# ELECTRONIC PROPERTIES
# ============================================================
"""
Advanced electronic structure analysis:
- Spin state estimation
- Orbital occupation
- Magnetic properties
- Jahn–Teller distortion prediction
"""


def low_spin_configuration(d_electrons):
    """
    Compute the low-spin electron configuration.

    Electrons preferentially pair in lower-energy
    orbitals before occupying higher levels.

    Parameters
    ----------
    d_electrons : int
        Number of d-electrons.

    Returns
    -------
    tuple[int, int]
        (paired_electrons, unpaired_electrons)
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
    Compute the high-spin electron configuration.

    Electrons occupy degenerate orbitals
    before pairing (Hund's rule).

    Parameters
    ----------
    d_electrons : int
        Number of d-electrons.

    Returns
    -------
    tuple[int, int]
        (paired_electrons, unpaired_electrons)
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

    Simplified model:
    - Strong-field ligands → low spin
    - Weak-field ligands → high spin

    Parameters
    ----------
    formula : str
        Coordination complex formula.

    Returns
    -------
    tuple
        (
            spin_label,
            (
                paired_electrons,
                unpaired_electrons,
            ),
        )
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
    Populate the five d-orbitals.

    Orbital ordering:
    t2g → eg

    Parameters
    ----------
    paired_electrons : int
        Number of electron pairs.

    unpaired_electrons : int
        Number of unpaired electrons.

    Returns
    -------
    list[int]
        Orbital occupation vector.
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
    Estimate the spin-only magnetic moment.

    Formula used:
        μ = √(n(n + 2))

    where n is the number of unpaired electrons.

    Parameters
    ----------
    formula : str
        Coordination complex formula.

    Returns
    -------
    float
        Magnetic moment in Bohr magnetons.
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
    Determine whether the complex is:
    - Diamagnetic
    - Paramagnetic

    Parameters
    ----------
    formula : str
        Coordination complex formula.

    Returns
    -------
    str
        Magnetic behaviour.
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
    Estimate the crystal field stabilization energy (CFSE).

    Octahedral approximation:
        CFSE = (-0.4 × t2g) + (0.6 × eg)

    Parameters
    ----------
    formula : str
        Coordination complex formula.

    Returns
    -------
    float
        Approximate CFSE value.
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
    Predict Jahn–Teller distortion tendencies.

    Distortion occurs when degenerate orbitals
    are asymmetrically occupied.

    Parameters
    ----------
    formula : str
        Coordination complex formula.

    Returns
    -------
    str
        Distortion prediction.
    """

    spin_state = determine_spin_state(
        formula
    )

    orbitals = fill_d_orbitals(
        spin_state[1][0],
        spin_state[1][1],
    )

    # --------------------------------------------------------
    # t2g asymmetry
    # --------------------------------------------------------

    for index in range(2):

        if orbitals[index] != orbitals[index + 1]:

            return (
                "Weak Jahn-Teller distortion"
            )

    # --------------------------------------------------------
    # eg asymmetry
    # --------------------------------------------------------

    for index in range(3, 4):

        if orbitals[index] != orbitals[index + 1]:

            return (
                "Strong Jahn-Teller distortion"
            )

    return "No Jahn-Teller distortion"


# ============================================================
# COORDINATION COMPOUND NOMENCLATURE
# ============================================================
"""
IUPAC-inspired nomenclature engine:
- Ligand prefixes
- Bridging ligands
- Counter ions
- Oxidation state notation
"""


# ============================================================
# NOMENCLATURE PREFIXES
# ============================================================

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


# ============================================================
# LIGAND NOMENCLATURE UTILITIES
# ============================================================

def ligand_nomenclature_name(ligand):
    """
    Return the nomenclature-compatible ligand name.

    Parameters
    ----------
    ligand : str
        Ligand identifier.

    Returns
    -------
    str
        Proper ligand nomenclature name.
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
    Determine whether multiplicative prefixes
    such as bis-, tris-, tetrakis- must be used.

    Parameters
    ----------
    ligand_name : str
        Ligand nomenclature name.

    Returns
    -------
    bool
        True if special prefixes are required.
    """

    for ligand in data_ligands.values():

        if (
            ligand["name"] == ligand_name
            and ligand.get("coeff") == "yes"
        ):
            return True

    return False


# ============================================================
# COUNTER-ION NOMENCLATURE
# ============================================================

def naming_counter_ions(
    formula_counter_ions,
):
    """
    Generate the counter-ion nomenclature sequence.

    Parameters
    ----------
    formula_counter_ions : str | None
        Counter-ion formula.

    Returns
    -------
    str
        Ordered counter-ion names.
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

# ============================================================
# LIGAND GEOMETRY UTILITIES
# ============================================================
"""
Internal ligand geometry generators used to build
more chemically realistic coordination compounds.

Supported ligand geometries:
- sphere
- linear
- double-linear
- bent
- tetrahedral
"""

# ============================================================
# LINEAR LIGAND GEOMETRY
# ============================================================

def ligand_linear(
    ligand,
    ligand_coord,
    r,
):
    """
    Generate coordinates for a linear ligand.

    Example:
        M–C≡O

    Parameters
    ----------
    ligand : str
        Ligand identifier.

    ligand_coord : tuple
        Coordination position around the metal.

    r : float
        Metal–ligand bond distance.

    Returns
    -------
    list[tuple]
        Atomic coordinates.
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


# ============================================================
# DOUBLE-LINEAR LIGAND GEOMETRY
# ============================================================

def ligand_dlinear(
    ligand,
    ligand_coord,
    r,
):
    """
    Generate coordinates for a double-linear
    ligand geometry.

    Example:
        –N≡C–

    Parameters
    ----------
    ligand : str
        Ligand identifier.

    ligand_coord : tuple
        Coordination position around the metal.

    r : float
        Metal–ligand bond distance.

    Returns
    -------
    list[tuple]
        Atomic coordinates.
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


# ============================================================
# BENT LIGAND GEOMETRY
# ============================================================

def ligand_bent(
    ligand,
    ligand_coord,
    r,
):
    """
    Generate coordinates for a bent ligand.

    Example:
        H₂O-like geometry

    Parameters
    ----------
    ligand : str
        Ligand identifier.

    ligand_coord : tuple
        Coordination position around the metal.

    r : float
        Metal–ligand bond distance.

    Returns
    -------
    list[tuple]
        Atomic coordinates.
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

    # --------------------------------------------------------
    # Temporary orthogonal vector
    # --------------------------------------------------------

    if abs(direction[0]) > 0.1:

        temp_vector = np.array(
            [0, 1, 0]
        )

    else:

        temp_vector = np.array(
            [1, 0, 0]
        )

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


# ============================================================
# TETRAHEDRAL LIGAND GEOMETRY
# ============================================================

def ligand_tetrahedral(
    ligand,
    ligand_coord,
    r,
):
    """
    Generate coordinates for a tetrahedral
    ligand geometry.

    Example:
        CH₄-like geometry

    Parameters
    ----------
    ligand : str
        Ligand identifier.

    ligand_coord : tuple
        Coordination position around the metal.

    r : float
        Metal–ligand bond distance.

    Returns
    -------
    list[tuple]
        Atomic coordinates.
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

    # --------------------------------------------------------
    # Local orthogonal basis
    # --------------------------------------------------------

    if abs(direction[0]) > 0.1:

        temp_vector = np.array(
            [0, 1, 0]
        )

    else:

        temp_vector = np.array(
            [1, 0, 0]
        )

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


# ============================================================
# LIGAND GEOMETRY LOOKUP
# ============================================================

def get_geometry_ligand(ligand_input):
    """
    Return the internal geometry type
    of a ligand from the database.

    Parameters
    ----------
    ligand_input : str
        Ligand identifier.

    Returns
    -------
    str | bool
        Geometry type if available,
        otherwise False.
    """

    geometry = (
        data_ligands[ligand_input]
        .get("geometry")
    )

    if geometry is not None:
        return geometry

    return False


# ============================================================
# COMPLETE ATOMIC POSITION GENERATION
# ============================================================

def atoms_position(
    formula,
    r=1.7,
):
    """
    Generate the complete 3D coordinates
    of a coordination compound.

    Parameters
    ----------
    formula : str
        Coordination compound formula.

    r : float, optional
        Metal–ligand distance.

    Returns
    -------
    list[tuple]
        Atomic coordinates.
    """

    # --------------------------------------------------------
    # Metal center position
    # --------------------------------------------------------

    positions = [(0, 0, 0)]

    ligand_positions = (
        get_geometry(formula, r)[0]
    )

    ligands = parse_ligands(formula)[0]

    # --------------------------------------------------------
    # Ligand placement
    # --------------------------------------------------------

    for index, ligand in enumerate(ligands):

        geometry = (
            get_geometry_ligand(ligand)
        )

        # ====================================================
        # SPHERICAL LIGAND
        # ====================================================

        if geometry == "sphere":

            positions += [
                ligand_positions[index]
            ]

        # ====================================================
        # LINEAR LIGAND
        # ====================================================

        elif geometry == "linear":

            positions += [
                ligand_positions[index]
            ]

            positions += ligand_linear(
                ligand,
                ligand_positions[index],
                r,
            )

        # ====================================================
        # DOUBLE-LINEAR LIGAND
        # ====================================================

        elif geometry == "dlinear":

            positions += [
                ligand_positions[index]
            ]

            positions += ligand_dlinear(
                ligand,
                ligand_positions[index],
                r,
            )

        # ====================================================
        # BENT LIGAND
        # ====================================================

        elif geometry == "bent":

            positions += [
                ligand_positions[index]
            ]

            positions += ligand_bent(
                ligand,
                ligand_positions[index],
                r,
            )

        # ====================================================
        # TETRAHEDRAL LIGAND
        # ====================================================

        elif geometry == "tetrahedral":

            positions += [
                ligand_positions[index]
            ]

            positions += ligand_tetrahedral(
                ligand,
                ligand_positions[index],
                r,
            )

        # ====================================================
        # UNSUPPORTED GEOMETRY
        # ====================================================

        else:

            raise ValueError(
                "Error: Ligand geometry "
                "not available in 3D."
            )

    return positions

# ============================================================
# ATOMIC SYMBOL EXTRACTION
# ============================================================

def get_atoms(ligand_input):
    """
    Extract all atomic symbols from a ligand.

    The donor atom is placed first in order
    to preserve chemically meaningful atom
    ordering during structure generation.

    Parameters
    ----------
    ligand_input : str
        Ligand formula identifier.

    Returns
    -------
    list[str]
        Ordered atomic symbols.
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

        # ----------------------------------------------------
        # Avoid duplicating donor atom
        # ----------------------------------------------------

        if symbol == donor_atoms[0]:
            continue

        result.extend([symbol] * count)

    return result


# ============================================================
# COMPLETE ATOMIC SYMBOL GENERATION
# ============================================================

def atom_symbols(formula):
    """
    Generate the complete ordered list
    of atomic symbols composing the
    coordination compound.

    Parameters
    ----------
    formula : str
        Coordination compound formula.

    Returns
    -------
    list[str]
        Ordered atomic symbols.
    """

    metals = parse_metal(formula)

    atoms_list = metals.copy()

    ligands = parse_ligands(formula)[0]

    for ligand in ligands:

        atoms_list += get_atoms(ligand)

    return atoms_list


# ============================================================
# ASE COMPOUND CONSTRUCTION
# ============================================================

def create_compound_render(formula):
    """
    Create the ASE Atoms object used
    for molecular rendering.

    Parameters
    ----------
    formula : str
        Coordination compound formula.

    Returns
    -------
    ase.Atoms
        Fully generated molecular object.
    """

    compound = Atoms(
        atom_symbols(formula),
        positions=atoms_position(formula),
    )

    return compound


# ============================================================
# 3D RENDERING UTILITIES
# ============================================================
"""
Interactive molecular visualization using py3Dmol.

Supported render styles:
- Ball and Stick
- Stick
- Sphere
- Lines
- VDW surface
"""


# ============================================================
# MAIN 3D RENDERING ENGINE
# ============================================================

def render_complex(
    compound,
    atoms_size=0.4,
    render_type="Ball and Stick",
):
    """
    Render a coordination complex in 3D.

    Parameters
    ----------
    compound : ase.Atoms
        Molecular object to render.

    atoms_size : float, optional
        Sphere scaling factor.

    render_type : str, optional
        Visualization style.

    Returns
    -------
    py3Dmol.view | str
        Interactive viewer or HTML content.
    """

    # --------------------------------------------------------
    # ASE -> XYZ conversion
    # --------------------------------------------------------

    xyz_string = io.StringIO()

    write(
        xyz_string,
        compound,
        format="xyz",
    )

    xyz_content = xyz_string.getvalue()

    # --------------------------------------------------------
    # py3Dmol viewer creation
    # --------------------------------------------------------

    view = py3Dmol.view(
        width=400,
        height=400,
    )

    view.addModel(
        xyz_content,
        "xyz",
    )

    # --------------------------------------------------------
    # Rendering style selection
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Automatic centering
    # --------------------------------------------------------

    view.zoomTo()

    # --------------------------------------------------------
    # Streamlit environment detection
    # --------------------------------------------------------

    is_streamlit = False

    if "streamlit" in sys.modules:

        from streamlit.runtime.scriptrunner import (
            get_script_run_ctx,
        )

        if get_script_run_ctx() is not None:
            is_streamlit = True

    # --------------------------------------------------------
    # Streamlit rendering
    # --------------------------------------------------------

    if is_streamlit:

        html_content = view._make_html()

        return html_content

    # --------------------------------------------------------
    # Notebook rendering
    # --------------------------------------------------------

    return view.show()

# ============================================================
# HIGH-LEVEL ANALYSIS ENGINE
# ============================================================
"""
Main orchestration utilities used to:
- Validate coordination compounds
- Compute chemical properties
- Generate formatted analyses
- Provide complete coordination chemistry reports
"""


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyse_complex(
    formula,
    formula_counter_ions=None,
):
    """
    Perform a complete coordination chemistry
    analysis of a complex.

    Parameters
    ----------
    formula : str
        Coordination sphere formula.

    formula_counter_ions : str | None
        Counter-ion formula.

    Returns
    -------
    str | Markdown
        Rendered analysis.
    """

    # --------------------------------------------------------
    # Initial validation
    # --------------------------------------------------------

    chemical_rules(formula)

    # --------------------------------------------------------
    # Core descriptors
    # --------------------------------------------------------

    metal = parse_metal(formula)[0]

    oxidation = metal_charge(
        formula,
        formula_counter_ions,
    )

    d_electrons = oxidation_state(
        formula
    )[0]

    electron_total = electron_count(
        formula
    )

    geometry = get_geometry(
        formula
    )[1]

    spin_state = determine_spin_state(
        formula
    )[0]

    magnetic_type = magnetic_behavior(
        formula
    )

    magnetic_value = magnetic_moment(
        formula
    )

    cfse = (
        crystal_field_stabilization_energy(
            formula
        )
    )

    jahn_teller = (
        jahn_teller_distortion(
            formula
        )
    )

    # --------------------------------------------------------
    # Stability analysis
    # --------------------------------------------------------

    stability_engine = StabilityEngine(
        formula
    )

    stability = (
        stability_engine.final_score()
    )

    # --------------------------------------------------------
    # Isomer analysis
    # --------------------------------------------------------

    stereoisomers, enantiomers = (
        isomers(formula)
    )

    # --------------------------------------------------------
    # Formula rendering
    # --------------------------------------------------------

    clean_formula = get_clean_formula(
        formula,
        formula_counter_ions,
    )

    # --------------------------------------------------------
    # Electronic configuration
    # --------------------------------------------------------

    electronic = electronic_structure(
        formula
    )

    noble_gas = electronic[0]

    period = electronic[1]

    s_e = electronic[2]

    d_e = electronic[3]

    configuration = (
        f"[{noble_gas}] "
        f"{period}s{s_e} "
        f"{period - 1}d{d_e}"
    )

    # --------------------------------------------------------
    # Remarks
    # --------------------------------------------------------

    oxidation_remark = (
        oxidation_state(formula)[1]
    )

    electron_remark = (
        electrons_probability(formula)
    )

    # --------------------------------------------------------
    # Formatted output
    # --------------------------------------------------------

    lines = []

    # ========================================================
    # HEADER
    # ========================================================

    lines.append(
        "# Coordination Compound Analysis"
    )

    lines.append("")

    lines.append(
        f"### {clean_formula}"
    )

    lines.append("")

    # ========================================================
    # NOMENCLATURE
    # ========================================================

    lines.append(
        "## Nomenclature"
    )

    lines.append(
        f"**IUPAC-inspired name:** "
        f"{naming_compound(formula, formula_counter_ions)}"
    )

    lines.append("")

    # ========================================================
    # METAL CENTER
    # ========================================================

    lines.append(
        "## Metal Center"
    )

    lines.append(
        f"**Metal:** {metal}"
    )

    lines.append(
        f"**Formal oxidation state:** "
        f"{oxidation:+}"
    )

    lines.append(
        f"**d-electron configuration:** "
        f"d{superscript_safe(d_electrons)}"
    )

    lines.append(
        f"**Electronic configuration:** "
        f"{configuration}"
    )

    if oxidation_remark:

        lines.append(
            f"⚠️ {oxidation_remark}"
        )

    lines.append("")

    # ========================================================
    # COORDINATION ENVIRONMENT
    # ========================================================

    lines.append(
        "## Coordination Environment"
    )

    lines.append(
        f"**Coordination number:** "
        f"{len(parse_ligands(formula)[0])}"
    )

    lines.append(
        f"**Predicted geometry:** "
        f"{geometry}"
    )

    lines.append(
        f"**Spin state:** "
        f"{spin_state}"
    )

    lines.append("")

    # ========================================================
    # ELECTRONIC PROPERTIES
    # ========================================================

    lines.append(
        "## Electronic Properties"
    )

    lines.append(
        f"**Electron count:** "
        f"{electron_total} e⁻"
    )

    lines.append(
        f"**Magnetic behaviour:** "
        f"{magnetic_type}"
    )

    lines.append(
        f"**Magnetic moment:** "
        f"{magnetic_value} BM"
    )

    lines.append(
        f"**CFSE:** "
        f"{cfse}"
    )

    lines.append(
        f"**Jahn–Teller effect:** "
        f"{jahn_teller}"
    )

    if electron_remark:

        lines.append(
            f"⚠️ {electron_remark}"
        )

    lines.append("")

    # ========================================================
    # ISOMERISM
    # ========================================================

    lines.append(
        "## Isomerism"
    )

    lines.append(
        f"**Stereoisomers:** "
        f"{stereoisomers}"
    )

    lines.append(
        f"**Enantiomeric pairs:** "
        f"{enantiomers}"
    )

    lines.append("")

    # ========================================================
    # STABILITY ANALYSIS
    # ========================================================

    lines.append(
        "## Stability Analysis"
    )

    lines.append(
        f"**Global stability score:** "
        f"{stability.total}/100"
    )

    lines.append("")

    lines.append(
        "### Partial Stability Scores"
    )

    lines.append(
        f"- Electron count: "
        f"{round(stability.electron, 1)}"
    )

    lines.append(
        f"- HSAB compatibility: "
        f"{round(stability.hsab, 1)}"
    )

    lines.append(
        f"- Chelation effect: "
        f"{round(stability.chelate, 1)}"
    )

    lines.append(
        f"- Ligand field: "
        f"{round(stability.field, 1)}"
    )

    lines.append(
        f"- Charge stabilization: "
        f"{round(stability.charge, 1)}"
    )

    lines.append(
        f"- Geometry preference: "
        f"{round(stability.geometry, 1)}"
    )

    lines.append(
        f"- Oxidation state preference: "
        f"{round(stability.oxidation, 1)}"
    )

    lines.append(
        f"- π-backbonding: "
        f"{round(stability.backbonding, 1)}"
    )

    lines.append(
        f"- Steric effects: "
        f"{round(stability.steric, 1)}"
    )

    lines.append("")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    lines.append(
        "## Summary"
    )

    if stability.total >= 85:

        lines.append(
            "This coordination compound is predicted "
            "to be highly stable."
        )

    elif stability.total >= 65:

        lines.append(
            "This coordination compound is predicted "
            "to be moderately stable."
        )

    elif stability.total >= 45:

        lines.append(
            "This coordination compound may present "
            "limited stability."
        )

    else:

        lines.append(
            "This coordination compound is predicted "
            "to be chemically unstable."
        )

    # --------------------------------------------------------
    # Final rendering
    # --------------------------------------------------------

    return render_analysis(lines)

# ============================================================
# SAFE DISPLAY UTILITIES
# ============================================================
"""
Formatting helpers used throughout the analysis engine.

These utilities ensure:
- Safe Unicode rendering
- Terminal compatibility
- Cleaner scientific formatting
"""


# ============================================================
# SAFE SUPERSCRIPT RENDERING
# ============================================================

def superscript_safe(value):
    """
    Safely render superscript values.

    Parameters
    ----------
    value : int | str

    Returns
    -------
    str
        Unicode-safe superscript string.
    """

    superscripts = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "+": "⁺",
        "-": "⁻",
    }

    value = str(value)

    return "".join(
        superscripts.get(char, char)
        for char in value
    )


# ============================================================
# SAFE SUBSCRIPT RENDERING
# ============================================================

def subscript_safe(value):
    """
    Safely render subscript values.

    Parameters
    ----------
    value : int | str

    Returns
    -------
    str
        Unicode-safe subscript string.
    """

    subscripts = {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
        "+": "₊",
        "-": "₋",
    }

    value = str(value)

    return "".join(
        subscripts.get(char, char)
        for char in value
    )


# ============================================================
# SAFE FLOAT FORMATTING
# ============================================================

def clean_float(
    value,
    precision=2,
):
    """
    Clean numerical formatting by removing
    unnecessary trailing zeros.

    Parameters
    ----------
    value : float

    precision : int, optional
        Decimal precision.

    Returns
    -------
    str
    """

    formatted = (
        f"{value:.{precision}f}"
    )

    formatted = formatted.rstrip("0")

    formatted = formatted.rstrip(".")

    return formatted


# ============================================================
# SAFE PERCENTAGE FORMATTING
# ============================================================

def format_percentage(value):
    """
    Format percentages consistently.

    Parameters
    ----------
    value : float

    Returns
    -------
    str
    """

    return (
        clean_float(value)
        + "%"
    )


# ============================================================
# SAFE CHEMICAL CHARGE DISPLAY
# ============================================================

def display_charge(charge):
    """
    Format a chemical charge for display.

    Examples
    --------
    +1 -> "+"
    +2 -> "2+"
    -1 -> "-"
    -3 -> "3-"

    Parameters
    ----------
    charge : int

    Returns
    -------
    str
    """

    if charge == 0:
        return "0"

    if charge == 1:
        return "+"

    if charge == -1:
        return "-"

    if charge > 0:
        return f"{charge}+"

    return f"{abs(charge)}-"


# ============================================================
# CHEMICAL FORMULA BEAUTIFIER
# ============================================================

def beautify_formula(formula):
    """
    Convert raw formulas into cleaner
    Unicode-rendered formulas.

    Parameters
    ----------
    formula : str

    Returns
    -------
    str
    """

    result = ""

    for character in formula:

        if character.isdigit():

            result += subscript_safe(
                character
            )

        else:

            result += character

    return result


# ============================================================
# SAFE TITLE GENERATOR
# ============================================================

def section_title(title):
    """
    Generate consistently formatted
    analysis section titles.

    Parameters
    ----------
    title : str

    Returns
    -------
    str
    """

    separator = "─" * len(title)

    return (
        f"{title}\n"
        f"{separator}"
    )


# ============================================================
# SAFE ANALYSIS BLOCK
# ============================================================

def analysis_block(
    title,
    content,
):
    """
    Create a formatted analysis block.

    Parameters
    ----------
    title : str

    content : list[str]

    Returns
    -------
    list[str]
    """

    block = []

    block.append(
        section_title(title)
    )

    block.append("")

    block.extend(content)

    block.append("")

    return block


# ============================================================
# TERMINAL COLOR STRIPPING
# ============================================================

def remove_ansi(text):
    """
    Remove ANSI escape sequences.

    Useful for:
    - Streamlit rendering
    - Notebook rendering
    - Markdown export

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

    ansi_pattern = re.compile(
        r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
    )

    return ansi_pattern.sub(
        "",
        text,
    )

# ============================================================
# ADVANCED CHEMICAL ANALYSIS UTILITIES
# ============================================================
"""
Additional high-level chemistry utilities used for:
- Coordination analysis
- Ligand statistics
- Metal classification
- Geometry interpretation
"""


# ============================================================
# TRANSITION SERIES CLASSIFICATION
# ============================================================

def transition_series(metal):
    """
    Return the transition series of a metal.

    Parameters
    ----------
    metal : str

    Returns
    -------
    str
    """

    first_row = {
        "Sc", "Ti", "V", "Cr", "Mn",
        "Fe", "Co", "Ni", "Cu", "Zn",
    }

    second_row = {
        "Y", "Zr", "Nb", "Mo", "Tc",
        "Ru", "Rh", "Pd", "Ag", "Cd",
    }

    third_row = {
        "La", "Hf", "Ta", "W", "Re",
        "Os", "Ir", "Pt", "Au", "Hg",
    }

    if metal in first_row:
        return "3d transition series"

    if metal in second_row:
        return "4d transition series"

    if metal in third_row:
        return "5d transition series"

    return "Unknown transition series"


# ============================================================
# METAL CLASSIFICATION
# ============================================================

def classify_metal(metal):
    """
    Classify a metal according to
    periodic block chemistry.

    Parameters
    ----------
    metal : str

    Returns
    -------
    str
    """

    group = data_metals[metal]["group"]

    if 3 <= group <= 7:
        return "Early transition metal"

    if 8 <= group <= 10:
        return "Middle transition metal"

    if 11 <= group <= 12:
        return "Late transition metal"

    return "Unknown classification"


# ============================================================
# LIGAND TYPE ANALYSIS
# ============================================================

def ligand_type_statistics(formula):
    """
    Analyse ligand categories present
    in the coordination sphere.

    Parameters
    ----------
    formula : str

    Returns
    -------
    dict
    """

    ligands = parse_ligands(formula)[0]

    statistics = {
        "neutral": 0,
        "anionic": 0,
        "cationic": 0,
        "pi_acceptor": 0,
        "strong_field": 0,
        "chelating": 0,
        "bridging": 0,
    }

    for ligand in ligands:

        ligand_name = ligand.replace(
            "m-",
            "",
        )

        ligand_data = data_ligands.get(
            ligand_name,
            {},
        )

        charge = ligand_data.get(
            "charge",
            0,
        )

        # ----------------------------------------------------
        # Charge classification
        # ----------------------------------------------------

        if charge == 0:
            statistics["neutral"] += 1

        elif charge < 0:
            statistics["anionic"] += 1

        else:
            statistics["cationic"] += 1

        # ----------------------------------------------------
        # π-acceptor ligands
        # ----------------------------------------------------

        if ligand_data.get(
            "pi_acceptor"
        ):
            statistics["pi_acceptor"] += 1

        # ----------------------------------------------------
        # Strong field ligands
        # ----------------------------------------------------

        if ligand_data.get(
            "field",
            1,
        ) >= 3:
            statistics["strong_field"] += 1

        # ----------------------------------------------------
        # Chelating ligands
        # ----------------------------------------------------

        if ligand_data.get(
            "denticity",
            1,
        ) > 1:
            statistics["chelating"] += 1

        # ----------------------------------------------------
        # Bridging ligands
        # ----------------------------------------------------

        if ligand.startswith("m-"):
            statistics["bridging"] += 1

    return statistics


# ============================================================
# COORDINATION SATURATION ANALYSIS
# ============================================================

def coordination_saturation(formula):
    """
    Estimate coordination saturation
    around the metal center.

    Parameters
    ----------
    formula : str

    Returns
    -------
    tuple
        saturation level, interpretation
    """

    coordination_number = len(
        parse_ligands(formula)[0]
    )

    if coordination_number <= 2:

        return (
            "Low coordination",
            "Highly unsaturated metal center",
        )

    if coordination_number <= 4:

        return (
            "Moderate coordination",
            "Partially saturated metal center",
        )

    if coordination_number == 5:

        return (
            "High coordination",
            "Near-saturated coordination sphere",
        )

    if coordination_number >= 6:

        return (
            "Fully coordinated",
            "Saturated coordination environment",
        )

    return (
        "Unknown",
        "Unknown coordination saturation",
    )


# ============================================================
# LIGAND FIELD CLASSIFICATION
# ============================================================

def ligand_field_classification(formula):
    """
    Estimate global ligand field strength.

    Parameters
    ----------
    formula : str

    Returns
    -------
    tuple
        field strength, interpretation
    """

    ligands = parse_ligands(formula)[0]

    total_field = 0

    for ligand in ligands:

        ligand_name = ligand.replace(
            "m-",
            "",
        )

        total_field += (
            data_ligands.get(
                ligand_name,
                {},
            ).get(
                "field",
                1,
            )
        )

    average_field = (
        total_field / max(1, len(ligands))
    )

    if average_field >= 3.5:

        return (
            "Strong-field",
            "Large crystal field splitting",
        )

    if average_field >= 2:

        return (
            "Intermediate-field",
            "Moderate crystal field splitting",
        )

    return (
        "Weak-field",
        "Small crystal field splitting",
    )


# ============================================================
# GEOMETRICAL DISTORTION ANALYSIS
# ============================================================

def geometry_distortion_risk(formula):
    """
    Estimate the probability of geometrical
    distortion around the metal center.

    Parameters
    ----------
    formula : str

    Returns
    -------
    tuple
        distortion risk, interpretation
    """

    d_electrons = electronic_structure(
        formula
    )[3]

    geometry = get_geometry(
        formula
    )[1]

    # --------------------------------------------------------
    # Strong Jahn–Teller candidates
    # --------------------------------------------------------

    if geometry == "octahedral":

        if d_electrons in {4, 7, 9}:

            return (
                "High distortion risk",
                "Strong Jahn–Teller activity likely",
            )

    if geometry == "tetrahedral":

        if d_electrons in {1, 4, 6, 9}:

            return (
                "Moderate distortion risk",
                "Electronic asymmetry possible",
            )

    return (
        "Low distortion risk",
        "Geometry expected to remain stable",
    )


# ============================================================
# REACTIVITY ESTIMATION
# ============================================================

def estimate_reactivity(formula):
    """
    Estimate global coordination reactivity.

    Parameters
    ----------
    formula : str

    Returns
    -------
    tuple
        reactivity level, interpretation
    """

    stability = (
        StabilityEngine(formula)
        .final_score()
        .total
    )

    if stability >= 85:

        return (
            "Low reactivity",
            "Very inert coordination sphere",
        )

    if stability >= 65:

        return (
            "Moderate reactivity",
            "Reasonably stable coordination sphere",
        )

    if stability >= 45:

        return (
            "Elevated reactivity",
            "Potentially labile coordination sphere",
        )

    return (
        "High reactivity",
        "Highly unstable coordination sphere",
    )

# ============================================================
# ATOMIC SYMBOL EXTRACTION
# ============================================================

def get_atoms(ligand_input):
    """
    Extract all atom symbols composing a ligand.

    The donor atom is always placed first in
    the returned list in order to preserve
    bonding consistency during 3D generation.

    Parameters
    ----------
    ligand_input : str
        Ligand formula key from the database.

    Returns
    -------
    list[str]
        Ordered atomic symbols.
    """

    ligand_data = data_ligands.get(
        ligand_input
    )

    donor_atoms = ligand_data.get(
        "donor_atoms"
    )

    atoms_list = (
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

        coefficient = (
            int(match.group(2))
            if match.group(2)
            else 1
        )

        # Avoid duplicating donor atom
        if symbol == donor_atoms[0]:
            continue

        atoms_list.extend(
            [symbol] * coefficient
        )

    return atoms_list


# ============================================================
# GLOBAL ATOMIC SYMBOL GENERATION
# ============================================================

def atom_symbols(formula):
    """
    Generate the complete ordered list of
    atomic symbols composing the complex.

    Parameters
    ----------
    formula : str
        Coordination sphere formula.

    Returns
    -------
    list[str]
        Ordered atomic symbols.
    """

    metals = parse_metal(formula)

    atoms_list = metals.copy()

    ligands = parse_ligands(formula)[0]

    for ligand in ligands:

        atoms_list += get_atoms(ligand)

    return atoms_list


# ============================================================
# ASE COMPOUND CONSTRUCTION
# ============================================================

def create_compound_render(formula):
    """
    Create the ASE Atoms object used for
    3D rendering.

    Parameters
    ----------
    formula : str
        Coordination sphere formula.

    Returns
    -------
    ase.Atoms
        Complete molecular object.
    """

    compound = Atoms(
        atom_symbols(formula),
        positions=atoms_position(formula),
    )

    return compound


# ============================================================
# 3D RENDERING ENGINE
# ============================================================

def render_complex(
    compound,
    atoms_size=0.4,
    render_type="Ball and Stick",
):
    """
    Render an interactive 3D coordination
    complex using py3Dmol.

    Supported render styles:
    - Ball and Stick
    - Stick
    - Sphere
    - Lines
    - VDW

    Parameters
    ----------
    compound : ase.Atoms
        Compound to render.

    atoms_size : float
        Sphere scaling factor.

    render_type : str
        Visualization style.

    Returns
    -------
    HTML | py3Dmol view
    """

    # ========================================================
    # ASE → XYZ CONVERSION
    # ========================================================

    xyz_buffer = io.StringIO()

    write(
        xyz_buffer,
        compound,
        format="xyz",
    )

    xyz_content = xyz_buffer.getvalue()

    # ========================================================
    # PY3DMOL VIEW INITIALIZATION
    # ========================================================

    view = py3Dmol.view(
        width=420,
        height=420,
    )

    view.addModel(
        xyz_content,
        "xyz",
    )

    # ========================================================
    # VISUAL STYLE SELECTION
    # ========================================================

    if render_type == "Ball and Stick":

        view.setStyle(
            {
                "stick": {
                    "radius": 0.12
                },
                "sphere": {
                    "scale": atoms_size
                },
            }
        )

    elif render_type == "Stick":

        view.setStyle(
            {
                "stick": {
                    "radius": 0.16
                }
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

        view.addSurface(
            py3Dmol.VDW
        )

    else:

        raise ValueError(
            f"Unknown render type: {render_type}"
        )

    # ========================================================
    # CAMERA SETTINGS
    # ========================================================

    view.zoomTo()

    view.setBackgroundColor(
        "white"
    )

    # ========================================================
    # STREAMLIT DETECTION
    # ========================================================

    is_streamlit = False

    if "streamlit" in sys.modules:

        from streamlit.runtime.scriptrunner import (
            get_script_run_ctx,
        )

        if get_script_run_ctx() is not None:
            is_streamlit = True

    # ========================================================
    # STREAMLIT OUTPUT
    # ========================================================

    if is_streamlit:

        html_content = view._make_html()

        return html_content

    # ========================================================
    # JUPYTER / STANDARD OUTPUT
    # ========================================================

    return view.show()


# ============================================================
# COMPLETE ANALYSIS REPORT
# ============================================================

def analyze_complex(
    formula,
    formula_counter_ions=None,
):
    """
    Generate a complete coordination
    chemistry analysis report.

    Parameters
    ----------
    formula : str
        Coordination sphere formula.

    formula_counter_ions : str | None
        Optional counter-ion formula.

    Returns
    -------
    str | Markdown
        Fully formatted analysis report.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    chemical_rules(formula)

    # ========================================================
    # BASIC DATA
    # ========================================================

    metal = parse_metal(formula)[0]

    metal_count = len(
        parse_metal(formula)
    )

    geometry = get_geometry(
        formula
    )[1]

    oxidation = metal_charge(
        formula,
        formula_counter_ions,
    )

    d_electrons = electronic_structure(
        formula
    )[3]

    electron_total = electron_count(
        formula
    )

    spin_state = determine_spin_state(
        formula
    )[0]

    magnetic_type = magnetic_behavior(
        formula
    )

    magnetic_value = magnetic_moment(
        formula
    )

    cfse = crystal_field_stabilization_energy(
        formula
    )

    jahn_teller = jahn_teller_distortion(
        formula
    )

    isomer_count, enantiomers = isomers(
        formula
    )

    stability = StabilityEngine(
        formula
    ).final_score()

    # ========================================================
    # FORMATTED OUTPUT
    # ========================================================

    lines = []

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    lines.append(
        "# Coordination Complex Analysis"
    )

    lines.append("")

    lines.append(
        f"### {get_clean_formula(formula, formula_counter_ions)}"
    )

    lines.append("")

    # --------------------------------------------------------
    # NOMENCLATURE
    # --------------------------------------------------------

    lines.append(
        "## Nomenclature"
    )

    lines.append(
        f"**IUPAC-inspired name:** "
        f"{naming_compound(formula, formula_counter_ions)}"
    )

    lines.append("")

    # --------------------------------------------------------
    # METAL CENTER
    # --------------------------------------------------------

    lines.append(
        "## Metal Center"
    )

    lines.append(
        f"**Metal:** {metal}"
    )

    lines.append(
        f"**Nuclearity:** {metal_count}"
    )

    lines.append(
        f"**Formal oxidation state:** "
        f"{oxidation:+d}"
    )

    lines.append(
        f"**d-electron configuration:** "
        f"d{d_electrons}"
    )

    lines.append("")

    # --------------------------------------------------------
    # ELECTRONIC STRUCTURE
    # --------------------------------------------------------

    lines.append(
        "## Electronic Structure"
    )

    lines.append(
        f"**Electron count:** "
        f"{electron_total} e⁻"
    )

    lines.append(
        f"**Spin state:** "
        f"{spin_state}"
    )

    lines.append(
        f"**Magnetic behavior:** "
        f"{magnetic_type}"
    )

    lines.append(
        f"**Magnetic moment:** "
        f"{magnetic_value} BM"
    )

    lines.append(
        f"**CFSE:** "
        f"{cfse}"
    )

    lines.append(
        f"**Jahn–Teller effect:** "
        f"{jahn_teller}"
    )

    lines.append("")

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    lines.append(
        "## Geometry"
    )

    lines.append(
        f"**Coordination geometry:** "
        f"{geometry}"
    )

    lines.append(
        f"**Coordination number:** "
        f"{len(parse_ligands(formula)[0])}"
    )

    lines.append("")

    # --------------------------------------------------------
    # ISOMERISM
    # --------------------------------------------------------

    lines.append(
        "## Stereochemistry"
    )

    lines.append(
        f"**Possible stereoisomers:** "
        f"{isomer_count}"
    )

    lines.append(
        f"**Enantiomeric pairs:** "
        f"{enantiomers}"
    )

    lines.append("")

    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    lines.append(
        "## Stability Analysis"
    )

    lines.append(
        f"**Global stability index:** "
        f"{stability.total}/100"
    )

    lines.append("")

    lines.append(
        "### Stability Contributions"
    )

    lines.append(
        f"- Electron count: "
        f"{round(stability.electron, 1)}"
    )

    lines.append(
        f"- HSAB compatibility: "
        f"{round(stability.hsab, 1)}"
    )

    lines.append(
        f"- Chelation effect: "
        f"{round(stability.chelate, 1)}"
    )

    lines.append(
        f"- Ligand field: "
        f"{round(stability.field, 1)}"
    )

    lines.append(
        f"- Charge stabilization: "
        f"{round(stability.charge, 1)}"
    )

    lines.append(
        f"- Geometry stabilization: "
        f"{round(stability.geometry, 1)}"
    )

    lines.append(
        f"- Oxidation state stability: "
        f"{round(stability.oxidation, 1)}"
    )

    lines.append(
        f"- π-backbonding: "
        f"{round(stability.backbonding, 1)}"
    )

    lines.append(
        f"- Steric contribution: "
        f"{round(stability.steric, 1)}"
    )

    lines.append("")

    # --------------------------------------------------------
    # WARNINGS
    # --------------------------------------------------------

    warning = electrons_probability(
        formula
    )

    if warning:

        lines.append(
            "## Warnings"
        )

        lines.append(
            warning
        )

        lines.append("")

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    return render_analysis(lines)

# ============================================================
# QUICK ANALYSIS SHORTCUT
# ============================================================

def quick_analyze(
    formula,
    formula_counter_ions=None,
):
    """
    Lightweight wrapper around the complete
    analysis engine.

    Parameters
    ----------
    formula : str
        Coordination sphere formula.

    formula_counter_ions : str | None
        Optional counter-ion formula.

    Returns
    -------
    str | Markdown
        Rendered analysis report.
    """

    return analyze_complex(
        formula,
        formula_counter_ions,
    )


# ============================================================
# FULL 3D ANALYSIS PIPELINE
# ============================================================

def analyze_and_render(
    formula,
    formula_counter_ions=None,
    atoms_size=0.4,
    render_type="Ball and Stick",
):
    """
    Perform a complete coordination chemistry
    analysis and generate the corresponding
    3D visualization.

    Parameters
    ----------
    formula : str
        Coordination sphere formula.

    formula_counter_ions : str | None
        Optional counter-ion formula.

    atoms_size : float
        Atomic sphere scaling factor.

    render_type : str
        Rendering style.

    Returns
    -------
    tuple
        (
            rendered_analysis,
            rendered_3D_view
        )
    """

    # ========================================================
    # ANALYSIS GENERATION
    # ========================================================

    analysis = analyze_complex(
        formula,
        formula_counter_ions,
    )

    # ========================================================
    # 3D STRUCTURE GENERATION
    # ========================================================

    compound = create_compound_render(
        formula
    )

    rendered_view = render_complex(
        compound,
        atoms_size=atoms_size,
        render_type=render_type,
    )

    return (
        analysis,
        rendered_view,
    )


# ============================================================
# EXPORT UTILITIES
# ============================================================

def export_xyz(
    formula,
    output_path="complex.xyz",
):
    """
    Export the coordination complex
    to XYZ format.

    Parameters
    ----------
    formula : str
        Coordination sphere formula.

    output_path : str
        Output XYZ file path.

    Returns
    -------
    str
        Exported file path.
    """

    compound = create_compound_render(
        formula
    )

    write(
        output_path,
        compound,
        format="xyz",
    )

    return output_path


def export_cif(
    formula,
    output_path="complex.cif",
):
    """
    Export the coordination complex
    to CIF format.

    Parameters
    ----------
    formula : str
        Coordination sphere formula.

    output_path : str
        Output CIF file path.

    Returns
    -------
    str
        Exported file path.
    """

    compound = create_compound_render(
        formula
    )

    write(
        output_path,
        compound,
        format="cif",
    )

    return output_path


# ============================================================
# DATABASE INSPECTION UTILITIES
# ============================================================

def available_metals():
    """
    Return all supported metal centers.

    Returns
    -------
    list[str]
    """

    return sorted(
        data_metals.keys()
    )


def available_ligands():
    """
    Return all supported ligands.

    Returns
    -------
    list[str]
    """

    return sorted(
        data_ligands.keys()
    )


def available_counter_ions():
    """
    Return all supported counter ions.

    Returns
    -------
    list[str]
    """

    return sorted(
        data_counter_ions.keys()
    )


# ============================================================
# DATABASE SUMMARY
# ============================================================

def database_summary():
    """
    Generate a quick summary of the
    chemistry databases.

    Returns
    -------
    dict
    """

    return {
        "metals": len(data_metals),
        "ligands": len(data_ligands),
        "counter_ions": len(
            data_counter_ions
        ),
    }


# ============================================================
# VALIDATION SHORTCUTS
# ============================================================

def validate_formula(
    formula,
    formula_counter_ions=None,
):
    """
    Validate a complete coordination
    compound before analysis.

    Parameters
    ----------
    formula : str
        Coordination sphere formula.

    formula_counter_ions : str | None
        Optional counter-ion formula.

    Returns
    -------
    bool
    """

    formula_verification(formula)

    if formula_counter_ions is not None:

        counter_ions_verification(
            formula_counter_ions
        )

        complex_charge(
            formula,
            formula_counter_ions,
        )

    chemical_rules(formula)

    return True


# ============================================================
# TEXT REPORT GENERATION
# ============================================================

def generate_text_report(
    formula,
    formula_counter_ions=None,
):
    """
    Generate a plain-text scientific report.

    Parameters
    ----------
    formula : str
        Coordination sphere formula.

    formula_counter_ions : str | None
        Optional counter-ion formula.

    Returns
    -------
    str
    """

    stability = StabilityEngine(
        formula
    ).final_score()

    geometry = get_geometry(
        formula
    )[1]

    report = []

    report.append(
        "COORDINATION COMPLEX REPORT"
    )

    report.append("=" * 40)

    report.append(
        f"Formula: {formula}"
    )

    report.append(
        f"Geometry: {geometry}"
    )

    report.append(
        f"Electron count: "
        f"{electron_count(formula)}"
    )

    report.append(
        f"Spin state: "
        f"{determine_spin_state(formula)[0]}"
    )

    report.append(
        f"Magnetic behavior: "
        f"{magnetic_behavior(formula)}"
    )

    report.append(
        f"Stability index: "
        f"{stability.total}/100"
    )

    report.append(
        f"IUPAC-inspired name: "
        f"{naming_compound(formula, formula_counter_ions)}"
    )

    warning = electrons_probability(
        formula
    )

    if warning:

        report.append("")
        report.append("Warning:")
        report.append(warning)

    return "\n".join(report)


# ============================================================
# DEBUG UTILITIES
# ============================================================

def debug_formula(formula):
    """
    Return all intermediate parsing data.

    Useful for development and debugging.

    Parameters
    ----------
    formula : str

    Returns
    -------
    dict
    """

    return {
        "formula": formula,
        "metals": parse_metal(formula),
        "ligands": parse_ligands(formula),
        "bond_order": bond_order(formula),
        "geometry": get_geometry(formula)[1],
        "electron_count": electron_count(
            formula
        ),
        "oxidation_state": oxidation_state(
            formula
        ),
        "spin_state": determine_spin_state(
            formula
        ),
    }


# ============================================================
# PACKAGE VERSION
# ============================================================

__version__ = "2.0.0"


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [

    # --------------------------------------------------------
    # Parsing
    # --------------------------------------------------------

    "parse_metal",
    "parse_ligands",
    "parse_counter_ions",
    "bond_order",

    # --------------------------------------------------------
    # Charges
    # --------------------------------------------------------

    "complex_charge",
    "metal_charge",
    "ligands_charge",

    # --------------------------------------------------------
    # Electronic structure
    # --------------------------------------------------------

    "electron_count",
    "electronic_structure",
    "determine_spin_state",
    "magnetic_moment",
    "magnetic_behavior",

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    "StabilityEngine",
    "StabilityResult",

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    "get_geometry",
    "create_compound_render",
    "render_complex",

    # --------------------------------------------------------
    # Nomenclature
    # --------------------------------------------------------

    "naming_compound",

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    "analyze_complex",
    "analyze_and_render",
    "generate_text_report",

    # --------------------------------------------------------
    # Exports
    # --------------------------------------------------------

    "export_xyz",
    "export_cif",

    # --------------------------------------------------------
    # Databases
    # --------------------------------------------------------

    "available_metals",
    "available_ligands",
    "available_counter_ions",
]

# ============================================================
# COMMAND-LINE EXECUTION
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # EXAMPLE COMPLEXES
    # ========================================================

    examples = [

        # ----------------------------------------------------
        # Classical octahedral complex
        # ----------------------------------------------------

        (
            "[Fe(NH3)6]3+",
            "(Cl)3",
        ),

        # ----------------------------------------------------
        # Square planar platinum complex
        # ----------------------------------------------------

        (
            "[PtCl2(NH3)2]",
            None,
        ),

        # ----------------------------------------------------
        # Carbonyl complex
        # ----------------------------------------------------

        (
            "[Fe(CO)5]",
            None,
        ),

        # ----------------------------------------------------
        # Cyanide complex
        # ----------------------------------------------------

        (
            "[Fe(CN)6]4-",
            "(K)4",
        ),
    ]

    # ========================================================
    # EXECUTION LOOP
    # ========================================================

    for formula, counter_ions in examples:

        print("\n")
        print("=" * 70)

        print(
            f" ANALYZING: {formula}"
        )

        print("=" * 70)
        print("\n")

        try:

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            validate_formula(
                formula,
                counter_ions,
            )

            # ------------------------------------------------
            # TEXT REPORT
            # ------------------------------------------------

            report = generate_text_report(
                formula,
                counter_ions,
            )

            print(report)

            # ------------------------------------------------
            # DEBUG DATA
            # ------------------------------------------------

            print("\n")
            print("-" * 70)
            print("DEBUG INFORMATION")
            print("-" * 70)

            debug_data = debug_formula(
                formula
            )

            for key, value in debug_data.items():

                print(
                    f"{key}: {value}"
                )

            # ------------------------------------------------
            # XYZ EXPORT
            # ------------------------------------------------

            xyz_filename = (
                formula
                .replace("[", "")
                .replace("]", "")
                .replace("(", "_")
                .replace(")", "")
                .replace("+", "p")
                .replace("-", "m")
            )

            xyz_filename += ".xyz"

            export_xyz(
                formula,
                xyz_filename,
            )

            print("\n")
            print(
                f"XYZ structure exported: "
                f"{xyz_filename}"
            )

        # ====================================================
        # ERROR HANDLING
        # ====================================================

        except Exception as error:

            print("\n")
            print(
                "ERROR DURING ANALYSIS"
            )

            print("-" * 70)

            print(str(error))

        print("\n")


# ============================================================
# END OF FILE
# ============================================================

"""
===============================================================
COORDINATION CHEMISTRY ENGINE
===============================================================

Main Features
--------------
✓ Coordination sphere parsing
✓ Counter-ion parsing
✓ Oxidation state determination
✓ Electron counting
✓ Ligand field analysis
✓ Spin-state prediction
✓ Magnetic behavior estimation
✓ Jahn–Teller prediction
✓ Stability scoring engine
✓ Stereoisomer analysis
✓ IUPAC-inspired nomenclature
✓ 3D molecular structure generation
✓ py3Dmol visualization
✓ ASE export support
✓ Streamlit compatibility
✓ Notebook compatibility

Supported Geometries
--------------------
✓ Linear
✓ Trigonal planar
✓ Tetrahedral
✓ Square planar
✓ Trigonal bipyramidal
✓ Octahedral

Supported Export Formats
------------------------
✓ XYZ
✓ CIF

Rendering Backends
------------------
✓ py3Dmol
✓ Jupyter Notebook
✓ Streamlit

===============================================================
"""