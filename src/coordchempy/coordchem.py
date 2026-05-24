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

BASE_DIR = Path(__file__).resolve().parent


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
        raise ValueError("Coordination sphere formula must be a string.")

    # --------------------------------------
    # Remove optional spaces
    # --------------------------------------

    clean_formula = formula.replace(" ", "")

    # --------------------------------------
    # Empty formula validation
    # --------------------------------------

    if clean_formula in {"", "[]"}:
        raise ValueError("Coordination sphere formula cannot be empty.")

    # --------------------------------------
    # Coordination sphere pattern
    # --------------------------------------
    #
    # Groups:
    # 1 → Metal-metal bond prefix
    # 2 → Metal symbol
    # 3 → Metal coefficient
    # 4 → Ligand block (m- = bridging)
    # 5 → Global charge
    #
    # Supported bond prefixes:
    # s → single
    # d → double
    # t → triple
    # q → quadruple
    #
    # Example:
    # [dRe2(m-Cl)8]2-
    #
    # 1 = d
    # 2 = Re
    # 3 = 2
    # 4 = (m-Cl)8
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
        raise ValueError("Invalid coordination sphere format.")

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
        raise ValueError("Counter-ion formula must be a string.")

    # --------------------------------------
    # Remove spaces
    # --------------------------------------

    clean_formula = formula_counter_ions.replace(" ", "")

    # --------------------------------------
    # Empty validation
    # --------------------------------------

    if clean_formula in {"", "()"}:
        raise ValueError("Counter-ion formula cannot be empty.")

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
        raise ValueError("Invalid counter-ion formula format.")

    return clean_formula


def chemical_rules(formula):
    """
    We use this function to verify that the compound follows the chemical rules
    This function regroups the cases which were not treated troughout the calculation,
    analysis, parsing ... functions)

    Parameters
    ----------
    formula : str

    "Returns"
    -------
    raise ValuesError : str

    """
    # We verify that there is no bridging ligand if there is only one metal center
    if len(parse_metal(formula)) == 1 and count_bridging_ligands(formula) > 0:
        raise ValueError("Error: A mononuclear complex cannot have bridging ligands")
    # We verify that if there is two metals both are connected
    if (
        len(parse_metal(formula)) == 2
        and count_bridging_ligands(formula) == 0
        and bond_order(formula) == 0
    ):
        raise ValueError(
            "Error: A binuclear compound must have at least one metal-metal bond or bridging ligand"
        )


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

    clean_formula = counter_ions_verification(formula_counter_ions)

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

        coefficient = int(coefficient) if coefficient else 1

        # ----------------------------------
        # Safety limit
        # ----------------------------------

        if coefficient > 12:
            raise ValueError("Counter-ion coefficient cannot exceed 12.")

        # ----------------------------------
        # Database lookup
        # ----------------------------------

        ion_key = find_counter_ion(counter_ion)

        if ion_key is False:
            raise ValueError(f"Unknown counter ion: {counter_ion}")

        # ----------------------------------
        # Expanded storage
        # ----------------------------------

        expanded_ions.extend([ion_key] * coefficient)

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

    match = formula_verification(formula)

    # --------------------------------------
    # Extract metal symbol
    # --------------------------------------

    metal = match.group(2)

    # --------------------------------------
    # Extract coefficient
    # --------------------------------------

    coefficient = int(match.group(3)) if match.group(3) else 1

    # --------------------------------------
    # Database validation
    # --------------------------------------

    if metal not in data_metals:
        raise ValueError(f"Unknown metal: {metal}")

    # --------------------------------------
    # Nuclearity validation
    # --------------------------------------

    if coefficient not in {1, 2}:
        raise ValueError("Only mono- and dinuclear complexes are supported.")

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

    match = formula_verification(formula)

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

    metal_count = int(match.group(3)) if match.group(3) else 1

    # --------------------------------------
    # Consistency validation
    # --------------------------------------

    if metal_count == 1 and order != 0:
        raise ValueError("Metal-metal bonds require two metal centres.")

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

    match = formula_verification(formula)

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

        coefficient = int(coefficient) if coefficient else 1

        # ----------------------------------
        # Safety limit
        # ----------------------------------

        if coefficient > 12:
            raise ValueError("Ligand coefficient cannot exceed 12.")

        # ----------------------------------
        # Bridging ligand detection
        # ----------------------------------

        is_bridging = ligand.startswith("m-")

        if is_bridging:
            ligand = ligand[2:]

            coefficient *= -1

        # ----------------------------------
        # Database lookup
        # ----------------------------------

        ligand_key = find_ligand(ligand)

        if ligand_key is False:
            raise ValueError(f"Unknown ligand: {ligand}")

        # ----------------------------------
        # Storage
        # ----------------------------------

        coefficients.append(coefficient)

        unique_ligands.append(ligand_key)

        expanded_ligands.extend([ligand_key] * abs(coefficient))

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

    _, _, coefficients = parse_ligands(formula)

    return sum(1 for coefficient in coefficients if coefficient < 0)


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

    elements.extend(parse_ligands(formula)[0])

    # --------------------------------------
    # Metals
    # --------------------------------------

    elements.extend(parse_metal(formula))

    return elements


# ==========================================
# COMPLEX CHARGE ANALYSIS
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

    counter_ions = parse_counter_ions(formula_counter_ions)[0]

    total_charge = 0

    # --------------------------------------
    # Sum charges
    # --------------------------------------

    for counter_ion in counter_ions:
        total_charge += data_counter_ions[counter_ion]["charge"]

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

    match = formula_verification(formula)

    # --------------------------------------
    # Sphere charge
    # --------------------------------------

    sphere_charge = transform_charge(match.group(5))

    # --------------------------------------
    # No counter ions
    # --------------------------------------

    if formula_counter_ions is None:
        return sphere_charge

    # --------------------------------------
    # Counter-ion charge
    # --------------------------------------

    counter_charge = counter_ions_charge(formula_counter_ions)

    # --------------------------------------
    # Neutrality validation
    # --------------------------------------

    if counter_charge == 0:
        raise ValueError("Counter ions cannot be globally neutral.")

    # --------------------------------------
    # Charge consistency validation
    # --------------------------------------

    if sphere_charge != 0 and sphere_charge != -counter_charge:
        raise ValueError("Counter-ion charge does not match sphere charge.")

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

    ligands = parse_ligands(formula)[0]

    total_charge = 0

    # --------------------------------------
    # Sum ligand charges
    # --------------------------------------

    for ligand in ligands:
        ligand = ligand.replace(
            "m-",
            "",
        )

        total_charge += data_ligands[ligand]["charge"]

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

    ligand_charge = ligands_charge(formula)

    # --------------------------------------
    # Number of metal centres
    # --------------------------------------

    metal_count = len(parse_metal(formula))

    # --------------------------------------
    # Formal oxidation state
    # --------------------------------------

    return (total_charge - ligand_charge) // metal_count


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

    metal = parse_metal(formula)[0]

    # --------------------------------------
    # d-electron count
    # --------------------------------------

    oxidation = data_metals[metal]["group"] - metal_charge(formula)

    # --------------------------------------
    # Known accessible oxidation states
    # --------------------------------------

    possible_states = data_metals[metal]["possible_ox_state"]

    # --------------------------------------
    # Chemical plausibility check
    # --------------------------------------

    if metal_charge(formula) not in possible_states:
        sign = "+" if metal_charge(formula) > 0 else ""

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
        raise ValueError("Impossible oxidation state detected.")

    return oxidation, remark


# ============================================================
# ELECTRONIC PROPERTIES
# ============================================================
"""
Advanced electronic structure analysis:
- Electron count
- Electronic structure
- Spin state estimation
- Orbital occupation
- Magnetic properties
- Jahn–Teller distortion prediction
"""


def electron_count(formula):
    """
    Function which does the electron counting. We use the ionic counting method
    """
    # We first set the number of electrons as the contributtion of the metal/s center/s
    electrons = oxidation_state(formula)[0] * len(parse_metal(formula))
    # We then add the contribution of each ligand according to the database and of the ligand type
    for ligand in parse_ligands(formula)[0]:
        if ligand.startswith("m-"):
            electrons += data_ligands[ligand[2:]]["bridging_e"]
        else:
            electrons += data_ligands[ligand]["donor_e"]
    # We add the contribution of the bond between the metal centers if there is one
    electrons += 2 * bond_order(formula)
    # We verify that the electron count yields an even number when dealing with a dinuclear complex because electrons are supposed to be equaly shared
    # beetween both metal centers
    if len(parse_metal(formula)) == 2 and electrons % 2 != 0:
        raise ValueError(
            "Error: The number of electrons is not an integer, check the formula."
        )
    # We finally return the number of electrons, however, if there are two metal centers,
    # we divide the number of electrons by 2 because the electrons are shared between the two metal centers (we can do that because all complexes are symetrical)
    return int(electrons) // 2 if len(parse_metal(formula)) == 2 else int(electrons)


def electrons_probable_complex(formula):
    """
    Function which return if the complexs follows the 16 or 18 electron rule
    """
    if electron_count(formula) == 16 or electron_count(formula) == 18:
        return ""
    elif electron_count(formula) > 22:
        return "This specific coordination complex is highly unstable and structurally unfeasible."
    else:
        return "This specific coordination complex does not follow the 16 or 18 electron rule."


def electronic_structure(formula):
    """
    Function which calulate the electronic structure of the metal
    """
    # We set the data needed and we create a list to stock the result
    metals = parse_metal(formula)
    per = 0
    list = []
    # We deal with the period and which noble gas is the base of the electronic structure
    # (both metals are the same so their period is also the same)
    per += data_metals[metals[0]]["period"]
    inert_gas = {4: "Ar", 5: "Kr", 6: "Xe"}
    # We deal with the As^b Cd^e part, also, we separate the cases: negative charge/ charge between 0-2 / charge > 2, because it is the s 2 electrons that are first removed
    if metal_charge(formula) == 0 or metal_charge(formula) == 2:
        s = 2 - metal_charge(formula)
        d = oxidation_state(formula)[0] - s
    elif (
        metal_charge(formula) >= 3 or metal_charge(formula) == 1
    ):  # Because a lonely electron in the s orbital fall into the d orbital as the latter is lower in energy
        s = 0
        d = oxidation_state(formula)[0]
    else:
        s = 2
        d = oxidation_state(formula)[0] - 2
    # We return the electronic structure as a list which we can easily use later to display it in the final result
    list.extend([inert_gas.get(per), per, s, d])
    return list


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

    unpaired_electrons = d_electrons - paired_electrons * 2

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

        strong_field_score += data_ligands[ligand].get("field", 1)

    average_field = strong_field_score / len(ligands)

    d_electrons = oxidation_state(formula)[0]

    if average_field >= 3:
        return (
            "Low spin",
            low_spin_configuration(d_electrons),
        )

    return (
        "High spin",
        high_spin_configuration(d_electrons),
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

    spin_state = determine_spin_state(formula)

    unpaired = spin_state[1][1]

    return round(
        math.sqrt(unpaired * (unpaired + 2)),
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

    spin_state = determine_spin_state(formula)

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

    d_electrons = oxidation_state(formula)[0]

    t2g = min(d_electrons, 6)

    eg = max(0, d_electrons - 6)

    cfse = -0.4 * t2g + 0.6 * eg

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

    spin_state = determine_spin_state(formula)

    orbitals = fill_d_orbitals(
        spin_state[1][0],
        spin_state[1][1],
    )

    # --------------------------------------------------------
    # t2g asymmetry
    # --------------------------------------------------------

    for index in range(2):
        if orbitals[index] != orbitals[index + 1]:
            return "Weak Jahn-Teller distortion"

    # --------------------------------------------------------
    # eg asymmetry
    # --------------------------------------------------------

    for index in range(3, 4):
        if orbitals[index] != orbitals[index + 1]:
            return "Strong Jahn-Teller distortion"

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
        if ligand["name"] == ligand_name and ligand.get("coeff") == "yes":
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

    ions = parse_counter_ions(formula_counter_ions)[1]

    ion_names = []

    for ion in ions:
        ion_names.append(data_counter_ions[ion]["name"])

    ion_names.sort(key=str.lower)

    return " ".join(ion_names)


# ============================================================
# COMPOUND FULL NOMENCLATURE
# ============================================================


def naming_compound(formula, formula_counter_ions=None):
    """
    Function which returns the IUPAC name of the input coordination compound
    """
    # We set the data nedded and empty list or string to stock the result
    parsed_data = parse_ligands(formula)
    ligands = parsed_data[1]
    coeffs = parsed_data[2]
    metals = parse_metal(formula)
    ligands_with_coeffs = []
    name = ""
    mu = "-" + "\u03bc" + "-"
    last_parenthesis = False

    # We first sort the ligands in alphabetic order and keep their respective coefficients by putting them in a list of tuple (ligand(sorted), coeff)
    for n in range(len(ligands)):
        ligand_name = ligand_nomenclature_name(ligands[n])
        ligands_with_coeffs.append((ligand_name, coeffs[n]))
    ligands_with_coeffs.sort(key=lambda x: x[0].lower())

    # We transform the metal as its name. We use the secondary name of the metal if the corrdination sphere is charged negatively
    # Also, we start by adding the counter ions if the sphere charge is negative
    if complex_charge(formula, formula_counter_ions) < 0:
        metal_name = data_metals[metals[0]]["secondary_name"]
        name = naming_counter_ions(formula_counter_ions)
    else:
        metal_name = data_metals[metals[0]]["name"]

    # We start the naming process by separating the cases :
    # 1. BRIDGING LIGANDS
    for ligand_name, coeff in ligands_with_coeffs:
        if coeff < 0:
            if (
                use_special_prefix(ligand_name) is True
            ):  # We seperate the case where we have to use the second type of prefixes according to the ligand (IUPAC rules)
                # If we use the second set of prefixes the ligand name must be within parenthesis
                prefixe_ligand = SPECIAL_PREFIXES[coeff * -1]
                name += f"{mu}{prefixe_ligand}({ligand_name})"  # We add the "mu" symbol in both case beacuse the ligand is bridging
            else:
                prefixe_ligand = STANDARD_PREFIXES[coeff * -1]
                name += mu + prefixe_ligand + ligand_name

    # 2. DINUCLEAR COMPLEXES WITH NON-BRIDGING LIGANDS
    n = 1  # We set n = 1 by default. This number will alows us to devide by two the coefficients in case of a symmetric binuclear complex
    if (
        len(metals) == 2
        and len(parse_ligands(formula)[0]) - count_bridging_ligands(formula)
        > 0  # We verify if there is at least one non-bridging ligand and in this case we use the second set of prefixes
    ):
        name += (
            SPECIAL_PREFIXES[2] + "("
        )  # We add parenthesis around the metal name and the ligands because of the use of the second set of prefixes
        n = 2
        last_parenthesis = True
    elif (
        len(metals) == 2
        and len(parse_ligands(formula)[0]) - count_bridging_ligands(formula) == 0
    ):  # If there is only bridging ligands we use the first set of prefixes
        name += STANDARD_PREFIXES[2]

    # Before deviding the coefficients by 2 in case of a dinuclear complex, we verify that the compound is well symmetric (to avoid having a float as the coefficient)
    if len(metals) == 2:
        for _, coeff in ligands_with_coeffs:
            if coeff > 0 and coeff % 2 != 0:
                raise ValueError(
                    "Error: The compound is not symmetric, the coefficients of the non-bridging ligands must all be even integers"
                )

    # 3. TERMINAL LIGANDS
    for ligand_name, coeff in ligands_with_coeffs:
        if coeff > 0:
            if use_special_prefix(ligand_name) is True:
                # We divide the coefficient by 1 or 2 to take into account the case of symmetric binuclear complexes
                prefixe_ligand = SPECIAL_PREFIXES[coeff // n]
                name += f"{prefixe_ligand}({ligand_name})"
            else:
                prefixe_ligand = STANDARD_PREFIXES[coeff // n]
                name += prefixe_ligand + ligand_name

    # We add the metal name
    name += metal_name

    # We add the charge according to roman number notation.

    charge = metal_charge(formula)
    charge_roman = roman.toRoman(abs(charge))

    # We treat if the metal charge is zero or negative because roman. does not handle zero or negative input
    if charge == 0:
        charge_roman = 0
    elif charge < 0:
        charge_roman = "-" + roman.toRoman(abs(charge))
    name += f"({charge_roman})"

    # We make some last adjustements (e.g. adding parenthesis for dinuclear complexes, removing the "-" at the beginning of the name if the first ligand is bridging)
    if last_parenthesis is True:
        name += ")"
    if name.startswith("-"):
        name = name[1:]
    name = re.sub(r" -μ-", " μ-", name)
    # We lastly add the counter ions, at the end of the formula, if the sphere charge is positive
    if complex_charge(formula, formula_counter_ions) > 0:
        name += " " + naming_counter_ions(formula_counter_ions)[:-1]
    name = (
        name[:1].capitalize() + name[1:]
    )  # We avoid the .capitalize to interact with the roman number
    return name


# ==========================================
# ISOMERS CALCULATION SECTION
# ==========================================

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


def isomers(formula):
    key = ""
    number = []
    alphabet = string.ascii_lowercase
    data = parse_ligands(formula)
    if len(parse_metal(formula)) == 1:
        key += "M"
    else:
        key += "M2"
    for n in range(len(data[2])):
        number.append(int(data[2][n]))
    number.sort(reverse=True)
    for n in range(len(data[2])):
        letter = alphabet[n]
        key += letter + str(number[n])

    if key == "Ma2b2" and get_geometry(formula)[1] == "Square planar":
        return 2, 0
    else:
        stereo = stereoisomers_dico.get(key)
        enantio = enantiomers_dico.get(key)
        return stereo, enantio


# ==========================================
# STABILITY ENGINE
# ==========================================

# ==========================================
# GLOBAL STATE (ANALYTICS ONLY)
# ==========================================

GLOBAL_SCORES: list[float] = []


def reset_global_scores():
    """Reset global analytics storage."""
    GLOBAL_SCORES.clear()


# ==========================================
# UTILS (ROBUST CORE)
# ==========================================


def clamp(x, a=0.0, b=10.0):
    """Clamp value safely between bounds."""
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(a, min(b, x))


def safe_mean(xs):
    """Compute mean ignoring invalid values."""
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def gaussian(x):
    """Smooth decay function for similarity scoring."""
    return math.exp(-(float(x) ** 2))


# ==========================================
# VISUAL (MINIMAL SAFE BACKWARD COMPAT)
# ==========================================


def score_icon(score):
    """Return emoji based on 0–10 score."""
    s = clamp(score)
    if s >= 8:
        return "🟢"
    if s >= 6:
        return "🟡"
    if s >= 4:
        return "🟠"
    return "🔴"


def score_bar(score, size=10):
    """ASCII energy bar."""
    s = clamp(score)
    filled = int((s / 10) * size)
    return "█" * filled + "░" * (size - filled)


# ==========================================
# DATA STRUCTURES
# ==========================================


@dataclass
class Criterion:
    value: float


@dataclass
class StabilityResult:
    total: float
    color: str
    electron: Criterion
    hsab: Criterion
    cfse: Criterion
    field: Criterion
    charge: Criterion
    geometry: Criterion
    oxidation: Criterion
    backbonding: Criterion
    steric: Criterion


# ==========================================
# CORE ENGINE
# ==========================================


class StabilityEngine:
    """Coordination chemistry stability evaluator (production-safe)."""

    def __init__(self, formula: str):
        self.formula = formula.replace(" ", "")
        metals = parse_metal(self.formula)
        if not metals:
            raise ValueError(f"No metal found in {self.formula}")
        self.metal = metals[0]

        self.ligands = parse_ligands(self.formula)[0]
        self.ligands = self.ligands or []

        self.cn = len(self.ligands)
        self.electrons = electron_count(self.formula)
        self.charge = metal_charge(self.formula)
        ox = oxidation_state(self.formula)
        self.ox = ox[0] if isinstance(ox, tuple) else ox

        self.m_data = data_metals.get(
            self.metal, {"hardness": 5, "possible_ox_state": [self.ox]}
        )

    # ======================================
    # ELECTRON COUNTING
    # ======================================
    def electron_score(self):
        target = 18 if self.metal in {"Fe", "Co", "Ni", "Pd", "Pt", "Cr", "Mn"} else 16
        diff = abs(self.electrons - target)
        base = max(0, 10 - diff * 1.0)  # moins agressif

        bonus_map = {"CO": 1.5, "CN": 1.2, "NH3": 0.5, "NO": 1.0, "PR3": 0.8}
        bonus = sum(bonus_map.get(i.replace("m-", ""), 0.2) for i in self.ligands)

        return clamp(base + bonus)

    # ======================================
    # HSAB
    # ======================================
    def hsab_score(self):
        mh = self.m_data.get("hardness", 5)
        vals = []
        for i in self.ligands:
            lh = data_ligands.get(i, {}).get("HSAB", {}).get("hardness", 5)
            vals.append(10 * gaussian(abs(mh - lh) / 1.5))  # plus sensible
        return clamp(safe_mean(vals))

    # ======================================
    # CFSE
    # ======================================
    def cfse_score(self):
        field_map = {"CO": 2.5, "CN": 2.3, "NH3": 1.2, "H2O": 1.0, "Cl": 0.8, "F": 0.7}
        vals = [field_map.get(i, 1.0) for i in self.ligands]
        field = safe_mean(vals)
        # CFSE ajusté avec une fonction non-linéaire
        score = field * 3 + 0.5 * (field**2)
        return clamp(score)

    # ======================================
    # FIELD STRENGTH
    # ======================================
    def field_score(self):
        vals = [data_ligands.get(i, {}).get("field", 1) for i in self.ligands]
        return clamp(safe_mean(vals) * 2.5)  # légèrement augmenté

    # ======================================
    # CHARGE
    # ======================================
    def charge_score(self):
        # On punit moins fortement les charges élevées
        return clamp(10 - abs(self.charge) * 1.2)

    # ======================================
    # GEOMETRY
    # ======================================
    def geometry_score(self):
        if self.cn >= 6:
            return 9.5
        if self.cn == 4:
            return 9.0 if self.metal in {"Pt", "Pd", "Ni", "Cu"} else 7.5
        if self.cn == 2:
            return 8.0
        return 5.0

    # ======================================
    # OXIDATION
    # ======================================
    def oxidation_score(self):
        possible = self.m_data.get("possible_ox_state", [self.ox])
        diff = min(abs(self.ox - p) for p in possible)
        return clamp(10 - diff * 2.0)  # moins punitif

    # ======================================
    # BACKBONDING
    # ======================================
    def backbonding_score(self):
        score = 0.2
        for i in self.ligands:
            info = data_ligands.get(i, {})
            if info.get("pi_acceptor"):
                score += 3.5  # plus impactant
            if i in {"CO", "CN"}:
                score += 2.5  # plus impactant
        return clamp(score)

    # ======================================
    # STERIC
    # ======================================
    def steric_score(self):
        vals = [data_ligands.get(i, {}).get("steric_bulk", 1) for i in self.ligands]
        return clamp(10 - safe_mean(vals) * 1.5)  # un peu plus permissif

    # ======================================
    # CRITERIA VECTOR
    # ======================================
    def criteria(self) -> dict[str, float]:
        return {
            "electron": self.electron_score(),
            "hsab": self.hsab_score(),
            "cfse": self.cfse_score(),
            "field": self.field_score(),
            "charge": self.charge_score(),
            "geometry": self.geometry_score(),
            "oxidation": self.oxidation_score(),
            "backbonding": self.backbonding_score(),
            "steric": self.steric_score(),
        }

    # ======================================
    # TOTAL SCORE
    # ======================================
    def total_score(self):
        c = self.criteria()
        raw = (
            c["electron"] * 0.22
            + c["hsab"] * 0.15
            + c["cfse"] * 0.20
            + c["field"] * 0.12
            + c["charge"] * 0.08
            + c["geometry"] * 0.08
            + c["oxidation"] * 0.05
            + c["backbonding"] * 0.05
            + c["steric"] * 0.05
        )
        score = clamp(raw)
        GLOBAL_SCORES.append(score)
        return round(score * 10, 2)

    # ======================================
    # FINAL OUTPUT (TEST + API SAFE)
    # ======================================
    def final_score(self):
        c = self.criteria()
        total = self.total_score()
        return StabilityResult(
            total=total,
            color=score_icon(total / 10),
            electron=Criterion(c["electron"]),
            hsab=Criterion(c["hsab"]),
            cfse=Criterion(c["cfse"]),
            field=Criterion(c["field"]),
            charge=Criterion(c["charge"]),
            geometry=Criterion(c["geometry"]),
            oxidation=Criterion(c["oxidation"]),
            backbonding=Criterion(c["backbonding"]),
            steric=Criterion(c["steric"]),
        )


# ==========================================
# DUEL SYSTEM (UNCHANGED + STABLE)
# ==========================================
def stability_duel(A, B):
    a = StabilityEngine(A)
    b = StabilityEngine(B)
    a_score = a.total_score()
    b_score = b.total_score()
    return {
        f"Score {A} ": a_score,
        f"Score {B}": b_score,
        "Most Stable": f"{A}"
        if a_score > b_score
        else f"{B}"
        if b_score > a_score
        else "Tie",
    }


# ==========================================
# COMPOUND 3D VISUALISATION SECTION
# ==========================================

# ==========================================
# COMPOUND MAIN GEOMETRY CALCULATION SECTION
# ==========================================

# For each geometry we set the coordinates of each ligand donor atom
# (the metal center is at the origin by default)


def linear(r):
    return [(r, 0, 0), (-r, 0, 0)]


def tetrahedral(r):
    base = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]])

    base = base / np.linalg.norm(base[0])  # Normalization
    base = r * base

    return [tuple(v) for v in base]


def octahedral(r):
    return [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]


def trigonal_planar(r):
    return [
        (r, 0, 0),
        (-r / 2, r * np.sqrt(3) / 2, 0),
        (-r / 2, -r * np.sqrt(3) / 2, 0),
    ]


def trigonal_bipyramidal(r):
    return [
        (0, 0, r),
        (0, 0, -r),
        (r, 0, 0),
        (r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0),
        (r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0),
    ]


def square_planar(r):
    array = [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0)]
    return array


def get_geometry(formula, r=0):
    cn = len(parse_ligands(formula)[0])
    if cn == 1:
        return [(r, 0, 0)], "Linear"
    elif cn == 2:
        return linear(r), "Linear"
    elif cn == 3:
        return trigonal_planar(r), "Trigonal planar"
    elif cn == 4 and oxidation_state(formula)[0] == 8:
        return square_planar(r), "Square planar"
    elif cn == 4 and not oxidation_state(formula)[0] == 8:
        return tetrahedral(r), "Tetrahedral"
    elif cn == 5:
        return trigonal_bipyramidal(r), "Trigonal bipyramidal"
    elif cn == 6:
        return octahedral(r), "Octahedral"
    else:
        raise ValueError("Error: The visualisation 3D does not work for CN over 6")


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

    ligand_position = np.array(ligand_coord)

    direction = ligand_position / np.linalg.norm(ligand_position)

    inter_distance = data_ligands[ligand]["inter_distance"]

    position = direction * (inter_distance + r)

    return [tuple(float(value) for value in position)]


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

    ligand_position = np.array(ligand_coord)

    direction = ligand_position / np.linalg.norm(ligand_position)

    inter_distance_1 = data_ligands[ligand]["inter_distance"]

    inter_distance_2 = data_ligands[ligand]["inter_distance2"]

    position_1 = direction * (inter_distance_1 + r)

    position_2 = direction * (inter_distance_1 + inter_distance_2 + r)

    return [
        tuple(float(value) for value in position_1),
        tuple(float(value) for value in position_2),
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

    ligand_position = np.array(ligand_coord)

    direction = ligand_position / np.linalg.norm(ligand_position)

    inter_distance_1 = data_ligands[ligand]["inter_distance"]

    inter_distance_2 = data_ligands[ligand]["inter_distance2"]

    # --------------------------------------------------------
    # Temporary orthogonal vector
    # --------------------------------------------------------

    if abs(direction[0]) > 0.1:
        temp_vector = np.array([0, 1, 0])

    else:
        temp_vector = np.array([1, 0, 0])

    perpendicular = np.cross(
        direction,
        temp_vector,
    )

    perpendicular /= np.linalg.norm(perpendicular)

    theta = np.deg2rad(60)

    position_1 = (np.cos(theta) * inter_distance_1 + r) * direction + (
        np.sin(theta) * inter_distance_1
    ) * perpendicular

    position_2 = (np.cos(theta) * inter_distance_2 + r) * direction + (
        -np.sin(theta) * inter_distance_2
    ) * perpendicular

    return [
        tuple(float(value) for value in position_1),
        tuple(float(value) for value in position_2),
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

    ligand_position = np.array(ligand_coord)

    direction = ligand_position / np.linalg.norm(ligand_position)

    inter_distance = data_ligands[ligand]["inter_distance"]

    # --------------------------------------------------------
    # Local orthogonal basis
    # --------------------------------------------------------

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
            (np.cos(theta) * inter_distance + r) * direction
            + (np.sin(theta) * np.cos(phi) * inter_distance) * u
            + (np.cos(theta) * np.sin(phi) * inter_distance) * w
        )

        positions.append(tuple(float(value) for value in position))

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

    geometry = data_ligands[ligand_input].get("geometry")

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

    ligand_positions = get_geometry(formula, r)[0]

    ligands = parse_ligands(formula)[0]

    # --------------------------------------------------------
    # Ligand placement
    # --------------------------------------------------------

    for index, ligand in enumerate(ligands):
        geometry = get_geometry_ligand(ligand)

        # ====================================================
        # SPHERICAL LIGAND
        # ====================================================

        if geometry == "sphere":
            positions += [ligand_positions[index]]

        # ====================================================
        # LINEAR LIGAND
        # ====================================================

        elif geometry == "linear":
            positions += [ligand_positions[index]]

            positions += ligand_linear(
                ligand,
                ligand_positions[index],
                r,
            )

        # ====================================================
        # DOUBLE-LINEAR LIGAND
        # ====================================================

        elif geometry == "dlinear":
            positions += [ligand_positions[index]]

            positions += ligand_dlinear(
                ligand,
                ligand_positions[index],
                r,
            )

        # ====================================================
        # BENT LIGAND
        # ====================================================

        elif geometry == "bent":
            positions += [ligand_positions[index]]

            positions += ligand_bent(
                ligand,
                ligand_positions[index],
                r,
            )

        # ====================================================
        # TETRAHEDRAL LIGAND
        # ====================================================

        elif geometry == "tetrahedral":
            positions += [ligand_positions[index]]

            positions += ligand_tetrahedral(
                ligand,
                ligand_positions[index],
                r,
            )

        # ====================================================
        # UNSUPPORTED GEOMETRY
        # ====================================================

        else:
            raise ValueError("Error: Ligand geometry not available in 3D.")

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

    ligand_info = data_ligands.get(ligand_input)

    donor_atoms = ligand_info.get("donor_atoms")

    result = [donor_atoms[0]] if donor_atoms[0] else []

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

        count = int(match.group(2)) if match.group(2) else 1

        # ----------------------------------------------------
        # Avoid duplicating donor atom
        # ----------------------------------------------------

        if symbol == donor_atoms[0]:
            if count > 1:
                result.extend([symbol] * (count - 1))
            else:
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
    if len(parse_metal(formula)) == 2:
        raise ValueError("Error: Dinuclear complexes are not available in 3D")

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
                "sphere": {"scale": atoms_size},
            }
        )

    elif render_type == "Stick":
        view.setStyle({"stick": {}})

    elif render_type == "Sphere":
        view.setStyle({"sphere": {"scale": atoms_size}})

    elif render_type == "Lines":
        view.setStyle({"line": {}})

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


def get_clean_formula(formula, formula_counter_ions=None):
    """
    We use a function to process the formula and return a clean LaTeX formula
    """
    clean_formula = formula.replace(" ", "")
    # We replace the m- by μ- for a better display in LaTeX
    clean_bridging = re.sub(r"m-", r"μ-", clean_formula)
    # We isolate the part of the formula with the metal and its coefficient to put the subscript in LaTeX
    end = clean_bridging.find("]")
    clean_sphere = re.sub(r"(\d+)", r"_{\1}", clean_bridging[: end + 1])
    clean_sphere = re.sub(r"\(([A-Z][a-z]?)\)", r"\1", clean_sphere)
    # We isolate the counter ions part and subscript in LaTeX
    if formula_counter_ions is None:
        clean_counter_ions = ""
    else:
        clean_counter_ions = re.sub(r"(\d+)", r"_{\1}", formula_counter_ions)
        clean_counter_ions = re.sub(r"\(([A-Z][a-z]?)\)", r"\1", clean_counter_ions)
    # We isolate the charge part to put it in superscript in LaTeX and we add the sign if the charge is positive
    charge = complex_charge(formula, formula_counter_ions)
    if charge > 0:
        return (
            "$"
            + clean_sphere
            + "^{"
            + "+"
            + str(charge)
            + "}"
            + clean_counter_ions
            + "$"
        )
    elif charge == 0:
        return "$" + clean_sphere + "$"
    else:
        return "$" + clean_counter_ions + clean_sphere + "^{" + str(charge) + "}" + "$"


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

    return "".join(superscripts.get(char, char) for char in value)


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

    return "".join(subscripts.get(char, char) for char in value)


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

    formatted = f"{value:.{precision}f}"

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

    return clean_float(value) + "%"


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

    return f"{title}\n{separator}"


# ============================================================
# COMPLETE ANALYSIS REPORT
# ============================================================


def analyse_compound(
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

    try:
        geometry = get_geometry(formula)[1]

    except ValueError:
        geometry = "Not specified"

    oxidation = metal_charge(
        formula,
        formula_counter_ions,
    )

    d_electrons = oxidation_state(formula)[0]

    electron_total = electron_count(formula)

    spin_state = determine_spin_state(formula)[0]

    magnetic_type = magnetic_behavior(formula)

    magnetic_value = magnetic_moment(formula)

    cfse = crystal_field_stabilization_energy(formula)

    jahn_teller = jahn_teller_distortion(formula)

    isomer_count, enantiomers = isomers(formula)

    stability = StabilityEngine(formula).final_score()

    electronic_struct = electronic_structure(formula)
    # Remarks
    remark1 = oxidation_state(formula)[1]
    remark2 = electrons_probable_complex(formula)

    # ========================================================
    # FORMATTED OUTPUT
    # ========================================================

    lines = []

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    lines.append("# Coordination Complex Analysis")

    lines.append(f"### {get_clean_formula(formula, formula_counter_ions)}")

    # --------------------------------------------------------
    # NOMENCLATURE
    # --------------------------------------------------------

    lines.append("## Nomenclature")

    lines.append(f"**IUPAC Name:** {naming_compound(formula, formula_counter_ions)}")

    # --------------------------------------------------------
    # METAL CENTER
    # --------------------------------------------------------

    lines.append("## Metal Center")

    lines.append(f"**Metal:** {metal}")

    lines.append(f"**Formal oxidation state:** {oxidation:+d}")

    lines.append(
        f"**Electronic structure** : "
        f"[{electronic_struct[0]}] {electronic_struct[1]}s{electronic_struct[2]} "
        f"{electronic_struct[1] - 1}d{electronic_struct[3]}"
    )

    lines.append(f"**Metal d-configuration:** d{d_electrons}")

    if remark1 != "":
        lines.append(f"**Remarks:** {remark1}")

    # --------------------------------------------------------
    # ELECTRONIC PROPERTIES
    # --------------------------------------------------------

    lines.append("## Electronic properties")

    lines.append(f"**Electron count:** {electron_total} e⁻")

    lines.append(f"**Spin state:** {spin_state}")

    lines.append(f"**Magnetic behavior:** {magnetic_type}")

    lines.append(f"**Magnetic moment:** {magnetic_value} BM")

    lines.append(f"**CFSE:** {cfse}")

    lines.append(f"**Jahn–Teller effect:** {jahn_teller}")

    if remark2 != "":
        lines.append(f"**Remarks:** {remark2}")

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    lines.append("## Geometry")

    lines.append(f"**Coordination geometry:** {geometry}")

    lines.append(f"**Coordination number:** {len(parse_ligands(formula)[0])}")

    # --------------------------------------------------------
    # ISOMERISM
    # --------------------------------------------------------

    lines.append("## Stereochemistry")

    lines.append(f"**Possible stereoisomers:** {isomer_count}")

    lines.append(f"**Enantiomeric pairs:** {enantiomers}")

    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    lines.append("## Stability Analysis")

    lines.append(f"**Global stability index:** {stability.total}/100")

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    return render_analysis(lines)


# ============================================================
# RENDER TYPE OUTPUT
# ============================================================


def render_analysis(lines):
    """
    Change the render depending on the interface (Notebook, Streamlit, Terminal)
    """
    is_streamlit = False
    is_notebook = False

    # Check Streamlit
    if "streamlit" in sys.modules:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            is_streamlit = True

    # Check Notebook
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            is_notebook = True
    except NameError:
        pass

    # Render depending on the interface
    if is_notebook:
        from IPython.display import Markdown, display

        markdown_text = "\n\n".join(lines)
        return display(Markdown(markdown_text))

    elif is_streamlit:
        markdown_text = "\n\n".join(lines)
        return markdown_text

    else:
        # Terminal (Standard .py)
        text = "\n".join(lines).replace("**", "")
        return text


# ============================================================
# FULL 3D ANALYSIS PIPELINE
# ============================================================


def analyse_and_render(
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

    analysis = analyse_compound(
        formula,
        formula_counter_ions,
    )

    # ========================================================
    # 3D STRUCTURE GENERATION
    # ========================================================

    compound = create_compound_render(formula)

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

    compound = create_compound_render(formula)

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

    compound = create_compound_render(formula)

    write(
        output_path,
        compound,
        format="cif",
    )

    return output_path


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
        counter_ions_verification(formula_counter_ions)

        complex_charge(
            formula,
            formula_counter_ions,
        )

    chemical_rules(formula)

    return True


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
✓ Jahn-Teller prediction
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
