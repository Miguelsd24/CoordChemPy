# ==========================================
# IMPORTS
# ==========================================
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
# LOADING DATA (JSON)
# ==========================================

# Importing the ligands and metals data from the json files
BASE_DIR = Path(__file__).resolve().parent.parent.parent

with open(BASE_DIR / "data" / "metals.json") as f:
    data_metals = json.load(f)

with open(BASE_DIR / "data" / "ligands.json") as f:
    data_ligands = json.load(f)

with open(BASE_DIR / "data" / "counter_ions.json") as f:
    data_counter_ions = json.load(f)

# ==========================================
# GENERAL FUNCTIONS HELPING THE VERIF AND PARSING
# ==========================================


# === Function which returns a ligand if its in the database by formula or abbr  === #
def find_ligand(ligand_input):
    # We perform a direct verification
    if ligand_input in data_ligands:
        return ligand_input
    # If not found, we perform a search by abbr
    for ligand_key, properties in data_ligands.items():
        if "abbr" in properties and properties["abbr"] == ligand_input:
            return ligand_key
    return False


# === Function which returns a counter ion if its in the database by formula  === #
def find_counter_ion(ion_input):
    # We perform a direct verification
    if ion_input in data_counter_ions:
        return ion_input
    # If not found, we perform a search by formula with charge
    for ion_key, properties in data_counter_ions.items():
        if "formula" in properties and properties["formula"] == ion_input:
            return ion_key
    # If not found, we perform a search by abbr
    for ion_key, properties in data_counter_ions.items():
        if "abbr" in properties and properties["abbr"] == ion_input:
            return ion_key
    return False


# === Function which transforms a charge string (i.e. "+, -, X+, +X, -X, X-") to a negative or positive int === #
def transform_charge(charge):
    # Case with nothing
    if charge is None or charge == "":
        return 0
    charge = charge.strip()

    # Case without sign
    try:
        return int(charge)
    except ValueError:
        pass

    # Case only the sign
    if charge == "+":
        return 1
    if charge == "-":
        return -1

    # Case with end sign (ex: 2+, 1-)
    match = re.match(r"(\d+)([+-])", charge)
    if match:
        value = int(match.group(1))
        sign = match.group(2)
        return value if sign == "+" else -value

    # Case with start sign (ex: +3, -2)
    match = re.match(r"([+-])(\d+)", charge)
    if match:
        sign = match.group(1)
        value = int(match.group(2))
        return value if sign == "+" else -value

    raise ValueError(f"Error: Invalid format charge: {charge}")


# ==========================================
# FORMAT VERIFICATION AND PARSING SECTION
# ==========================================


# === We use a function to verify the format of the coordination sphere formula and begin the first step of the parsing (seperate metal/ligands/sphere charge) === #
def formula_verif_and_parsing(formula):
    # We verify that the formula is a string
    if not isinstance(formula, str):
        raise ValueError("Error: Formula (coordination sphere) must be a string")
    # We discard spaces and verify that the formula is not empty
    clean_formula = formula.replace(" ", "")
    if clean_formula == "" or clean_formula == "[]":
        raise ValueError("Error: Formula (coordination sphere) cannot be empty")
    # We verify that the formula has the appropriate format with re.match()
    match = re.match(
        r"\[([sdt])?([A-Z][a-z]?)([1-9]\d*)?(\((?:.+)\)(?:[1-9]\d*)?)*\]([0-9+-]+)?$",
        clean_formula,
    )
    if not match:
        raise ValueError(
            "Error: Invalid formula format, see coordination sphere input format rules in the README.md file."
        )
    # We return the match object for later use in the parsing functions
    return match


# === We use a function to verify sthe format of the counter ion/s formula === #
def counter_ions_verif(formula_counter_ions):
    # We verify that the formula is a string
    if not isinstance(formula_counter_ions, str):
        raise ValueError("Error: Formula (counter ions) must be a string")
    # We discard spaces and verify that the formula is not empty
    clean_formula = formula_counter_ions.replace(" ", "")
    if clean_formula == "" or clean_formula == "()":
        raise ValueError("Error: Formula (counter ions) cannot be empty")
    # We verify that the formula has the appropriate format with re.match()
    verif = re.match(
        r"(\((.*?)\)([1-9]\d*))*",
        clean_formula,
    )
    if not verif:
        raise ValueError(
            "Error: Invalid formula format, see counter ions input format rules in the README.md file."
        )
    return clean_formula


# === We use a function to parse counter ion/s formula === #
def parse_counter_ions(formula_counter_ions):
    clean_formula = counter_ions_verif(formula_counter_ions)
    match = re.findall(r"\((.*?)\)(\d*)", clean_formula)
    # For each counter ion, we isolate it and its stoechiometric coefficient
    ions_coeff_list = []
    ions_list = []
    coeffs_list = []
    for counter_ion, coeff in match:
        coeff = int(coeff) if coeff != "" else 1
        # We put a limit of 12 identical counter ions
        if coeff > 12:
            raise ValueError(
                "Error: No counter ion can have a coefficient superior to 12"
            )
        # We verify if the counter ion is in the database and treat the output if not
        if find_counter_ion(counter_ion) is False:
            raise ValueError(
                f"Error: Counter ion of formula: {counter_ion} is not in the database."
            )
        else:
            # We return a list with each counter ion times its coefficient, a list with only each counter ion one time and a list with the coefficient
            # this will be usefull in others functions
            ions_coeff_list.extend([counter_ion] * coeff)
            ions_list.extend([counter_ion])
            coeffs_list.append(coeff)
    return ions_coeff_list, ions_list, coeffs_list


# === We use a function to extract the metal/s and its stoechiometric coefficient from a given formula. It also verifies if the metal is in the json database === #
def parse_metal(formula):
    # We import the match result from the formula format verification function to avoid doing it twice
    match = formula_verif_and_parsing(formula)
    metal = match.group(2)
    if match.group(3) is None:
        coeff = 1
    else:
        coeff = int(match.group(3))
    # We test if the metal is present in the database and treat the output if not
    if metal not in data_metals:
        raise ValueError(f"Error: Metal {metal} is not in the database.")
    # We test if the coefficient has the appropriate value (1 or 2) and treat the output if not
    if coeff != 1 and coeff != 2:
        raise ValueError("Error: Invalid metal coefficient. Only 1 or 2 are allowed.")
    # We return a list of the metal times its coefficient
    metals = []
    metals.extend([metal] * coeff)
    return metals


# ===  Function which extracts form the formula if the metals are bonded (single, double ,triple) or not === #
def bond_order(formula):
    # We import the match result from the formula format verification function to avoid doing it twice
    match = formula_verif_and_parsing(formula)
    # We stock a dico to link the letter to a number of metal-metal bond
    order_dico = {"s": 1, "d": 2, "t": 3, "q": 4}
    # We extract the letter accordint to the metal coefficient value (if no coeff, cooef = 1)
    order = order_dico.get(match.group(1), 0)
    if match.group(3) is None:
        coeff = 1
    else:
        coeff = int(match.group(3))
    # We return an error if there is a bond_order specified for a mononuclear complex and we return the bond order otherwise
    if coeff == 1 and order != 0:
        raise ValueError(
            "Error: s,d,t,q are only to specifiy the bond between two metals center not one"
        )
    return order


# ===  Function which extract a list of the ligand/s from a given raw formula. It also verifies if the ligand/s is/are in the json database === #
def parse_ligands(formula):
    # We import the match result from the formula format verification function to avoid doing it twice
    match = formula_verif_and_parsing(formula)
    ligands_str = match.group(4)
    # We use re.findall to extract the ligands and their stoechiometric coefficient from the match.group(4) result
    match = re.findall(r"\((.*?)\)(\d*)", ligands_str)
    # For each ligand in the ligands, we isolate the stoechiometric coefficient and test if the ligands are in the database
    ligand_list = []
    coeff_list = []
    each_ligand = []
    for ligand, coeff in match:
        coeff = int(coeff) if coeff != "" else 1
        if (
            coeff > 12
        ):  # We put a limit of 12 identical ligands (at most: 6 ligands per metal) which is the last probable coordination number
            raise ValueError("Error: No ligand can have a coefficient superior to 12")
        if ligand.startswith("m-"):  # We identify if a ligand is bridging
            ligand = ligand[2:]
            coeff *= -1  # We put the coefficient in negative to identify it as a bridging ligand later
        # We verify if the ligand is in the database and treat the output if not
        if find_ligand(ligand) is False:
            raise ValueError(f"Error: Ligand {ligand} not in the database")
        else:
            # We return a list of the ligand times its coefficient (the coefficient is negative if the ligand is bridging)
            coeff_list.extend([coeff])
            ligand_list.extend([find_ligand(ligand)])
            each_ligand.extend([find_ligand(ligand)] * coeff)
    return each_ligand, ligand_list, coeff_list


# ===  Function which counts the number of bridging ligands === #
def count_bridging_ligands(formula):
    # For each bridging ligand in the ligands list we add + 1 to num and we return num at the end
    num = 0
    for ligand in parse_ligands(formula)[0]:
        if ligand.startswith("m-"):
            num += 1
    return num


# ===  Function which put the metal and ligands in a same list with their respective stoechiometric coefficient === #
def parse_elements(formula):
    elements = []
    elements.extend(parse_ligands(formula)[0])
    elements.extend(parse_metal(formula))
    return elements


# ==========================================
# COMPLEXE ANALYSIS SECTION
# ==========================================


def counter_ions_charge(formula_counter_ions):
    counter_ions = parse_counter_ions(formula_counter_ions)[0]
    charge = 0
    for counter_ion in counter_ions:
        charge += data_counter_ions[counter_ion]["charge"]
    return charge


# === Function which return the charge of the coordination sphere as an int === #
def complexe_charge(formula, formula_counter_ions=None):
    match = formula_verif_and_parsing(formula)
    charge = transform_charge(match.group(5))
    if formula_counter_ions is None:
        return charge if charge else 0
    else:
        if charge != 0 and -charge != counter_ions_charge(formula_counter_ions):
            raise ValueError(
                "Error: Counter ions total charge doesn't match the charge in the coordination sphere formula"
            )
        if counter_ions_charge(formula_counter_ions) == 0:
            raise ValueError(
                "Error: The counter ions cannot have a neutral total charge"
            )
        else:
            return -counter_ions_charge(formula_counter_ions)


# === Function which calulate the sum of all ligands' charges === #
def ligands_charge(formula):
    ligands = parse_ligands(formula)[0]
    charge = 0
    # We seperate the case of terminal or chelating/bridging ligands
    for ligand in ligands:
        if not ligand.startswith("m-"):
            charge += data_ligands[ligand]["charge"]
        else:
            charge += data_ligands[ligand[2:]]["charge"]
    return charge


# === Function which calulate the charge of the metal center (i.e. MX+ or MX-) === #
def metal_charge(formula, formula_counter_ions=None):
    charge = (
        complexe_charge(formula, formula_counter_ions) - ligands_charge(formula)
    ) // len(parse_metal(formula))
    return charge


# === Function which calulate the oxidation state of the metal center (i.e. dX) === #
def oxidation_state(formula):
    metals = parse_metal(formula)
    # The oxidation state is calculated by : group of the metal - charge of the metal center
    ox_state = data_metals[metals[0]]["group"] - metal_charge(formula)

    possible_ox_state = data_metals[metals[0]]["possible_ox_state"]
    # We compare the calculated oxidation state with the database and return a remark if the metal is not likely to be in this oxidation state
    if metal_charge(formula) not in possible_ox_state:
        sign = "+" if metal_charge(formula) > 0 else ""
        remark = (
            "The following compound is not likely to exist because the metal center cannot be in the oxidation state "
            + sign
            + str(metal_charge(formula))
            + " according to the database."
        )
    else:
        remark = ""
    # We return an error if the oxidation state is too large or too small to rule very impossible compounds, thus the limit is +10 and -5
    # (We let +10,+9,+8... and some negative values because we'd like our program to compute even not possible compounds to let the user understand why they are not possible)
    if ox_state < 0 or ox_state > 12:
        raise ValueError(
            f"Error: The oxidation state {sign}{str(metal_charge(formula))} of the metal center is impossible, try changing the formula"
        )
    return ox_state, remark


# === Function which does the electron counting. We use the ionic counting method === #
def electron_count(formula):
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


# === Function which return if the complexs follows the 16 or 18 electron rule === #
def electrons_probable_complex(formula):
    if electron_count(formula) == 16 or electron_count(formula) == 18:
        return ""
    elif electron_count(formula) > 22:
        return "This specific coordination complex is highly unstable and structurally unfeasible."
    else:
        return "This specific coordination complex does not follow the 16 or 18 electron rule, thus, it is probably not very stable. It can however exist."


# === Function which calulate the electroni structure of the metal === #
def electronic_structure(formula):
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


# ==========================================
# NAMING COORDINATION COMPOUNDS SECTION
# ==========================================

# === We first set two sets of useful prefixes for the nomenclature of compounds,
# the first one is used in general but the second one is used for some specific ligands (see IUPAC rules) === #

coeff_name1 = {
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
    # No experimentally confirmed coordination compound with more than 16 ligands exist so we stop at 16.
    # Even 16 is exceptional and only for small ligands like H,O ...
}

coeff_name2 = {
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
    # No experimentally confirmed coordination compound with more than 16 ligands exist so we stop at 16.
    # Even 16 is exceptional and only for small ligands like H,O ...
}


# === Function which returns the name of the ligand while considering bridging ligands and nomenclature exceptions === #
def name_ligand(ligand):
    n = 0
    # If the ligand is bridging we set n = 2 to remove the "m-" at the beginning of the ligand name
    if ligand.startswith("m-"):
        n = 2
    if (
        data_ligands[ligand[n:]].get("nomenclature") is not None
    ):  # The ligand's name is almost always used in the nomenclature. We add if there is an exception a nomenclature key in the database
        return data_ligands[ligand[n:]]["nomenclature"]
    else:
        return data_ligands[ligand[n:]]["name"]


# === Function which decides whether to use the second set of prefixes from the database === #
def should_use_the_coeff_name2(ligand_name):
    for ligand in data_ligands.values():
        if (
            ligand["name"] == ligand_name and ligand.get("coeff") == "yes"
        ):  # The coeff key in the database is set to yes if the second set of prefixes should be used
            return True
    return False


# === Function used for naming the counter ions part of the complex name === #
def naming_counter_ions(formula_counter_ions):
    if formula_counter_ions is None:
        return ""
    counter_ions_list = parse_counter_ions(formula_counter_ions)
    ions = counter_ions_list[1]
    ions_with_coeffs = []
    # We first sort the ligands in alphabetic order and keep their respective coefficients by putting them in a list of tuple (ligand(sorted), coeff)
    for n in range(len(ions)):
        ion_name = data_counter_ions[ions[n]]["name"]
        ions_with_coeffs.append(ion_name)
    ions_with_coeffs.sort(key=lambda x: x.lower())

    counter_ions_naming = ""
    for ions in ions_with_coeffs:
        counter_ions_naming += ions + " "
    return counter_ions_naming


# === Function which returns the IUPAC name of the input coordination compound === #
def naming_compound(formula, formula_counter_ions=None):
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
        ligand_name = name_ligand(ligands[n])
        ligands_with_coeffs.append((ligand_name, coeffs[n]))
    ligands_with_coeffs.sort(key=lambda x: x[0].lower())

    # We transform the metal as its name. We use the secondary name of the metal if the corrdination sphere is charged negatively
    # Also, we start by adding the counter ions if the sphere charge is negative
    if complexe_charge(formula, formula_counter_ions) < 0:
        metal_name = data_metals[metals[0]]["secondary_name"]
        name = naming_counter_ions(formula_counter_ions)
    else:
        metal_name = data_metals[metals[0]]["name"]

    # We start the naming process by separating the cases :
    # 1. BRIDGING LIGANDS
    for ligand_name, coeff in ligands_with_coeffs:
        if coeff < 0:
            if (
                should_use_the_coeff_name2(ligand_name) is True
            ):  # We seperate the case where we have to use the second type of prefixes according to the ligand (IUPAC rules)
                # If we use the second set of prefixes the ligand name must be within parenthesis
                prefixe_ligand = coeff_name2[coeff * -1]
                name += f"{mu}{prefixe_ligand}({ligand_name})"  # We add the "mu" symbol in both case beacuse the ligand is bridging
            else:
                prefixe_ligand = coeff_name1[coeff * -1]
                name += mu + prefixe_ligand + ligand_name

    # 2. DINUCLEAR COMPLEXES WITH NON-BRIDGING LIGANDS
    n = 1  # We set n = 1 by default. This number will alows us to devide by two the coefficients in case of a symmetric binuclear complex
    if (
        len(metals) == 2
        and len(parse_ligands(formula)[0]) - count_bridging_ligands(formula)
        > 0  # We verify if there is at least one non-bridging ligand and in this case we use the second set of prefixes
    ):
        name += (
            coeff_name2[2] + "("
        )  # We add parenthesis around the metal name and the ligands because of the use of the second set of prefixes
        n = 2
        last_parenthesis = True
    elif (
        len(metals) == 2
        and len(parse_ligands(formula)[0]) - count_bridging_ligands(formula) == 0
    ):  # If there is only bridging ligands we use the first set of prefixes
        name += coeff_name1[2]

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
            if should_use_the_coeff_name2(ligand_name) is True:
                # We divide the coefficient by 1 or 2 to take into account the case of symmetric binuclear complexes
                prefixe_ligand = coeff_name2[coeff / n]
                name += f"{prefixe_ligand}({ligand_name})"
            else:
                prefixe_ligand = coeff_name1[coeff / n]
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
    if complexe_charge(formula, formula_counter_ions) > 0:
        name += " " + naming_counter_ions(formula_counter_ions)[:-1]
    name = (
        name[:1].capitalize() + name[1:]
    )  # We avoid the .capitalize to interact with the roman number
    return name


# ==========================================
# STABILITY ESTIMATION SECTION
# ==========================================


@dataclass
class StabilityResult:
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


# Stability class combining all function to calculate the stability score
class StabilityEngine:
    def __init__(self, formula):
        self.formula = formula.replace(" ", "")

        metals = parse_metal(self.formula)
        if not metals:
            raise ValueError("No metal found")

        self.metal = metals[0]
        self.ligands = parse_ligands(self.formula)[0]

        self.cn = len(self.ligands)
        self.electrons = electron_count(self.formula)

        self.charge = metal_charge(self.formula)
        self.ox, _ = oxidation_state(self.formula)

        self.m_data = data_metals.get(self.metal, {"hardness": 5, "group": 10})

        self.series_bonus = self._series_bonus()

    # ==========================================
    # 🔥 SERIES BONUS
    # ==========================================

    def _series_bonus(self):
        d5 = {"Pd", "Ag", "Cd", "Pt", "Au", "Hg"}
        d4 = {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh"}

        if self.metal in d5:
            return 10
        if self.metal in d4:
            return 5
        return 0

    # ==========================================
    # ⚡ ELECTRON COUNT (18e rule improved)
    # ==========================================

    def electron_score(self):
        # distinction CO / CN / phosphines
        strong_field = 18
        moderate_field = 16

        cn_boost_metal = {"Ni", "Pd", "Pt"}

        target = strong_field if self.metal in cn_boost_metal else moderate_field

        gap = abs(self.electrons - target)

        score = 100 - (gap * 10)

        # bonus Ni(CO)4 / Pt(CO) etc
        if "CO" in self.formula:
            score += 8

        return max(0, min(100, score + self.series_bonus))

    # ==========================================
    # 🧲 HSAB (more contrast)
    # ==========================================

    def hsab_score(self):
        m_h = self.m_data.get("hardness", 5)

        scores = []

        for lig in self.ligands:
            lig = lig.replace("m-", "")
            ligand = data_ligands.get(lig, {})

            l_h = ligand.get("HSAB", {}).get("hardness", 5)
            field = ligand.get("field", 1)

            # stronger penalty for mismatch
            match = math.exp(-0.45 * abs(m_h - l_h))

            scores.append(match * (1 + field / 4))

        return sum(scores) / len(scores) * 100 if scores else 50

    # ==========================================
    # 🧷 CHELATION (IMPORTANT FIX)
    # ==========================================

    def chelate_score(self):
        dent = 0

        for lig in self.ligands:
            lig = lig.replace("m-", "")
            dent += data_ligands.get(lig, {}).get("denticity", 1)

        # real chelate effect boost
        bonus = dent - self.cn

        # stronger exponential reward
        return max(0, min(100, 60 + bonus * 25))

    # ==========================================
    # ⚛️ FIELD (NORMALIZED BY CN)
    # ==========================================

    def field_score(self):
        score = 0

        for lig in self.ligands:
            lig = lig.replace("m-", "")
            score += data_ligands.get(lig, {}).get("field", 1)

        # CN normalization (CRUCIAL FIX)
        normalized = score / max(1, self.cn)

        return min(100, normalized * 25)

    # ==========================================
    # ⚖️ CHARGE
    # ==========================================

    def charge_score(self):
        return max(0, 100 - abs(self.charge) * 8)

    # ==========================================
    # 🧬 GEOMETRY (d8 FIXED)
    # ==========================================

    def geometry_score(self):
        cn = self.cn

        if cn == 6:
            return 95

        if cn == 4:
            # REAL FIX: d8 square planar dominance
            if self.metal in {"Ni", "Pd", "Pt"}:
                return 98  # IMPORTANT BOOST
            return 80

        if cn == 5:
            return 78

        if cn == 2:
            return 85

        return 60

    # ==========================================
    # 🔥 OXIDATION
    # ==========================================

    def oxidation_score(self):
        preferred = data_metals.get(self.metal, {}).get("possible_ox_state", [self.ox])

        diff = min(abs(self.ox - p) for p in preferred)

        return max(30, 100 - diff * 18)

    # ==========================================
    # 🔗 BACKBONDING (CO vs CN FIX)
    # ==========================================

    def backbonding_score(self):
        score = 0

        for lig in self.ligands:
            lig = lig.replace("m-", "")

            info = data_ligands.get(lig, {})

            if info.get("pi_acceptor"):
                score += 35  # stronger separation

            # CN stronger than CO in this model
            if lig == "CN":
                score += 10

            if lig == "CO":
                score += 15

        return min(100, score + self.series_bonus)

    # ==========================================
    # 🧱 STERIC
    # ==========================================

    def steric_score(self):
        bulk = 0

        for lig in self.ligands:
            lig = lig.replace("m-", "")
            bulk += data_ligands.get(lig, {}).get("steric_bulk", 1)

        return max(0, 100 - bulk * 3)

    # ==========================================
    # 🏆 FINAL SCORE (REBALANCED)
    # ==========================================

    def final_score(self):
        parts = {
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

        total = sum(parts[k] * weights[k] for k in parts)

        return StabilityResult(**parts, total=round(max(0, min(100, total)), 2))


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

    if key == "Ma2b2" and get_geometry(formula)[1] == "square planar":
        return 2, 0
    else:
        stereo = stereoisomers_dico.get(key)
        enantio = enantiomers_dico.get(key)
        return stereo, enantio


# ==========================================
# COMPOUND ANALYSIS AND DISPLAY SECTION
# ==========================================


# === We use a function to process the formula and return a clean LaTeX formula === #
def get_clean_formula(formula, formula_counter_ions=None):
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
    charge = complexe_charge(formula, formula_counter_ions)
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


# === We use this function to verify that the compound follows the chemical rules
# (This function regroups the cases which were not treated troughout the calculation, analysis, parsing ... functions) === #
def chemical_rules(formula):
    # We verify that there is no bridging ligand if there is only one metal center
    if len(parse_metal(formula)) == 1 and count_bridging_ligands(formula) > 0:
        raise ValueError("Error: A mononuclear complex cannot have bridging ligands")


# Final function which prints all the relevant information
def analyse_compound(formula, formula_counter_ions=None):
    # We first verify the chemical rules
    chemical_rules(formula)

    # We set a list which will contain each result line
    lines = []

    # Formula
    lines.append(f"**Formula** : {get_clean_formula(formula, formula_counter_ions)}")

    # Nomenclature
    name = naming_compound(formula, formula_counter_ions)
    lines.append(f"**IUPAC Name** : {name}")

    # Metal charge
    metals = parse_metal(formula)
    charge = metal_charge(formula)
    charge_str = f"{charge}+" if charge > 0 else f"{charge}"
    lines.append(f"**Metal oxidation state** : {metals[0]} ({charge_str})")

    # Electronic structure
    e_list = electronic_structure(formula)

    lines.append(
        f"**Electronic structure** : "
        f"[{e_list[0]}] {e_list[1]}s{e_list[2]} "
        f"{e_list[1] - 1}d{e_list[3]}"
    )

    # Electrons counting
    count = electron_count(formula)
    lines.append(f"**Electron count** : {count}")

    # Isomers
    iso = isomers(formula)

    if iso[0] is None or iso[1] is None:
        lines.append(
            "**Isomers:** The number of isomers of this compound is not specified"
        )

    else:
        lines.append(
            f"**Isomers:** This compound has "
            f"{iso[0]} stereoisomers and "
            f"{iso[1]} enantiomeres pairs"
        )

    # Stability
    engine = StabilityEngine(formula)
    result = engine.final_score()

    lines.append(f"**Stability index** : {result.total}/100")

    # Geometry
    if len(parse_ligands(formula)[0]) < 6:
        geometry = get_geometry(formula)[1].capitalize()
        lines.append(f"**Probable geometry** : {geometry}")

    # Remarks
    remark1 = oxidation_state(formula)[1]
    remark2 = electrons_probable_complex(formula)

    if remark1 != "" or remark2 != "":
        lines.append(f"**Remarks:** {remark1} {remark2}")
    return render_analysis(lines)


# Change the render depending on the interface (Notebook, Streamlit, Terminal)
def render_analysis(lines):
    is_streamlit = False
    is_notebook = False

    # Check Streamlit
    if "streamlit" in sys.modules:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            is_streamlit = True

    # Check Notebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            is_notebook = True
    except NameError:
        pass

    # Render depending on the interface
    if is_notebook:
        markdown_text = "\n".join(f"* {line}" for line in lines)
        return display(Markdown(markdown_text))

    elif is_streamlit:
        markdown_text = "\n\n".join(lines)
        return markdown_text

    else:
        # Terminal (Standard .py)
        text = "\n".join(lines).replace("**", "")
        return text


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
        return [(r, 0, 0)], "linear"
    elif cn == 2:
        return linear(r), "linear"
    elif cn == 3:
        return trigonal_planar(r), "trigonal planar"
    elif cn == 4 and oxidation_state(formula)[0] == 8:
        return square_planar(r), "square planar"
    elif cn == 4 and not oxidation_state(formula)[0] == 8:
        return tetrahedral(r), "tetrahedral"
    elif cn == 5:
        return trigonal_bipyramidal(r), "trigonal bipyramidal"
    elif cn == 6:
        return octahedral(r), "octahedral"
    else:
        raise ValueError("Error: The visualisation 3D does not work for CN over 6")


# ==========================================
# LIGANDS GEOMETRY CALCULATION SECTION
# ==========================================

# For each geometry of the ligands  we set the internal coordinates of each ligand


def ligand_linear(ligand, ligand_coord, r):
    ligand_position = np.array(ligand_coord)
    v = ligand_position / np.linalg.norm(ligand_position)

    inter_distance = data_ligands[ligand]["inter_distance"]
    position = v * (inter_distance + r)
    return [tuple(float(x) for x in position)]


def ligand_dlinear(ligand, ligand_coord, r):
    ligand_position = np.array(ligand_coord)
    v = ligand_position / np.linalg.norm(ligand_position)

    inter_distance = data_ligands[ligand]["inter_distance"]
    inter_distance2 = data_ligands[ligand]["inter_distance2"]
    position1 = v * (inter_distance + r)
    position2 = v * (inter_distance2 + inter_distance + r)
    return [tuple(float(x) for x in position1)] + [tuple(float(x) for x in position2)]


def ligand_bent(ligand, ligand_coord, r):
    ligand_position = np.array(ligand_coord)
    v = ligand_position / np.linalg.norm(ligand_position)

    inter_distance = data_ligands[ligand]["inter_distance"]
    inter_distance2 = data_ligands[ligand]["inter_distance2"]

    if abs(v[0]) > 0.1:
        temp_vec = np.array([0, 1, 0])
    else:
        temp_vec = np.array([1, 0, 0])

    perp = np.cross(v, temp_vec)
    perp /= np.linalg.norm(perp)

    theta = np.deg2rad(60)
    position1 = (np.cos(theta) * inter_distance + r) * v + (
        np.sin(theta) * inter_distance
    ) * perp
    position2 = (np.cos(theta) * inter_distance2 + r) * v + (
        -np.sin(theta) * inter_distance2
    ) * perp

    return [tuple(float(x) for x in position1)] + [tuple(float(x) for x in position2)]


def ligand_tetrahedral(ligand, ligand_coord, r):
    ligand_position = np.array(ligand_coord)
    v = ligand_position / np.linalg.norm(ligand_position)

    inter_distance = data_ligands[ligand]["inter_distance"]

    if abs(v[0]) > 0.1:
        temp_vec = np.array([0, 1, 0])
    else:
        temp_vec = np.array([1, 0, 0])

    u = np.cross(v, temp_vec)
    u /= np.linalg.norm(u)
    w = np.cross(v, u)

    positions = []
    theta = np.deg2rad(-54.75)
    for i in range(3):
        phi = np.deg2rad(i * 120)
        pos = (
            (np.cos(theta) * inter_distance + r) * v
            + np.sin(theta) * np.cos(phi) * inter_distance * u
            + np.cos(theta) * np.sin(phi) * inter_distance * w
        )
        positions.append(tuple(float(x) for x in pos))

    return positions


def get_geometry_ligand(ligand_input):
    geometry = data_ligands[ligand_input].get("geometry")
    if geometry is not None:
        return geometry
    return False


# ==========================================
# COMPOUND 3D RENDERING SECTION
# ==========================================


# Function which returns a list with the position of each atom of the complex.
# We use the coordinates of the ligand donor atoms and the internal coordination of each ligand
def atoms_position(
    formula, r=1.7
):  # The bond length is set to 1.7 A as py3Dmol can create automatically each bond using this value
    nb_of_atoms = 0
    position = [(0, 0, 0)]
    big_array = get_geometry(formula, r)[0]
    ligand_list = parse_ligands(formula)[0]
    for i, ligand in enumerate(ligand_list):
        if get_geometry_ligand(ligand) == "sphere":
            nb_of_atoms += 1
            position += [big_array[i]]
        elif get_geometry_ligand(ligand) == "linear":
            nb_of_atoms += 2
            position += [big_array[i]]
            position += ligand_linear(ligand, big_array[i], r)
        elif get_geometry_ligand(ligand) == "dlinear":
            nb_of_atoms += 3
            position += [big_array[i]]
            position += ligand_dlinear(ligand, big_array[i], r)
        elif get_geometry_ligand(ligand) == "bent":
            nb_of_atoms += 3
            position += [big_array[i]]
            position += ligand_bent(ligand, big_array[i], r)
        elif get_geometry_ligand(ligand) == "tetrahedral":
            nb_of_atoms += 4
            position += [big_array[i]]
            position += ligand_tetrahedral(ligand, big_array[i], r)
        else:
            raise ValueError("Error: Geometry of the ligand not available in 3D")
    return position


# Function which return all atoms symbols in a list
def get_atoms(ligand_input):
    ligand_info = data_ligands.get(ligand_input)
    donor_atom = ligand_info.get("donor_atoms")
    result = [donor_atom[0]] if donor_atom[0] else []
    atoms = re.findall(r"[A-Z][a-z]?\d*", ligand_input)

    for atom in atoms:
        match = re.match(r"([A-Z][a-z]?)(\d*)", atom)
        symbol = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1

        if symbol == donor_atom[0]:
            continue
        else:
            result.extend([symbol] * count)
    return result


# Function which returns a list with each atom which will be used to connect each atom to its corresponding sphere in the rendering
def atom_symbols(formula):
    metal = parse_metal(formula)
    atoms_list = metal

    ligand_list = parse_ligands(formula)[0]
    for ligand in ligand_list:
        atoms_list += get_atoms(ligand)
    return atoms_list


# Function which creates the compound to render as a ASE object
def create_compound_render(formula):
    compound = Atoms(atom_symbols(formula), positions=atoms_position(formula))
    return compound


# Function which convertes the ASE to a render. It optimises the render for a Notebook
def render_complex(compound, atoms_size=0.4, render_type="Ball and Stick"):
    # ASE atoms -> XYZ string
    xyz_str = io.StringIO()
    write(xyz_str, compound, format="xyz")
    xyz_content = xyz_str.getvalue()

    # Creation of the render zone
    view = py3Dmol.view(width=400, height=400)
    view.addModel(xyz_content, "xyz")

    # Style render type
    if render_type == "Ball and Stick":
        view.setStyle({"stick": {}, "sphere": {"scale": atoms_size}})
    elif render_type == "Stick":
        view.setStyle({"stick": {}})
    elif render_type == "Sphere":
        view.setStyle({"sphere": {"scale": atoms_size}})
    elif render_type == "Lines":
        view.setStyle({"line": {}})
    elif render_type == "VDW":
        view.addSurface(py3Dmol.VDW)
    view.zoomTo()

    # Render config depending on the interface
    is_streamlit = False
    if "streamlit" in sys.modules:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            is_streamlit = True

    if is_streamlit:
        html_content = view._make_html()
        return html_content
    else:
        return view.show()  # Notebook


def lowspin(nb_electrons):
    # calculations of the repartition of the electrons in the orbitals for low spin

    # Low spin is the repartition of the electrons in the orbitals with the lowest energy (t2g) forming pairs in these and then filling the higher energy orbitals (eg), which leads to more paired electrons and a lower total spin.
    # 5 orbitals (_ _ _ lowest   _ _ highest energy) adding order : the three lower, the two higher

    nb_pair = nb_electrons % 2
    nb_free = nb_electrons - nb_pair * 2

    return nb_pair, nb_free


def highspin(nb_electrons):
    # calculations of the repartition of the electrons in the orbitals for high spin

    # High spin is the repartition of the electrons in the orbitals with the highest energy (eg) before forming pairs in the lower energy orbitals (t2g), which leads to more unpaired electrons and a higher total spin.
    # 5 orbitals (_ _ _ lowest   _ _ highest energy) adding order : the three lower, the two higher

    nb_pair = 0
    nb_free = 0

    if nb_electrons <= 5:
        nb_free = nb_electrons
        nb_pair = 0
    else:
        electrons_remaining = nb_electrons - 5

        nb_free = 5 - electrons_remaining
        nb_pair = electrons_remaining

    return nb_pair, nb_free


def find_type_spin(nb_electrons):
    # on cherche si c'est high spin ou low spin ( DUR, ligand field theory or metal configuration and position in the periodic table) )
    return lowspin(nb_electrons)
    # todo
    # return highspin(nb_electrons)


def Fill_orbitals(p, s):
    result = [0, 0, 0, 0, 0]  # 3 first = low energy, 2 last = high energy

    for i in range(len(result)):
        if p > 0:
            result[i] += 2
            p -= 1
        else:
            if s > 0:
                result[i] += 1
                s -= 1

    return result


def Jahn_Teller_distorsion(orbitals):
    # uneven filing of the the t2g or eg orbitals leads to a Jahn-Teller distorsion
    for i in range(2):
        if orbitals[i] != orbitals[i + 1]:
            return "Weak Jahn-Teller distorsion"
    for i in range(3, 4):
        if orbitals[i] != orbitals[i + 1]:
            return "Strong Jahn-Teller distorsion"
    return "No Jahn-Teller distorsion"


nb_elec = 8
p, s = highspin(nb_elec)
print("p:", p, "s:", s)
print(Fill_orbitals(p, s))
print(Jahn_Teller_distorsion(Fill_orbitals(p, s)))
