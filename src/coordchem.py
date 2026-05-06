# ==========================================
# IMPORTS
# ==========================================

import json
import re
import string
from pathlib import Path

import numpy as np
import roman
from ase import Atoms
from IPython.display import Markdown, display

# ==========================================
# JUPYTER ERROR HANDLER
# ==========================================


# ???


# ==========================================
# LOADING DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "data" / "metals.json") as f:
    data_metals = json.load(f)

with open(BASE_DIR / "data" / "ligands.json") as f:
    data_ligands = json.load(f)


# ==========================================
# GENERAL FUNCTIONS HELPING THE VERIF AND PARSING
# ==========================================


# === Function which returns a ligand if its in the database by name or abbr  === #
def find_ligand(ligand_input):
    # We perform a direct verification
    if ligand_input in data_ligands:
        return ligand_input
    # If not found, we perform a search by abbr
    for ligand_key, properties in data_ligands.items():
        if "abbr" in properties and properties["abbr"] == ligand_input:
            return ligand_key
    return False


# === Function which transforms a charge string (i.e. "+, -, X+, +X, -X, X-") to a negative or positive int === #
def transform_charge(charge):
    # Case with nothing
    if charge is None:
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

    raise ValueError(
        f"Error: Invalid format charge for the coordination sphere: {charge}"
    )


# ==========================================
# FORMAT VERIFICATION AND PARSING SECTION
# ==========================================

formula = "[Fe2(CN)4(m-H2O)2(en)2]2-"


# === We use a function to verify the format of the formula === #
def formula_format_verification(formula):
    # We verify that the formula is a string
    if not isinstance(formula, str):
        raise ValueError("Error: Formula must be a string")
    # We discard spaces and verify that the formula is not empty
    clean_formula = formula.replace(" ", "")
    if clean_formula == "" or clean_formula == "[]":
        raise ValueError("Error: Formula cannot be empty")
    # We verify that the formula has the appropriate format with re.match()
    match = re.match(
        r"\[([sdt]?)([A-Z][a-z]?)(0|[1-9]\d*)?(\((.+)\)(0|[1-9]\d*))*\]([0-9+-]+)?$",
        clean_formula,
    )
    if not match:
        raise ValueError("Error: Invalid formula format")
    # We return the match object for later use in the parsing functions
    return match


# === We use a function to process the formula and return a clean LaTeX formula === #
def get_clean_formula(formula):
    # We replace the m- by μ- for a better display in LaTeX
    clean_bridging = re.sub(r"m-", r"μ-", formula)
    # We isolate the part of the formula with the metal and its coefficient to put the subscript in LaTeX
    end = clean_bridging.find("]")
    clean_sphere = re.sub(r"(\d+)", r"_{\1}", clean_bridging[: end + 1])
    # We isolate the charge part to put it in superscript in LaTeX and we add the sign if the charge is positive
    signe = ""
    if complexe_charge(formula) > 0:
        signe = "+"
    elif complexe_charge(formula) == 0:
        return (
            "$" + clean_sphere + "$"
        )  # If the charge is 0 we don't put anything in superscript
    # If there is a charge, we put the charge in superscript with the appropriate sign
    clean_subscript = "^{" + signe + str(complexe_charge(formula)) + "}"
    return "$" + clean_sphere + clean_subscript + "$"


# === We use a function to extract the metal/s and its stoechiometric coefficient from a given formula. It also verifies if the metal is in the json database === #
def parse_metal(formula):
    # We import the match result from the formula format verification function to avoid doing it twice
    match = formula_format_verification(formula)
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
    match = formula_format_verification(formula)
    # We stock a dico to link the letter to a number of metal-metal bond
    order_dico = {"s": 1, "d": 2, "t": 3}
    # We extract the letter accordint to the metal coefficient value (if no coeff, cooef = 1)
    order = order_dico.get(match.group(1), 0)
    if match.group(3) is None:
        coeff = 1
    else:
        coeff = int(match.group(3))
    # We return an error if there is a bond_order specified for a mononuclear complex and we return the bond order otherwise
    if coeff == 1 and order != 0:
        raise ValueError(
            "Error: s,d,t are only to specifiy the bond between two metals center not one"
        )
    return order


# ===  Function which extract a list of the ligand/s from a given raw formula. It also verifies if the ligand/s is/are in the json database === #
def parse_ligands(formula):
    # We import the match result from the formula format verification function to avoid doing it twice
    match = formula_format_verification(formula)
    ligands_str = match.group(4)
    # We use re.findall to extract the ligands and their stoechiometric coefficient from the match.group(4) result
    match = re.findall(r"\((.*?)\)(\d*)", ligands_str)
    # For each ligand in the ligands, we isolate the stoechiometric coefficient and test if the ligands are in the database
    ligand_list = []
    coeff_list = []
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
    return ligand_list, coeff_list


# === Function which creates a list with all the ligands times their coeficient.
# This list is useful to use loops in the calculation function like that There is non need to deal with number later === #
def ligands_list(formula):
    ligands_list = []
    ligands = parse_ligands(formula)[0]
    coeff = parse_ligands(formula)[1]
    for n in range(len(ligands)):
        if coeff[n] < 0:
            ligands_list.extend(["m-" + ligands[n]] * (coeff[n] * -1))
        else:
            ligands_list.extend([ligands[n]] * coeff[n])
    return ligands_list


# ===  Function which counts the number of bridging ligands === #
def count_bridging_ligands(formula):
    # For each bridging ligand in the ligands list we add + 1 to num and we return num at the end
    num = 0
    for ligand in ligands_list(formula):
        if ligand.startswith("m-"):
            num += 1
    return num


# ===  Function which put the metal and ligands in a same list with their respective stoechiometric coefficient === #
def parse_elements(formula):
    elements = []
    elements.extend(ligands_list(formula))
    elements.extend(parse_metal(formula))
    # We also verify that there is no bridging ligand if there is only one metal center
    if len(parse_metal(formula)) == 1 and count_bridging_ligands(formula) > 0:
        raise ValueError("Error: A mononuclear complex cannot have bridging ligands")
    return elements


# ==========================================
# COMPLEXE ANALYSIS SECTION
# ==========================================


# === Function which return the charge of the coordination sphere as an int === #
def complexe_charge(formula):
    match = formula_format_verification(formula)
    return transform_charge(match.group(7)) if match.group(7) else 0


# === Function which calulate the sum of all ligands' charges === #
def ligands_charge(formula):
    ligands = ligands_list(formula)
    charge = 0
    # We seperate the case of terminal or chelating/bridging ligands
    for ligand in ligands:
        if not ligand.startswith("m-"):
            charge += data_ligands[ligand]["charge"]
        else:
            charge += data_ligands[ligand[2:]]["charge"]
    return charge


# === Function which calulate the charge of the metal center (i.e. MX+ or MX-) === #
def metal_charge(formula):
    charge = (complexe_charge(formula) - ligands_charge(formula)) // len(
        parse_metal(formula)
    )
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
            + " according to the database"
        )
    else:
        remark = None
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
    for ligand in ligands_list(formula):
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

# We first set two sets of useful prefixes for the nomenclature of compounds.
# first one is used in general but the second one is used for some specific ligands (see IUPAC rules)

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


# === Function which returns the IUPAC name of the input coordination compound === #
def naming_compound(formula):
    # We set the data nedded and empty list or string to stock the result
    parsed_data = parse_ligands(formula)
    ligands = parsed_data[0]
    coeffs = parsed_data[1]
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
    if complexe_charge(formula) < 0:
        metal_name = data_metals[metals[0]]["secondary_name"]
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
        and len(ligands_list(formula)) - count_bridging_ligands(formula)
        > 0  # We verify if there is at least one non-bridging ligand and in this case we use the second set of prefixes
    ):
        name += (
            coeff_name2[2] + "("
        )  # We add parenthesis around the metal name and the ligands because of the use of the second set of prefixes
        n = 2
        last_parenthesis = True
    elif (
        len(metals) == 2
        and len(ligands_list(formula)) - count_bridging_ligands(formula) == 0
    ):  # If there is only bridging ligands we use the first set of prefixes
        name += coeff_name1[2]

    # Before deviding the coefficients by 2 in case of a dinuclear complex, we verify that the compound is well symmetric (to avoid having a float as the coefficient)
    if len(metals) == 2:
        for ligand_name, coeff in ligands_with_coeffs:
            if coeff > 0 and coeff % 2 != 0:
                raise ValueError(
                    "Error: The compound is not symmetric, the coefficients of the non-bridging ligands must all be even integers"
                )

    # 3. TERMINAL LIGANDS
    for i, (ligand_name, coeff) in enumerate(ligands_with_coeffs):
        if coeff > 0:
            if should_use_the_coeff_name2(ligand_name) is True:
                # We divide the coefficient by 1 or 2 to take into account the case of symmetric binuclear complexes
                prefixe_ligand = coeff_name2[coeff / n]
                name += f"{prefixe_ligand}({ligand_name})"
            else:
                prefixe_ligand = coeff_name1[coeff / n]
                name += prefixe_ligand + ligand_name

    # We add the metal name and already put the capital at the begining (avoid unwanted interactions between .capitalize and roman numbers later)
    name += metal_name
    name = name.capitalize()

    # We add the charge according to preference selected in the site (roman/integer) !!not implemented yet, we use roman by default for the moment!!

    # charge_int = complexe_charge(formula)
    # name += (f"({charge_int})")

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

    return name


# ==========================================
# STABILITY ESTIMATION SECTION
# ==========================================

# ------------------------------------------------------------------------------------------------------------------------------- VERIFIE JUSQU ICI -------------------------------------------------------------------------------------------------------------------------------


def ligand_field_strength(formula):
    """
    Estimation simple du champ de ligand (qualitative -> numérique)
    """
    ligands = ligands_list(formula)

    field_score = 0

    for lig in ligands:
        if lig.startswith("m-"):
            lig = lig[2:]

        info = data_ligands.get(lig, {})

        # classification simple (tu peux enrichir plus tard)
        if info.get("field") == "strong":
            field_score += 2
        elif info.get("field") == "medium":
            field_score += 1
        else:
            field_score += 0

    return field_score


def crystal_field_stabilization(formula):
    """
    Approximation CFSE (très simplifiée)
    """
    electrons = electron_count(formula)

    # approximation: d electron count influence
    return (electrons - 6) * ligand_field_strength(formula)


def stability_index(formula):
    """
    Score global de stabilité (0-100)
    """
    cfse = crystal_field_stabilization(formula)
    charge = abs(complexe_charge(formula))
    lig_score = ligand_field_strength(formula)

    score = 50

    # CFSE stabilise
    score += cfse * 5

    # charge élevée = moins stable
    score -= charge * 5

    # ligands forts stabilisent
    score += lig_score * 3

    # clamp 0-100
    return max(0, min(100, score))


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
    for n in range(len(data[1])):
        number.append(int(data[1][n]))
    number.sort(reverse=True)
    for n in range(len(data[1])):
        letter = alphabet[n]
        key += letter + str(number[n])

    if key == "Ma2b2" and find_geometry(formula, 1) == [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
    ]:
        return 2, 0
    else:
        stereo = stereoisomers_dico.get(key)
        enantio = enantiomers_dico.get(key)
        return stereo, enantio


# =============================================================================================================================================================== #


# Final function which prints all the relevant information about the coordination compound
def analyze_complexe(formula):
    parse_elements(formula)
    lines = []

    # Formula
    lines.append(f"* **Formula** : {get_clean_formula(formula)}")

    # Nomenclature
    name = naming_compound(formula)
    lines.append(f"* **IUPAC Name** : {name}")

    # Metal charge
    metals = parse_metal(formula)
    charge = metal_charge(formula)
    charge_str = f"{charge}+" if charge > 0 else f"{charge}"
    lines.append(f"* **Metal** : {metals[0]} ({charge_str})")

    # Electronic structure
    e_list = electronic_structure(formula)
    lines.append(
        f"* **Electronic structure** : [{e_list[0]}] {e_list[1]}s{e_list[2]} {e_list[1] - 1}d{e_list[3]}"
    )

    # Electrons counting
    count = electron_count(formula)
    lines.append(f"* **Electron count** : {count}")

    # Isomers
    if isomers(formula)[0] is None or isomers(formula)[1] is None:
        lines.append(
            "* **Isomers:** The number of isomers of this compound is not specified"
        )
    else:
        lines.append(
            f"* **Isomers:** This compound has {isomers(formula)[0]} stereoisomers and {isomers(formula)[1]} enantiomeres pairs"
        )

    # Remarks
    remarks = oxidation_state(formula)[1]
    if remarks is not None:
        lines.append(f"* **Remarks:** {remarks}")

    # Stability (NEW)
    stability = stability_index(formula)
    lines.append(f"* **Stability index** : {stability}/100")

    return lines, "\n".join(lines)


def show_analysis(formula):
    return display(Markdown(analyze_complexe(formula)[1]))


# =============================================================================================================================================================== #
# =============================================================================================================================================================== #
# =============================================================================================================================================================== #
# 3D part of the code
# =============================================================================================================================================================== #
# =============================================================================================================================================================== #
# =============================================================================================================================================================== #

# ------------------------------------------------------------
# ------------------------------------------------------------
# GEOMETRY ---- METAL - LIGAND -------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------


def linear(r):
    return [(r, 0, 0), (-r, 0, 0)]


def tetrahedral(r):
    base = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]])

    base = base / np.linalg.norm(base[0])  # normalisation
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


def find_geometry(formula, r):
    cn = len(ligands_list(formula))
    if cn == 1:
        return [(r, 0, 0)]
    elif cn == 2:
        return linear(r)
    elif cn == 3:
        return trigonal_planar(r)
    elif cn == 4 and oxidation_state(formula)[0] == 8:
        return square_planar(r)
    elif cn == 4 and not oxidation_state(formula)[0] == 8:
        return tetrahedral(r)
    elif cn == 5:
        return trigonal_bipyramidal(r)
    elif cn == 6:
        return octahedral(r)
    else:
        raise ValueError("Error: The visualisation 3D does not work for CN over 6")


# ------------------------------------------------------------
# ------------------------------------------------------------
# GEOMETRY INTERNE ---- LIGANDS -----------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------


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
    if geometry != None:
        return geometry
    return False


# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# -------- 3D visualisation and coumpound creation -----------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------


def atoms_position_and_bond(formula):
    r = 1.7
    bonding = []
    nb_of_atoms = 0
    position = [(0, 0, 0)]
    big_array = find_geometry(formula, r)
    ligand_list = ligands_list(formula)
    for i, ligand in enumerate(ligand_list):
        if get_geometry_ligand(ligand) == "sphere":
            nb_of_atoms += 1
            position += [big_array[i]]
            bonding += (0, nb_of_atoms)
        elif get_geometry_ligand(ligand) == "linear":
            nb_of_atoms += 2
            position += [big_array[i]]
            position += ligand_linear(ligand, big_array[i], r)
            bonding += (0, nb_of_atoms - 1)
        elif get_geometry_ligand(ligand) == "dlinear":
            nb_of_atoms += 3
            position += [big_array[i]]
            position += ligand_dlinear(ligand, big_array[i], r)
            bonding += (0, nb_of_atoms - 2)
        elif get_geometry_ligand(ligand) == "bent":
            nb_of_atoms += 3
            position += [big_array[i]]
            position += ligand_bent(ligand, big_array[i], r)
            bonding += (0, nb_of_atoms - 2)
        elif get_geometry_ligand(ligand) == "tetrahedral":
            nb_of_atoms += 4
            position += [big_array[i]]
            position += ligand_tetrahedral(ligand, big_array[i], r)
            bonding += (0, nb_of_atoms - 3)
        else:
            raise ValueError("Error: Geometry of the ligand not available in 3D")
    return position, bonding


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


def atom_symbols(formula):
    metal = parse_metal(formula)
    atoms_list = metal

    ligand_list = ligands_list(formula)
    for ligand in ligand_list:
        atoms_list += get_atoms(ligand)
    return atoms_list


def create_compound_render(formula):
    compound = Atoms(
        atom_symbols(formula), positions=atoms_position_and_bond(formula)[0]
    )
    return compound


"""
def metal_radii(formula):
    metal = parse_metal(formula)[0]
    coordination = len(ligands_list(formula))
    charge = metal_charge(formula)
    radii = get_ionic_radii(metal, charge, coordination)
    return radii

def ligand_radii(formula):
    ligands = ligands_list(formula)
    r = 0
    for ligand in ligands:
        if data_ligands[ligand].get("radii") != False:
            r += data_ligands[ligand]["radii"]
        else:
            donor_atom = data_ligands[ligand]["donor_atoms"][0]
            r += covalent_radii[atomic_numbers[donor_atom]]
    return r/len(ligands)
"""
