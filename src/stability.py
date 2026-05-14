# ==========================================
# 🧪 STABILITY ENGINE
# ==========================================

import math
from dataclasses import dataclass

try:
    from src.coordchem import (
        data_ligands,
        data_metals,
        electron_count,
        ligands_list,
        metal_charge,
        oxidation_state,
        parse_metal,
    )
except:
    from coordchempy import (
        data_ligands,
        data_metals,
        electron_count,
        ligands_list,
        metal_charge,
        oxidation_state,
        parse_metal,
    )


# ==========================================
# 📦 RESULT
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


# ==========================================
# 🔧 NORMALIZATION
# ==========================================


def normalize(f):
    return f.replace(" ", "")


# ==========================================
# 🧪 ENGINE
# ==========================================


class StabilityEngine:
    def __init__(self, formula):
        self.formula = normalize(formula)

        metals = parse_metal(self.formula)
        if not metals:
            raise ValueError("No metal found")

        self.metal = metals[0]
        self.ligands = ligands_list(self.formula)

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
            l = data_ligands.get(lig, {})

            l_h = l.get("HSAB", {}).get("hardness", 5)
            field = l.get("field", 1)

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
