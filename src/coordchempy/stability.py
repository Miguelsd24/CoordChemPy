# ==========================================
# 🧪 STABILITY ENGINE V9 — IMPROVED HEURISTICS
# ==========================================

import math
from dataclasses import dataclass
from typing import Dict, List

from coordchempy.coordchem import (
    parse_ligands,
    parse_metal,
    electron_count,
    metal_charge,
    oxidation_state,
    data_ligands,
    data_metals,
)

# ==========================================
# GLOBAL STATE (ANALYTICS ONLY)
# ==========================================

GLOBAL_SCORES: List[float] = []

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
    return math.exp(-float(x) ** 2)

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

        self.ligands, _ = parse_ligands(self.formula)
        self.ligands = self.ligands or []

        self.cn = len(self.ligands)
        self.electrons = electron_count(self.formula)
        self.charge = metal_charge(self.formula)
        ox = oxidation_state(self.formula)
        self.ox = ox[0] if isinstance(ox, tuple) else ox

        self.m_data = data_metals.get(
            self.metal,
            {"hardness": 5, "possible_ox_state": [self.ox]}
        )

    # ======================================
    # ELECTRON COUNTING
    # ======================================
    def electron_score(self):
        target = 18 if self.metal in {"Fe","Co","Ni","Pd","Pt","Cr","Mn"} else 16
        diff = abs(self.electrons - target)
        base = max(0, 10 - diff * 1.0)  # moins agressif

        bonus_map = {"CO": 1.5, "CN": 1.2, "NH3": 0.5, "NO":1.0, "PR3":0.8}
        bonus = sum(bonus_map.get(l.replace("m-", ""), 0.2) for l in self.ligands)

        return clamp(base + bonus)

    # ======================================
    # HSAB
    # ======================================
    def hsab_score(self):
        mh = self.m_data.get("hardness", 5)
        vals = []
        for l in self.ligands:
            lh = data_ligands.get(l, {}).get("HSAB", {}).get("hardness", 5)
            vals.append(10 * gaussian(abs(mh - lh) / 1.5))  # plus sensible
        return clamp(safe_mean(vals))

    # ======================================
    # CFSE
    # ======================================
    def cfse_score(self):
        field_map = {"CO": 2.5, "CN": 2.3, "NH3": 1.2, "H2O": 1.0, "Cl": 0.8, "F": 0.7}
        vals = [field_map.get(l, 1.0) for l in self.ligands]
        field = safe_mean(vals)
        # CFSE ajusté avec une fonction non-linéaire
        score = field * 3 + 0.5 * (field ** 2)
        return clamp(score)

    # ======================================
    # FIELD STRENGTH
    # ======================================
    def field_score(self):
        vals = [data_ligands.get(l, {}).get("field", 1) for l in self.ligands]
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
            return 9.0 if self.metal in {"Pt","Pd","Ni","Cu"} else 7.5
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
        for l in self.ligands:
            info = data_ligands.get(l, {})
            if info.get("pi_acceptor"):
                score += 3.5  # plus impactant
            if l in {"CO", "CN"}:
                score += 2.5  # plus impactant
        return clamp(score)

    # ======================================
    # STERIC
    # ======================================
    def steric_score(self):
        vals = [data_ligands.get(l, {}).get("steric_bulk", 1) for l in self.ligands]
        return clamp(10 - safe_mean(vals) * 1.5)  # un peu plus permissif

    # ======================================
    # CRITERIA VECTOR
    # ======================================
    def criteria(self) -> Dict[str, float]:
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
            c["electron"] * 0.22 +
            c["hsab"] * 0.15 +
            c["cfse"] * 0.20 +
            c["field"] * 0.12 +
            c["charge"] * 0.08 +
            c["geometry"] * 0.08 +
            c["oxidation"] * 0.05 +
            c["backbonding"] * 0.05 +
            c["steric"] * 0.05
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
def duel(A, B):
    a = StabilityEngine(A)
    b = StabilityEngine(B)
    cA = a.criteria()
    cB = b.criteria()
    breakdown = {}
    A_w = B_w = 0
    for k in cA:
        if cA[k] > cB[k]:
            winner = "A"
            A_w += 1
        elif cB[k] > cA[k]:
            winner = "B"
            B_w += 1
        else:
            winner = "Tie"
        breakdown[k] = {"A": cA[k], "B": cB[k], "winner": winner}
    return {
        "A_total": a.total_score(),
        "B_total": b.total_score(),
        "winner": "A" if A_w > B_w else "B",
        "A_wins": A_w,
        "B_wins": B_w,
        "breakdown": breakdown
    }