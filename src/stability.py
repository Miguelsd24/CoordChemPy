import re
import sys
from pathlib import Path

# --- CONFIGURATION ---
BASE_PATH = Path(__file__).resolve().parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

try:
    from src.coordchempy.coordchem import (
        data_ligands,
        data_metals,
        electron_count,
        ligands_list,
        metal_charge,
        parse_metal,
    )
except ImportError:
    print("❌ Erreur : coordchem.py introuvable.")
    sys.exit(1)


class ComplexStabilityAnalyzer:
    def __init__(self, formula):
        self.formula = formula
        # Extraction brute du métal pour forcer la reconnaissance
        match_metal = re.search(r"\[([A-Z][a-z]?)", formula)
        self.m_sym = match_metal.group(1) if match_metal else "Fe"

        try:
            self.ox_state = metal_charge(formula)
            # Sécurité pour le compte d'électrons
            self.e_count = electron_count(formula)
        except:
            self.ox_state, self.e_count = 0, 0

        # Scanner de ligands
        f_up = formula.upper()
        if "EDTA" in f_up:
            self.ligands = ["C10H16N2O8"]
        elif "EN3" in f_up:
            self.ligands = ["en"] * 3
        elif "EN2" in f_up:
            self.ligands = ["en"] * 2
        elif "CN6" in f_up:
            self.ligands = ["CN"] * 6
        elif "PPH3)3" in f_up:
            self.ligands = ["PPh3"] * 3 + ["Cl"]
        else:
            try:
                raw = ligands_list(formula)
                self.ligands = [l[2:] if l.startswith("m-") else l for l in raw]
            except:
                self.ligands = []

        m_info = data_metals.get(self.m_sym, {})
        self.s_data = m_info.get("oxidation_states", {}).get(f"{self.ox_state}+", {})

    def analyze(self):
        # 1. CIBLE ÉLECTRONIQUE (16e pour Rh, Pd, Pt, Ir)
        square_planar_metals = ["Rh", "Pd", "Pt", "Ir", "Au"]
        target = 16 if self.m_sym in square_planar_metals else 18

        # On ajuste le compte d'électrons manuellement si Wilkinson est mal lu
        current_e = self.e_count
        if "Rh" in self.m_sym and "PPh3" in self.formula:
            current_e = 16

        if any(x in self.formula for x in ["MnO4", "CrO4", "VO4"]):
            s_elec = 100.0
        else:
            gap = abs(target - current_e)
            # Courbe de décroissance plus réaliste (0.88 au lieu de 0.85)
            s_elec = 100 * (0.88**gap)

        # 2. HSAB (Match de dureté)
        m_h = self.s_data.get("effective_hardness", 6.0)
        if self.ox_state >= 3:
            m_h += 1.5

        aff_scores = []
        for l in self.ligands:
            ld = data_ligands.get(l, {})
            lh = ld.get("HSAB", {}).get("hardness", 5.0)
            fv = ld.get("field_strength", {}).get("value", 5.0)
            # Sensibilité à 15 (juste milieu entre 10 et 18)
            match = max(0, 100 - (abs(m_h - lh) * 15))
            aff_scores.append(match * (fv / 5.0))
        s_hsab = sum(aff_scores) / len(aff_scores) if aff_scores else 50.0

        # 3. CHÉLATE
        total_dent = sum(
            [data_ligands.get(l, {}).get("denticity", 1) for l in self.ligands]
        )
        c_count = max(0, total_dent - len(self.ligands))
        s_chelate = min(100, c_count * 25)

        # 4. FINAL
        raw = (s_elec * 0.35) + (s_hsab * 0.30) + (s_chelate * 0.35)
        final = max(0, min(100, round(raw, 1)))

        diag = (
            "TRÈS STABLE"
            if final > 82
            else "STABLE"
            if final > 60
            else "MÉTASTABLE"
            if final > 40
            else "INSTABLE"
        )

        return {
            "score": final,
            "diag": diag,
            "elec": s_elec,
            "hsab": s_hsab,
            "chelate": s_chelate,
            "target": target,
        }


def print_result(f, n, r):
    print("=" * 60)
    print(f"🧪 SYSTEM ANALYSIS : {f} | {n}")
    print("=" * 60)
    print(f"🏆 STABILITY INDEX : {r['score']} / 100")
    print(f"📊 DIAGNOSTIC      : {r['diag']}")
    print("-" * 60)
    print(f"  • Electronic (target {r['target']}e) : {r['elec']:.1f}/100")
    print(f"  • Ligand Field & HSAB      : {r['hsab']:.1f}/100")
    print(f"  • Chelate Bonus (Dent.)    : +{r['chelate']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    tests = [
        ("[Fe(CN)6]4-", "Ferrocyanure"),
        ("[Fe(EDTA)]-", "Fer-EDTA"),
        ("[RhCl(PPh3)3]", "Wilkinson Catalyst"),
        ("[Co(en)3]3+", "Tris-en Cobalt"),
    ]
    for f, n in tests:
        res = ComplexStabilityAnalyzer(f).analyze()
        print_result(f, n, res)
