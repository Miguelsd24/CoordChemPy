# ==========================================
# 🧪 TEST STABILITY ENGINE
# ==========================================

from coordchempy import StabilityEngine

# ==========================================
# 🧪 TEST LIST (50 COMPLEXES)
# ==========================================

TESTS = [
    # cyanides
    "[Fe(CN)6]4-",
    "[Fe(CN)6]3-",
    "[Co(CN)6]3-",
    "[Ni(CN)4]2-",
    "[Pd(CN)4]2-",
    "[Pt(CN)4]2-",
    # carbonyls
    "[Ni(CO)4]",
    "[Fe(CO)5]",
    "[Cr(CO)6]",
    "[Mo(CO)6]",
    "[W(CO)6]",
    "[Mn(CO)6]+",
    "[V(CO)6]-",
    # ammines
    "[Co(NH3)6]3+",
    "[Rh(NH3)6]3+",
    "[Ir(NH3)6]3+",
    "[Ni(NH3)4]2+",
    "[Cu(NH3)4]2+",
    "[Zn(NH3)4]2+",
    "[Pd(NH3)4]2+",
    "[Pt(NH3)4]2+",
    "[Ag(NH3)2]+",
    "[Au(NH3)2]+",
    # aqua
    "[Cr(H2O)6]3+",
    "[Fe(H2O)6]3+",
    "[Co(H2O)6]2+",
    "[Ni(H2O)6]2+",
    "[Cu(H2O)6]2+",
    "[Zn(H2O)6]2+",
    # fluorides
    "[TiF6]2-",
    "[CrF6]3-",
    "[FeF6]3-",
    "[CoF6]3-",
    "[AlF6]3-",
    # chlorides
    "[CoCl4]2-",
    "[NiCl4]2-",
    "[CuCl4]2-",
    "[PdCl4]2-",
    "[PtCl6]2-",
    "[AuCl4]-",
    # chelates
    "[Co(en)3]3+",
    "[Ni(en)3]2+",
    "[Fe(en)3]2+",
    "[Cr(en)3]3+",
    # phosphines
    "[Ni(PPh3)4]",
    "[Pd(PPh3)4]",
    "[Pt(PPh3)4]",
    # mixed
    "[Ru(NH3)5Cl]2+",
    "[Co(NH3)5Cl]2+",
    "[Pt(NH3)2Cl2]",
    # oxalates
    "[Fe(C2O4)3]3-",
    "[Cr(C2O4)3]3-",
]


# ==========================================
# 🧪 RUN
# ==========================================

print("\n🧪 TEST STABILITY ENGINE\n")

print("=" * 60)

results = []

for formula in TESTS:
    try:
        engine = StabilityEngine(formula)

        score = engine.final_score().total

        results.append((formula, score))

        print(f"{formula:<25} | {score:6.2f}")

    except Exception as e:
        print(f"❌ {formula:<25} | {e}")


# ==========================================
# 🏆 RANKING
# ==========================================

results.sort(key=lambda x: x[1], reverse=True)

print("\n🏆 RANKING\n")

for i, (formula, score) in enumerate(results, 1):
    print(f"{i:2}. {formula:<25} -> {score:6.2f}")

print("\n💾 TEST FINISHED\n")
