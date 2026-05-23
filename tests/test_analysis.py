# ==========================================
# 🧪 TEST ANALYSIS COMPOUND WITH VALIDATION
# ==========================================

from coordchempy import analyse_compound

# ==========================================
# 🧪 TEST LIST WITH THEORETICAL VALUES
# ==========================================
# Format : ( "Formule", { "IUPAC": ..., "Oxidation": ..., "Config": ..., "Electrons": ..., "Geometry": ... } )

TESTS = [
    # --- Classical & Mixed Monodentate Ligands ---
    (
        "[Co(NH3)5(Cl)]2+",
        {
            "IUPAC": "pentaamminechlorocobalt(III)",
            "Oxidation": "Co (3+)",
            "Config": "[Ar] 4s0 3d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Pt(NH3)2(Cl)2]",
        {
            "IUPAC": "diamminedichloroplatinum(II)",
            "Oxidation": "Pt (2+)",
            "Config": "[Xe] 6s0 5d8",
            "Electrons": 16,
            "Geometry": "Square planar",
        },
    ),
    (
        "[Ru(NH3)5(H2O)]2+",
        {
            "IUPAC": "pentaammineaquaruthenium(II)",
            "Oxidation": "Ru (2+)",
            "Config": "[Kr] 5s0 4d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Cr(H2O)4(Cl)2]+",
        {
            "IUPAC": "tetraaquadichlorochromium(III)",
            "Oxidation": "Cr (3+)",
            "Config": "[Ar] 4s0 3d3",
            "Electrons": 15,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Fe(CN)5(NO)]2-",
        {
            "IUPAC": "pentacyanonitrosylferrate(III)",
            "Oxidation": "Fe (3+)",
            "Config": "[Ar] 4s0 3d5",
            "Electrons": 17,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Co(NH3)4(NO2)2]+",
        {
            "IUPAC": "tetraamminedinitritocobalt(III)",
            "Oxidation": "Co (3+)",
            "Config": "[Ar] 4s0 3d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Re(PR3)2(CO)2(CH3)(C2H4)]",
        {
            "IUPAC": "Carbonylmethylbis(phosphane)ethenerhénium",
            "Oxidation": "Re (1+)",
            "Config": "[Xe] 4s0 3d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Cu(py)2(Cl)2]",
        {
            "IUPAC": "dichlorodipyridinecopper(II)",
            "Oxidation": "Cu (2+)",
            "Config": "[Ar] 4s0 3d9",
            "Electrons": 17,
            "Geometry": "Square planar",
        },
    ),
    (
        "[Ni(NH3)3(H2O)3]2+",
        {
            "IUPAC": "triaquatriaaminenickel(II)",
            "Oxidation": "Ni (2+)",
            "Config": "[Ar] 4s0 3d8",
            "Electrons": 20,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Mo(CO)3(CH3CN)3]",
        {
            "IUPAC": "tris(acetonitrile)tricarbonylmolybdenum(0)",
            "Oxidation": "Mo (0)",
            "Config": "[Kr] 5s0 4d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Ni(PPh3)2(Cl)2]",
        {
            "IUPAC": "dichlorobis(triphenylphosphine)nickel(II)",
            "Oxidation": "Ni (2+)",
            "Config": "[Ar] 4s0 3d8",
            "Electrons": 16,
            "Geometry": "Tetrahedral",
        },
    ),
    (
        "[Pd(PPh3)4]",
        {
            "IUPAC": "tetrakis(triphenylphosphine)palladium(0)",
            "Oxidation": "Pd (0)",
            "Config": "[Kr] 5s0 4d10",
            "Electrons": 18,
            "Geometry": "Tetrahedral",
        },
    ),
    (
        "[Ir(CO)(Cl)(PPh3)2]",
        {
            "IUPAC": "carbonylchlorobis(triphenylphosphine)iridium(I)",
            "Oxidation": "Ir (1+)",
            "Config": "[Xe] 6s0 5d8",
            "Electrons": 16,
            "Geometry": "Square planar",
        },
    ),
    (
        "[Rh(PPh3)3(Cl)]",
        {
            "IUPAC": "chlorotris(triphenylphosphine)rhodium(I)",
            "Oxidation": "Rh (1+)",
            "Config": "[Kr] 5s0 4d8",
            "Electrons": 16,
            "Geometry": "Square planar",
        },
    ),
    (
        "[Fe(CO)4(PPh3)]",
        {
            "IUPAC": "tetracarbonyl(triphenylphosphine)iron(0)",
            "Oxidation": "Fe (0)",
            "Config": "[Ar] 4s0 3d8",
            "Electrons": 18,
            "Geometry": "Trigonal bipyramidal",
        },
    ),
    (
        "[W(CO)5(py)]",
        {
            "IUPAC": "pentacarbonyl(pyridine)tungsten(0)",
            "Oxidation": "W (0)",
            "Config": "[Xe] 6s0 5d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Mn(CO)5(Br)]",
        {
            "IUPAC": "bromopentacarbonylmanganese(I)",
            "Oxidation": "Mn (1+)",
            "Config": "[Ar] 4s0 3d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Co(CO)3(PPh3)2]+",
        {
            "IUPAC": "tricarbonylbis(triphenylphosphine)cobalt(I)",
            "Oxidation": "Co (1+)",
            "Config": "[Ar] 4s0 3d8",
            "Electrons": 18,
            "Geometry": "Trigonal bipyramidal",
        },
    ),
    (
        "[Re(CO)5(Cl)]",
        {
            "IUPAC": "chloropentacarbonylrhenium(I)",
            "Oxidation": "Re (1+)",
            "Config": "[Xe] 6s0 5d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Pt(PPh3)2(C2H4)]",
        {
            "IUPAC": "(ethene)bis(triphenylphosphine)platinum(0)",
            "Oxidation": "Pt (0)",
            "Config": "[Xe] 6s0 5d10",
            "Electrons": 16,
            "Geometry": "Three-coordinate / Trigonal planar",
        },
    ),
    (
        "[Co(en)3]3+",
        {
            "IUPAC": "tris(ethane-1,2-diamine)cobalt(III)",
            "Oxidation": "Co (3+)",
            "Config": "[Ar] 4s0 3d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Ni(en)2(H2O)2]2+",
        {
            "IUPAC": "diaquabis(ethane-1,2-diamine)nickel(II)",
            "Oxidation": "Ni (2+)",
            "Config": "[Ar] 4s0 3d8",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Cu(en)2]2+",
        {
            "IUPAC": "bis(ethane-1,2-diamine)copper(II)",
            "Oxidation": "Cu (2+)",
            "Config": "[Ar] 4s0 3d9",
            "Electrons": 19,
            "Geometry": "Square planar",
        },
    ),
    (
        "[Cr(en)2(Ox)]+",
        {
            "IUPAC": "bis(ethane-1,2-diamine)oxalatochromium(III)",
            "Oxidation": "Cr (3+)",
            "Config": "[Ar] 4s0 3d3",
            "Electrons": 15,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Fe(bipy)3]2+",
        {
            "IUPAC": "tris(2,2'-bipyridine)iron(II)",
            "Oxidation": "Fe (2+)",
            "Config": "[Ar] 4s0 3d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Ru(bipy)3]2+",
        {
            "IUPAC": "tris(2,2'-bipyridine)ruthenium(II)",
            "Oxidation": "Ru (2+)",
            "Config": "[Kr] 5s0 4d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Fe(ox)3]3-",
        {
            "IUPAC": "trioxalatoferrate(III)",
            "Oxidation": "Fe (3+)",
            "Config": "[Ar] 4s0 3d5",
            "Electrons": 17,
            "Geometry": "Octahedral",
        },
    ),
    "[Co(NCS)4]2-",
    {
        "IUPAC": "tetraisothiocyanatocobaltate(II)",
        "Oxidation": "Co (2+)",
        "Config": "[Ar] 4s0 3d7",
        "Electrons": 15,
        "Geometry": "Tetrahedral",
    },
    (
        "[Hg(SCN)4]2-",
        {
            "IUPAC": "tetrathiocyanato-S-mercurate(II)",
            "Oxidation": "Hg (2+)",
            "Config": "[Xe] 6s0 5d10",
            "Electrons": 18,
            "Geometry": "Tetrahedral",
        },
    ),
    (
        "[Ag(S2O3)2]3-",
        {
            "IUPAC": "bis(thiosulfato)argentate(I)",
            "Oxidation": "Ag (1+)",
            "Config": "[Kr] 5s0 4d10",
            "Electrons": 18,
            "Geometry": "Linear",
        },
    ),
    (
        "[Fe(N3)6]3-",
        {
            "IUPAC": "hexaazidoferrate(III)",
            "Oxidation": "Fe (3+)",
            "Config": "[Ar] 4s0 3d5",
            "Electrons": 17,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Co(NH3)5(NCS)]2+",
        {
            "IUPAC": "pentaammineisothiocyanatocobalt(III)",
            "Oxidation": "Co (3+)",
            "Config": "[Ar] 4s0 3d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Zn(NCS)4]2-",
        {
            "IUPAC": "tetraisothiocyanatozincate(II)",
            "Oxidation": "Zn (2+)",
            "Config": "[Ar] 4s0 3d10",
            "Electrons": 18,
            "Geometry": "Tetrahedral",
        },
    ),
    (
        "[Pt(SCN)4]2-",
        {
            "IUPAC": "tetrathiocyanato-S-platinate(II)",
            "Oxidation": "Pt (2+)",
            "Config": "[Xe] 6s0 5d8",
            "Electrons": 16,
            "Geometry": "Square planar",
        },
    ),
    (
        "[Cu(edta)]2-",
        {
            "IUPAC": "ethylenediaminetetraacetatocuprate(II)",
            "Oxidation": "Cu (2+)",
            "Config": "[Ar] 4s0 3d9",
            "Electrons": 21,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Co(edta)]-",
        {
            "IUPAC": "ethylenediaminetetraacetatocobalt(III)",
            "Oxidation": "Co (3+)",
            "Config": "[Ar] 4s0 3d6",
            "Electrons": 18,
            "Geometry": "Octahedral",
        },
    ),
    (
        "[Au(CN)2]-",
        {
            "IUPAC": "dicyanoaurate(I)",
            "Oxidation": "Au (1+)",
            "Config": "[Xe] 6s0 5d10",
            "Electrons": 14,
            "Geometry": "Linear",
        },
    ),
    (
        "[Ag(CN)2]-",
        {
            "IUPAC": "dicyanoargentate(I)",
            "Oxidation": "Ag (1+)",
            "Config": "[Kr] 5s0 4d10",
            "Electrons": 14,
            "Geometry": "Linear",
        },
    ),
    (
        "[Hg(I)4]2-",
        {
            "IUPAC": "tetraiodomercurate(II)",
            "Oxidation": "Hg (2+)",
            "Config": "[Xe] 6s0 5d10",
            "Electrons": 18,
            "Geometry": "Tetrahedral",
        },
    ),
]

# ==========================================
# 🧪 RUN
# ==========================================

print("\n🧪 STARTING COMPOUND ANALYSIS TESTS WITH VALIDATION\n")
print("=" * 60)

for i, (formula, theoretical) in enumerate(TESTS, 1):
    print(f"🔬 TEST {i}/{len(TESTS)} : {formula}")
    print("-" * 60)

    try:
        print("🖥️  CALCULATED ANALYSIS:")
        analysis_result = analyse_compound(formula)
        print(analysis_result)

        print("\n🎯 THEORETICAL VALUES:")
        for key, value in theoretical.items():
            print(f"   • {key:<10} : {value}")

    except Exception as e:
        print(f"❌ Error analyzing {formula}")
        print(f"   Reason: {e}")

    print("=" * 60)

print("\n💾 ALL TESTS FINISHED\n")
