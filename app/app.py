import streamlit as st
import streamlit.components.v1 as components

import coordchempy as cc

# ============================================================
# Title / Subheaders of the app
# ============================================================

st.title("CoordChemPy", text_alignment="left")
st.subheader(
    "The Essential Python Toolkit for Coordination Chemistry!",
    divider="gray",
    text_alignment="left",
)
#
st.subheader("Coordination compound information finder")

# ============================================================
# Initialisation of the session
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "compound_ase" not in st.session_state:
    st.session_state.compound_ase = None

# ============================================================
# Test input
# ============================================================

counter_ions = st.text_input(
    "_Enter the counter ion formula following the correct format._",
    placeholder="Type here (optional) ...",
    help="Input format rules are explained in the README.md file",
)

coord_compound = st.text_input(
    "_Enter the coordination compound formula following the correct format. Try [Pt(Cl)2(NH3)2] !_",
    placeholder="Type here ...",
    help="Input format rules are explained in the README.md file",
)


# ============================================================
# Analysis button
# ============================================================

if st.button("Analysis"):
    # Vérification : La formule principale ne peut pas être vide
    if not coord_compound:
        st.warning("Please enter the coordination compound formula.")
        st.session_state.analysis_result = None
    else:
        try:
            # Si counter_ions est vide, on appelle la fonction avec un seul argument
            # Sinon, on passe les deux.
            if counter_ions:
                result = cc.analyse_compound(coord_compound, counter_ions)
            else:
                result = cc.analyse_compound(coord_compound)
            st.session_state.analysis_result = result

        except Exception as e:
            st.error(f"Analysis error: {e}")
            st.session_state.analysis_result = None

# ============================================================
# Render analysis compound
# ============================================================

if st.session_state.analysis_result:
    st.divider()
    st.markdown(st.session_state.analysis_result)
    st.success("Successful analysis")

st.divider()

# ============================================================
# Render visualtion 3D complex
# ============================================================

st.subheader("Coordination compound 3D rendering")

size_range = [i / 10 for i in range(1, 11)]
render_options = ["Ball and Stick", "Stick", "Sphere", "Lines", "VDW"]
with st.form("render_form"):
    atoms_size = st.select_slider("Atom size", options=size_range, value=0.3)
    render_type = st.segmented_control(
        "Rendring type",
        options=render_options,
        selection_mode="single",
        default="Ball and Stick",
        help="VDW: Van der Waals",
    )
    submit = st.form_submit_button("3D render")

if submit:
    if coord_compound:
        try:
            st.session_state.compound_ase = cc.create_compound_render(coord_compound)
        except ValueError as e:
            st.error(str(e))
            st.session_state.analysis_result = None
            st.session_state.compound_ase = None

    else:
        st.warning("Please enter a formula.")
        st.session_state.analysis_result = None
        st.session_state.compound_ase = None

if st.session_state.compound_ase:
    try:
        view_html = cc.render_complex(
            st.session_state.compound_ase, atoms_size, render_type
        )
        components.html(view_html, height=400, width=400)
        st.success("Successful render")

    except ValueError as e:
        st.error(str(e))
        st.session_state.compound_ase = None

st.divider()
# ============================================================
# Stability Duel Section
# ============================================================

st.subheader("Stability Duel")
st.write(
    "Compare the current compound with another formula to see which one is more stable."
)


comp_A = st.text_input("Compound A :", placeholder="Type the first compound here...")
comp_B = st.text_input("Compound B :", placeholder="Type the second compound here...")

if st.button("Run Stability Duel"):
    if not comp_A or not comp_B:
        st.warning("Please make sure both compound fields are filled to run the duel.")
    else:
        try:
            comp_A_clean = comp_A.strip()
            comp_B_clean = comp_B.strip()

            # Appel direct de la fonction depuis ton package coordchempy
            duel_results = cc.stability_duel(comp_A_clean, comp_B_clean)

            # Extraction dynamique des scores pour éviter les KeyErrors liés aux espaces
            score_A = next(
                (
                    v
                    for k, v in duel_results.items()
                    if k.startswith("Score") and comp_A_clean in k
                ),
                "N/A",
            )
            score_B = next(
                (
                    v
                    for k, v in duel_results.items()
                    if k.startswith("Score") and comp_B_clean in k
                ),
                "N/A",
            )

            # Affichage des scores
            duel_col1, duel_col2 = st.columns(2)
            duel_col1.metric(label=f"Score {comp_A_clean}", value=score_A)
            duel_col2.metric(label=f"Score {comp_B_clean}", value=score_B)

            # Affichage du gagnant
            winner = duel_results.get("Most Stable", "Tie")
            if winner == "Tie":
                st.info("🤝 It's a tie!")
            else:
                st.success(f"🏆 Most Stable: **{winner}**")

        except Exception as e:
            st.error(f"Stability analysis error: {e}")

st.divider()
