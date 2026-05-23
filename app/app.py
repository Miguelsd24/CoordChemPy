import streamlit as st
import streamlit.components.v1 as components

import coordchempy as cc

# Title of the app
st.title("CoordChemPy", text_alignment="left")
st.subheader(
    "The best python based tool for coordination chemist!",
    divider="gray",
    text_alignment="left",
)
#
st.subheader("Coordination compound information finder")


# 1. Initialisation
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "compound_ase" not in st.session_state:
    st.session_state.compound_ase = None

# 2. Input text

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

# 3. Analysis button
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

# 4. Affichage du résultat (si il existe dans le state)
if st.session_state.analysis_result:
    st.divider()
    st.markdown(st.session_state.analysis_result)
    st.success("Successful analysis")

st.divider()
# ----------------------------------------------------------------------------------------


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
            st.session_state.compound_ase = coord_compound
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
