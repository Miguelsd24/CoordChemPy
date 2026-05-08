import io
import os
import sys

import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from ase.io import write

# 1. Trouve le dossier où se trouve app.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Remonte d'un niveau et va dans 'src'
src_path = os.path.join(current_dir, "..", "src")

# 3. Ajoute-le au système
if src_path not in sys.path:
    sys.path.append(src_path)

import coordchem as cc

# pour lancer mettre dans le terminal : python -m streamlit run c:/Users/migus/git/ppchem/app/app.py


def render_molecule(compound):
    xyz_str = io.StringIO()
    write(xyz_str, compound, format="xyz")
    xyz_content = xyz_str.getvalue()

    view = py3Dmol.view(width=400, height=400)
    view.addModel(xyz_content, "xyz")

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

    view_html = view._make_html()
    components.html(view_html, height=400, width=400)


# Title of the app
st.title("CoordChemPy", text_alignment="left")
st.subheader(
    "The best python based tool for coordination chemist! :atom_symbol:",
    divider="gray",
    text_alignment="left",
)
#
st.subheader("Coordination compound information finder")


# 1. Initialisation (crucial pour éviter les erreurs de clés manquantes)
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
# Initialisation au début du script
if "compound_ase" not in st.session_state:
    st.session_state.compound_ase = None

# 2. Les champs de saisie (on utilise des variables simples ici pour le contrôle)

counter_ions = st.text_input(
    "_Enter the counter ion formula following the correct format. Try (K)3 !_",
    placeholder="Type here (optional) ...",
    help="Input format rules are explained in the README.md file",
)

coord_compound = st.text_input(
    "_Enter the coordination compound formula following the correct format. Try [Pt(Cl)2(NH3)2] !_",
    placeholder="Type here ...",
    help="Input format rules are explained in the README.md file",
)


# 3. Le bouton Analysis (Le seul déclencheur)
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
                result = cc.analyze_complexe(coord_compound, counter_ions)
            else:
                result = cc.analyze_complexe(coord_compound)

            st.session_state.analysis_result = result[0]
        except Exception as e:
            st.error(f"Analysis error: {e}")
            st.session_state.analysis_result = None

# 4. Affichage du résultat (si il existe dans le state)
if st.session_state.analysis_result:
    st.divider()
    for line in st.session_state.analysis_result:
        st.write(line)
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
    render_molecule(st.session_state.compound_ase)
    st.success("Successful render")


st.divider()
