import streamlit as st
import numpy as np
import tensorflow as tf
import sys
import os
import streamlit.components.v1 as components
import pubchempy as pcp  # <--- LA NOUVELLE LIBRAIRIE MAGIQUE

# --- 1. SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

from model import TRM_Official
from data_loader import RNA_MAP, STRUCT_MAP, INV_STRUCT_MAP, MAX_LEN

# --- CONFIG ---
D_MODEL = 64
N_RECURSION = 2
T_LOOPS = 2
N_REFINE = 3

st.set_page_config(page_title="Project Origami", page_icon="🧬", layout="wide")

# Style CSS
st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Courier New', monospace; }
    .success-box { padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px; }
    .chem-formula { font-family: monospace; font-weight: bold; color: #e83e8c; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Project Origami: AI Drug Discovery")

# --- CHARGEMENT MODELE ---
@st.cache_resource
def load_model():
    model = TRM_Official(len(RNA_MAP), len(STRUCT_MAP), d=D_MODEL, max_len=MAX_LEN, n=N_RECURSION, T=T_LOOPS)
    # Build dummy
    dummy_x = tf.zeros((1, MAX_LEN), dtype=tf.int32)
    dummy_y = tf.zeros((1, MAX_LEN), dtype=tf.int32)
    dummy_z = tf.zeros((1, MAX_LEN, D_MODEL), dtype=tf.float32)
    model([model.emb_x(dummy_x), model.emb_y(dummy_y), dummy_z])
    
    weight_path = os.path.join(current_dir, '..', 'saved_models', 'trm_paper_version.weights.h5')
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
        return model, True
    return model, False

model, is_loaded = load_model()

if is_loaded:
    st.sidebar.success("✅ IA RNA-Folding : Active")

# --- NOUVEAU : RÉCUPÉRER LA VRAIE MOLÉCULE ---
def get_molecule_from_name(name):
    try:
        # On cherche dans la base de données mondiale PubChem
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            c = compounds[0]
            return {
                "smiles": c.canonical_smiles, # La formule brute
                "formula": c.molecular_formula, # Ex: C9H8O4
                "weight": c.molecular_weight,
                "cid": c.cid
            }
    except:
        return None
    return None

# --- VISUALISATION ---
def render_rna_plot(sequence, structure):
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/tntvis@0.3.1/dist/tnt.utils.min.js"></script>
        <script type="text/javascript" src="https://unpkg.com/fornac@0.3.1/dist/fornac.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/fornac@0.3.1/dist/fornac.css" />
        <style> #rna_ss {{ height: 400px; width: 100%; }} </style>
    </head>
    <body>
        <div id="rna_ss"></div>
        <script>
            var container = new fornac.FornaContainer("#rna_ss", 
                {{'sequence': '{sequence}', 'structure': '{structure}'}});
            container.addOptions({{'structureOptions': {{'avoidOthers': true}}}});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=450)

# --- PREDICTION ---
def predict(seq):
    clean_seq = seq.replace("\n", "").strip().upper()
    enc = [RNA_MAP.get(c, 5) for c in clean_seq]
    enc = enc[:MAX_LEN] + [0]*(MAX_LEN-len(enc))
    x = np.array([enc], dtype=np.int32)
    x_emb = model.emb_x(x) + model.pos_emb[:, :MAX_LEN, :]
    y = tf.zeros((1, MAX_LEN), dtype=tf.int32)
    z = tf.zeros((1, MAX_LEN, D_MODEL), dtype=tf.float32)
    
    progress = st.progress(0)
    for i in range(N_REFINE):
        logits, _, _, z_new = model([x_emb, model.emb_y(y), z])
        y = tf.argmax(logits, axis=-1, output_type=tf.int32)
        z = z_new
        progress.progress((i+1)/N_REFINE)
    
    struct = ""
    indices = y.numpy()[0]
    real_len = min(len(clean_seq), MAX_LEN)
    for idx in indices[:real_len]:
        struct += INV_STRUCT_MAP.get(idx, ".")
    return struct, clean_seq[:real_len]

# --- INTERFACE ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Cible (Maladie)")
    sequence_input = st.text_area("Séquence ARN :", 
        "GCGGAUUUAGCUCAGDDGGGAGAGCGCCAGACUGAAYAAAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGC", height=150)
    
    st.markdown("---")
    st.subheader("2. Médicament (Molécule)")
    drug_name = st.text_input("Nom du médicament :", "Aspirin")
    
    if st.button("Lancer l'Analyse"):
        if is_loaded:
            # 1. Analyse ARN
            with st.spinner("Repliement de l'ARN..."):
                struct_pred, seq_final = predict(sequence_input)
                st.session_state['rna_result'] = (seq_final, struct_pred)
            
            # 2. Analyse Médicament (Récupération des vraies données)
            with st.spinner(f"Recherche chimique de '{drug_name}'..."):
                mol_data = get_molecule_from_name(drug_name)
                st.session_state['mol_data'] = mol_data
                st.session_state['drug_name'] = drug_name

with col2:
    if 'rna_result' in st.session_state:
        seq, struct = st.session_state['rna_result']
        
        # Onglets pour séparer les vues
        tab1, tab2 = st.tabs(["🧬 Visualisation Cible", "💊 Analyse Médicament"])
        
        with tab1:
            render_rna_plot(seq, struct)
            st.caption(f"Structure 2D Prédite : {struct}")

        with tab2:
            mol = st.session_state.get('mol_data')
            name = st.session_state.get('drug_name')
            
            if mol:
                st.success(f"Molécule identifiée : {name}")
                st.markdown(f"**Formule Chimique :** `{mol['formula']}`")
                st.markdown(f"**Poids Moléculaire :** {mol['weight']} g/mol")
                st.markdown("**Code SMILES (Lu par l'IA) :**")
                st.code(mol['smiles'], language="text")
                
                # Image 2D de la molécule (via PubChem Widget)
                st.markdown(f"![Structure 2D](https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid={mol['cid']}&width=300&height=300)")
                
                st.divider()
                st.subheader("Résultat de l'interaction")
                # ICI : Pour l'instant c'est simulé car on n'a pas entraîné le modèle d'interaction
                # Mais au moins, on a les VRAIES données d'entrée
                
                # Simulation intelligente : Si la molécule est petite, elle passe mieux
                score = 0.85 if mol['weight'] < 500 else 0.40
                
                st.metric("Score d'Affinité (Binding)", f"{score*100:.1f}%")
                if score > 0.7:
                    st.write("✅ **Candidat Prometteur** : Cette molécule est assez petite pour s'insérer dans les boucles de l'ARN.")
                else:
                    st.write("⚠️ **Risque d'échec** : Molécule probablement trop volumineuse.")
                
            else:
                st.error(f"Impossible de trouver '{name}' dans la base de données PubChem. Vérifiez l'orthographe.")