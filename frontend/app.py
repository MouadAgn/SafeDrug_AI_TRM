import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw
import random

# ==============================================================================
# 1. CONSTANTES & CLASSES (Architecture du modèle RNA)
# ==============================================================================

MAX_LEN = 150
D_MODEL = 32
N_REC = 1
N_SUP = 3 # Attention à bien garder la valeur utilisée lors de l'entraînement
SEQ_MAP = {'A': 1, 'C': 2, 'G': 3, 'U': 4, 'T': 4, 'N': 0}
REV_STRUCT_MAP = {0: '', 1: '.', 2: '(', 3: ')'}

class TinyBlock(layers.Layer):
    def __init__(self, d, **kwargs):
        super().__init__(**kwargs)
        self.ln = layers.LayerNormalization()
        self.fc1 = layers.Dense(4 * d, activation="gelu")
        self.fc2 = layers.Dense(d)
        self.d = d

    def call(self, u):
        h = self.ln(u)
        h = self.fc1(h)
        h = self.fc2(h)
        return u + h
        
    def get_config(self):
        config = super().get_config()
        config.update({"d": self.d})
        return config

class RNATRM(keras.Model):
    def __init__(self, input_vocab=5, output_vocab=4, d=64, max_len=150, n_rec=2, T=3, Nsup=4, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.n_rec = n_rec
        self.T = T
        self.Nsup = Nsup
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_len = max_len

        self.emb = layers.Embedding(input_vocab, d)
        self.pos = self.add_weight(shape=(1, max_len, d), initializer="random_normal", trainable=True)
        self.y0 = self.add_weight(shape=(1, max_len, d), initializer="zeros", trainable=True)
        self.z0 = self.add_weight(shape=(1, max_len, d), initializer="zeros", trainable=True)
        self.block1 = TinyBlock(d)
        self.block2 = TinyBlock(d)
        self.to_structure = layers.Dense(output_vocab) 
        self.halt_head = layers.Dense(1)

    def tiny_net(self, u):
        u = self.block1(u)
        u = self.block2(u)
        return u

    def update_z(self, x, y, z): return self.tiny_net(x + y + z)
    def update_y(self, y, z): return self.tiny_net(y + z)

    def call(self, x_tokens, y_true=None, training=False):
        B = tf.shape(x_tokens)[0]
        L = tf.shape(x_tokens)[1]
        x = self.emb(x_tokens) + self.pos[:, :L, :]
        y = tf.tile(self.y0[:, :L, :], [B, 1, 1])
        z = tf.tile(self.z0[:, :L, :], [B, 1, 1])

        for step in range(self.Nsup):
            for t in range(self.T):
                if t < self.T - 1:
                    for _ in range(self.n_rec):
                        z = tf.stop_gradient(self.update_z(x, y, z))
                    y = tf.stop_gradient(self.update_y(y, z))
                else:
                    for _ in range(self.n_rec):
                        z = self.update_z(x, y, z)
                    y = self.update_y(y, z)
            logits = self.to_structure(y)
            halt_p = tf.sigmoid(tf.reduce_mean(self.halt_head(y), axis=1))
            y = tf.stop_gradient(y)
            z = tf.stop_gradient(z)
        return logits, halt_p
    
    def get_config(self):
        config = super().get_config()
        config.update({"input_vocab": self.input_vocab, "output_vocab": self.output_vocab, "d": self.d, "max_len": self.max_len, "n_rec": self.n_rec, "T": self.T, "Nsup": self.Nsup})
        return config

# ==============================================================================
# 2. FONCTIONS UTILITAIRES (RNA + DRUG)
# ==============================================================================

def preprocess_sequence(seq, max_len=150):
    seq = seq.strip().upper()
    x_seq = [SEQ_MAP.get(c, 0) for c in seq]
    if len(x_seq) > max_len:
        x_seq = x_seq[:max_len]
    else:
        x_seq = x_seq + [0] * (max_len - len(x_seq))
    return np.array([x_seq]), len(seq)

@st.cache_resource
def load_rna_model(weights_path):
    model = RNATRM(input_vocab=5, output_vocab=4, d=D_MODEL, max_len=MAX_LEN, n_rec=N_REC, T=3, Nsup=N_SUP)
    dummy_x = tf.zeros((1, MAX_LEN))
    model(dummy_x)
    try:
        model.load_weights(weights_path)
    except ValueError:
        try:
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        except Exception as e:
            return None
    except Exception as e:
        return None
    return model

# --- NOUVEAU : Fonctions Médicaments ---

def get_drug_from_pubchem(name):
    """Récupère le SMILES et l'image d'un médicament via PubChem."""
    try:
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            c = compounds[0]
            smiles = c.isomeric_smiles
            mol = Chem.MolFromSmiles(smiles)
            img = Draw.MolToImage(mol, size=(300, 150))
            return c.cid, smiles, img
    except Exception as e:
        return None, None, None
    return None, None, None

def predict_efficacy_dummy(rna_seq, drug_smiles):
    """
    SIMULATION : Calcule une efficacité basée sur des heuristiques simples
    car le modèle RNA actuel ne prend pas de médicament en entrée.
    """
    # On utilise un hash pour que le résultat soit constant pour une même paire (RNA, Drug)
    seed_val = hash(rna_seq + drug_smiles)
    random.seed(seed_val)
    
    # Simulation d'affinité entre 0% et 100%
    base_score = random.uniform(20, 95)
    
    # Petit bonus si l'ARN est riche en GC (plus stable)
    gc_content = (rna_seq.count('G') + rna_seq.count('C')) / len(rna_seq)
    final_score = base_score + (gc_content * 10)
    
    return min(99.9, max(0.1, final_score))

# ==============================================================================
# 3. INTERFACE UTILISATEUR
# ==============================================================================

st.set_page_config(page_title="SafeDrug AI: RNA & Interaction", page_icon="💊", layout="wide")

st.title("💊 SafeDrug AI : RNA & Drug Interaction")
st.markdown("Analyse de structure d'ARN et estimation d'efficacité thérapeutique.")

# Sidebar : Chargement Modèle
st.sidebar.header("1. Cerveau IA")
weights_file = st.sidebar.file_uploader("Charger les poids (.h5)", type=["h5", "keras"])

if weights_file:
    with open("temp_weights.h5", "wb") as f:
        f.write(weights_file.getbuffer())
    
    model = load_rna_model("temp_weights.h5")
    
    if model:
        st.sidebar.success("Modèle RNA chargé !")
        
        # --- Colonne Gauche : Séquence RNA ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧬 Séquence ARN")
            seq_input = st.text_area("Entrez la séquence cible", 
                                    value="GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCAC",
                                    height=150)
            
            x_val, original_len = preprocess_sequence(seq_input, MAX_LEN)
            
            # Prédiction Structure
            logits, _ = model(x_val, training=False)
            preds = np.argmax(logits, axis=-1)[0]
            limit = min(original_len, MAX_LEN)
            pred_str = "".join([REV_STRUCT_MAP.get(t, '.') for t in preds])[:limit]
            
            st.info(f"Structure prédite (Dot-Bracket) :\n{pred_str}")

        # --- Colonne Droite : Médicament ---
        with col2:
            st.subheader("💊 Médicament Candidat")
            drug_name = st.text_input("Nom du médicament (ex: Aspirin, Paclitaxel)", value="Caffeine")
            
            if drug_name:
                with st.spinner(f"Recherche de '{drug_name}' sur PubChem..."):
                    cid, smiles, img = get_drug_from_pubchem(drug_name)
                
                if cid:
                    st.image(img, caption=f"CID: {cid}")
                    st.caption(f"**SMILES:** `{smiles}`")
                else:
                    st.error("Médicament non trouvé sur PubChem.")
                    smiles = None

        st.divider()

        # --- Section : Interaction & Efficacité ---
        if st.button("Lancer l'analyse d'interaction", type="primary"):
            if smiles and seq_input:
                st.subheader("📊 Résultats de l'analyse")
                
                # 1. Visualisation ARN colorée
                html_view = "<div style='font-family: monospace; font-size: 1.2em; letter-spacing: 2px; margin-bottom: 20px;'>"
                for base, struc in zip(seq_input[:limit], pred_str):
                    color = "black"
                    if struc == '(': color = "#1E88E5" # Bleu
                    elif struc == ')': color = "#D32F2F" # Rouge
                    elif struc == '.': color = "#9E9E9E" # Gris
                    html_view += f"<span style='color: {color}; font-weight:bold;' title='{struc}'>{base}</span>"
                html_view += "</div>"
                st.markdown("#### Repliement de la cible :")
                st.markdown(html_view, unsafe_allow_html=True)
                
                # 2. Score d'efficacité (SIMULÉ)
                efficacy = predict_efficacy_dummy(seq_input, smiles)
                
                # Jauge d'efficacité
                col_res1, col_res2 = st.columns([1, 3])
                with col_res1:
                    st.metric(label="Efficacité Estimée", value=f"{efficacy:.1f} %")
                with col_res2:
                    st.progress(efficacy / 100)
                    if efficacy > 75:
                        st.success("Interaction forte détectée : Candidat prometteur.")
                    elif efficacy > 40:
                        st.warning("Interaction modérée : Optimisation requise.")
                    else:
                        st.error("Interaction faible : Molécule peu probable d'agir.")
                
                # Avertissement honnête pour le prof/jury
                st.warning("""
                ⚠️ **Note Technique :** Le pourcentage ci-dessus est une simulation pour la démo. 
                Le modèle `.h5` chargé prédit uniquement la structure de l'ARN. 
                Pour une vraie prédiction, il faudrait entraîner un réseau (ex: Graph Neural Network) prenant en entrée le SMILES et l'Embedding de l'ARN.
                """)
            else:
                st.error("Veuillez entrer une séquence valide et un nom de médicament correct.")

else:
    st.info("👈 Chargez d'abord le fichier de poids dans le menu.")