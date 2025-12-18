# 🧬 SafeDrug AI - TRM (Transformer-based RNA Modeling)

> **Intelligence Artificielle pour la Découverte de Médicaments via Prédiction de Structure ARN**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table des Matières

- [À Propos](#-à-propos)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du Projet](#-structure-du-projet)
- [Technologies](#-technologies)
- [Contribution](#-contribution)
- [Licence](#-licence)

## 🎯 À Propos

**SafeDrug AI - TRM** est un système d'intelligence artificielle avancé conçu pour la découverte de médicaments en analysant les interactions entre les molécules thérapeutiques et les structures ARN. Le projet utilise un modèle Transformer récursif (TRM) pour prédire la structure secondaire de l'ARN, permettant ainsi d'identifier les candidats médicaments les plus prometteurs.

### Objectifs

- **Prédiction de Structure ARN** : Prédire la structure secondaire de séquences ARN avec une haute précision
- **Analyse d'Interaction** : Évaluer l'affinité entre des molécules thérapeutiques et des cibles ARN
- **Interface Interactive** : Fournir une interface web intuitive pour l'analyse en temps réel
- **Accélération de la Recherche** : Réduire le temps et les coûts de développement pharmaceutique

## ✨ Fonctionnalités

### 🧪 Prédiction de Structure ARN
- Modèle Transformer récursif (TRM) optimisé pour le repliement ARN
- Support de séquences jusqu'à 128 nucléotides
- Visualisation interactive 2D des structures prédites
- Encodage robuste des séquences ARN (A, C, G, U, N)

### 💊 Analyse de Médicaments
- Intégration avec la base de données **PubChem** pour récupérer les données moléculaires
- Extraction automatique des propriétés chimiques (formule, poids moléculaire, SMILES)
- Calcul de score d'affinité (binding score) pour évaluer les interactions
- Visualisation 2D des structures moléculaires

### 🖥️ Interface Web (Project Origami)
- Interface Streamlit moderne et responsive
- Analyse en temps réel avec feedback visuel
- Onglets séparés pour la visualisation ARN et l'analyse médicament
- Design professionnel avec support CSS personnalisé

## 🏗️ Architecture

### Modèle TRM (Transformer-based RNA Model)

Le modèle utilise une architecture Transformer récursive avec les composants suivants :

- **Embeddings** : Encodage des séquences ARN et des structures
- **Positional Encoding** : Encodage positionnel appris
- **Tiny Network** : Blocs de transformation légers (2 couches)
- **Récursion Latente** : Mécanisme de récursion interne (n itérations)
- **Récursion Externe** : Boucle de raffinement (T itérations)
- **Deep Supervision** : Supervision à plusieurs niveaux pour l'entraînement

### Pipeline de Prédiction

```
Séquence ARN → Encodage → Embeddings → TRM Model → Structure Prédite → Visualisation
```

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Git (pour cloner le dépôt)

### Étapes d'Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/MouadAgn/SafeDrug_AI_TRM.git
   cd SafeDrug_AI_TRM
   ```

2. **Créer un environnement virtuel** (recommandé)
   ```bash
   python -m venv venv
   
   # Sur Windows
   venv\Scripts\activate
   
   # Sur Linux/Mac
   source venv/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Télécharger le modèle pré-entraîné** (optionnel)
   
   Le modèle pré-entraîné doit être placé dans le dossier `saved_models/` :
   ```
   saved_models/
   └── trm_paper_version.weights.h5
   ```

## 💻 Utilisation

### Interface Web (Streamlit)

Lancer l'application web :

```bash
streamlit run frontend/app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

#### Utilisation de l'Interface

1. **Saisir une Séquence ARN** : Entrez la séquence ARN cible dans la zone de texte
2. **Entrer un Nom de Médicament** : Saisissez le nom du médicament à analyser (ex: "Aspirin")
3. **Lancer l'Analyse** : Cliquez sur "Lancer l'Analyse"
4. **Consulter les Résultats** :
   - **Onglet "Visualisation Cible"** : Structure 2D prédite de l'ARN
   - **Onglet "Analyse Médicament"** : Propriétés chimiques et score d'affinité

### Entraînement du Modèle

Pour entraîner le modèle sur vos propres données :

```bash
python src/train.py
```

**Paramètres configurables** (dans `src/train.py`) :
- `D_MODEL` : Dimension du modèle (défaut: 64)
- `N_SUP` : Nombre de pas de deep supervision (défaut: 3)
- `EPOCHS` : Nombre d'époques (défaut: 5)
- `BATCH_SIZE` : Taille du batch (défaut: 16)
- `MAX_LEN` : Longueur maximale des séquences (défaut: 128)

### Utilisation Programmatique

```python
from src.model import TRM_Official
from src.data_loader import RNA_MAP, STRUCT_MAP, MAX_LEN
import tensorflow as tf
import numpy as np

# Initialiser le modèle
model = TRM_Official(
    input_vocab_size=len(RNA_MAP),
    output_vocab_size=len(STRUCT_MAP),
    d=64,
    max_len=MAX_LEN,
    n=2,
    T=2
)

# Charger les poids pré-entraînés
model.load_weights('saved_models/trm_paper_version.weights.h5')

# Prédire une structure
sequence = "GCGGAUUUAGCUCAGDDGGGAGAGCGCCAGACUGAAYAAAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGC"
# ... (voir frontend/app.py pour l'exemple complet)
```

## 📁 Structure du Projet

```
SafeDrug_AI_TRM/
│
├── frontend/
│   └── app.py                 # Application Streamlit
│
├── src/
│   ├── model.py               # Architecture TRM
│   ├── train.py               # Script d'entraînement
│   └── data_loader.py         # Chargement et préprocessing des données
│
├── saved_models/              # Modèles pré-entraînés (ignoré par Git)
│   └── trm_paper_version.weights.h5
│
├── requirements.txt           # Dépendances Python
├── .gitignore                # Fichiers ignorés par Git
└── README.md                 # Documentation principale
```

## 🛠️ Technologies

### Backend & ML
- **TensorFlow 2.x** : Framework de deep learning
- **Keras** : API haut niveau pour TensorFlow
- **NumPy** : Calculs numériques
- **Pandas** : Manipulation de données

### Frontend
- **Streamlit** : Framework web pour applications ML
- **HTML/CSS/JavaScript** : Visualisation interactive (Forna.js)

### Données
- **HuggingFace Datasets** : Chargement du dataset bpRNA
- **PubChemPy** : API pour récupérer les données moléculaires

### Visualisation
- **Forna.js** : Visualisation 2D des structures ARN
- **Matplotlib** : Graphiques et visualisations

## 📊 Dataset

Le modèle est entraîné sur le dataset **bpRNA** disponible sur HuggingFace :
- **Source** : `multimolecule/bprna`
- **Format** : Séquences ARN avec structures secondaires annotées
- **Taille** : Configurable (défaut: 2000 séquences pour l'entraînement)

## 🔬 Méthodologie

### Encodage des Séquences

- **ARN** : `{'<PAD>': 0, 'A': 1, 'C': 2, 'G': 3, 'U': 4, 'N': 5}`
- **Structures** : `{'.': 1, '(': 2, ')': 3, '[': 4, ']': 5, '{': 6, '}': 7, '>': 8, '<': 9}`

### Entraînement

- **Loss Function** : Sparse Categorical Crossentropy avec masquage
- **Optimizer** : Adam (learning rate: 1e-3)
- **Gradient Clipping** : Global norm = 1.0
- **Deep Supervision** : Supervision à plusieurs niveaux pour améliorer la convergence

---

**Note** : Ce projet est en développement actif. Les fonctionnalités peuvent évoluer.

