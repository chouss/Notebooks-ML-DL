# Système de Recommandation de Recettes

Ce projet est une application web interactive construite avec
**Streamlit** qui implémente un système de recommandation de recettes.\
Elle propose deux modes de découverte : un filtrage multi‑critères et un
moteur de recommandation hybride pondéré.

------------------------------------------------------------------------

## 🚀 Fonctionnalités

### 🔍 Page d'Exploration

Explorez des milliers de recettes avec : - Temps de cuisson\
- Calories\
- Protéines\
- Note moyenne\
- Tags (catégories)

------------------------------------------------------------------------

## 🧠 Méthodes de Recommandation

### **Méthode 1 -- Filtrage Basé sur les Contraintes**

Filtre strict sur : - Temps de cuisson maximum\
- Calories maximales\
- Protéines minimales\
- Note minimale\
- Tags / catégories

### **Méthode 2 -- Recommandation Hybride**

Un moteur utilisant une combinaison pondérée : - **50% Filtrage
Collaboratif** -- (Surprise `KNNWithMeans`)\
- **30% Filtrage de Contenu** -- (TF‑IDF + similarité cosinus)\
- **20% Popularité** -- (basé sur les avis récents)

------------------------------------------------------------------------

## 🧪 Page d'Évaluation

Validation croisée rapide sur un échantillon de 10 000 avis :\
➡️ RMSE / MAE du modèle collaboratif Surprise

------------------------------------------------------------------------

## 📦 Dataset Utilisé

Dataset : *Food.com Recipes and Reviews* (Kaggle)\
- 100k avis utilisés pour la recommandation temps réel\
- 10k avis pour l'évaluation

------------------------------------------------------------------------

## 🛠️ Technologies

-   **Streamlit** -- Application web\
-   **Pandas / PyArrow** -- Données & Parquet\
-   **scikit‑surprise** -- Filtrage collaboratif\
-   **Scikit‑learn** -- Filtrage de contenu\
-   **Anaconda** -- Gestion des dépendances

------------------------------------------------------------------------

## ⚙️ Installation et Exécution

### **Étape 1 : Télécharger les données**

Téléchargez :

    recipes.parquet
    reviews.parquet

Placez-les à la racine du projet.

------------------------------------------------------------------------

### **Étape 2 : Créer l'environnement Conda**

``` bash
conda create -n reco_env python=3.11
conda activate reco_env
```

------------------------------------------------------------------------

### **Étape 3 : Installer les dépendances**

Placez `app.py` et `requirements.txt` dans le dossier du projet.

Dans le terminal :

``` bash
cd C:\chemin\vers\MonProjetRecos
pip install -r requirements.txt
```

------------------------------------------------------------------------

### **Étape 4 : Lancer l'application**

``` bash
streamlit run app.py
```

L'application sera disponible à :\
👉 http://localhost:8501

------------------------------------------------------------------------

## 📁 Structure du Projet

    MonProjetRecos/
    │
    ├── app.py
    ├── requirements.txt
    ├── recipes.parquet
    └── reviews.parquet

------------------------------------------------------------------------

