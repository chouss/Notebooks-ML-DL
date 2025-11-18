import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import cross_validate

RECIPES_PATH = 'recipes.parquet'
REVIEWS_PATH = 'reviews.parquet'

# Échantillon pour l'application (plus grand pour éviter les erreurs de division par zéro)
APP_SAMPLE_SIZE = 1000 
# Échantillon pour l'évaluation (plus petit pour la vitesse, comme vous l'avez demandé)
EVAL_SAMPLE_SIZE = 1000

def parse_iso8601_duration(duration_str):
    if not isinstance(duration_str, str):
        return None
    hours = 0
    minutes = 0
    match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?', duration_str)
    if not match:
        return None
    if match.group(1):
        hours = int(match.group(1))
    if match.group(2):
        minutes = int(match.group(2))
    return (hours * 60) + minutes

@st.cache_data
def load_data(recipes_path, reviews_path, sample_size):
    if not os.path.exists(recipes_path) or not os.path.exists(reviews_path):
        st.error(f"Erreur : Fichiers de données non trouvés. Vérifiez {recipes_path} et {reviews_path}.")
        return None, None, [], []

    try:
        reviews_df = pd.read_parquet(reviews_path)
        if len(reviews_df) > sample_size:
            reviews_df = reviews_df.sample(n=sample_size, random_state=42)
    except Exception as e:
        st.error(f"Erreur lors du chargement de {reviews_path}: {e}")
        return None, None, [], []
    
    review_cols_needed = ['RecipeId', 'AuthorId', 'Rating', 'DateSubmitted']
    if not all(col in reviews_df.columns for col in review_cols_needed):
        st.error(f"Colonnes manquantes dans reviews.parquet. Requis : {review_cols_needed}")
        return None, None, [], []
        
    sampled_recipe_ids = reviews_df['RecipeId'].unique()
    
    recipe_cols_needed = [
        'RecipeId', 'Name', 'TotalTime', 'RecipeCategory', 'Description', 
        'AggregatedRating', 'ReviewCount', 'Calories', 'ProteinContent'
    ]
    
    try:
        recipes_table = pq.read_table(recipes_path, columns=recipe_cols_needed)
        recipes_df = recipes_table.filter(
            pa.compute.is_in(recipes_table['RecipeId'], pa.array(sampled_recipe_ids))
        ).to_pandas()
    except Exception as e:
        st.error(f"Erreur lors du chargement de {recipes_path}. Vérifiez les colonnes : {e}")
        return None, None, [], []

    recipes_df['minutes'] = recipes_df['TotalTime'].apply(parse_iso8601_duration)
    recipes_df['tags'] = recipes_df['RecipeCategory'].astype(str)
    recipes_df['Description'] = recipes_df['Description'].astype(str)
    recipes_df['Rating'] = pd.to_numeric(recipes_df['AggregatedRating'], errors='coerce')
    recipes_df['ReviewCount'] = pd.to_numeric(recipes_df['ReviewCount'], errors='coerce').fillna(0).astype(int)
    recipes_df['Calories'] = pd.to_numeric(recipes_df['Calories'], errors='coerce')
    recipes_df['ProteinContent'] = pd.to_numeric(recipes_df['ProteinContent'], errors='coerce')

    unique_users = sorted(reviews_df['AuthorId'].unique())
    unique_recipes = recipes_df[['RecipeId', 'Name']].sort_values('Name').set_index('RecipeId')
    
    return recipes_df, reviews_df, unique_users, unique_recipes

def recommend_by_constraints(df, use_time, max_time, use_tags, tags_str, 
                               use_calories, max_calories, use_protein, min_protein, 
                               use_rating, min_rating):
    results = df.copy()
    
    if use_time:
        results = results.dropna(subset=['minutes'])
        results = results[results['minutes'] <= max_time]
    
    if use_calories:
        results = results.dropna(subset=['Calories'])
        results = results[results['Calories'] <= max_calories]
        
    if use_protein:
        results = results.dropna(subset=['ProteinContent'])
        results = results[results['ProteinContent'] >= min_protein]
        
    if use_rating:
        results = results.dropna(subset=['Rating'])
        results = results[results['Rating'] >= min_rating]
    
    if use_tags and tags_str:
        required_tags = [tag.strip().lower() for tag in tags_str.split(',')]
        for tag in required_tags:
            if tag:
                results = results[results['tags'].str.contains(tag, case=False, na=False)]
            
    return results[['Name', 'minutes', 'tags', 'Calories', 'ProteinContent', 'Rating']].head(20)

@st.cache_resource
def get_hybrid_models(_recipes_df, _reviews_df):
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(_recipes_df['tags'].fillna(''))
    cosine_sim_content = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(_reviews_df[['AuthorId', 'RecipeId', 'Rating']], reader)
    trainset = data.build_full_trainset()
    
    sim_options = {'name': 'cosine', 'user_based': True, 'min_support': 3}
    cf_algo = KNNWithMeans(sim_options=sim_options)
    
    cf_algo.fit(trainset)
    
    _reviews_df['date'] = pd.to_datetime(_reviews_df['DateSubmitted'], errors='coerce')
    max_date = _reviews_df['date'].max()
    recent_date = max_date - pd.Timedelta(days=90)
    recent_reviews = _reviews_df[_reviews_df['date'] > recent_date]
    popularity_scores = recent_reviews.groupby('RecipeId').size().to_frame('pop_score')
    
    recipe_indices = pd.Series(_recipes_df.index, index=_recipes_df['RecipeId'])
    
    return cf_algo, cosine_sim_content, popularity_scores, recipe_indices

def get_hybrid_recommendations(user_id, ref_recipe_id, recipes_df, reviews_df, models, weights):
    cf_algo, cosine_sim, pop_scores, recipe_indices = models
    all_recipe_ids = recipes_df['RecipeId'].values
    rated_recipes = reviews_df[reviews_df['AuthorId'] == user_id]['RecipeId'].values
    
    ref_idx = recipe_indices.get(ref_recipe_id)
    if ref_idx is None:
        return pd.DataFrame()
    
    score_table = pd.DataFrame({'RecipeId': all_recipe_ids})
    score_table['cf_score'] = [cf_algo.predict(user_id, r_id).est for r_id in all_recipe_ids]
    content_sims = cosine_sim[ref_idx]
    
    score_table['content_score'] = score_table['RecipeId'].map(
        lambda r_id: content_sims[recipe_indices[r_id]] if r_id in recipe_indices else 0
    )
    
    score_table = score_table.merge(pop_scores, on='RecipeId', how='left').fillna(0)
    score_table = score_table[~score_table['RecipeId'].isin(rated_recipes)]
    
    scaler = MinMaxScaler()
    score_columns = ['cf_score', 'content_score', 'pop_score']
    for col in score_columns:
        if score_table[col].std() > 0:
            score_table[col] = scaler.fit_transform(score_table[[col]])
        else:
            score_table[col] = 0.0
            
    score_table['final_score'] = (
        weights['cf'] * score_table['cf_score'] +
        weights['content'] * score_table['content_score'] +
        weights['pop'] * score_table['pop_score']
    )
    
    score_table = score_table.merge(recipes_df[['RecipeId', 'Name']], on='RecipeId')
    final_cols = ['Name', 'final_score', 'cf_score', 'content_score', 'pop_score']
    
    return score_table.sort_values(by='final_score', ascending=False)[final_cols].head(10)

# --- FONCTION D'ÉVALUATION MODIFIÉE ---
@st.cache_resource
def run_evaluation(reviews_path, sample_size):
    st.write(f"Chargement d'un échantillon de {sample_size} avis pour l'évaluation...")
    try:
        reviews_df_full = pd.read_parquet(reviews_path, columns=['AuthorId', 'RecipeId', 'Rating'])
        
        # Applique l'échantillonnage
        if len(reviews_df_full) > sample_size:
            reviews_df_full = reviews_df_full.sample(n=sample_size, random_state=42)
            
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

    st.write("Données chargées. Préparation du format 'Surprise'...")
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(reviews_df_full[['AuthorId', 'RecipeId', 'Rating']], reader)

    sim_options = {'name': 'cosine', 'user_based': True, 'min_support': 3}
    algo = KNNWithMeans(sim_options=sim_options)

    st.write("Lancement de la validation croisée (3-folds)...")
    
    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=False, n_jobs=-1)
    
    st.write("Évaluation terminée !")
    return results
# --- FIN DE LA MODIFICATION ---


st.set_page_config(layout="wide")
st.title("🍲 Système de Recommandation de Recettes")

# Utilise APP_SAMPLE_SIZE pour l'application principale
recipes_df, reviews_df, user_list, recipe_list_dict = load_data(RECIPES_PATH, REVIEWS_PATH, APP_SAMPLE_SIZE)

if recipes_df is None or recipes_df.empty:
    st.error("Impossible de charger et de traiter les données. L'application s'arrête.")
    st.stop()

recipe_list = recipe_list_dict.to_dict().get('Name', {})

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisissez votre page :",
    ('Explorer les Recettes', 'Recommandation (Contraintes)', 'Recommandation (Hybride)', 'Évaluation du Modèle')
)

if page == 'Explorer les Recettes':
    st.header("Explorer les Recettes de l'Échantillon")
    st.info(f"Parcourez les {len(recipes_df)} recettes de notre échantillon (basé sur {APP_SAMPLE_SIZE} avis).")
    
    search_term = st.text_input("Rechercher par nom :").lower()
    
    if search_term:
        filtered_recipes_df = recipes_df[recipes_df['Name'].str.lower().contains(search_term, na=False)]
    else:
        filtered_recipes_df = recipes_df
    
    st.subheader(f"{len(filtered_recipes_df)} recettes affichées")
    
    st.dataframe(filtered_recipes_df[[
        'Name', 'minutes', 'tags', 'Rating', 'ReviewCount', 'Calories', 'ProteinContent'
    ]], use_container_width=True)
    
    st.subheader("Voir les détails d'une recette")
    selected_name = st.selectbox("Sélectionnez une recette :", filtered_recipes_df['Name'])
    
    if selected_name:
        recipe_details = filtered_recipes_df[filtered_recipes_df['Name'] == selected_name].iloc[0]
        
        st.write(f"### {recipe_details['Name']}")
        st.write(recipe_details['Description'])
        st.write(f"**Tags:** {recipe_details['tags']}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Note", f"{recipe_details['Rating']:.1f}/5", f"{recipe_details['ReviewCount']} avis")
        c2.metric("Temps", f"{recipe_details['minutes']:.0f} min" if pd.notna(recipe_details['minutes']) else "N/A")
        c3.metric("Calories", f"{recipe_details['Calories']:.0f}" if pd.notna(recipe_details['Calories']) else "N/A")
        c4.metric("Protéines", f"{recipe_details['ProteinContent']:.0f} g" if pd.notna(recipe_details['ProteinContent']) else "N/A")

elif page == 'Recommandation (Contraintes)':
    st.header("Méthode 1: Recommandation par contraintes")
    st.write("Cochez les contraintes que vous souhaitez appliquer.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_time = st.checkbox("Filtrer par temps", value=True)
        max_time = st.slider("Temps de cuisson max (minutes) :", 5, 600, 60, disabled=not use_time)
        
        use_calories = st.checkbox("Filtrer par calories", value=True)
        max_calories = st.slider("Calories Max :", 0, 3000, 1000, disabled=not use_calories)

    with col2:
        use_tags = st.checkbox("Filtrer par tags", value=True)
        tags = st.text_input("Tags requis (basé sur RecipeCategory) :", "easy", disabled=not use_tags)
        
        use_protein = st.checkbox("Filtrer par protéines", value=True)
        min_protein = st.slider("Protéines Min (g) :", 0, 100, 20, disabled=not use_protein)
        
    use_rating = st.checkbox("Filtrer par note", value=True)
    min_rating = st.slider("Note Minimale (0-5) :", 0.0, 5.0, 3.5, 0.5, disabled=not use_rating)

    if st.button("Trouver des recettes"):
        results = recommend_by_constraints(
            recipes_df, 
            use_time=use_time, max_time=max_time, 
            use_tags=use_tags, tags_str=tags,
            use_calories=use_calories, max_calories=max_calories,
            use_protein=use_protein, min_protein=min_protein,
            use_rating=use_rating, min_rating=min_rating
        )
        st.subheader(f"{len(results)} recettes trouvées :")
        
        for _, row in results.iterrows():
            st.write(f"**{row['Name']}** ({row['minutes']} min)")
            st.write(f"Note: {row['Rating']:.1f} | Calories: {row['Calories']:.0f} | Protéines: {row['ProteinContent']:.0f}g")
            st.write(f"Tags: {row['tags']}")
            st.markdown("---")

elif page == 'Recommandation (Hybride)':
    st.header("Méthode 2: Recommandation Hybride Pondérée")
    
    with st.spinner(f"Préparation des modèles (sur {APP_SAMPLE_SIZE} avis)..."):
        models = get_hybrid_models(recipes_df, reviews_df)
    
    weights = {'cf': 0.5, 'content': 0.3, 'pop': 0.2}
    
    c1, c2 = st.columns(2)
    with c1:
        user_id = st.selectbox("Pour quel utilisateur ?", user_list)
    with c2:
        recipe_id = st.selectbox("Recette de référence (consultée) :", recipe_list.keys(), format_func=lambda x: recipe_list.get(x, f"ID: {x}"))
        
    if st.button("Obtenir les recommandations hybrides"):
        with st.spinner("Calcul des scores hybrides..."):
            results = get_hybrid_recommendations(user_id, recipe_id, recipes_df, reviews_df, models, weights)
            st.subheader("Vos recommandations personnalisées :")
            
            for _, row in results.iterrows():
                st.write(f"**{row['Name']}** (Score: {row['final_score']:.3f})")
                st.write(f"Scores (CF: {row['cf_score']:.2f}, Contenu: {row['content_score']:.2f}, Pop: {row['pop_score']:.2f})")
                st.markdown("---")

elif page == 'Évaluation du Modèle':
    st.header("Évaluation Offline du Modèle de Filtrage Collaboratif")
    st.info(f"""
    Cette page évalue la performance du cœur de notre Méthode 2 (KNNWithMeans).
    Nous utilisons la validation croisée (3-folds) sur un **échantillon de {EVAL_SAMPLE_SIZE} avis**
    pour mesurer la précision des prédictions de notes.
    """)
    
    st.subheader("Lancement de l'évaluation...")
    st.warning("Ceci ne s'exécute qu'une fois. Le premier chargement sera rapide.")
    
    with st.spinner(f"Calcul des métriques (RMSE, MAE) sur {EVAL_SAMPLE_SIZE} avis..."):
        # Appelle la fonction d'évaluation avec l'échantillon de 10k
        evaluation_results = run_evaluation(REVIEWS_PATH, sample_size=EVAL_SAMPLE_SIZE)

    st.subheader("Résultats de l'évaluation (KNNWithMeans)")
    
    if evaluation_results:
        avg_rmse = pd.Series(evaluation_results['test_rmse']).mean()
        avg_mae = pd.Series(evaluation_results['test_mae']).mean()
        
        st.metric("RMSE Moyen (Root Mean Squared Error)", f"{avg_rmse:.4f}")
        st.metric("MAE Moyen (Mean Absolute Error)", f"{avg_mae:.4f}")
        
        st.markdown(f"""
        **Qu'est-ce que cela signifie ?**
        - **RMSE** : En moyenne, la prédiction de note du système (ex: 4.5 étoiles) se trompe d'environ **{avg_rmse:.4f}** étoiles par rapport à la note réelle donnée par l'utilisateur.
        - **Note :** Ces métriques sont basées sur un très petit échantillon de 10 000 avis. Elles peuvent ne pas être représentatives de la performance sur l'ensemble du dataset.
        """)
        
        st.subheader("Résultats détaillés (par fold)")
        st.dataframe(pd.DataFrame(evaluation_results))
    else:
        st.error("L'évaluation a échoué. Cela peut être dû à un échantillon trop petit (10k) qui provoque une `ZeroDivisionError`.")