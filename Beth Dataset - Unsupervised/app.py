"""
AI Clustering & Insight Dashboard for BETH Dataset - Enhanced Version
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AI Clustering Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Color palette for clusters
CLUSTER_COLORS = px.colors.qualitative.Set2

# ==================== DATA FUNCTIONS ====================

@st.cache_data
def load_data():
    """Load and combine BETH dataset files"""
    try:
        df_train = pd.read_csv('labelled_training_data.csv')
        df_test = pd.read_csv('labelled_testing_data.csv')
        df_val = pd.read_csv('labelled_validation_data.csv')
        df = pd.concat([df_train, df_test, df_val], ignore_index=True)
        return df
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None

@st.cache_data
def encode_categorical_features(df, cat_columns):
    """Encode categorical features using Label Encoding or Frequency Encoding"""
    df_encoded = df.copy()
    encoding_info = {}
    
    for col in cat_columns:
        if col in df_encoded.columns:
            if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
                # Frequency encoding - replace category with its frequency
                freq_map = df_encoded[col].value_counts(normalize=True).to_dict()
                df_encoded[f'{col}_encoded'] = df_encoded[col].map(freq_map)
                
                # Also create label encoding for reference
                unique_vals = df_encoded[col].unique()
                label_map = {val: idx for idx, val in enumerate(unique_vals)}
                df_encoded[f'{col}_label'] = df_encoded[col].map(label_map)
                
                encoding_info[col] = {
                    'type': 'frequency',
                    'freq_map': freq_map,
                    'label_map': label_map,
                    'n_unique': len(unique_vals)
                }
    
    return df_encoded, encoding_info

@st.cache_data
def preprocess_data(df, sample_size, features_to_use):
    """Preprocess data for clustering"""
    if sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df.copy()
    
    X = df_sample[features_to_use].copy()
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, df_sample, scaler

# ==================== PCA FUNCTIONS ====================

@st.cache_data
def perform_full_pca(X_scaled, feature_names):
    """Perform full PCA analysis"""
    n_components = min(len(feature_names), X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    return pca, X_pca, loadings

@st.cache_data
def reduce_dimensions(X_scaled, method, n_components=2, perplexity=30):
    """Apply dimensionality reduction"""
    if method == "PCA":
        reducer = PCA(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
        explained_var = reducer.explained_variance_ratio_
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        X_reduced = reducer.fit_transform(X_scaled)
        explained_var = None
    return X_reduced, explained_var

# ==================== CLUSTERING FUNCTIONS ====================

def perform_clustering(X, algorithm, params):
    """Perform clustering with selected algorithm"""
    if algorithm == "K-Means":
        model = KMeans(n_clusters=params['n_clusters'], random_state=42, n_init=10)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    elif algorithm == "GMM":
        model = GaussianMixture(n_components=params['n_clusters'], random_state=42)
    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=params['n_clusters'])
    
    labels = model.fit_predict(X)
    return labels, model

def calculate_metrics(X, labels):
    """Calculate clustering evaluation metrics"""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum() if -1 in labels else 0
    
    if n_clusters < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None, 
                "n_clusters": n_clusters, "n_noise": n_noise}
    
    mask = labels != -1
    if mask.sum() < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
                "n_clusters": n_clusters, "n_noise": n_noise}
    
    try:
        sil = silhouette_score(X[mask], labels[mask])
        db = davies_bouldin_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])
    except:
        sil, db, ch = None, None, None
    
    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
            "n_clusters": n_clusters, "n_noise": n_noise}

# ==================== EVALUATION FUNCTIONS ====================

@st.cache_data
def evaluate_kmeans_elbow(X_scaled, max_k=15):
    """Elbow method for K-Means"""
    inertias = []
    silhouettes = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
    
    return list(K_range), inertias, silhouettes

@st.cache_data
def evaluate_all_algorithms(X_scaled, k_values=[3, 4, 5, 6, 7]):
    """Compare all algorithms across different k values"""
    results = []
    
    algorithms = {
        'K-Means': lambda k: KMeans(n_clusters=k, random_state=42, n_init=10),
        'GMM': lambda k: GaussianMixture(n_components=k, random_state=42),
        'Agglomerative': lambda k: AgglomerativeClustering(n_clusters=k)
    }
    
    for algo_name, algo_func in algorithms.items():
        for k in k_values:
            model = algo_func(k)
            labels = model.fit_predict(X_scaled)
            
            try:
                sil = silhouette_score(X_scaled, labels)
                db = davies_bouldin_score(X_scaled, labels)
                ch = calinski_harabasz_score(X_scaled, labels)
            except:
                sil, db, ch = None, None, None
            
            results.append({
                'Algorithm': algo_name,
                'K': k,
                'Silhouette': sil,
                'Davies-Bouldin': db,
                'Calinski-Harabasz': ch
            })
    
    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters >= 2:
            mask = labels != -1
            try:
                sil = silhouette_score(X_scaled[mask], labels[mask])
                db = davies_bouldin_score(X_scaled[mask], labels[mask])
                ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
            except:
                sil, db, ch = None, None, None
            
            results.append({
                'Algorithm': f'DBSCAN (eps={eps})',
                'K': n_clusters,
                'Silhouette': sil,
                'Davies-Bouldin': db,
                'Calinski-Harabasz': ch
            })
    
    return pd.DataFrame(results)

def compute_silhouette_samples_data(X, labels):
    """Compute silhouette values for each sample"""
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return None, None
    
    sample_silhouettes = silhouette_samples(X[mask], labels[mask])
    return sample_silhouettes, labels[mask]

# ==================== INSIGHT FUNCTIONS ====================

def generate_ai_insights(df_sample, labels, features):
    """Generate AI-powered insights about clusters"""
    df_temp = df_sample.copy()
    df_temp['Cluster'] = labels
    
    insights = []
    cluster_means = df_temp.groupby('Cluster')[features].mean()
    overall_means = df_temp[features].mean()
    overall_stds = df_temp[features].std()
    
    for cluster in sorted(cluster_means.index):
        if cluster == -1:
            insights.append({
                'cluster': -1,
                'title': 'Noise Points (Outliers)',
                'size': (labels == -1).sum(),
                'pct': (labels == -1).sum() / len(labels) * 100,
                'characteristics': ['Points that do not fit any cluster pattern'],
                'risk_level': 'Unknown'
            })
            continue
        
        cluster_size = (labels == cluster).sum()
        cluster_pct = cluster_size / len(labels) * 100
        
        z_scores = (cluster_means.loc[cluster] - overall_means) / overall_stds.replace(0, 1)
        
        high_features = z_scores.nlargest(3)
        low_features = z_scores.nsmallest(3)
        
        characteristics = []
        for feat, z in high_features.items():
            if z > 0.5:
                characteristics.append(f"High {feat} (z={z:.2f})")
        for feat, z in low_features.items():
            if z < -0.5:
                characteristics.append(f"Low {feat} (z={z:.2f})")
        
        risk_level = 'Low'
        if 'evil' in df_temp.columns:
            evil_rate = df_temp[df_temp['Cluster'] == cluster]['evil'].mean()
            if evil_rate > 0.3:
                risk_level = 'High'
            elif evil_rate > 0.1:
                risk_level = 'Medium'
        
        insights.append({
            'cluster': cluster,
            'title': f'Cluster {cluster}',
            'size': cluster_size,
            'pct': cluster_pct,
            'characteristics': characteristics[:5],
            'risk_level': risk_level
        })
    
    return insights

def interpret_pca_component(loadings, pc_num, feature_names, top_n=3):
    """Generate interpretation for a principal component"""
    pc_col = f'PC{pc_num}'
    pc_loadings = loadings[pc_col].sort_values(key=abs, ascending=False)
    
    positive_features = [(f, v) for f, v in pc_loadings.items() if v > 0.2][:top_n]
    negative_features = [(f, v) for f, v in pc_loadings.items() if v < -0.2][:top_n]
    
    interpretation = f"**PC{pc_num}** captures "
    
    if positive_features:
        pos_str = ", ".join([f"{f} (+{v:.2f})" for f, v in positive_features])
        interpretation += f"high values of {pos_str}"
    
    if positive_features and negative_features:
        interpretation += " versus "
    
    if negative_features:
        neg_str = ", ".join([f"{f} ({v:.2f})" for f, v in negative_features])
        interpretation += f"high values of {neg_str}"
    
    return interpretation

# ==================== MAIN APP ====================

def main():
    st.markdown('<h1 class="main-header">🔬 AI Clustering & Insight Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">BETH Network Security Dataset Analysis - Enhanced Version</p>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        st.stop()
    
    # ==================== SIDEBAR ====================
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("📊 Data Sampling")
    sample_size = st.sidebar.slider("Sample Size", 1000, min(50000, len(df)), 5000, step=1000)
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = ['processName', 'hostName', 'eventName']
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    # Encode categorical features
    df_encoded, encoding_info = encode_categorical_features(df, categorical_cols)
    
    # Add encoded columns to available features
    encoded_feature_cols = []
    for col in categorical_cols:
        if f'{col}_encoded' in df_encoded.columns:
            encoded_feature_cols.append(f'{col}_encoded')
    
    # All available features (numeric + encoded categorical)
    all_available_features = numeric_cols + encoded_feature_cols
    
    # Default features - exclude labels and IDs
    exclude_cols = ['evil', 'timestamp', 'processId', 'threadId', 'userId', 'sus']
    default_features = [c for c in all_available_features if c not in exclude_cols][:12]
    
    # Make sure mountNamespace and eventId are included if available
    priority_features = ['mountNamespace', 'eventId']
    for pf in priority_features:
        if pf in all_available_features and pf not in default_features:
            default_features.append(pf)
    
    st.sidebar.subheader("🎯 Feature Selection")
    
    features_to_use = st.sidebar.multiselect("Select Features", options=all_available_features, default=default_features)
    
    if len(features_to_use) < 2:
        st.warning("Please select at least 2 features.")
        st.stop()
    
    st.sidebar.subheader("📉 Dimensionality Reduction")
    dim_method = st.sidebar.selectbox("Method", ["PCA", "t-SNE"])
    
    perplexity = 30
    if dim_method == "t-SNE":
        perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
    
    st.sidebar.subheader("🔮 Clustering Algorithm")
    algorithm = st.sidebar.selectbox("Algorithm", ["K-Means", "DBSCAN", "GMM", "Agglomerative"])
    
    params = {}
    if algorithm in ["K-Means", "GMM", "Agglomerative"]:
        params['n_clusters'] = st.sidebar.slider("Number of Clusters (k)", 2, 15, 5)
    if algorithm == "DBSCAN":
        params['eps'] = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 0.1)
        params['min_samples'] = st.sidebar.slider("Min Samples", 2, 20, 5)
    
    # ==================== PROCESS DATA ====================
    with st.spinner("Processing data..."):
        X_scaled, df_sample, scaler = preprocess_data(df_encoded, sample_size, features_to_use)
        pca_full, X_pca_full, loadings = perform_full_pca(X_scaled, features_to_use)
        X_reduced, explained_var = reduce_dimensions(X_scaled, dim_method, 2, perplexity)
        labels, model = perform_clustering(X_scaled, algorithm, params)
        metrics = calculate_metrics(X_scaled, labels)
    
    # ==================== TABS ====================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Clustering", "📉 PCA Analysis", "🎯 Evaluation", 
        "📈 Algorithm Comparison", "🤖 AI Insights", "📋 Data Explorer"
    ])
    
    # ==================== TAB 1: CLUSTERING VIEW ====================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Clustering Visualization ({dim_method})")
            
            plot_df = pd.DataFrame({
                'Dim1': X_reduced[:, 0],
                'Dim2': X_reduced[:, 1],
                'Cluster': labels.astype(str)
            })
            
            fig = px.scatter(
                plot_df, x='Dim1', y='Dim2', color='Cluster',
                title=f'{algorithm} Clustering with {dim_method}',
                color_discrete_sequence=CLUSTER_COLORS,
                opacity=0.7
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            if dim_method == "PCA" and len(features_to_use) >= 3:
                with st.expander("🎮 3D PCA View"):
                    pca_3d = PCA(n_components=3, random_state=42)
                    X_3d = pca_3d.fit_transform(X_scaled)
                    
                    fig_3d = px.scatter_3d(
                        x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
                        color=labels.astype(str),
                        title="3D PCA Clustering View",
                        labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                        color_discrete_sequence=CLUSTER_COLORS,
                        opacity=0.7
                    )
                    fig_3d.update_layout(height=600)
                    st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            st.subheader("Quick Metrics")
            st.metric("Clusters", metrics['n_clusters'])
            if metrics['silhouette']:
                st.metric("Silhouette", f"{metrics['silhouette']:.3f}")
            if metrics['n_noise'] > 0:
                st.metric("Noise Points", metrics['n_noise'])
            
            st.subheader("Cluster Distribution")
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            fig_pie = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {i}" for i in cluster_counts.index],
                color_discrete_sequence=CLUSTER_COLORS
            )
            fig_pie.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # ==================== TAB 2: PCA ANALYSIS ====================
    with tab2:
        st.subheader("📉 Principal Component Analysis (PCA)")
        
        # Educational explanation
        with st.expander("📚 What is PCA? Understanding the Graphs", expanded=True):
            st.markdown("""
            ### What is PCA?
            **Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance (information) as possible.
            
            ### Key Concepts:
            
            **🔹 Principal Components (PCs)**
            - PCs are new axes created from linear combinations of original features
            - PC1 captures the most variance, PC2 the second most, and so on
            - They are orthogonal (perpendicular) to each other
            
            **🔹 Explained Variance**
            - Shows how much information each PC captures
            - Example: If PC1 = 45%, it means 45% of the data's variability is captured by PC1
            
            **🔹 Loadings**
            - Show how much each original feature contributes to each PC
            - High positive loading: feature increases along this PC
            - High negative loading: feature decreases along this PC
            
            ---
            
            ### Understanding the Graphs:
            
            | Graph | What it Shows | How to Interpret |
            |-------|---------------|------------------|
            | **Scree Plot** | Variance per PC | Look for the "elbow" - where variance drops significantly |
            | **Cumulative Variance** | Total variance captured | Choose enough PCs to capture 80-90% variance |
            | **Loadings Heatmap** | Feature-PC relationships | Dark colors = strong contribution |
            | **Biplot** | Data points + feature directions | Arrows show feature influence direction |
            
            ### For BETH Security Dataset:
            - High loadings on syscall-related features → PC captures system activity patterns
            - High loadings on network features → PC captures communication patterns
            """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scree Plot
            st.markdown("#### 📊 Scree Plot (Explained Variance)")
            n_pcs = min(10, len(pca_full.explained_variance_ratio_))
            
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(n_pcs)],
                y=pca_full.explained_variance_ratio_[:n_pcs] * 100,
                name='Individual Variance',
                marker_color='steelblue'
            ))
            fig_scree.add_trace(go.Scatter(
                x=[f'PC{i+1}' for i in range(n_pcs)],
                y=np.cumsum(pca_full.explained_variance_ratio_[:n_pcs]) * 100,
                name='Cumulative Variance',
                mode='lines+markers',
                marker_color='red',
                line=dict(width=3)
            ))
            fig_scree.add_hline(y=80, line_dash="dash", line_color="green", 
                               annotation_text="80% threshold")
            fig_scree.update_layout(
                yaxis_title='Explained Variance (%)',
                xaxis_title='Principal Component',
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig_scree, use_container_width=True)
            
            # Find number of components for 80% variance
            cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
            n_for_80 = np.argmax(cumvar >= 80) + 1
            st.info(f"📌 **{n_for_80} components** needed to capture 80% of variance")
        
        with col2:
            # Loadings Heatmap
            st.markdown("#### 🎨 PCA Loadings Heatmap")
            n_show = min(6, loadings.shape[1])
            
            fig_loadings = px.imshow(
                loadings.iloc[:, :n_show],
                labels=dict(x="Principal Component", y="Feature", color="Loading"),
                color_continuous_scale="RdBu_r",
                aspect="auto",
                text_auto='.2f'
            )
            fig_loadings.update_layout(height=400)
            st.plotly_chart(fig_loadings, use_container_width=True)
        
        # Variance Table
        st.markdown("#### 📋 Variance Explained Table")
        var_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(n_pcs)],
            'Eigenvalue': pca_full.explained_variance_[:n_pcs].round(3),
            'Variance (%)': (pca_full.explained_variance_ratio_[:n_pcs] * 100).round(2),
            'Cumulative (%)': (np.cumsum(pca_full.explained_variance_ratio_[:n_pcs]) * 100).round(2)
        })
        st.dataframe(var_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # PC Interpretations
        st.markdown("#### 🔍 Principal Component Interpretations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(interpret_pca_component(loadings, 1, features_to_use))
            st.markdown(f"*Explains {pca_full.explained_variance_ratio_[0]*100:.1f}% of variance*")
        
        with col2:
            st.markdown(interpret_pca_component(loadings, 2, features_to_use))
            st.markdown(f"*Explains {pca_full.explained_variance_ratio_[1]*100:.1f}% of variance*")
        
        st.markdown("---")
        
        # Biplot
        st.markdown("#### 🎯 PCA Biplot")
        st.markdown("*Biplot shows data points colored by cluster with feature loading vectors (arrows)*")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            scale_factor = st.slider("Arrow Scale", 1.0, 10.0, 3.0, 0.5)
            show_points = st.checkbox("Show Data Points", value=True)
            top_n_arrows = st.slider("Top N Features", 3, len(features_to_use), min(6, len(features_to_use)))
        
        with col1:
            fig_biplot = go.Figure()
            
            # Get top contributing features
            feature_importance = loadings[['PC1', 'PC2']].apply(lambda x: np.sqrt(x['PC1']**2 + x['PC2']**2), axis=1)
            top_features = feature_importance.nlargest(top_n_arrows).index.tolist()
            
            # Data points colored by cluster
            if show_points:
                unique_labels = sorted(set(labels))
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                    fig_biplot.add_trace(go.Scatter(
                        x=X_pca_full[mask, 0], 
                        y=X_pca_full[mask, 1],
                        mode='markers',
                        marker=dict(color=color, size=5, opacity=0.5),
                        name=f'Cluster {label}',
                        legendgroup=f'cluster_{label}'
                    ))
            
            # Loading vectors for top features
            for feature in top_features:
                idx = features_to_use.index(feature)
                x_end = loadings.iloc[idx, 0] * scale_factor
                y_end = loadings.iloc[idx, 1] * scale_factor
                
                fig_biplot.add_trace(go.Scatter(
                    x=[0, x_end],
                    y=[0, y_end],
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f'{feature}: ({loadings.iloc[idx, 0]:.3f}, {loadings.iloc[idx, 1]:.3f})'
                ))
                
                fig_biplot.add_annotation(
                    x=x_end, y=y_end,
                    text=feature,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowcolor='red',
                    font=dict(size=10, color='red'),
                    ax=20, ay=-20
                )
            
            fig_biplot.update_layout(
                xaxis_title=f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)',
                yaxis_title=f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)',
                height=550,
                title="Biplot: Data Points + Feature Loading Vectors",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_biplot, use_container_width=True)
        
        # Feature Contributions Bar Charts
        st.markdown("#### 📊 Top Feature Contributions per Component")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**PC1 Contributors**")
            pc1_contrib = loadings['PC1'].sort_values(key=abs, ascending=True).tail(6)
            colors = ['#ef5350' if v < 0 else '#42a5f5' for v in pc1_contrib.values]
            fig_pc1 = go.Figure(go.Bar(
                x=pc1_contrib.values, y=pc1_contrib.index, orientation='h',
                marker_color=colors
            ))
            fig_pc1.update_layout(height=250, xaxis_title='Loading', yaxis_title='')
            st.plotly_chart(fig_pc1, use_container_width=True)
        
        with col2:
            st.markdown("**PC2 Contributors**")
            pc2_contrib = loadings['PC2'].sort_values(key=abs, ascending=True).tail(6)
            colors = ['#ef5350' if v < 0 else '#42a5f5' for v in pc2_contrib.values]
            fig_pc2 = go.Figure(go.Bar(
                x=pc2_contrib.values, y=pc2_contrib.index, orientation='h',
                marker_color=colors
            ))
            fig_pc2.update_layout(height=250, xaxis_title='Loading', yaxis_title='')
            st.plotly_chart(fig_pc2, use_container_width=True)
        
        # Correlation Circle
        st.markdown("#### 🔵 Correlation Circle")
        st.markdown("*Shows how features are correlated with PC1 and PC2. Features close together are positively correlated.*")
        
        fig_circle = go.Figure()
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        fig_circle.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode='lines', line=dict(color='gray', dash='dash'),
            showlegend=False
        ))
        
        # Plot feature loadings as points
        for i, feature in enumerate(features_to_use):
            corr_pc1 = loadings.iloc[i, 0] * np.sqrt(pca_full.explained_variance_[0])
            corr_pc2 = loadings.iloc[i, 1] * np.sqrt(pca_full.explained_variance_[1])
            
            # Normalize to unit circle
            norm = np.sqrt(corr_pc1**2 + corr_pc2**2)
            if norm > 0:
                corr_pc1_norm = corr_pc1 / max(norm, 1)
                corr_pc2_norm = corr_pc2 / max(norm, 1)
            else:
                corr_pc1_norm, corr_pc2_norm = 0, 0
            
            fig_circle.add_trace(go.Scatter(
                x=[0, corr_pc1_norm], y=[0, corr_pc2_norm],
                mode='lines+markers+text',
                line=dict(color='steelblue', width=2),
                marker=dict(size=[0, 8], color='steelblue'),
                text=['', feature],
                textposition='top right',
                showlegend=False,
                hoverinfo='text',
                hovertext=f'{feature}'
            ))
        
        fig_circle.update_layout(
            xaxis=dict(range=[-1.2, 1.2], title='PC1', zeroline=True, zerolinecolor='lightgray'),
            yaxis=dict(range=[-1.2, 1.2], title='PC2', zeroline=True, zerolinecolor='lightgray', scaleanchor='x'),
            height=500,
            title="Correlation Circle"
        )
        st.plotly_chart(fig_circle, use_container_width=True)
    
    # ==================== TAB 3: EVALUATION ====================
    with tab3:
        st.subheader("🎯 Clustering Evaluation")
        
        # Explanation
        with st.expander("📚 Understanding Evaluation Metrics", expanded=False):
            st.markdown("""
            ### Clustering Evaluation Metrics Explained
            
            | Metric | Range | Optimal | What it Measures |
            |--------|-------|---------|------------------|
            | **Silhouette Score** | -1 to 1 | Higher (→1) | How similar points are to their own cluster vs others |
            | **Davies-Bouldin Index** | 0 to ∞ | Lower (→0) | Average similarity between clusters (lower = more separated) |
            | **Calinski-Harabasz** | 0 to ∞ | Higher | Ratio of between-cluster to within-cluster variance |
            | **Inertia** | 0 to ∞ | Lower | Sum of squared distances to cluster centers (K-Means) |
            
            ### Elbow Method
            The elbow method helps find optimal k by plotting inertia vs k. The "elbow" point where the curve bends indicates a good k value.
            
            ### Silhouette Analysis
            - **Values near +1**: Points are well-matched to their cluster
            - **Values near 0**: Points are on cluster boundaries
            - **Values near -1**: Points may be in wrong cluster
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Elbow Method & Silhouette Curve")
            with st.spinner("Computing elbow curve..."):
                k_range, inertias, silhouettes = evaluate_kmeans_elbow(X_scaled, max_k=12)
            
            fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
            fig_elbow.add_trace(
                go.Scatter(x=k_range, y=inertias, mode='lines+markers', name='Inertia', 
                          line=dict(color='blue', width=2)),
                secondary_y=False
            )
            fig_elbow.add_trace(
                go.Scatter(x=k_range, y=silhouettes, mode='lines+markers', name='Silhouette', 
                          line=dict(color='red', width=2)),
                secondary_y=True
            )
            fig_elbow.update_xaxes(title_text="Number of Clusters (k)")
            fig_elbow.update_yaxes(title_text="Inertia", secondary_y=False, color='blue')
            fig_elbow.update_yaxes(title_text="Silhouette Score", secondary_y=True, color='red')
            fig_elbow.update_layout(height=400, title="Elbow Method & Silhouette Score")
            st.plotly_chart(fig_elbow, use_container_width=True)
            
            optimal_k = k_range[np.argmax(silhouettes)]
            st.success(f"📌 **Suggested optimal k** (highest silhouette): **{optimal_k}**")
        
        with col2:
            st.markdown("#### 📊 Silhouette Distribution")
            
            sil_samples, sil_labels = compute_silhouette_samples_data(X_scaled, labels)
            
            if sil_samples is not None:
                sil_df = pd.DataFrame({'Silhouette': sil_samples, 'Cluster': sil_labels.astype(str)})
                
                fig_sil = px.histogram(
                    sil_df, x='Silhouette', color='Cluster',
                    nbins=50, barmode='overlay',
                    color_discrete_sequence=CLUSTER_COLORS,
                    title="Silhouette Score Distribution by Cluster"
                )
                if metrics['silhouette']:
                    fig_sil.add_vline(x=metrics['silhouette'], line_dash="dash", 
                                     annotation_text=f"Mean: {metrics['silhouette']:.3f}")
                fig_sil.update_layout(height=400)
                st.plotly_chart(fig_sil, use_container_width=True)
            else:
                st.warning("Cannot compute silhouette for current clustering.")
        
        # Silhouette per cluster bar chart
        if sil_samples is not None:
            st.markdown("#### 📊 Mean Silhouette Score per Cluster")
            sil_by_cluster = sil_df.groupby('Cluster')['Silhouette'].mean().sort_index()
            
            fig_sil_bar = px.bar(
                x=[f"Cluster {c}" for c in sil_by_cluster.index],
                y=sil_by_cluster.values,
                color=sil_by_cluster.values,
                color_continuous_scale='RdYlGn',
                title="Mean Silhouette Score by Cluster"
            )
            if metrics['silhouette']:
                fig_sil_bar.add_hline(y=metrics['silhouette'], line_dash="dash", 
                                      annotation_text="Overall Mean")
            fig_sil_bar.update_layout(height=350, showlegend=False, xaxis_title='', yaxis_title='Silhouette Score')
            st.plotly_chart(fig_sil_bar, use_container_width=True)
        
        # Current clustering metrics summary
        st.markdown("#### 📋 Current Clustering Metrics Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Clusters Found", metrics['n_clusters'])
        with col2:
            sil_val = f"{metrics['silhouette']:.4f}" if metrics['silhouette'] else "N/A"
            st.metric("Silhouette Score", sil_val)
        with col3:
            db_val = f"{metrics['davies_bouldin']:.4f}" if metrics['davies_bouldin'] else "N/A"
            st.metric("Davies-Bouldin Index", db_val)
        with col4:
            ch_val = f"{metrics['calinski_harabasz']:.1f}" if metrics['calinski_harabasz'] else "N/A"
            st.metric("Calinski-Harabasz", ch_val)
    
    # ==================== TAB 4: ALGORITHM COMPARISON ====================
    with tab4:
        st.subheader("📈 Algorithm Comparison")
        
        with st.spinner("Evaluating all algorithms (this may take a moment)..."):
            comparison_df = evaluate_all_algorithms(X_scaled, k_values=[3, 4, 5, 6, 7])
        
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_metric = st.selectbox("Select Metric to Compare", 
                                          ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"])
        
        # Comparison chart
        fig_compare = px.bar(
            comparison_df.dropna(subset=[selected_metric]),
            x='Algorithm', y=selected_metric, color='K',
            barmode='group',
            title=f'{selected_metric} Score by Algorithm and K',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_compare.update_layout(height=450)
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Best configurations table
        st.markdown("#### 🏆 Top 10 Configurations")
        
        if selected_metric == "Davies-Bouldin":
            best_configs = comparison_df.dropna(subset=[selected_metric]).nsmallest(10, selected_metric)
        else:
            best_configs = comparison_df.dropna(subset=[selected_metric]).nlargest(10, selected_metric)
        
        st.dataframe(
            best_configs.round(4).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
        
        # Heatmap comparison
        st.markdown("#### 🔥 Algorithm Performance Heatmap")
        
        # Pivot for heatmap (only for algorithms with K)
        pivot_df = comparison_df[~comparison_df['Algorithm'].str.contains('DBSCAN')].copy()
        if not pivot_df.empty:
            heatmap_data = pivot_df.pivot(index='Algorithm', columns='K', values=selected_metric)
            
            fig_heat = px.imshow(
                heatmap_data,
                labels=dict(x="Number of Clusters (K)", y="Algorithm", color=selected_metric),
                color_continuous_scale='RdYlGn' if selected_metric != 'Davies-Bouldin' else 'RdYlGn_r',
                text_auto='.3f',
                aspect='auto'
            )
            fig_heat.update_layout(height=300)
            st.plotly_chart(fig_heat, use_container_width=True)
        
        # Full results table
        with st.expander("📋 Full Comparison Results"):
            st.dataframe(comparison_df.round(4), use_container_width=True, hide_index=True)
    
    # ==================== TAB 5: AI INSIGHTS ====================
    with tab5:
        st.subheader("🤖 AI-Generated Cluster Insights")
        
        # Cluster insights
        insights = generate_ai_insights(df_sample, labels, features_to_use)
        
        # Cluster characteristic cards
        st.markdown("### 🎴 Cluster Profiles")
        cols = st.columns(min(3, len(insights)))
        for i, insight in enumerate(insights):
            with cols[i % 3]:
                risk_color = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴', 'Unknown': '⚪'}
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                            padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4>{insight['title']}</h4>
                    <p><strong>Size:</strong> {insight['size']} ({insight['pct']:.1f}%)</p>
                    <p><strong>Risk:</strong> {risk_color.get(insight['risk_level'], '⚪')} {insight['risk_level']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if insight['characteristics']:
                    st.markdown("**Characteristics:**")
                    for char in insight['characteristics']:
                        st.markdown(f"- {char}")
        
        st.markdown("---")
        
        # Security context for BETH
        st.markdown("### 🔒 Security Analysis (BETH Dataset)")
        if 'evil' in df_sample.columns:
            df_temp = df_sample.copy()
            df_temp['Cluster'] = labels
            evil_by_cluster = df_temp.groupby('Cluster')['evil'].agg(['mean', 'sum', 'count'])
            evil_by_cluster.columns = ['Malicious Rate', 'Malicious Count', 'Total Count']
            evil_by_cluster['Malicious Rate'] = evil_by_cluster['Malicious Rate'] * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_evil = px.bar(
                    x=[f"Cluster {c}" for c in evil_by_cluster.index],
                    y=evil_by_cluster['Malicious Rate'].values,
                    title="Malicious Activity Rate by Cluster (%)",
                    color=evil_by_cluster['Malicious Rate'].values,
                    color_continuous_scale='Reds'
                )
                fig_evil.update_layout(height=400, showlegend=False, xaxis_title='', yaxis_title='Malicious Rate (%)')
                st.plotly_chart(fig_evil, use_container_width=True)
            
            with col2:
                st.markdown("#### Cluster Security Summary")
                st.dataframe(evil_by_cluster.round(2), use_container_width=True)
                
                high_risk = evil_by_cluster[evil_by_cluster['Malicious Rate'] > evil_by_cluster['Malicious Rate'].mean()]
                if not high_risk.empty:
                    st.warning(f"⚠️ **High-risk clusters:** {list(high_risk.index)}")
        
        # Feature profiles heatmap
        st.markdown("### 📊 Cluster Feature Profiles")
        df_temp = df_sample.copy()
        df_temp['Cluster'] = labels
        cluster_means = df_temp.groupby('Cluster')[features_to_use].mean()
        
        # Normalize for better visualization
        cluster_means_norm = (cluster_means - cluster_means.mean()) / cluster_means.std()
        
        fig_profile = px.imshow(
            cluster_means_norm.T,
            labels=dict(x="Cluster", y="Feature", color="Z-Score"),
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Normalized Feature Values by Cluster (Z-Score)"
        )
        fig_profile.update_layout(height=450)
        st.plotly_chart(fig_profile, use_container_width=True)
        
        # Radar chart
        st.markdown("### 🎯 Cluster Radar Chart")
        top_features = features_to_use[:min(8, len(features_to_use))]
        
        scaler = MinMaxScaler()
        radar_data = cluster_means[top_features].copy()
        radar_scaled = pd.DataFrame(
            scaler.fit_transform(radar_data),
            index=radar_data.index,
            columns=radar_data.columns
        )
        
        fig_radar = go.Figure()
        for cluster in radar_scaled.index:
            if cluster != -1:
                values = radar_scaled.loc[cluster].values.tolist()
                values.append(values[0])  # Close the polygon
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=top_features + [top_features[0]],
                    name=f'Cluster {cluster}',
                    fill='toself',
                    opacity=0.6
                ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500,
            title="Cluster Profiles (Normalized)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # ==================== TAB 6: DATA EXPLORER ====================
    with tab6:
        st.subheader("📋 Data Explorer")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Sample Size", f"{len(df_sample):,}")
        with col2:
            st.metric("Features Used", len(features_to_use))
            st.metric("Algorithm", algorithm)
        with col3:
            st.metric("Dim. Reduction", dim_method)
            st.metric("Clusters Found", metrics['n_clusters'])
        
        st.markdown("---")
        
        # Categorical encoding details section
        if encoding_info:
            st.markdown("### 🏷️ Categorical Feature Encoding")
            st.markdown("""
            **Encoding Method: Frequency Encoding**  
            Categorical features are encoded by their frequency (proportion) in the dataset.
            This preserves the relative importance of each category.
            """)
            
            # Display encoding info cards
            col1, col2, col3 = st.columns(3)
            for idx, (col, info) in enumerate(encoding_info.items()):
                with [col1, col2, col3][idx % 3]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                                padding: 15px; border-radius: 10px; margin: 5px 0; border-left: 4px solid #00d4ff;">
                        <h4 style="color: #00d4ff; margin: 0;">{col}</h4>
                        <p style="color: #ffffff; margin: 5px 0;">→ <code>{col}_encoded</code></p>
                        <p style="color: #aaaaaa; margin: 0;">Unique values: <strong>{info['n_unique']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed encoding tables
            with st.expander("📊 View Encoding Details (Top 10 per Feature)"):
                for col, info in encoding_info.items():
                    st.markdown(f"**{col}** ({info['n_unique']} unique values)")
                    top_values = sorted(info['freq_map'].items(), key=lambda x: x[1], reverse=True)[:10]
                    top_df = pd.DataFrame(top_values, columns=[col, 'Frequency (Encoded Value)'])
                    top_df['Frequency (Encoded Value)'] = top_df['Frequency (Encoded Value)'].apply(lambda x: f"{x:.6f}")
                    st.dataframe(top_df, use_container_width=True, hide_index=True)
                    st.markdown("---")
            
            st.markdown("---")
        
        # Filter by cluster
        st.markdown("#### 🔍 Filter Data by Cluster")
        selected_clusters = st.multiselect(
            "Select Clusters to View",
            options=sorted(set(labels)),
            default=sorted(set(labels))
        )
        
        df_display = df_sample.copy()
        df_display['Cluster'] = labels
        df_filtered = df_display[df_display['Cluster'].isin(selected_clusters)]
        
        st.markdown(f"**Showing {len(df_filtered):,} records**")
        st.dataframe(df_filtered.head(500), use_container_width=True, height=400)
        
        # Download options
        st.markdown("#### 📥 Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_full = df_display.to_csv(index=False)
            st.download_button(
                "📥 Download Full Clustered Data",
                csv_full,
                "clustered_data_full.csv",
                "text/csv"
            )
        
        with col2:
            csv_filtered = df_filtered.to_csv(index=False)
            st.download_button(
                "📥 Download Filtered Data",
                csv_filtered,
                "clustered_data_filtered.csv",
                "text/csv"
            )
        
        # Statistics
        st.markdown("#### 📊 Feature Statistics by Cluster")
        stats_feature = st.selectbox("Select Feature", features_to_use)
        
        fig_box = px.box(
            df_display, x='Cluster', y=stats_feature,
            color='Cluster',
            color_discrete_sequence=CLUSTER_COLORS,
            title=f"Distribution of {stats_feature} by Cluster"
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

if __name__ == "__main__":
    main()