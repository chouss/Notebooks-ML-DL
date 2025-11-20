# 🔬 AI Clustering & Insight Dashboard

An interactive **unsupervised machine learning dashboard** built with Streamlit for exploring, clustering, and visualizing the BETH (Background Events from Threat Hunting) network security dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Dashboard Tabs](#-dashboard-tabs)
- [Algorithms](#-algorithms)
- [Evaluation Metrics](#-evaluation-metrics)
- [Screenshots](#-screenshots)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Authors](#-authors)

---

## 🎯 Overview

This project implements an **AI-powered clustering dashboard** that enables users to:

- Perform **unsupervised clustering** on network security data
- Apply **dimensionality reduction** (PCA, t-SNE) for visualization
- **Compare multiple algorithms** with various hyperparameters
- Generate **AI-powered insights** about cluster characteristics
- Identify **security threats** through behavioral pattern analysis

The dashboard is designed for the **BETH Dataset** from Kaggle, which contains labeled host-based intrusion detection data.

---

## ✨ Features

### 🔮 Clustering Algorithms
- **K-Means** - Centroid-based clustering
- **DBSCAN** - Density-based spatial clustering
- **GMM** - Gaussian Mixture Models (probabilistic)
- **Agglomerative** - Hierarchical clustering

### 📉 Dimensionality Reduction
- **PCA** (Principal Component Analysis) - Linear projection
- **t-SNE** - Non-linear embedding for visualization

### 📊 Comprehensive PCA Analysis
- **Scree Plot** - Explained variance per component
- **Cumulative Variance** - Total information captured
- **Loadings Heatmap** - Feature-component relationships
- **Biplot** - Data points with feature vectors
- **Correlation Circle** - Feature correlations visualization
- **Component Interpretation** - Automatic PC explanations

### 🎯 Evaluation & Comparison
- **Elbow Method** - Optimal K detection
- **Silhouette Analysis** - Cluster quality assessment
- **Davies-Bouldin Index** - Cluster separation metric
- **Calinski-Harabasz Index** - Variance ratio criterion
- **Multi-algorithm Comparison** - Side-by-side benchmarking

### 🤖 AI-Powered Insights
- Automatic cluster profiling with Z-score analysis
- Security risk assessment for each cluster
- Feature importance identification
- Radar charts for cluster comparison
- **Categorical feature analysis per cluster**
- **Dominant category identification**
- **Category distribution heatmaps**

### 🏷️ Categorical Feature Encoding
- **Frequency Encoding** for text/categorical columns
- Support for: `processName`, `hostName`, `eventName`
- Automatic encoding info display
- Per-cluster category distribution analysis

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
```bash
# Create project directory
mkdir clustering-dashboard
cd clustering-dashboard
```

### Step 2: Download BETH Dataset
Download the dataset from Kaggle:
- [BETH Dataset on Kaggle](https://www.kaggle.com/datasets/katehighnam/beth-dataset)

Place these files in the project directory:
- `labelled_training_data.csv`
- `labelled_testing_data.csv`
- `labelled_validation_data.csv`

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas numpy plotly scikit-learn
```

### Step 4: Run the Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 💻 Usage

### Sidebar Configuration

| Option | Description |
|--------|-------------|
| **Sample Size** | Number of data points to analyze (1,000 - 50,000) |
| **Feature Selection** | Choose which numeric features to include |
| **Dim. Reduction** | Select PCA or t-SNE |
| **Algorithm** | Choose clustering algorithm |
| **Hyperparameters** | Adjust k, epsilon, min_samples, etc. |

### Workflow

1. **Load Data** - Dashboard automatically loads BETH dataset
2. **Configure** - Adjust settings in sidebar
3. **Explore** - Navigate through tabs to analyze results
4. **Compare** - Use evaluation tab to find optimal parameters
5. **Export** - Download clustered data for further analysis

---

## 📁 Dataset

### BETH Dataset (Background Events from Threat Hunting)

The BETH dataset contains **host-based intrusion detection data** collected from honeypots. It includes system call information and behavioral patterns labeled as malicious or benign.

### Key Features

| Feature | Description |
|---------|-------------|
| `processId` | Process identifier |
| `threadId` | Thread identifier |
| `parentProcessId` | Parent process ID |
| `userId` | User identifier |
| `mountNamespace` | Mount namespace ID |
| `processName` | Name of the process |
| `hostName` | Host machine name |
| `eventId` | Event type identifier |
| `eventName` | Name of the event |
| `stackAddresses` | Stack trace addresses |
| `argsNum` | Number of arguments |
| `returnValue` | System call return value |
| `sus` | Suspicious activity score |
| `evil` | **Label** (1 = malicious, 0 = benign) |

### Dataset Statistics
- **Training Set**: ~1M records
- **Testing Set**: ~200K records
- **Validation Set**: ~100K records

---

## 📑 Dashboard Tabs

### Tab 1: 📊 Clustering View
- 2D scatter plot of clustered data
- 3D PCA visualization (optional)
- Cluster distribution pie chart
- Quick metrics summary

### Tab 2: 📉 PCA Analysis
- **Educational explanations** of PCA concepts
- Scree plot with 80% variance threshold
- Loadings heatmap
- Interactive biplot with adjustable arrows
- Correlation circle
- Feature contribution bar charts

### Tab 3: 🎯 Evaluation
- Elbow method curve
- Silhouette score analysis
- Per-cluster silhouette distribution
- Optimal K recommendation
- All metrics summary

### Tab 4: 📈 Algorithm Comparison
- Multi-algorithm benchmarking
- Performance heatmap
- Top configurations ranking
- Interactive metric selection

### Tab 5: 🤖 AI Insights
- Cluster characteristic cards
- Security risk assessment
- Malicious activity rate by cluster
- Feature profile heatmap
- Radar chart comparison

### Tab 6: 📋 Data Explorer
- Filter data by cluster
- View raw data with cluster labels
- Download clustered datasets
- Feature distribution box plots

---

## 🔧 Algorithms

### K-Means
```
Objective: Minimize within-cluster sum of squares
Parameters: n_clusters (k)
Best for: Spherical clusters of similar size
```

### DBSCAN
```
Objective: Find dense regions separated by sparse regions
Parameters: eps (neighborhood radius), min_samples
Best for: Arbitrary shaped clusters, outlier detection
```

### Gaussian Mixture Models (GMM)
```
Objective: Maximize likelihood of Gaussian mixture
Parameters: n_components
Best for: Overlapping clusters, soft assignments
```

### Agglomerative Clustering
```
Objective: Build hierarchy by merging closest clusters
Parameters: n_clusters, linkage method
Best for: Hierarchical structure discovery
```

---

## 📏 Evaluation Metrics

| Metric | Range | Optimal | Interpretation |
|--------|-------|---------|----------------|
| **Silhouette Score** | [-1, 1] | → 1 | How well points match their cluster vs others |
| **Davies-Bouldin** | [0, ∞) | → 0 | Average similarity ratio between clusters |
| **Calinski-Harabasz** | [0, ∞) | → ∞ | Between-cluster / within-cluster variance |
| **Inertia** | [0, ∞) | → 0 | Sum of squared distances to centroids |

### Interpretation Guide

- **Silhouette > 0.5**: Strong cluster structure
- **Silhouette 0.25-0.5**: Moderate structure
- **Silhouette < 0.25**: Weak or overlapping clusters
- **Davies-Bouldin < 1**: Good cluster separation

---

## 📸 Screenshots

### Clustering Visualization
```
┌─────────────────────────────────────────┐
│  🔬 AI Clustering & Insight Dashboard   │
├─────────────────────────────────────────┤
│  [Scatter Plot: Clustered Data Points]  │
│                                         │
│  Clusters: 5    Silhouette: 0.342      │
└─────────────────────────────────────────┘
```

### PCA Biplot
```
┌─────────────────────────────────────────┐
│  PCA Biplot                             │
│  ↗ feature1                             │
│    ↘ feature2    • • •  •              │
│        • • • • • • •  • •              │
│  ← feature3  • • • • • • →             │
│              • • • • •                  │
└─────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
clustering-dashboard/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── labelled_training_data.csv      # BETH training data
├── labelled_testing_data.csv       # BETH testing data
└── labelled_validation_data.csv    # BETH validation data
```

---

## 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Programming language |
| **Streamlit** | Web dashboard framework |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Plotly** | Interactive visualizations |
| **scikit-learn** | Machine learning algorithms |

---

## 📚 References

1. **BETH Dataset**: Highnam, K., et al. (2021). "BETH Dataset for Host-Based Intrusion Detection"
2. **PCA**: Jolliffe, I.T. (2002). "Principal Component Analysis"
3. **K-Means**: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
4. **DBSCAN**: Ester, M., et al. (1996). "A density-based algorithm for discovering clusters"
5. **Silhouette Score**: Rousseeuw, P.J. (1987). "Silhouettes: a graphical aid to interpretation"

---

## 👥 Authors

- **Student Name** - *Initial work* - Data Mining Project

### Course Information
- **Module**: Data Mining
- **Instructor**: Dr-ing Rym Besrour
- **Academic Year**: 2025/2026

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Kaggle for hosting the BETH dataset
- Streamlit team for the amazing framework
- scikit-learn contributors for ML algorithms
- Plotly team for interactive visualizations

---

<p align="center">
  Made with ❤️ for Data Mining Course
</p>