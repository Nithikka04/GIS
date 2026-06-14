import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("master_dataset_with_GIS.csv")

print("Dataset Shape:", df.shape)

# =====================================================
# PREPARE FEATURES
# =====================================================

X = df.drop(columns=["Sample"])

# Standardize features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature Matrix Shape:", X_scaled.shape)

# =====================================================
# FIND BEST K
# =====================================================

print("\nEvaluating K values...")

results = []

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    sil_score = silhouette_score(
        X_scaled,
        labels
    )

    db_score = davies_bouldin_score(
        X_scaled,
        labels
    )

    results.append([
        k,
        sil_score,
        db_score
    ])

    print(
        f"K={k} | "
        f"Silhouette={sil_score:.4f} | "
        f"Davies-Bouldin={db_score:.4f}"
    )

results_df = pd.DataFrame(
    results,
    columns=[
        "K",
        "Silhouette",
        "Davies_Bouldin"
    ]
)

print("\nClustering Evaluation Results")
print(results_df)

# =====================================================
# SELECT BEST K
# =====================================================

best_k = results_df.loc[
    results_df["Silhouette"].idxmax(),
    "K"
]

best_k = int(best_k)

print("\nBest K selected:", best_k)

# =====================================================
# FINAL CLUSTERING
# =====================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

cluster_labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_labels

# =====================================================
# SAVE CLUSTERED DATASET
# =====================================================

df.to_csv(
    "clustered_dataset.csv",
    index=False
)

print("\nClustered dataset saved.")

# =====================================================
# CLUSTER DISTRIBUTION
# =====================================================

print("\nCluster Distribution")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)

# =====================================================
# PCA FOR VISUALIZATION
# =====================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame()

pca_df["PC1"] = X_pca[:, 0]
pca_df["PC2"] = X_pca[:, 1]
pca_df["Cluster"] = cluster_labels

print(
    "\nExplained Variance Ratio:"
)

print(
    pca.explained_variance_ratio_
)

# =====================================================
# SAVE PCA COORDINATES
# =====================================================

pca_df.to_csv(
    "pca_coordinates.csv",
    index=False
)

# =====================================================
# PCA SCATTER PLOT
# =====================================================

plt.figure(figsize=(10, 7))

for cluster in sorted(
    pca_df["Cluster"].unique()
):

    subset = pca_df[
        pca_df["Cluster"] == cluster
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Cluster {cluster}"
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(
    "PCA Visualization of Genomic Instability Clusters"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "cluster_pca_plot.png",
    dpi=300
)

plt.show()

# =====================================================
# CLUSTER-WISE GIS SUMMARY
# =====================================================

print("\nCluster GIS Statistics")

gis_summary = df.groupby(
    "Cluster"
)["GIS"].agg(
    [
        "count",
        "mean",
        "std",
        "min",
        "max"
    ]
)

print(gis_summary)

gis_summary.to_csv(
    "cluster_gis_summary.csv"
)

# =====================================================
# CLUSTER FEATURE MEANS
# =====================================================

cluster_means = (
    df.groupby("Cluster")
    .mean(numeric_only=True)
)

cluster_means.to_csv(
    "cluster_feature_means.csv"
)

print(
    "\nCluster feature means saved."
)

print("\nAnalysis Complete.")


df = pd.read_csv("clustered_dataset.csv")

outliers = df[df["Cluster"] == 1]

print(outliers[["Sample","GIS"]])

cluster_means = (
    df.groupby("Cluster")
    .mean(numeric_only=True)
)

difference = (
    cluster_means.loc[1]
    - cluster_means.loc[0]
)

difference = (
    difference.abs()
    .sort_values(ascending=False)
)

print(difference.head(15))