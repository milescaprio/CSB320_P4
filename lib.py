# flake8: noqa: F401

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from IPython.display import display
from joblib import Memory
from scipy.cluster.hierarchy import dendrogram

memory = Memory(location="./cache_dir", verbose=0)


# Visualize PCA in 2D with outliers in a different color
# PCA is fitted on both inliers and outliers (because it is for visualization purposes)
def visualize_pca_2d_with_outliers(
    df_all_scaled, df_inliers_scaled, df_outliers_scaled
):
    # Fit PCA on the scaled inlier data (excluding outliers)
    pca = PCA(n_components=2)
    # Remove 'Cluster' column if present for PCA
    cols = [col for col in df_all_scaled.columns]
    pca.fit(df_all_scaled[cols])

    # Transform inlier and outlier data
    inlier_pca = pca.transform(df_inliers_scaled[cols])
    outlier_pca = pca.transform(df_outliers_scaled[cols])

    plt.figure(figsize=(10, 6))
    plt.scatter(inlier_pca[:, 0], inlier_pca[:, 1], alpha=0.5, label="Inliers")
    plt.scatter(
        outlier_pca[:, 0],
        outlier_pca[:, 1],
        color="red",
        alpha=0.7,
        label="Outliers",
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Result in 2D (Outliers Highlighted)")
    plt.legend()
    plt.grid()
    plt.show()


def visualize_pca_3d_with_outliers(
    df_all_scaled, df_inliers_scaled, df_outliers_scaled
):
    # Fit PCA on all cleaned data
    pca = PCA(n_components=3)
    cols = df_all_scaled.columns
    pca.fit(df_all_scaled[cols])

    # Transform inliers and outliers
    inlier_pca = pca.transform(df_inliers_scaled[cols])
    outlier_pca = pca.transform(df_outliers_scaled[cols])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        inlier_pca[:, 0],
        inlier_pca[:, 1],
        inlier_pca[:, 2],
        alpha=0.5,
        label="Inliers",
    )
    ax.scatter(
        outlier_pca[:, 0],
        outlier_pca[:, 1],
        outlier_pca[:, 2],
        color="red",
        alpha=0.7,
        label="Outliers",
    )
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    plt.title("PCA Result in 3D (Outliers Highlighted)")
    plt.legend()
    plt.show()


def visualize_pca_2d(df_scaled):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.25)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Result in 2D")
    plt.grid()
    plt.show()


def visualize_pca_3d(df_scaled):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df_scaled)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], alpha=0.25)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    plt.title("PCA Result in 3D")
    plt.show()


def visualize_clusters_2d(df_scaled, clusters, smaller_points=False):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled.drop(columns=["Cluster"]))

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=df_scaled["Cluster"],
        cmap="viridis",
        alpha=0.5 if not smaller_points else 0.25,
        s=10 if smaller_points else 50,
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"PCA Result in 2D with {clusters} Clusters")
    plt.colorbar(scatter, label="Cluster")
    plt.grid()
    plt.show()


def visualize_clusters_3d(df_scaled, clusters, smaller_points=False):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df_scaled.drop(columns=["Cluster"]))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        pca_result[:, 2],
        c=df_scaled["Cluster"],
        cmap="viridis",
        alpha=0.5 if not smaller_points else 0.36,
        s=7 if smaller_points else 50,
    )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    plt.title(f"PCA Result in 3D with {clusters} Clusters")
    plt.colorbar(scatter, label="Cluster")
    plt.show()


def elbow_method(df_scaled, estimator, max_clusters=10):
    inertia = []
    for n in range(1, max_clusters + 1):
        clustering = estimator.set_params(n_clusters=n)
        clustering.fit(df_scaled)
        inertia.append(clustering.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), inertia, marker="o")
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.xticks(range(1, max_clusters + 1, 2))  # Set x labels every two units
    plt.grid()
    plt.show()


def silhouette_plot(df_scaled, n_clusters, estimator, legend_fix=False):
    cluster_labels = estimator.fit_predict(df_scaled)
    silhouette_vals = silhouette_samples(df_scaled, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        cluster_silhouette_vals.sort()
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_vals,
            alpha=0.7,
            label=f"Cluster {i}",
        )
        y_lower = y_upper + 10

    plt.title("Silhouette Plot")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.axvline(
        x=silhouette_score(df_scaled, cluster_labels),
        color="red",
        linestyle="--",
        label="Average Silhouette Score",
    )
    if legend_fix:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Clusters")
    else:
        plt.legend()
    plt.show()


# Plot dendrogram to visualize the hierarchical clustering
def plot_dendrogram(df_scaled, linkage_matx):
    plt.figure(figsize=(10, 7))
    dendrogram(
        linkage_matx,
        orientation="top",
        labels=df_scaled.index,
        distance_sort="descending",
        show_leaf_counts=True,
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()


# Plot clusters vs linkage distance
def plot_clusters_vs_linkage_distance(df_scaled, linkage_matx, max_clusters=20):
    n_samples = df_scaled.shape[0]
    linkage_distances = linkage_matx[:, 2]
    # num_clusters = np.arange(n_samples - 1, 0, -1)
    cluster_counts = n_samples - np.arange(
        len(linkage_distances)
    )  # from 2 to n_samples
    idx = cluster_counts <= max_clusters
    cluster_counts = cluster_counts[idx]
    linkage_plot = linkage_distances[idx]

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_counts, linkage_plot, marker="o")
    plt.title("Linkage Distance vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Linkage Distance")
    plt.xticks(cluster_counts)
    plt.grid(True)
    plt.show()


def plot_clusters_vs_silhouette_score(df_scaled, estimator, max_clusters=20):
    silhouette_scores = []
    for n in range(2, max_clusters + 1):
        clustering = estimator.set_params(n_clusters=n)
        cluster_labels = clustering.fit_predict(df_scaled)
        score = silhouette_score(df_scaled, cluster_labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker="o")
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.xticks(range(2, max_clusters + 1))
    plt.grid(True)
    plt.show()
