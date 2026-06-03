
# ============================================================
# CLUSTERSENSE AI - CUSTOMER SEGMENTATION
# COMPLETE CLUSTERING PIPELINE
# ============================================================

import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path):

    df = pd.read_csv(path)

    print("\n[DATA LOADED]", df.shape)

    return df


# ============================================================
# 2. MISSING VALUE HANDLING
# ============================================================

def handle_missing(df):

    df = df.copy()

    for col in df.columns:

        if df[col].dtype != np.number:

            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

        else:

            df[col] = df[col].fillna(
                df[col].median()
            )

    print("\n[MISSING VALUES HANDLED]")

    return df


# ============================================================
# 3. WINSORIZATION
# ============================================================

def winsorize(df, num_cols):

    df = df.copy()

    for col in num_cols:

        q1 = df[col].quantile(0.25)

        q3 = df[col].quantile(0.75)

        lower = q1 - 1.5 * (q3 - q1)

        upper = q3 + 1.5 * (q3 - q1)

        df[col] = np.clip(
            df[col],
            lower,
            upper
        )

    print("\n[WINSORIZATION COMPLETED]")

    return df


# ============================================================
# 4. ENCODING
# ============================================================

def encode(df):

    df = pd.get_dummies(
        df,
        drop_first=True
    )

    print("\n[ENCODING COMPLETED]")

    return df


# ============================================================
# 5. SCALING
# ============================================================

def scale(df):

    scaler = StandardScaler()

    scaled = scaler.fit_transform(df)

    print("\n[SCALING COMPLETED]")

    return scaled


# ============================================================
# 6. ELBOW METHOD
# ============================================================

def elbow_method(X):

    print("\n===== ELBOW METHOD =====")

    wcss = []

    for i in range(2, 11):

        model = KMeans(
            n_clusters=i,
            random_state=42
        )

        model.fit(X)

        wcss.append(model.inertia_)

        print(f"K={i} | WCSS={model.inertia_:.2f}")

    return wcss


# ============================================================
# 7. KMEANS CLUSTERING
# ============================================================

def kmeans_clustering(X, n_clusters=4):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    print("\n[KMEANS RESULTS]")

    print("Clusters:", n_clusters)

    print("Silhouette Score:", round(score, 4))

    return labels, score


# ============================================================
# 8. AGGLOMERATIVE CLUSTERING
# ============================================================

def agglomerative_clustering(X, n_clusters=4):

    model = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    print("\n[AGGLOMERATIVE RESULTS]")

    print("Clusters:", n_clusters)

    print("Silhouette Score:", round(score, 4))

    return labels, score


# ============================================================
# 9. DBSCAN CLUSTERING
# ============================================================

def dbscan_clustering(X):

    model = DBSCAN(
        eps=1.5,
        min_samples=3
    )

    labels = model.fit_predict(X)

    unique_clusters = len(set(labels))

    if unique_clusters > 1:

        score = silhouette_score(X, labels)

    else:

        score = -1

    print("\n[DBSCAN RESULTS]")

    print("Clusters Found:", unique_clusters)

    print("Silhouette Score:", round(score, 4))

    return labels, score


# ============================================================
# 10. RUN PIPELINE
# ============================================================

def run_pipeline(path):

    df = load_data(path)

    # ========================================================
    # MISSING VALUES
    # ========================================================

    df = handle_missing(df)

    # ========================================================
    # NUMERICAL COLUMNS
    # ========================================================

    num_cols = df.select_dtypes(
        include=np.number
    ).columns

    # ========================================================
    # WINSORIZATION
    # ========================================================

    df = winsorize(df, num_cols)

    # ========================================================
    # ENCODING
    # ========================================================

    df = encode(df)

    # ========================================================
    # SCALING
    # ========================================================

    X_scaled = scale(df)

    # ========================================================
    # ELBOW METHOD
    # ========================================================

    elbow_method(X_scaled)

    # ========================================================
    # KMEANS
    # ========================================================

    k_labels, k_score = kmeans_clustering(
        X_scaled,
        n_clusters=4
    )

    # ========================================================
    # AGGLOMERATIVE
    # ========================================================

    a_labels, a_score = agglomerative_clustering(
        X_scaled,
        n_clusters=4
    )

    # ========================================================
    # DBSCAN
    # ========================================================

    d_labels, d_score = dbscan_clustering(
        X_scaled
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    results = pd.DataFrame({

        "Model": [
            "KMeans",
            "Agglomerative",
            "DBSCAN"
        ],

        "Silhouette Score": [
            round(k_score, 4),
            round(a_score, 4),
            round(d_score, 4)
        ]
    })

    print("\n========== FINAL COMPARISON ==========\n")

    print(
        results.sort_values(
            "Silhouette Score",
            ascending=False
        )
    )

    return results


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    run_pipeline(
        "artifacts/clustersense_customer_segmentation_dataset.csv"
    )
