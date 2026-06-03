# ============================================================
# CLUSTERSENSE AI - PIPELINE
# BEST MODEL: KMeans (K=3)
# STRATEGY  : Winsorized Data + Scaling
# ============================================================

import logging
import time
import os
import pickle

import pandas as pd
import numpy as np

from sklearn.preprocessing  import StandardScaler
from sklearn.cluster        import KMeans
from sklearn.metrics        import silhouette_score


# ============================================================
# LOGGING SETUP
# ============================================================

os.makedirs("logs",      exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

logging.basicConfig(
    filename="logs/clustersense.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.Formatter.converter = time.localtime


# ============================================================
# SAVE PICKLE
# ============================================================

def save_pickle(obj, path):

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    print(f"\nSaved: {path}")
    logging.info(f"Saved artifact: {path}")


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path):

    df = pd.read_csv(path)

    print("\n[DATA LOADED]", df.shape)
    logging.info(f"Data loaded — shape: {df.shape}")

    return df


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

def handle_missing(df):

    df = df.copy()

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].fillna(df[col].median())

        else:

            df[col] = df[col].fillna(df[col].mode()[0])

    logging.info("Missing values handled.")
    print("\n[MISSING VALUES HANDLED]")

    return df


# ============================================================
# 3. WINSORIZATION (treat outliers)
# ============================================================

def winsorize(df, num_cols):

    df = df.copy()

    for col in num_cols:

        q1    = df[col].quantile(0.25)
        q3    = df[col].quantile(0.75)
        iqr   = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = np.clip(df[col], lower, upper)

    logging.info("Winsorization completed.")
    print("\n[WINSORIZATION COMPLETED]")

    return df


# ============================================================
# 4. ENCODING
# ============================================================

def encode(df):

    df_enc = pd.get_dummies(df, drop_first=True)

    logging.info(f"Encoding done — Features: {df_enc.shape[1]}")
    print(f"\n[ENCODING DONE] Features: {df_enc.shape[1]}")

    return df_enc


# ============================================================
# 5. SCALING
# ============================================================

def scale(df_enc):

    scaler = StandardScaler()
    X      = scaler.fit_transform(df_enc)

    save_pickle(scaler,                  "artifacts/scaler.pkl")
    save_pickle(df_enc.columns.tolist(), "artifacts/model_columns.pkl")

    logging.info("Scaling completed.")
    print("\n[SCALING COMPLETED]")

    return X, scaler


# ============================================================
# 6. ELBOW METHOD
# ============================================================

def elbow_method(X, k_range=range(2, 11)):

    wcss_values = []

    print("\n===== ELBOW METHOD =====")

    for k in k_range:

        km   = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        wcss = km.inertia_

        wcss_values.append(wcss)
        print(f"K={k} | WCSS={wcss:,.2f}")

    logging.info("Elbow method completed.")

    return list(k_range), wcss_values


# ============================================================
# 7. FIND BEST K — silhouette score
# ============================================================

def find_best_k(X, k_range=range(2, 11)):

    best_k     = 2
    best_score = -1
    scores     = []

    print("\n===== SILHOUETTE SCORES =====")

    for k in k_range:

        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score  = silhouette_score(
            X, labels,
            sample_size=2000,
            random_state=42
        )

        scores.append(score)
        print(f"K={k} | Silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k     = k

    logging.info(f"Best K={best_k} | Silhouette={best_score:.4f}")
    print(f"\nBest K = {best_k} | Silhouette Score = {best_score:.4f}")

    return best_k, best_score, list(k_range), scores


# ============================================================
# 8. TRAIN BEST MODEL — KMeans
# ============================================================

def train_model(X, best_k):

    model  = KMeans(
        n_clusters   = best_k,
        random_state = 42,
        n_init       = 10
    )

    labels = model.fit_predict(X)

    sil_score = silhouette_score(X, labels)

    print("\n" + "="*60)
    print("   MODEL RESULTS")
    print("="*60)
    print(f"  Model      : KMeans")
    print(f"  Clusters   : {best_k}")
    print(f"  Silhouette : {sil_score:.4f}")
    print(f"  WCSS       : {model.inertia_:,.2f}")
    print("="*60)

    logging.info(
        f"KMeans trained — "
        f"K={best_k} | "
        f"Silhouette={sil_score:.4f} | "
        f"WCSS={model.inertia_:,.2f}"
    )

    return model, labels


# ============================================================
# 9. CLUSTER PROFILING
# ============================================================

def profile_clusters(df, labels):

    df_profile            = df.copy()
    df_profile["Cluster"] = labels

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    profile  = df_profile.groupby("Cluster")[num_cols].mean().round(2)

    print("\n===== CLUSTER PROFILES =====")
    print(profile.to_string())

    logging.info("Cluster profiling completed.")

    return df_profile, profile


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def run_pipeline():

    print("\n" + "="*60)
    print("   CLUSTERSENSE AI — CUSTOMER SEGMENTATION PIPELINE")
    print("   Best Model : KMeans")
    print("   Strategy   : Winsorized Data + Scaling")
    print("="*60)

    logging.info("="*50)
    logging.info("CLUSTERSENSE AI PIPELINE STARTED")
    logging.info("="*50)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_data(
        "artifacts/clustersense_customer_segmentation_dataset.csv"
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    df = handle_missing(df)

    # --------------------------------------------------------
    # NUMERICAL COLUMNS
    # --------------------------------------------------------

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # --------------------------------------------------------
    # WINSORIZATION
    # --------------------------------------------------------

    df = winsorize(df, num_cols)

    # --------------------------------------------------------
    # ENCODING
    # --------------------------------------------------------

    df_enc = encode(df)

    # --------------------------------------------------------
    # SCALING
    # --------------------------------------------------------

    X, scaler = scale(df_enc)

    # --------------------------------------------------------
    # ELBOW METHOD
    # --------------------------------------------------------

    k_list, wcss_list = elbow_method(X)

    # --------------------------------------------------------
    # FIND BEST K
    # --------------------------------------------------------

    best_k, best_score, k_list2, sil_list = find_best_k(X)

    # --------------------------------------------------------
    # TRAIN BEST MODEL
    # --------------------------------------------------------

    model, labels = train_model(X, best_k)

    # --------------------------------------------------------
    # CLUSTER PROFILING
    # --------------------------------------------------------

    df_profile, profile = profile_clusters(df, labels)

    # --------------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------------

    save_pickle(model,  "artifacts/best_model.pkl")
    save_pickle(best_k, "artifacts/best_k.pkl")

    df_profile.to_csv(
        "artifacts/clustered_customers.csv", index=False
    )
    profile.to_csv("artifacts/cluster_profiles.csv")

    print("\nSaved: artifacts/clustered_customers.csv")
    print("Saved: artifacts/cluster_profiles.csv")

    # --------------------------------------------------------
    # SAVE ELBOW + SILHOUETTE DATA FOR STREAMLIT
    # --------------------------------------------------------

    pd.DataFrame({
        "K":    k_list,
        "WCSS": wcss_list
    }).to_csv("artifacts/elbow_data.csv", index=False)

    pd.DataFrame({
        "K":          k_list2,
        "Silhouette": sil_list
    }).to_csv("artifacts/silhouette_data.csv", index=False)

    print("\nPipeline completed successfully.")
    logging.info("PIPELINE COMPLETED SUCCESSFULLY.")

    return best_k, best_score


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    run_pipeline()