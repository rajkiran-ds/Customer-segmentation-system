# ============================================================
# CLUSTERSENSE AI — STREAMLIT APP
# Run: streamlit run app.py
# ============================================================

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ClusterSense AI",
    page_icon="🎯",
    layout="wide"
)

# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model   = pickle.load(open("artifacts/best_model.pkl",    "rb"))
    scaler  = pickle.load(open("artifacts/scaler.pkl",        "rb"))
    columns = pickle.load(open("artifacts/model_columns.pkl", "rb"))
    best_k  = pickle.load(open("artifacts/best_k.pkl",        "rb"))

    return model, scaler, columns, best_k


# ============================================================
# TITLE
# ============================================================

st.title("🎯 ClusterSense AI — Customer Segmentation")
st.write("Identify customer segments using KMeans Clustering.")
st.markdown("---")

# ============================================================
# CHECK ARTIFACTS
# ============================================================

if not os.path.exists("artifacts/best_model.pkl"):

    st.warning(
        "Model not found. "
        "Please run `python main.py` first."
    )
    st.stop()

model, scaler, columns, best_k = load_artifacts()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎯 ClusterSense AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Cluster Analysis", "🔮 Predict Segment"]
)

# ============================================================
# PAGE — HOME
# ============================================================

if page == "🏠 Home":

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model",       "KMeans")
    c2.metric("Optimal Clusters", str(best_k))
    c3.metric("Strategy",         "Winsorized + Scaled")
    c4.metric("Dataset Size",     "10,000 customers")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📉 Elbow Method")

        if os.path.exists("artifacts/elbow_data.csv"):

            elbow_df = pd.read_csv("artifacts/elbow_data.csv")

            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(
                elbow_df["K"], elbow_df["WCSS"],
                marker="o", color="#534AB7", linewidth=2
            )
            ax.axvline(
                x=best_k, color="red",
                linestyle="--", label=f"Best K={best_k}"
            )
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("WCSS")
            ax.set_title("Elbow Method")
            ax.legend()
            ax.set_facecolor("#f7f6f2")
            fig.patch.set_facecolor("#f7f6f2")
            st.pyplot(fig)
            plt.close()

    with col2:

        st.markdown("### 📈 Silhouette Scores")

        if os.path.exists("artifacts/silhouette_data.csv"):

            sil_df = pd.read_csv("artifacts/silhouette_data.csv")

            fig, ax  = plt.subplots(figsize=(6, 3.5))
            colors   = [
                "#2d6a00" if k == best_k else "#aaa"
                for k in sil_df["K"]
            ]
            ax.bar(sil_df["K"], sil_df["Silhouette"], color=colors)
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Score per K")
            ax.set_facecolor("#f7f6f2")
            fig.patch.set_facecolor("#f7f6f2")
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    st.success(
        f"✅ KMeans with K={best_k} gave the best Silhouette Score."
    )
    st.info(
        "📌 Winsorization treated outliers before clustering."
    )
    st.warning(
        "⚠️ DBSCAN found 164 clusters — too fragmented, not useful."
    )

# ============================================================
# PAGE — CLUSTER ANALYSIS
# ============================================================

elif page == "📊 Cluster Analysis":

    st.markdown("# 📊 Cluster Analysis")

    if not os.path.exists("artifacts/clustered_customers.csv"):
        st.info("Run `python main.py` first.")
        st.stop()

    df      = pd.read_csv("artifacts/clustered_customers.csv")
    profile = pd.read_csv(
        "artifacts/cluster_profiles.csv", index_col=0
    )

    st.markdown("### Cluster Sizes")

    sizes  = df["Cluster"].value_counts().sort_index()
    colors = ["#534AB7", "#1D9E75", "#D85A30", "#378ADD"]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(
        [f"Cluster {i}" for i in sizes.index],
        sizes.values,
        color=colors[:len(sizes)]
    )
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customers per Cluster")
    ax.set_facecolor("#f7f6f2")
    fig.patch.set_facecolor("#f7f6f2")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("### Cluster Profiles (Mean Values)")
    st.dataframe(profile, use_container_width=True)

    st.markdown("---")
    st.markdown("### Feature Comparison Across Clusters")

    feature = st.selectbox(
        "Select Feature",
        profile.columns.tolist()
    )

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(
        [f"Cluster {i}" for i in profile.index],
        profile[feature],
        color=colors[:len(profile)]
    )
    ax.set_xlabel(feature)
    ax.set_title(f"{feature} by Cluster")
    ax.set_facecolor("#f7f6f2")
    fig.patch.set_facecolor("#f7f6f2")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("### Full Clustered Dataset")

    cluster_filter = st.multiselect(
        "Filter by Cluster",
        options=sorted(df["Cluster"].unique()),
        default=sorted(df["Cluster"].unique())
    )

    filtered = df[df["Cluster"].isin(cluster_filter)]
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Clustered Data",
        filtered.to_csv(index=False),
        file_name="clustered_customers.csv",
        mime="text/csv"
    )

# ============================================================
# PAGE — PREDICT SEGMENT
# ============================================================

elif page == "🔮 Predict Segment":

    st.markdown("# 🔮 Predict Customer Segment")
    st.write(
        "Enter customer details to find "
        "which segment they belong to."
    )
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:

        age = st.number_input(
            "Customer Age",
            min_value=18, max_value=69, value=35
        )

        income = st.number_input(
            "Annual Income",
            min_value=10000, max_value=200000, value=65000
        )

        spending = st.number_input(
            "Spending Score",
            min_value=1, max_value=100, value=50
        )

        purchase_freq = st.number_input(
            "Purchase Frequency",
            min_value=1, max_value=100, value=20
        )

    with c2:

        web_visits = st.number_input(
            "Website Visits",
            min_value=0, max_value=100, value=15
        )

        time_app = st.slider(
            "Time on App (hrs)",
            0.0, 10.0, 3.0
        )

        discount = st.slider(
            "Discount Usage",
            0.0, 10.0, 5.0
        )

        days_since = st.number_input(
            "Days Since Last Purchase",
            min_value=0, max_value=365, value=30
        )

    with c3:

        loyalty = st.number_input(
            "Loyalty Points",
            min_value=100, max_value=9999, value=5000
        )

        social = st.slider(
            "Social Media Engagement",
            1.0, 10.0, 5.0
        )

        payment = st.selectbox(
            "Preferred Payment Mode",
            ["Cash", "Credit Card", "Debit Card",
             "Net Banking", "UPI"]
        )

        city = st.selectbox(
            "City Tier",
            ["Tier1", "Tier2", "Tier3"]
        )

    if st.button("🎯 Predict Segment", use_container_width=True):

        row = {
            "customer_age":              age,
            "annual_income":             income,
            "spending_score":            spending,
            "purchase_frequency":        purchase_freq,
            "website_visits":            web_visits,
            "time_on_app":               time_app,
            "discount_usage":            discount,
            "days_since_last_purchase":  days_since,
            "loyalty_points":            loyalty,
            "social_media_engagement":   social,
            "preferred_payment_mode_Credit Card":
                1 if payment == "Credit Card"  else 0,
            "preferred_payment_mode_Debit Card":
                1 if payment == "Debit Card"   else 0,
            "preferred_payment_mode_Net Banking":
                1 if payment == "Net Banking"  else 0,
            "preferred_payment_mode_UPI":
                1 if payment == "UPI"          else 0,
            "city_tier_Tier2":
                1 if city    == "Tier2"        else 0,
            "city_tier_Tier3":
                1 if city    == "Tier3"        else 0,
        }

        X_input  = pd.DataFrame([row]).reindex(
            columns=columns, fill_value=0
        )
        X_scaled = scaler.transform(X_input)
        cluster  = model.predict(X_scaled)[0]

        segment_labels = {
            0: (
                "💎 Premium Customer",
                "#534AB7",
                "High income, high spending, loyal customer."
            ),
            1: (
                "🛒 Regular Shopper",
                "#1D9E75",
                "Moderate income, frequent purchases."
            ),
            2: (
                "🌱 Occasional Buyer",
                "#D85A30",
                "Low engagement, price-sensitive customer."
            ),
        }

        label, color, desc = segment_labels.get(
            cluster,
            (f"Cluster {cluster}", "#888", "Segment identified.")
        )

        st.markdown("---")

        _, col_b, _ = st.columns([1, 2, 1])

        with col_b:

            st.markdown(f"""
            <div style="background:#fff;border-radius:14px;
                        padding:30px;text-align:center;
                        border:2px solid {color};">
                <div style="font-size:13px;color:#888;
                            font-weight:600;">
                    CUSTOMER SEGMENT
                </div>
                <div style="font-size:34px;font-weight:700;
                            color:{color};margin:10px 0;">
                    {label}
                </div>
                <div style="font-size:13px;color:#555;">
                    Cluster {cluster} of {best_k}
                </div>
                <div style="font-size:12px;color:#888;
                            margin-top:8px;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            "Model: KMeans (K=3) | "
            "Strategy: Winsorized + Scaled"
        )