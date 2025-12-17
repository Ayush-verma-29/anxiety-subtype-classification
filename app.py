import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import umap

import os
import praw

# =============================
# Page config
# =============================
st.set_page_config(
    page_title="Anxiety Subtype Classification Dashboard",
    layout="wide"
)

# #  Optional Safety Check (TEMPORARY)
# st.write(
#     "Secrets loaded:",
#     os.getenv("REDDIT_CLIENT_ID") is not None
# )

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)



st.title(" Anxiety Subtype Classification Dashboard")

# =============================
# Load data
# =============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/reddit_with_predictions.csv")
    emb = np.load("features/sbert_embeddings_all.npy")
    return df, emb

df, embeddings = load_data()

# =============================
# Sidebar controls
# =============================
st.sidebar.header("Controls")

label_type = st.sidebar.selectbox(
    "Choose label type",
    ["final_subtype", "predicted_subtype"]
)

show_unknown = st.sidebar.checkbox("Include 'unknown' labels", value=False)

if not show_unknown:
    df_vis = df[df["final_subtype"] != "unknown"]
else:
    df_vis = df.copy()



# =============================
# Tabs
# =============================
tabs = st.tabs([
    " Overview",
    "EDA",
    " Sentiment Analysis",
    " Confusion Matrix",
    " Model Metrics",
    " Embedding Visualization",
    # " GAD Analysis",
    "⬇️ Download"
])


# =============================
# TAB 1: Overview
# =============================
with tabs[0]:
    st.subheader("Class Distribution")

    fig, ax = plt.subplots(figsize=(6,4))
    df_vis[label_type].value_counts().plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Subtype Distribution")
    st.pyplot(fig)

    st.write("Sample data")
    st.dataframe(df_vis[["clean_text", "final_subtype", "sentiment"]].head(10))


# =============================
# TAB 1: EDA
# =============================
with tabs[1]:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Posts", len(df))
    col2.metric("Labeled Posts", (df["subtype_label"] != "unknown").sum())
    col3.metric("Unlabeled Posts", (df["subtype_label"] == "unknown").sum())

    st.write("Sample records")
    st.dataframe(df.sample(5), use_container_width=True)


    st.subheader("Anxiety Subtype Distribution")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(
        data=df,
        x="final_subtype",
        order=df["final_subtype"].value_counts().index,
        ax=ax
    )

    ax.set_xlabel("Subtype")
    ax.set_ylabel("Number of Posts")
    ax.set_title("Distribution of Anxiety Subtypes")

    st.pyplot(fig)



    st.subheader("Keyword Frequency by Subtype")

    kw_cols = ["gad_kw_freq", "panic_kw_freq", "social_kw_freq"]

    kw_df = df.groupby("final_subtype")[kw_cols].mean().reset_index()
    kw_melt = kw_df.melt(
        id_vars="final_subtype",
        var_name="keyword_type",
        value_name="avg_frequency"
    )

    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(
        data=kw_melt,
        x="final_subtype",
        y="avg_frequency",
        hue="keyword_type",
        ax=ax
    )

    ax.set_ylabel("Average Frequency")
    ax.set_xlabel("Subtype")

    st.pyplot(fig)


    st.subheader("Feature Correlation")

    corr_features = df[
        ["sentiment", "gad_kw_freq", "panic_kw_freq", "social_kw_freq"]
    ]

    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(
        corr_features.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)


    st.subheader("Filter by Subtype")

    selected_subtype = st.selectbox(
        "Choose subtype",
        df["final_subtype"].unique()
    )

    filtered = df[df["final_subtype"] == selected_subtype]

    st.write(f"Showing {len(filtered)} posts")
    st.dataframe(
        filtered[["post_id", "clean_text", "sentiment"]],
        use_container_width=True
    )

# =============================
# TAB 2: Sentiment
# =============================
with tabs[2]:
    st.subheader("Sentiment Distribution")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df_vis["sentiment"], bins=30, kde=True, ax=ax)
    ax.set_title("Sentiment Score Distribution (-1 to 1)")
    st.pyplot(fig)

    st.subheader("Sentiment by Subtype")

    fig, ax = plt.subplots(figsize=(7,4))
    sns.boxplot(
        data=df_vis,
        x=label_type,
        y="sentiment",
        showfliers=False,
        ax=ax
    )
    ax.set_title("Sentiment vs Subtype")
    st.pyplot(fig)

# =============================
# TAB 3: Confusion Matrix
# =============================
with tabs[3]:
    st.subheader("Confusion Matrix (Labeled Data Only)")

    labeled = df[df["subtype_label"] != "unknown"]

    y_true = labeled["subtype_label"]
    y_pred = labeled["predicted_subtype"]
    labels = sorted(y_true.unique())

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# =============================
# TAB 4: Metrics
# =============================
with tabs[4]:
    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

    @st.cache_data
    def load_metrics():
     return pd.read_csv("models/model_metrics.csv")

    metrics_df = load_metrics()

    st.header(" Model Performance Comparison")

    st.subheader("Overall Metrics (Macro Averaged)")
    st.dataframe(
        metrics_df.style.format({
            "accuracy": "{:.3f}",
            "precision_macro": "{:.3f}",
            "recall_macro": "{:.3f}",
            "f1_macro": "{:.3f}"
        }),
        use_container_width=True
    )

    metric = st.selectbox(
        "Select Metric",
        ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        data=metrics_df,
        x="model",
        y=metric,
        ax=ax
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("")
    ax.set_title(f"Model Comparison – {metric.replace('_', ' ').title()}")

    st.pyplot(fig)

    best_model = metrics_df.sort_values("f1_macro", ascending=False).iloc[0]

    st.success(
        f" Best Overall Model: **{best_model['model']}** "
        f"(F1 = {best_model['f1_macro']:.3f})"
    )

    st.subheader("Macro vs Weighted F1")

    fig, ax = plt.subplots(figsize=(5,4))
    df_report.loc[
        ["macro avg", "weighted avg"],
        ["precision", "recall", "f1-score"]
    ].plot(kind="bar", ax=ax)
    ax.set_ylim(0,1)
    ax.set_title("Overall Performance")
    st.pyplot(fig)

# =============================
# TAB 5: Embedding Visualization
# =============================
with tabs[5]:
    st.subheader("Embedding Visualization")

    method = st.selectbox("Choose method", ["UMAP", "t-SNE"])

    sample_size = st.slider("Sample size", 500, 2000, 1000)

    idx = np.random.choice(len(df_vis), sample_size, replace=False)
    emb_sample = embeddings[idx]
    labels_sample = df_vis.iloc[idx][label_type]

    if method == "UMAP":
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)

    emb_2d = reducer.fit_transform(emb_sample)

    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(
        x=emb_2d[:,0],
        y=emb_2d[:,1],
        hue=labels_sample,
        alpha=0.6,
        ax=ax
    )
    ax.set_title(f"{method} Projection of SBERT Embeddings")
    st.pyplot(fig)

# =============================
# TAB 6: GAD Analysis
# =============================
# with tabs[5]:
#     st.subheader("Generalized Anxiety Disorder (GAD) Analysis")

#     gad_df = labeled.copy()
#     gad_df["is_gad_true"] = gad_df["subtype_label"] == "generalized_anxiety"
#     gad_df["is_gad_pred"] = gad_df["predicted_subtype"] == "generalized_anxiety"

#     gad_counts = pd.DataFrame({
#         "True GAD": gad_df["is_gad_true"].value_counts(),
#         "Predicted GAD": gad_df["is_gad_pred"].value_counts()
#     })

#     st.dataframe(gad_counts)

#     fig, ax = plt.subplots(figsize=(5,4))
#     gad_counts.plot(kind="bar", ax=ax)
#     ax.set_title("GAD True vs Predicted Counts")
#     st.pyplot(fig)

# =============================
# TAB 7: Download
# =============================
with tabs[6]:
    st.subheader("Download Results")

    st.download_button(
        label="Download Final Dataset",
        data=df.to_csv(index=False),
        file_name="reddit_with_predictions.csv",
        mime="text/csv"
    )


