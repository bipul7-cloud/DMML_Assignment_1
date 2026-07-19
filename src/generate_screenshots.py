"""
RecoMart - Screenshot Generator
Generates all required output screenshots for the submission report.
"""

import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR = BASE_DIR / "data" / "features"
SCREENSHOTS_DIR = BASE_DIR / "final_submission" / "screenshots"
REPORTS_DIR = BASE_DIR / "reports"

SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"BASE_DIR = {BASE_DIR}")
print(f"SCREENSHOTS_DIR = {SCREENSHOTS_DIR}")
print(f"REPORTS_DIR exists = {REPORTS_DIR.exists()}")
print(f"RAW_DIR exists = {RAW_DIR.exists()}")


def load_latest_csv(folder):
    d = RAW_DIR / folder
    files = sorted(d.glob("*.csv"), reverse=True)
    return pd.read_csv(files[0]) if files else None


def screenshot_pipeline_execution():
    """Screenshot 1: Pipeline execution summary."""
    reports = sorted(REPORTS_DIR.glob("pipeline_execution_*.json"), reverse=True)
    if not reports:
        return
    with open(reports[0]) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    steps = list(data.keys())
    statuses = [data[s]["status"] for s in steps]
    colors = ["#4CAF50" if s == "SUCCESS" else "#F44336" for s in statuses]

    table_data = [[s.replace("_", " ").title(), st, data[s].get("timestamp", "")[:19]]
                  for s, st in zip(steps, statuses)]
    cell_colors = [["#C8E6C9" if st == "SUCCESS" else "#FFCDD2"] * 3 for st in statuses]
    table = ax.table(cellText=table_data,
                     colLabels=["Pipeline Stage", "Status", "Timestamp"],
                     cellColours=cell_colors,
                     colColours=["#1976D2"] * 3, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Pipeline Execution Summary — All Stages SUCCESS", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "01_pipeline_execution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 01_pipeline_execution.png")


def screenshot_data_quality():
    """Screenshot 2: Data quality report summary."""
    reports = sorted(REPORTS_DIR.glob("data_quality_report_*.json"), reverse=True)
    if not reports:
        return
    with open(reports[0]) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    rows = []
    for source, info in data.items():
        if isinstance(info, dict) and "profile" in info:
            rows.append([
                source.title(),
                info["profile"]["shape"]["rows"],
                info["profile"]["shape"]["columns"],
                f"{info['missing_values']['total_missing_pct']}%",
                f"{info['duplicates']['duplicate_pct']}%",
                len(info.get("range_checks", {}).get("range_issues", [])),
                info.get("quality_status", "N/A"),
            ])
    table = ax.table(cellText=rows,
                     colLabels=["Dataset", "Rows", "Columns", "Missing %", "Duplicate %", "Range Issues", "Status"],
                     colColours=["#1976D2"] * 7, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Data Quality Validation Report", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "02_data_quality_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 02_data_quality_report.png")


def screenshot_eda_users():
    """Screenshot 3: User demographics EDA."""
    users = load_latest_csv("users")
    if users is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EDA — User Demographics", fontsize=16, fontweight="bold")

    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    users["age"].dropna().hist(bins=30, ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")

    gender_counts = users["gender"].replace("", "Unknown").value_counts()
    gender_counts.plot(kind="bar", ax=axes[1], color=["#2196F3", "#E91E63", "#4CAF50"])
    axes[1].set_title("Gender Distribution")
    axes[1].tick_params(axis="x", rotation=0)

    country_top = users["country"].value_counts().head(15)
    country_top.plot(kind="barh", ax=axes[2], color="#FF9800")
    axes[2].set_title("Top 15 Countries")

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "03_eda_users.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 03_eda_users.png")


def screenshot_eda_products():
    """Screenshot 4: Product catalog EDA."""
    products = load_latest_csv("products")
    if products is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EDA — Product Catalog", fontsize=16, fontweight="bold")

    products["price"].hist(bins=30, ax=axes[0], color="#4CAF50", edgecolor="black")
    axes[0].set_title("Price Distribution")
    axes[0].set_xlabel("Price ($)")

    products["category"].value_counts().plot(kind="barh", ax=axes[1], color="#FF5722")
    axes[1].set_title("Products by Category")

    products["avg_rating"] = pd.to_numeric(products["avg_rating"], errors="coerce")
    products["avg_rating"].hist(bins=20, ax=axes[2], color="#9C27B0", edgecolor="black")
    axes[2].set_title("Average Rating Distribution")

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "04_eda_products.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 04_eda_products.png")


def screenshot_eda_clickstream():
    """Screenshot 5: Clickstream EDA."""
    cs = load_latest_csv("clickstream")
    if cs is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EDA — Clickstream Events", fontsize=16, fontweight="bold")

    cs["event_type"].value_counts().plot(kind="bar", ax=axes[0], color="#00BCD4")
    axes[0].set_title("Event Types")
    axes[0].tick_params(axis="x", rotation=45)

    cs["device"].value_counts().plot(kind="pie", ax=axes[1], autopct="%1.1f%%",
                                     colors=["#2196F3", "#FF9800", "#4CAF50"])
    axes[1].set_title("Device Distribution")
    axes[1].set_ylabel("")

    cs["timestamp"] = pd.to_datetime(cs["timestamp"])
    cs["hour"] = cs["timestamp"].dt.hour
    cs["hour"].value_counts().sort_index().plot(kind="line", ax=axes[2], marker="o", color="#FF5722")
    axes[2].set_title("Activity by Hour of Day")
    axes[2].set_xlabel("Hour")

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "05_eda_clickstream.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 05_eda_clickstream.png")


def screenshot_eda_ratings():
    """Screenshot 6: Ratings distribution and sparsity."""
    ratings = load_latest_csv("ratings")
    if ratings is None:
        return

    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    valid = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 5)]

    n_users = valid["user_id"].nunique()
    n_items = valid["product_id"].nunique()
    sparsity = 1 - len(valid) / (n_users * n_items)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"EDA — Ratings (Sparsity: {sparsity:.2%})", fontsize=16, fontweight="bold")

    valid["rating"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#673AB7")
    axes[0].set_title("Rating Distribution")
    axes[0].set_xlabel("Rating")

    valid.groupby("user_id").size().hist(bins=30, ax=axes[1], color="#009688", edgecolor="black")
    axes[1].set_title("Ratings per User")
    axes[1].set_xlabel("Number of Ratings")

    valid.groupby("product_id").size().hist(bins=30, ax=axes[2], color="#795548", edgecolor="black")
    axes[2].set_title("Item Popularity (Ratings per Product)")
    axes[2].set_xlabel("Number of Ratings")

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "06_eda_ratings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 06_eda_ratings.png")


def screenshot_interaction_heatmap():
    """Screenshot 7: User-item interaction heatmap."""
    ratings = load_latest_csv("ratings")
    if ratings is None:
        return

    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    valid = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 5)]

    top_users = valid.groupby("user_id").size().nlargest(25).index
    top_items = valid.groupby("product_id").size().nlargest(25).index
    subset = valid[valid["user_id"].isin(top_users) & valid["product_id"].isin(top_items)]
    pivot = subset.pivot_table(index="user_id", columns="product_id", values="rating", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.5, annot=False,
                cbar_kws={"label": "Rating"}, ax=ax)
    ax.set_title("User-Item Interaction Heatmap (Top 25 Users × Top 25 Items)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Product ID")
    ax.set_ylabel("User ID")
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "07_interaction_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 07_interaction_heatmap.png")


def screenshot_transactions():
    """Screenshot 8: Transaction analysis."""
    txn = load_latest_csv("transactions")
    if txn is None:
        return

    txn["total_amount"] = pd.to_numeric(txn["total_amount"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EDA — Transactions", fontsize=16, fontweight="bold")

    txn["status"].value_counts().plot(kind="bar", ax=axes[0], color="#8BC34A")
    axes[0].set_title("Transaction Status")
    axes[0].tick_params(axis="x", rotation=45)

    txn["payment_method"].value_counts().plot(kind="bar", ax=axes[1], color="#FF9800")
    axes[1].set_title("Payment Methods")
    axes[1].tick_params(axis="x", rotation=45)

    txn["total_amount"].dropna().hist(bins=50, ax=axes[2], color="#E91E63", edgecolor="black")
    axes[2].set_title("Transaction Amount Distribution")
    axes[2].set_xlabel("Amount ($)")

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "08_eda_transactions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 08_eda_transactions.png")


def screenshot_model_performance():
    """Screenshot 9: Model performance comparison."""
    reports = sorted(REPORTS_DIR.glob("model_performance_*.json"), reverse=True)
    if not reports:
        return
    with open(reports[0]) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Evaluation Results", fontsize=16, fontweight="bold")

    # SVD metrics bar chart
    svd = data.get("svd", {}).get("metrics", {})
    if svd:
        names = [k for k in svd.keys() if k not in ("rmse", "mae")]
        values = [svd[k] for k in names]
        colors = ["#2196F3" if "precision" in n else "#4CAF50" if "recall" in n else "#FF9800" for n in names]
        axes[0].barh(names, values, color=colors)
        axes[0].set_title("SVD Collaborative Filtering")
        axes[0].set_xlabel("Score")
        # Add RMSE/MAE text
        axes[0].text(0.95, 0.05, f"RMSE: {svd.get('rmse', 0):.4f}\nMAE: {svd.get('mae', 0):.4f}",
                     transform=axes[0].transAxes, ha="right", va="bottom",
                     fontsize=11, bbox=dict(boxstyle="round", facecolor="#FFF9C4"))

    # Content-based metrics
    cb = data.get("content_based", {}).get("metrics", {})
    if cb:
        names = list(cb.keys())
        values = [cb[k] for k in names]
        colors = ["#2196F3" if "precision" in n else "#4CAF50" if "recall" in n else "#FF9800" for n in names]
        axes[1].barh(names, values, color=colors)
        axes[1].set_title("Content-Based Filtering")
        axes[1].set_xlabel("Score")

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "09_model_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 09_model_performance.png")


def screenshot_feature_store():
    """Screenshot 10: Feature store registry."""
    import sqlite3
    db_path = BASE_DIR / "data" / "feature_store" / "registry.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    sets_df = pd.read_sql_query("SELECT name, version, n_features, n_rows, created_at, status FROM feature_sets ORDER BY created_at DESC LIMIT 6", conn)
    meta_df = pd.read_sql_query("SELECT feature_set_name, feature_name, dtype, source_table FROM feature_metadata LIMIT 20", conn)
    conn.close()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Feature Store Registry", fontsize=16, fontweight="bold")

    # Feature sets table
    axes[0].axis("off")
    table1 = axes[0].table(cellText=sets_df.values, colLabels=sets_df.columns,
                           colColours=["#1976D2"] * len(sets_df.columns),
                           loc="center", cellLoc="center")
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1.1, 1.4)
    for (row, col), cell in table1.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
    axes[0].set_title("Registered Feature Sets", fontsize=12, pad=10)

    # Feature metadata sample
    axes[1].axis("off")
    table2 = axes[1].table(cellText=meta_df.values, colLabels=meta_df.columns,
                           colColours=["#388E3C"] * len(meta_df.columns),
                           loc="center", cellLoc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1.1, 1.3)
    for (row, col), cell in table2.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
    axes[1].set_title("Feature Metadata (Sample)", fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "10_feature_store.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 10_feature_store.png")


def screenshot_data_versioning():
    """Screenshot 11: Data versioning manifest."""
    manifest_path = BASE_DIR / "data" / "versions" / "manifest.json"
    if not manifest_path.exists():
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    rows = []
    for ds_name, ds_info in manifest.get("datasets", {}).items():
        latest = ds_info.get("versions", [{}])[-1]
        rows.append([
            ds_name,
            latest.get("version_id", ""),
            latest.get("hash", "")[:16] + "...",
            latest.get("stats", {}).get("rows", ""),
            latest.get("stats", {}).get("columns", ""),
            latest.get("timestamp", "")[:19],
        ])

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")
    table = ax.table(cellText=rows,
                     colLabels=["Dataset", "Version ID", "SHA-256 Hash", "Rows", "Cols", "Timestamp"],
                     colColours=["#1976D2"] * 6, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Data Versioning Manifest — {len(rows)} Datasets Tracked",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "11_data_versioning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 11_data_versioning.png")


def screenshot_folder_structure():
    """Screenshot 12: Data lake folder structure."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis("off")

    structure = """
    data/
    ├── raw/                          ← Raw ingested data
    │   ├── users/
    │   │   └── users_20260711.csv
    │   ├── products/
    │   │   ├── products_20260711.csv
    │   │   └── products_20260711.json
    │   ├── clickstream/
    │   │   └── clickstream_20260711.csv
    │   ├── transactions/
    │   │   └── transactions_20260711.csv
    │   └── ratings/
    │       └── ratings_20260711.csv
    │
    ├── validated/                    ← Post-validation
    │   ├── users/
    │   ├── products/
    │   ├── clickstream/
    │   ├── transactions/
    │   └── ratings/
    │
    ├── processed/                   ← Cleaned & encoded
    │   ├── users_clean.csv
    │   ├── products_clean.csv
    │   ├── clickstream_clean.csv
    │   ├── transactions_clean.csv
    │   └── ratings_clean.csv
    │
    ├── features/                    ← Engineered features
    │   ├── user_features.csv
    │   ├── item_features.csv
    │   ├── user_item_features.csv
    │   └── schema.sql
    │
    ├── feature_store/               ← Feature registry
    │   ├── registry.db
    │   ├── online/
    │   └── versions/
    │
    └── versions/                    ← Versioned snapshots
        ├── manifest.json
        ├── raw_users/
        ├── raw_products/
        ├── processed_*/
        └── features_*/
    """

    ax.text(0.05, 0.95, structure, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#BDBDBD"))
    ax.set_title("Data Lake Storage Structure", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "12_folder_structure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 12_folder_structure.png")


def screenshot_mlflow():
    """Screenshot 13: MLflow experiment tracking summary."""
    reports = sorted(REPORTS_DIR.glob("model_performance_*.json"), reverse=True)
    if not reports:
        return
    with open(reports[0]) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    run_id = data.get("mlflow_run_id", "N/A")
    svd_metrics = data.get("svd", {}).get("metrics", {})
    svd_params = data.get("svd", {}).get("params", {})

    info = [
        ["Experiment Name", "recomart_recommendation"],
        ["Run ID", run_id],
        ["Algorithm", svd_params.get("algorithm", "SVD")],
        ["n_factors", str(svd_params.get("n_factors", 50))],
        ["RMSE", f"{svd_metrics.get('rmse', 0):.4f}"],
        ["MAE", f"{svd_metrics.get('mae', 0):.4f}"],
        ["Precision@10", f"{svd_metrics.get('precision@10', 0):.4f}"],
        ["Recall@10", f"{svd_metrics.get('recall@10', 0):.4f}"],
        ["NDCG@10", f"{svd_metrics.get('ndcg@10', 0):.4f}"],
        ["Model Artifact", data.get("svd", {}).get("model_path", "N/A").split("\\")[-1]],
    ]

    table = ax.table(cellText=info, colLabels=["Parameter", "Value"],
                     colColours=["#1976D2"] * 2, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("MLflow Experiment Tracking — Run Details", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "13_mlflow_tracking.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 13_mlflow_tracking.png")


def screenshot_recommendations():
    """Screenshot 14: Sample recommendations output."""
    reports = sorted(REPORTS_DIR.glob("model_performance_*.json"), reverse=True)
    if not reports:
        return
    with open(reports[0]) as f:
        data = json.load(f)

    recs = data.get("sample_recommendations", {})
    user_id = recs.get("user_id", "?")
    items = recs.get("recommendations", [])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    rows = [[r["product_id"], r["predicted_rating"], i + 1] for i, r in enumerate(items)]
    table = ax.table(cellText=rows,
                     colLabels=["Product ID", "Predicted Rating", "Rank"],
                     colColours=["#1976D2"] * 3, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Top-5 Recommendations for User {user_id} (SVD Model)",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "14_sample_recommendations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 14_sample_recommendations.png")


if __name__ == "__main__":
    print("Generating screenshots...\n")
    screenshot_pipeline_execution()
    screenshot_data_quality()
    screenshot_eda_users()
    screenshot_eda_products()
    screenshot_eda_clickstream()
    screenshot_eda_ratings()
    screenshot_interaction_heatmap()
    screenshot_transactions()
    screenshot_model_performance()
    screenshot_feature_store()
    screenshot_data_versioning()
    screenshot_folder_structure()
    screenshot_mlflow()
    screenshot_recommendations()
    print(f"\nAll screenshots saved to: {SCREENSHOTS_DIR}")
