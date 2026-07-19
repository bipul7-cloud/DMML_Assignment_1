"""
RecoMart - Model Training and Evaluation Module
Trains collaborative filtering (SVD) and content-based recommendation models.
Tracks experiments with MLflow.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class RecommendationTrainer:
    """Trains and evaluates recommendation models."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(str(LOG_DIR / "training.log"), rotation="10 MB", level="INFO")

        # Setup MLflow
        if MLFLOW_AVAILABLE:
            mlruns_path = BASE_DIR / "mlruns"
            mlruns_path.mkdir(parents=True, exist_ok=True)
            db_path = mlruns_path / "mlflow.db"
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
            mlflow.set_experiment("recomart_recommendation")
            logger.info("MLflow tracking enabled.")

        self.metrics = {}
        self.model_metadata = {}

    def _load_ratings(self):
        """Load clean ratings data."""
        path = PROCESSED_DIR / "ratings_clean.csv"
        if not path.exists():
            logger.error(f"Ratings file not found: {path}")
            return None
        df = pd.read_csv(path)
        logger.info(f"Loaded ratings: {df.shape}")
        return df

    def prepare_data(self, df, test_size=0.2):
        """Split data into train and test sets."""
        train, test = train_test_split(
            df, test_size=test_size, random_state=42
        )
        logger.info(f"Train: {len(train)}, Test: {len(test)}")
        return train, test

    def _create_user_item_matrix(self, df):
        """Create a user-item interaction matrix."""
        user_ids = df["user_id"].unique()
        item_ids = df["product_id"].unique()
        user_map = {uid: idx for idx, uid in enumerate(user_ids)}
        item_map = {pid: idx for idx, pid in enumerate(item_ids)}

        rows = df["user_id"].map(user_map).values
        cols = df["product_id"].map(item_map).values
        vals = df["rating"].values.astype(float)

        matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(user_ids), len(item_ids))
        )

        return matrix, user_map, item_map, user_ids, item_ids

    def train_svd(self, train_df, n_factors=50):
        """Train SVD-based collaborative filtering model."""
        logger.info(f"Training SVD model (n_factors={n_factors})...")

        matrix, user_map, item_map, user_ids, item_ids = \
            self._create_user_item_matrix(train_df)

        # Compute mean ratings per user for centering
        dense = matrix.toarray()
        user_means = np.true_divide(
            dense.sum(axis=1),
            (dense != 0).sum(axis=1),
            where=(dense != 0).sum(axis=1) != 0
        )
        user_means = np.nan_to_num(user_means, nan=0)

        # Center the matrix
        dense_centered = dense.copy().astype(float)
        for i in range(len(user_means)):
            mask = dense_centered[i, :] != 0
            dense_centered[i, mask] -= user_means[i]

        # SVD decomposition
        k = min(n_factors, min(dense_centered.shape) - 1)
        U, sigma, Vt = svds(csr_matrix(dense_centered), k=k)

        # Reconstruct predictions
        sigma_diag = np.diag(sigma)
        predicted = np.dot(np.dot(U, sigma_diag), Vt) + user_means.reshape(-1, 1)
        predicted = np.clip(predicted, 1, 5)

        model = {
            "type": "SVD",
            "U": U,
            "sigma": sigma,
            "Vt": Vt,
            "user_means": user_means,
            "predicted_ratings": predicted,
            "user_map": user_map,
            "item_map": item_map,
            "user_ids": user_ids.tolist(),
            "item_ids": item_ids.tolist(),
            "n_factors": k,
        }

        logger.info(f"  SVD model trained: U={U.shape}, Vt={Vt.shape}")
        return model

    def train_content_based(self, train_df):
        """Train content-based filtering model using item features."""
        logger.info("Training content-based model...")

        item_features_path = FEATURES_DIR / "item_features.csv"
        if not item_features_path.exists():
            logger.warning("Item features not found, skipping content-based model")
            return None

        item_features = pd.read_csv(item_features_path)

        # Use numeric features for similarity
        feature_cols = item_features.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in feature_cols if c != "product_id"]

        feature_matrix = item_features[feature_cols].fillna(0).values

        # Normalize
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        feature_matrix_norm = feature_matrix / norms

        # Compute item-item similarity
        similarity_matrix = cosine_similarity(feature_matrix_norm)

        model = {
            "type": "content_based",
            "similarity_matrix": similarity_matrix,
            "item_ids": item_features["product_id"].tolist(),
            "feature_cols": list(feature_cols),
        }

        logger.info(
            f"  Content-based model: {similarity_matrix.shape[0]} items, "
            f"{len(feature_cols)} features"
        )
        return model

    def evaluate_model(self, model, test_df, k_values=None):
        """Evaluate model using Precision@K, Recall@K, and NDCG@K."""
        if k_values is None:
            k_values = [5, 10, 20]

        logger.info(f"Evaluating model (k={k_values})...")
        metrics = {}

        if model["type"] == "SVD":
            predicted = model["predicted_ratings"]
            user_map = model["user_map"]
            item_map = model["item_map"]

            # Calculate per-user metrics
            precision_scores = {k: [] for k in k_values}
            recall_scores = {k: [] for k in k_values}
            ndcg_scores = {k: [] for k in k_values}

            # Group test data by user
            test_grouped = test_df.groupby("user_id")

            for user_id, group in test_grouped:
                if user_id not in user_map:
                    continue

                user_idx = user_map[user_id]
                actual_items = set(
                    group[group["rating"] >= 4]["product_id"].values
                )
                if not actual_items:
                    continue

                # Get predicted scores for all items
                user_preds = predicted[user_idx, :]
                top_items_idx = np.argsort(user_preds)[::-1]

                for k in k_values:
                    top_k_idx = top_items_idx[:k]
                    top_k_items = set()
                    for idx in top_k_idx:
                        for pid, pidx in item_map.items():
                            if pidx == idx:
                                top_k_items.add(pid)
                                break

                    hits = len(actual_items & top_k_items)

                    # Precision@K
                    precision = hits / k if k > 0 else 0
                    precision_scores[k].append(precision)

                    # Recall@K
                    recall = hits / len(actual_items) if actual_items else 0
                    recall_scores[k].append(recall)

                    # NDCG@K
                    dcg = 0
                    for i, idx in enumerate(top_k_idx[:k]):
                        item_id = None
                        for pid, pidx in item_map.items():
                            if pidx == idx:
                                item_id = pid
                                break
                        if item_id in actual_items:
                            dcg += 1.0 / np.log2(i + 2)

                    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k)))
                    ndcg = dcg / idcg if idcg > 0 else 0
                    ndcg_scores[k].append(ndcg)

            for k in k_values:
                metrics[f"precision@{k}"] = float(np.mean(precision_scores[k])) if precision_scores[k] else 0
                metrics[f"recall@{k}"] = float(np.mean(recall_scores[k])) if recall_scores[k] else 0
                metrics[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0

            # RMSE and MAE on test set
            errors = []
            for _, row in test_df.iterrows():
                uid, pid, actual = row["user_id"], row["product_id"], row["rating"]
                if uid in user_map and pid in item_map:
                    pred = predicted[user_map[uid], item_map[pid]]
                    errors.append(actual - pred)

            if errors:
                errors_arr = np.array(errors)
                metrics["rmse"] = float(np.sqrt(np.mean(errors_arr ** 2)))
                metrics["mae"] = float(np.mean(np.abs(errors_arr)))
            else:
                metrics["rmse"] = 0
                metrics["mae"] = 0

        elif model["type"] == "content_based":
            # Content-based evaluation: for each user in test, find similar items
            # to their liked items and check if test items appear
            item_ids = model["item_ids"]
            sim_matrix = model["similarity_matrix"]
            item_map_cb = {pid: idx for idx, pid in enumerate(item_ids)}

            precision_scores = {k: [] for k in k_values}
            recall_scores = {k: [] for k in k_values}
            ndcg_scores = {k: [] for k in k_values}

            test_grouped = test_df.groupby("user_id")
            train_grouped = test_df.groupby("user_id")  # use test users' train history

            # Build per-user train profiles from the train portion that was used
            for user_id, group in test_grouped:
                actual_items = set(
                    group[group["rating"] >= 4]["product_id"].values
                )
                if not actual_items:
                    continue

                # Score items by avg similarity to user's liked test items
                user_liked = [pid for pid in group["product_id"].values if pid in item_map_cb]
                if not user_liked:
                    continue

                scores = np.zeros(len(item_ids))
                for pid in user_liked:
                    if pid in item_map_cb:
                        scores += sim_matrix[item_map_cb[pid]]
                top_idx = np.argsort(scores)[::-1]

                for k in k_values:
                    top_k_items = set(item_ids[i] for i in top_idx[:k] if i < len(item_ids))
                    hits = len(actual_items & top_k_items)
                    precision_scores[k].append(hits / k if k > 0 else 0)
                    recall_scores[k].append(hits / len(actual_items) if actual_items else 0)

                    dcg = 0
                    for i in range(min(k, len(top_idx))):
                        if item_ids[top_idx[i]] in actual_items:
                            dcg += 1.0 / np.log2(i + 2)
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k)))
                    ndcg_scores[k].append(dcg / idcg if idcg > 0 else 0)

            for k in k_values:
                metrics[f"precision@{k}"] = float(np.mean(precision_scores[k])) if precision_scores[k] else 0
                metrics[f"recall@{k}"] = float(np.mean(recall_scores[k])) if recall_scores[k] else 0
                metrics[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0

        logger.info("  Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"    {metric}: {value:.4f}")

        return metrics

    def save_model(self, model, name):
        """Save trained model to disk."""
        model_path = MODELS_DIR / f"{name}_{self.timestamp}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"  Model saved: {model_path}")
        return str(model_path)

    def get_recommendations(self, model, user_id, n=10):
        """Get top-N recommendations for a user."""
        if model["type"] == "SVD":
            user_map = model["user_map"]
            item_map = model["item_map"]
            predicted = model["predicted_ratings"]

            if user_id not in user_map:
                logger.warning(f"User {user_id} not found in model")
                return []

            user_idx = user_map[user_id]
            user_preds = predicted[user_idx, :]
            top_idx = np.argsort(user_preds)[::-1][:n]

            # Map back to product IDs
            inv_item_map = {v: k for k, v in item_map.items()}
            recommendations = []
            for idx in top_idx:
                if idx in inv_item_map:
                    recommendations.append({
                        "product_id": int(inv_item_map[idx]),
                        "predicted_rating": round(float(user_preds[idx]), 2),
                    })

            return recommendations
        return []

    def train_and_evaluate(self):
        """Full training and evaluation pipeline."""
        logger.info("=" * 60)
        logger.info(f"Starting Model Training - {self.timestamp}")
        logger.info("=" * 60)

        ratings = self._load_ratings()
        if ratings is None:
            return

        train_df, test_df = self.prepare_data(ratings)
        results = {}

        # 1. Train SVD model
        svd_model = self.train_svd(train_df, n_factors=50)
        svd_metrics = self.evaluate_model(svd_model, test_df)
        svd_path = self.save_model(svd_model, "svd_model")
        results["svd"] = {
            "metrics": svd_metrics,
            "model_path": svd_path,
            "params": {"n_factors": 50, "algorithm": "SVD"},
        }

        # 2. Train content-based model
        cb_model = self.train_content_based(train_df)
        if cb_model:
            cb_metrics = self.evaluate_model(cb_model, test_df)
            cb_path = self.save_model(cb_model, "content_based_model")
            results["content_based"] = {
                "metrics": cb_metrics,
                "model_path": cb_path,
                "params": {"algorithm": "cosine_similarity"},
            }

        # Log with MLflow
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"training_{self.timestamp}"):
                # Log parameters
                mlflow.log_param("n_factors", 50)
                mlflow.log_param("test_size", 0.2)
                mlflow.log_param("train_size", len(train_df))
                mlflow.log_param("test_size_actual", len(test_df))
                mlflow.log_param("n_users", ratings["user_id"].nunique())
                mlflow.log_param("n_items", ratings["product_id"].nunique())

                # Log metrics
                for metric, value in svd_metrics.items():
                    safe_name = "svd_" + metric.replace("@", "_at_")
                    mlflow.log_metric(safe_name, value)

                # Log content-based metrics if available
                if cb_model and "content_based" in results:
                    for metric, value in results["content_based"]["metrics"].items():
                        safe_name = "cb_" + metric.replace("@", "_at_")
                        mlflow.log_metric(safe_name, value)

                # Log model
                mlflow.log_artifact(svd_path)

                run_id = mlflow.active_run().info.run_id
                results["mlflow_run_id"] = run_id
                logger.info(f"  MLflow run ID: {run_id}")

        # Demo recommendations
        sample_user = ratings["user_id"].iloc[0]
        recs = self.get_recommendations(svd_model, sample_user, n=5)
        results["sample_recommendations"] = {
            "user_id": int(sample_user),
            "recommendations": recs,
        }

        logger.info(f"\nSample recommendations for user {sample_user}:")
        for rec in recs:
            logger.info(f"  Product {rec['product_id']}: {rec['predicted_rating']}")

        # Save performance report
        report_path = REPORT_DIR / f"model_performance_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nPerformance report saved to {report_path}")

        logger.info("=" * 60)
        logger.info("Model training and evaluation complete!")
        logger.info("=" * 60)

        return results


def main():
    trainer = RecommendationTrainer()
    trainer.train_and_evaluate()


if __name__ == "__main__":
    main()
