"""
RecoMart - Feature Engineering and Transformation Module
Creates features for recommendation algorithms and stores them
in a structured format.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR = BASE_DIR / "data" / "features"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"


# SQL schema for the feature warehouse
SQL_SCHEMA = """
-- RecoMart Feature Warehouse Schema

CREATE TABLE IF NOT EXISTS user_features (
    user_id INTEGER PRIMARY KEY,
    total_ratings INTEGER,
    avg_rating REAL,
    rating_std REAL,
    total_purchases INTEGER,
    total_spend REAL,
    avg_order_value REAL,
    total_clicks INTEGER,
    view_count INTEGER,
    cart_add_count INTEGER,
    purchase_frequency REAL,
    activity_score REAL,
    account_age_days INTEGER,
    is_premium INTEGER,
    gender_encoded INTEGER
);

CREATE TABLE IF NOT EXISTS item_features (
    product_id INTEGER PRIMARY KEY,
    total_ratings INTEGER,
    avg_user_rating REAL,
    rating_variance REAL,
    total_purchases INTEGER,
    total_revenue REAL,
    total_views INTEGER,
    view_to_purchase_ratio REAL,
    popularity_score REAL,
    price REAL,
    price_normalized REAL,
    category_encoded INTEGER,
    in_stock INTEGER
);

CREATE TABLE IF NOT EXISTS user_item_features (
    user_id INTEGER,
    product_id INTEGER,
    rating REAL,
    n_interactions INTEGER,
    has_purchased INTEGER,
    has_viewed INTEGER,
    has_carted INTEGER,
    PRIMARY KEY (user_id, product_id)
);
"""


class FeatureEngineer:
    """Creates features for recommendation models."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(str(LOG_DIR / "transformation.log"), rotation="10 MB", level="INFO")
        self.feature_metadata = {}

    def _load_processed(self, name):
        """Load processed dataset."""
        path = PROCESSED_DIR / f"{name}_clean.csv"
        if path.exists():
            return pd.read_csv(path)
        logger.warning(f"Processed file not found: {path}")
        return None

    def create_user_features(self):
        """Create user-level features from multiple data sources."""
        logger.info("Creating user features...")

        users = self._load_processed("users")
        ratings = self._load_processed("ratings")
        transactions = self._load_processed("transactions")
        clickstream = self._load_processed("clickstream")

        if users is None:
            return None

        user_features = users[["user_id"]].copy()

        # Rating features
        if ratings is not None:
            rating_agg = ratings.groupby("user_id").agg(
                total_ratings=("rating", "count"),
                avg_rating=("rating", "mean"),
                rating_std=("rating", "std"),
            ).reset_index()
            rating_agg["rating_std"] = rating_agg["rating_std"].fillna(0)
            user_features = user_features.merge(rating_agg, on="user_id", how="left")

        # Transaction features
        if transactions is not None:
            txn_completed = transactions[transactions["status"] == "completed"]
            txn_agg = txn_completed.groupby("user_id").agg(
                total_purchases=("transaction_id", "count"),
                total_spend=("total_amount", "sum"),
                avg_order_value=("total_amount", "mean"),
            ).reset_index()
            user_features = user_features.merge(txn_agg, on="user_id", how="left")

        # Clickstream features
        if clickstream is not None:
            click_agg = clickstream.groupby("user_id").agg(
                total_clicks=("event_id", "count"),
            ).reset_index()

            # Event-specific counts
            for event_type in ["view", "add_to_cart"]:
                col_name = f"{event_type.replace(' ', '_')}_count"
                event_counts = (
                    clickstream[clickstream["event_type"] == event_type]
                    .groupby("user_id")
                    .size()
                    .reset_index(name=col_name)
                )
                click_agg = click_agg.merge(event_counts, on="user_id", how="left")

            user_features = user_features.merge(click_agg, on="user_id", how="left")

        # Merge user metadata
        if "account_age_days" in users.columns:
            user_features = user_features.merge(
                users[["user_id", "account_age_days", "is_premium", "gender_encoded"]],
                on="user_id", how="left"
            )

        # Fill NAs
        user_features = user_features.fillna(0)

        # Derived features
        user_features["purchase_frequency"] = np.where(
            user_features.get("account_age_days", 0) > 0,
            user_features.get("total_purchases", 0) / (user_features.get("account_age_days", 1) + 1),
            0
        )
        user_features["activity_score"] = (
            user_features.get("total_clicks", 0) * 0.3
            + user_features.get("total_purchases", 0) * 0.5
            + user_features.get("total_ratings", 0) * 0.2
        )

        # Save
        user_features.to_csv(FEATURES_DIR / "user_features.csv", index=False)
        logger.info(f"  User features: {user_features.shape}")

        self.feature_metadata["user_features"] = {
            "n_features": len(user_features.columns) - 1,
            "n_users": len(user_features),
            "features": list(user_features.columns),
            "description": {
                "total_ratings": "Number of ratings given by user",
                "avg_rating": "Average rating given by user",
                "rating_std": "Standard deviation of user ratings",
                "total_purchases": "Total completed purchases",
                "total_spend": "Total amount spent",
                "avg_order_value": "Average transaction amount",
                "total_clicks": "Total clickstream events",
                "purchase_frequency": "Purchases per day since signup",
                "activity_score": "Weighted engagement score",
            }
        }

        return user_features

    def create_item_features(self):
        """Create item-level features."""
        logger.info("Creating item features...")

        products = self._load_processed("products")
        ratings = self._load_processed("ratings")
        transactions = self._load_processed("transactions")
        clickstream = self._load_processed("clickstream")

        if products is None:
            return None

        item_features = products[["product_id"]].copy()

        # Rating features
        if ratings is not None:
            rating_agg = ratings.groupby("product_id").agg(
                total_ratings=("rating", "count"),
                avg_user_rating=("rating", "mean"),
                rating_variance=("rating", "var"),
            ).reset_index()
            rating_agg["rating_variance"] = rating_agg["rating_variance"].fillna(0)
            item_features = item_features.merge(rating_agg, on="product_id", how="left")

        # Transaction features
        if transactions is not None:
            txn_completed = transactions[transactions["status"] == "completed"]
            txn_agg = txn_completed.groupby("product_id").agg(
                total_purchases=("transaction_id", "count"),
                total_revenue=("total_amount", "sum"),
            ).reset_index()
            item_features = item_features.merge(txn_agg, on="product_id", how="left")

        # Clickstream features
        if clickstream is not None:
            view_counts = (
                clickstream[clickstream["event_type"] == "view"]
                .groupby("product_id")
                .size()
                .reset_index(name="total_views")
            )
            item_features = item_features.merge(view_counts, on="product_id", how="left")

        # Merge product metadata
        meta_cols = ["product_id", "price", "category_encoded", "in_stock"]
        available_cols = [c for c in meta_cols if c in products.columns]
        if available_cols:
            item_features = item_features.merge(
                products[available_cols], on="product_id", how="left"
            )

        if "price_normalized" in products.columns:
            item_features = item_features.merge(
                products[["product_id", "price_normalized"]], on="product_id", how="left"
            )

        # Fill NAs
        item_features = item_features.fillna(0)

        # Derived features
        item_features["view_to_purchase_ratio"] = np.where(
            item_features.get("total_views", 0) > 0,
            item_features.get("total_purchases", 0) / (item_features.get("total_views", 1) + 1),
            0
        )
        item_features["popularity_score"] = (
            item_features.get("total_purchases", 0) * 0.4
            + item_features.get("total_ratings", 0) * 0.3
            + item_features.get("total_views", 0) * 0.01 * 0.3
        )

        # Save
        item_features.to_csv(FEATURES_DIR / "item_features.csv", index=False)
        logger.info(f"  Item features: {item_features.shape}")

        self.feature_metadata["item_features"] = {
            "n_features": len(item_features.columns) - 1,
            "n_items": len(item_features),
            "features": list(item_features.columns),
            "description": {
                "total_ratings": "Number of ratings received",
                "avg_user_rating": "Average rating from users",
                "total_purchases": "Total purchase count",
                "total_revenue": "Total revenue from product",
                "total_views": "Total view count",
                "view_to_purchase_ratio": "Conversion rate from view to purchase",
                "popularity_score": "Weighted popularity metric",
            }
        }

        return item_features

    def create_user_item_features(self):
        """Create user-item interaction features."""
        logger.info("Creating user-item interaction features...")

        ratings = self._load_processed("ratings")
        transactions = self._load_processed("transactions")
        clickstream = self._load_processed("clickstream")

        if ratings is None:
            return None

        # Start with ratings as base
        ui_features = ratings[["user_id", "product_id", "rating"]].copy()

        # Add interaction counts from clickstream
        if clickstream is not None:
            interaction_counts = (
                clickstream.groupby(["user_id", "product_id"])
                .size()
                .reset_index(name="n_interactions")
            )
            ui_features = ui_features.merge(
                interaction_counts, on=["user_id", "product_id"], how="left"
            )

            # Has viewed
            viewed = (
                clickstream[clickstream["event_type"] == "view"]
                .groupby(["user_id", "product_id"])
                .size()
                .reset_index(name="has_viewed")
            )
            viewed["has_viewed"] = 1
            ui_features = ui_features.merge(
                viewed[["user_id", "product_id", "has_viewed"]],
                on=["user_id", "product_id"], how="left"
            )

            # Has carted
            carted = (
                clickstream[clickstream["event_type"] == "add_to_cart"]
                .groupby(["user_id", "product_id"])
                .size()
                .reset_index(name="has_carted")
            )
            carted["has_carted"] = 1
            ui_features = ui_features.merge(
                carted[["user_id", "product_id", "has_carted"]],
                on=["user_id", "product_id"], how="left"
            )

        # Has purchased
        if transactions is not None:
            purchased = (
                transactions[transactions["status"] == "completed"]
                .groupby(["user_id", "product_id"])
                .size()
                .reset_index(name="has_purchased")
            )
            purchased["has_purchased"] = 1
            ui_features = ui_features.merge(
                purchased[["user_id", "product_id", "has_purchased"]],
                on=["user_id", "product_id"], how="left"
            )

        ui_features = ui_features.fillna(0)
        ui_features.to_csv(FEATURES_DIR / "user_item_features.csv", index=False)
        logger.info(f"  User-item features: {ui_features.shape}")

        self.feature_metadata["user_item_features"] = {
            "n_pairs": len(ui_features),
            "features": list(ui_features.columns),
        }

        return ui_features

    def transform_all(self):
        """Run all feature engineering transformations."""
        logger.info("=" * 60)
        logger.info(f"Starting Feature Engineering - {self.timestamp}")
        logger.info("=" * 60)

        self.create_user_features()
        self.create_item_features()
        self.create_user_item_features()

        # Save SQL schema
        schema_path = FEATURES_DIR / "schema.sql"
        with open(schema_path, "w") as f:
            f.write(SQL_SCHEMA)
        logger.info(f"SQL schema saved to {schema_path}")

        # Save feature metadata
        meta_path = REPORT_DIR / f"feature_metadata_{self.timestamp}.json"
        with open(meta_path, "w") as f:
            json.dump(self.feature_metadata, f, indent=2, default=str)
        logger.info(f"Feature metadata saved to {meta_path}")

        logger.info("=" * 60)
        logger.info("Feature engineering complete!")
        logger.info("=" * 60)


def main():
    engineer = FeatureEngineer()
    engineer.transform_all()


if __name__ == "__main__":
    main()
