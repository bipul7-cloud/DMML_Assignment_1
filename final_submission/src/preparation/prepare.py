"""
RecoMart - Data Preparation Module
Handles data cleaning, preprocessing, encoding, normalization,
and generates EDA summaries.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"


class DataPreparator:
    """Cleans, preprocesses, and prepares data for feature engineering."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(str(LOG_DIR / "preparation.log"), rotation="10 MB", level="INFO")
        self.label_encoders = {}
        self.scalers = {}
        self.eda_stats = {}

    def _load_latest(self, source_name):
        """Load the latest CSV from raw data directory."""
        source_dir = RAW_DIR / source_name
        csv_files = sorted(source_dir.glob("*.csv"), reverse=True)
        if not csv_files:
            return None
        return pd.read_csv(csv_files[0])

    def clean_users(self):
        """Clean user data."""
        logger.info("Cleaning users data...")
        df = self._load_latest("users")
        if df is None:
            return None

        # Handle missing ages - fill with median
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        median_age = df["age"].median()
        df["age"] = df["age"].fillna(median_age).astype(int)

        # Handle missing gender
        df["gender"] = df["gender"].replace("", np.nan)
        df["gender"] = df["gender"].fillna("Unknown")

        # Encode gender
        le = LabelEncoder()
        df["gender_encoded"] = le.fit_transform(df["gender"])
        self.label_encoders["gender"] = le

        # Encode is_premium
        df["is_premium"] = df["is_premium"].map(
            {"True": 1, "False": 0, True: 1, False: 0}
        ).fillna(0).astype(int)

        # Parse signup date
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
        df["account_age_days"] = (
            pd.Timestamp.now() - df["signup_date"]
        ).dt.days.fillna(0).astype(int)

        # EDA stats
        self.eda_stats["users"] = {
            "total_users": len(df),
            "age_distribution": {
                "mean": float(df["age"].mean()),
                "std": float(df["age"].std()),
                "min": int(df["age"].min()),
                "max": int(df["age"].max()),
            },
            "gender_distribution": df["gender"].value_counts().to_dict(),
            "premium_rate": float(df["is_premium"].mean()),
            "countries": int(df["country"].nunique()),
        }

        logger.info(f"  Users cleaned: {len(df)} rows")
        return df

    def clean_products(self):
        """Clean product data."""
        logger.info("Cleaning products data...")
        df = self._load_latest("products")
        if df is None:
            return None

        # Clean price
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["price"] = df["price"].fillna(df["price"].median())

        # Normalize price
        scaler = MinMaxScaler()
        df["price_normalized"] = scaler.fit_transform(df[["price"]])
        self.scalers["price"] = scaler

        # Clean avg_rating
        df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce")
        df["avg_rating"] = df["avg_rating"].clip(1, 5)
        df["avg_rating"] = df["avg_rating"].fillna(df["avg_rating"].median())

        # Encode category
        le = LabelEncoder()
        df["category_encoded"] = le.fit_transform(df["category"])
        self.label_encoders["category"] = le

        # Boolean encoding
        df["in_stock"] = df["in_stock"].map(
            {"True": 1, "False": 0, True: 1, False: 0}
        ).fillna(1).astype(int)

        # EDA stats
        self.eda_stats["products"] = {
            "total_products": len(df),
            "categories": df["category"].nunique(),
            "category_distribution": df["category"].value_counts().to_dict(),
            "price_stats": {
                "mean": float(df["price"].mean()),
                "std": float(df["price"].std()),
                "min": float(df["price"].min()),
                "max": float(df["price"].max()),
            },
            "avg_rating_stats": {
                "mean": float(df["avg_rating"].mean()),
                "std": float(df["avg_rating"].std()),
            },
            "in_stock_rate": float(df["in_stock"].mean()),
        }

        logger.info(f"  Products cleaned: {len(df)} rows")
        return df

    def clean_clickstream(self):
        """Clean clickstream data."""
        logger.info("Cleaning clickstream data...")
        df = self._load_latest("clickstream")
        if df is None:
            return None

        # Remove duplicates
        n_before = len(df)
        df = df.drop_duplicates(subset=["event_id"])
        n_removed = n_before - len(df)
        logger.info(f"  Removed {n_removed} duplicate events")

        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

        # Encode event_type
        le = LabelEncoder()
        df["event_type_encoded"] = le.fit_transform(df["event_type"])
        self.label_encoders["event_type"] = le

        # Encode device
        le_device = LabelEncoder()
        df["device_encoded"] = le_device.fit_transform(df["device"])
        self.label_encoders["device"] = le_device

        # EDA stats
        self.eda_stats["clickstream"] = {
            "total_events": len(df),
            "unique_users": int(df["user_id"].nunique()),
            "unique_products": int(df["product_id"].nunique()),
            "event_distribution": df["event_type"].value_counts().to_dict(),
            "device_distribution": df["device"].value_counts().to_dict(),
            "hourly_distribution": df["hour_of_day"].value_counts().sort_index().to_dict(),
            "duplicates_removed": n_removed,
        }

        logger.info(f"  Clickstream cleaned: {len(df)} rows")
        return df

    def clean_transactions(self):
        """Clean transaction data."""
        logger.info("Cleaning transactions data...")
        df = self._load_latest("transactions")
        if df is None:
            return None

        # Handle missing total_amount
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
        mask = df["total_amount"].isna()
        df.loc[mask, "total_amount"] = df.loc[mask, "unit_price"] * df.loc[mask, "quantity"]

        # Remove cancelled/refunded for clean dataset
        df_completed = df[df["status"] == "completed"].copy()

        # Parse date
        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"], errors="coerce"
        )

        # Encode payment method
        le = LabelEncoder()
        df["payment_encoded"] = le.fit_transform(df["payment_method"])
        self.label_encoders["payment_method"] = le

        # EDA stats
        self.eda_stats["transactions"] = {
            "total_transactions": len(df),
            "completed_transactions": len(df_completed),
            "unique_buyers": int(df["user_id"].nunique()),
            "status_distribution": df["status"].value_counts().to_dict(),
            "payment_distribution": df["payment_method"].value_counts().to_dict(),
            "amount_stats": {
                "mean": float(df["total_amount"].mean()),
                "median": float(df["total_amount"].median()),
                "total_revenue": float(df_completed["total_amount"].sum()),
            },
        }

        logger.info(f"  Transactions cleaned: {len(df)} rows")
        return df

    def clean_ratings(self):
        """Clean ratings data."""
        logger.info("Cleaning ratings data...")
        df = self._load_latest("ratings")
        if df is None:
            return None

        # Filter valid ratings (1-5)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        invalid_count = int(((df["rating"] < 1) | (df["rating"] > 5)).sum())
        df = df[(df["rating"] >= 1) & (df["rating"] <= 5)].copy()
        logger.info(f"  Removed {invalid_count} invalid ratings")

        # Remove duplicates (keep latest)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").drop_duplicates(
            subset=["user_id", "product_id"], keep="last"
        )

        # Interaction matrix sparsity
        n_users = df["user_id"].nunique()
        n_products = df["product_id"].nunique()
        n_ratings = len(df)
        sparsity = 1 - (n_ratings / (n_users * n_products))

        self.eda_stats["ratings"] = {
            "total_ratings": n_ratings,
            "unique_users": n_users,
            "unique_products": n_products,
            "rating_distribution": df["rating"].value_counts().sort_index().to_dict(),
            "avg_rating": float(df["rating"].mean()),
            "sparsity": round(sparsity, 4),
            "invalid_removed": invalid_count,
            "ratings_per_user": {
                "mean": float(df.groupby("user_id").size().mean()),
                "median": float(df.groupby("user_id").size().median()),
            },
            "ratings_per_product": {
                "mean": float(df.groupby("product_id").size().mean()),
                "median": float(df.groupby("product_id").size().median()),
            },
        }

        logger.info(f"  Ratings cleaned: {len(df)} rows, sparsity={sparsity:.2%}")
        return df

    def prepare_all(self):
        """Run preparation on all datasets."""
        logger.info("=" * 60)
        logger.info(f"Starting Data Preparation - {self.timestamp}")
        logger.info("=" * 60)

        datasets = {}

        users = self.clean_users()
        if users is not None:
            users.to_csv(PROCESSED_DIR / "users_clean.csv", index=False)
            datasets["users"] = users

        products = self.clean_products()
        if products is not None:
            products.to_csv(PROCESSED_DIR / "products_clean.csv", index=False)
            datasets["products"] = products

        clickstream = self.clean_clickstream()
        if clickstream is not None:
            clickstream.to_csv(PROCESSED_DIR / "clickstream_clean.csv", index=False)
            datasets["clickstream"] = clickstream

        transactions = self.clean_transactions()
        if transactions is not None:
            transactions.to_csv(PROCESSED_DIR / "transactions_clean.csv", index=False)
            datasets["transactions"] = transactions

        ratings = self.clean_ratings()
        if ratings is not None:
            ratings.to_csv(PROCESSED_DIR / "ratings_clean.csv", index=False)
            datasets["ratings"] = ratings

        # Save EDA stats
        eda_path = REPORT_DIR / f"eda_summary_{self.timestamp}.json"
        with open(eda_path, "w") as f:
            json.dump(self.eda_stats, f, indent=2, default=str)
        logger.info(f"EDA summary saved to {eda_path}")

        logger.info("=" * 60)
        logger.info("Data preparation complete!")
        logger.info("=" * 60)

        return datasets


def main():
    preparator = DataPreparator()
    preparator.prepare_all()


if __name__ == "__main__":
    main()
