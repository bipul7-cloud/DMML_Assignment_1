"""
RecoMart - Feature Store Implementation
A simple feature store with metadata registry, versioning, and retrieval.
"""

import json
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
STORE_DIR = BASE_DIR / "data" / "feature_store"
LOG_DIR = BASE_DIR / "logs"


class FeatureStore:
    """
    Simple feature store with:
    - SQLite-based metadata registry
    - Versioned feature storage
    - Feature retrieval for training and inference
    """

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        (STORE_DIR / "versions").mkdir(parents=True, exist_ok=True)
        (STORE_DIR / "online").mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(str(LOG_DIR / "feature_store.log"), rotation="10 MB", level="INFO")

        self.registry_path = STORE_DIR / "registry.db"
        self._init_registry()

    def _init_registry(self):
        """Initialize the metadata registry database."""
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT,
                source TEXT,
                n_features INTEGER,
                n_rows INTEGER,
                feature_names TEXT,
                created_at TEXT,
                file_path TEXT,
                status TEXT DEFAULT 'active',
                UNIQUE(name, version)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_set_name TEXT,
                feature_name TEXT,
                description TEXT,
                dtype TEXT,
                source_table TEXT,
                transformation TEXT,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Feature store registry initialized.")

    def register_feature_set(self, name, df, description="", source=""):
        """Register a feature set in the store with versioning."""
        version = f"v_{self.timestamp}"
        logger.info(f"Registering feature set: {name} ({version})")

        # Save versioned copy
        version_dir = STORE_DIR / "versions" / name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        file_path = version_dir / f"{name}.csv"
        df.to_csv(file_path, index=False)

        # Save to online store (latest version)
        online_path = STORE_DIR / "online" / f"{name}.csv"
        df.to_csv(online_path, index=False)

        # Register in metadata
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO feature_sets
            (name, version, description, source, n_features, n_rows,
             feature_names, created_at, file_path, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, version, description, source,
            len(df.columns), len(df),
            json.dumps(list(df.columns)),
            datetime.now().isoformat(),
            str(file_path),
            "active"
        ))
        conn.commit()
        conn.close()

        logger.info(f"  Registered {name}: {df.shape[0]} rows, {df.shape[1]} features")
        return version

    def register_feature_metadata(self, feature_set_name, feature_descriptions):
        """Register individual feature descriptions."""
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()

        for feat_name, info in feature_descriptions.items():
            cursor.execute("""
                INSERT INTO feature_metadata
                (feature_set_name, feature_name, description, dtype,
                 source_table, transformation, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                feature_set_name,
                feat_name,
                info.get("description", ""),
                info.get("dtype", ""),
                info.get("source_table", ""),
                info.get("transformation", ""),
                datetime.now().isoformat(),
            ))

        conn.commit()
        conn.close()
        logger.info(
            f"  Registered {len(feature_descriptions)} feature descriptions "
            f"for {feature_set_name}"
        )

    def get_feature_set(self, name, version=None):
        """Retrieve a feature set (latest or specific version)."""
        if version:
            path = STORE_DIR / "versions" / name / version / f"{name}.csv"
        else:
            path = STORE_DIR / "online" / f"{name}.csv"

        if not path.exists():
            logger.warning(f"Feature set not found: {path}")
            return None

        df = pd.read_csv(path)
        logger.info(f"Retrieved {name}: {df.shape}")
        return df

    def list_feature_sets(self):
        """List all registered feature sets."""
        conn = sqlite3.connect(str(self.registry_path))
        df = pd.read_sql_query(
            "SELECT name, version, n_features, n_rows, created_at, status "
            "FROM feature_sets ORDER BY created_at DESC",
            conn
        )
        conn.close()
        return df

    def list_versions(self, name):
        """List all versions of a feature set."""
        conn = sqlite3.connect(str(self.registry_path))
        df = pd.read_sql_query(
            "SELECT version, n_features, n_rows, created_at, status "
            "FROM feature_sets WHERE name = ? ORDER BY created_at DESC",
            conn,
            params=(name,)
        )
        conn.close()
        return df

    def get_feature_metadata_report(self):
        """Generate feature metadata documentation."""
        conn = sqlite3.connect(str(self.registry_path))
        sets_df = pd.read_sql_query("SELECT * FROM feature_sets", conn)
        meta_df = pd.read_sql_query("SELECT * FROM feature_metadata", conn)
        conn.close()
        return sets_df, meta_df


def main():
    logger.info("=" * 60)
    logger.info("Initializing Feature Store")
    logger.info("=" * 60)

    store = FeatureStore()

    # Register feature sets from feature engineering output
    feature_files = {
        "user_features": {
            "description": "User-level aggregated features for recommendation",
            "source": "users, ratings, transactions, clickstream",
        },
        "item_features": {
            "description": "Item-level aggregated features for recommendation",
            "source": "products, ratings, transactions, clickstream",
        },
        "user_item_features": {
            "description": "User-item interaction features",
            "source": "ratings, transactions, clickstream",
        },
    }

    for name, info in feature_files.items():
        path = FEATURES_DIR / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            version = store.register_feature_set(
                name, df,
                description=info["description"],
                source=info["source"]
            )

            # Register feature-level metadata
            feature_descs = {}
            for col in df.columns:
                feature_descs[col] = {
                    "description": f"Feature: {col}",
                    "dtype": str(df[col].dtype),
                    "source_table": info["source"],
                    "transformation": "aggregation",
                }
            store.register_feature_metadata(name, feature_descs)
        else:
            logger.warning(f"Feature file not found: {path}")

    # Demonstrate retrieval
    logger.info("\n--- Feature Store Demo ---")
    print("\nRegistered Feature Sets:")
    print(store.list_feature_sets().to_string())

    # Retrieve and show sample
    for name in feature_files:
        df = store.get_feature_set(name)
        if df is not None:
            print(f"\n{name} sample (first 3 rows):")
            print(df.head(3).to_string())

    logger.info("=" * 60)
    logger.info("Feature store operations complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
