"""
RecoMart - Data Profiling and Validation Module
Validates data quality: missing values, duplicates, schema mismatches,
range checks, and generates a comprehensive data quality report.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
VALIDATED_DIR = BASE_DIR / "data" / "validated"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

# Expected schemas for each dataset
SCHEMAS = {
    "users": {
        "required_columns": [
            "user_id", "username", "email", "age", "gender",
            "country", "signup_date", "is_premium"
        ],
        "dtypes": {
            "user_id": "int64",
            "username": "object",
            "email": "object",
            "age": "float64",
            "gender": "object",
            "country": "object",
            "signup_date": "object",
            "is_premium": "object",
        },
    },
    "products": {
        "required_columns": [
            "product_id", "name", "category", "subcategory",
            "price", "brand", "avg_rating", "n_reviews", "in_stock", "description"
        ],
        "dtypes": {
            "product_id": "int64",
            "price": "float64",
            "avg_rating": "float64",
            "n_reviews": "int64",
        },
    },
    "clickstream": {
        "required_columns": [
            "event_id", "user_id", "product_id", "event_type",
            "timestamp", "session_id", "device", "page_url"
        ],
        "dtypes": {
            "user_id": "int64",
            "product_id": "int64",
        },
    },
    "transactions": {
        "required_columns": [
            "transaction_id", "user_id", "product_id", "quantity",
            "unit_price", "total_amount", "payment_method",
            "transaction_date", "status"
        ],
        "dtypes": {
            "transaction_id": "int64",
            "user_id": "int64",
            "product_id": "int64",
            "quantity": "int64",
        },
    },
    "ratings": {
        "required_columns": [
            "user_id", "product_id", "rating", "review_text", "timestamp"
        ],
        "dtypes": {
            "user_id": "int64",
            "product_id": "int64",
            "rating": "int64",
        },
    },
}


class DataValidator:
    """Validates datasets and generates quality reports."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(str(LOG_DIR / "validation.log"), rotation="10 MB", level="INFO")
        self.quality_report = {}

    def _load_latest(self, source_name):
        """Load the latest file from a raw data source directory."""
        source_dir = RAW_DIR / source_name
        if not source_dir.exists():
            logger.warning(f"Directory not found: {source_dir}")
            return None

        csv_files = sorted(source_dir.glob("*.csv"), reverse=True)
        if not csv_files:
            logger.warning(f"No CSV files in {source_dir}")
            return None

        df = pd.read_csv(csv_files[0])
        logger.info(f"Loaded {csv_files[0].name}: {df.shape}")
        return df

    def check_schema(self, df, source_name):
        """Validate schema: column names and types."""
        schema = SCHEMAS.get(source_name, {})
        issues = []

        # Check required columns
        required = schema.get("required_columns", [])
        missing_cols = [c for c in required if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in required]

        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        if extra_cols:
            issues.append(f"Extra columns: {extra_cols}")

        return {
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "column_count": len(df.columns),
            "expected_count": len(required),
            "issues": issues,
        }

    def check_missing_values(self, df, source_name):
        """Check for missing values across columns."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        total_missing = int(missing.sum())
        total_missing_pct = round(total_missing / (len(df) * len(df.columns)) * 100, 2)

        return {
            "total_missing_cells": total_missing,
            "total_missing_pct": total_missing_pct,
            "per_column": {
                col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
                for col in df.columns if missing[col] > 0
            },
        }

    def check_duplicates(self, df, source_name):
        """Check for duplicate rows."""
        n_duplicates = int(df.duplicated().sum())
        dup_pct = round(n_duplicates / len(df) * 100, 2)

        return {
            "total_duplicates": n_duplicates,
            "duplicate_pct": dup_pct,
        }

    def check_ranges(self, df, source_name):
        """Validate value ranges for known fields."""
        issues = []

        if source_name == "ratings" and "rating" in df.columns:
            invalid = df[
                pd.to_numeric(df["rating"], errors="coerce").apply(
                    lambda x: x < 1 or x > 5 if pd.notna(x) else False
                )
            ]
            if len(invalid) > 0:
                issues.append(
                    f"Ratings out of range [1-5]: {len(invalid)} rows"
                )

        if source_name == "products" and "price" in df.columns:
            negative_prices = df[
                pd.to_numeric(df["price"], errors="coerce") < 0
            ]
            if len(negative_prices) > 0:
                issues.append(f"Negative prices: {len(negative_prices)} rows")

        if source_name == "transactions" and "quantity" in df.columns:
            invalid_qty = df[
                pd.to_numeric(df["quantity"], errors="coerce") <= 0
            ]
            if len(invalid_qty) > 0:
                issues.append(f"Invalid quantities (<=0): {len(invalid_qty)} rows")

        return {"range_issues": issues, "n_issues": len(issues)}

    def profile_data(self, df, source_name):
        """Generate a data profile summary."""
        profile = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }

        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            desc = df[numeric_cols].describe().round(2).to_dict()
            profile["numeric_summary"] = {
                col: {k: float(v) for k, v in stats.items()}
                for col, stats in desc.items()
            }

        # Categorical summary
        cat_cols = df.select_dtypes(include=["object"]).columns
        if len(cat_cols) > 0:
            profile["categorical_summary"] = {}
            for col in cat_cols:
                profile["categorical_summary"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(5).to_dict(),
                }

        return profile

    def validate_source(self, source_name):
        """Run all validation checks on a data source."""
        logger.info(f"{'=' * 40}")
        logger.info(f"Validating: {source_name}")
        logger.info(f"{'=' * 40}")

        df = self._load_latest(source_name)
        if df is None:
            return {"status": "SKIPPED", "reason": "No data found"}

        report = {
            "source": source_name,
            "timestamp": self.timestamp,
            "profile": self.profile_data(df, source_name),
            "schema": self.check_schema(df, source_name),
            "missing_values": self.check_missing_values(df, source_name),
            "duplicates": self.check_duplicates(df, source_name),
            "range_checks": self.check_ranges(df, source_name),
        }

        # Determine overall quality status
        issues = []
        if report["schema"]["missing_columns"]:
            issues.append("Schema mismatch")
        if report["missing_values"]["total_missing_pct"] > 15:
            issues.append("High missing rate")
        if report["duplicates"]["duplicate_pct"] > 5:
            issues.append("High duplicate rate")
        if report["range_checks"]["n_issues"] > 0:
            issues.append("Range violations")

        report["quality_status"] = "PASS" if not issues else "WARN"
        report["quality_issues"] = issues

        # Log summary
        logger.info(f"  Rows: {report['profile']['shape']['rows']}")
        logger.info(f"  Missing: {report['missing_values']['total_missing_pct']}%")
        logger.info(f"  Duplicates: {report['duplicates']['duplicate_pct']}%")
        logger.info(f"  Status: {report['quality_status']}")
        if issues:
            for issue in issues:
                logger.warning(f"  Issue: {issue}")

        # Save validated data (keeping all rows, flagged)
        output_dir = VALIDATED_DIR / source_name
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            output_dir / f"{source_name}_validated_{self.timestamp}.csv",
            index=False,
        )

        return report

    def validate_all(self):
        """Run validation on all data sources."""
        logger.info("=" * 60)
        logger.info(f"Starting Data Validation Pipeline - {self.timestamp}")
        logger.info("=" * 60)

        sources = ["users", "products", "clickstream", "transactions", "ratings"]
        for source in sources:
            self.quality_report[source] = self.validate_source(source)

        # Save full quality report
        report_path = REPORT_DIR / f"data_quality_report_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.quality_report, f, indent=2, default=str)

        logger.info(f"Quality report saved to {report_path}")
        logger.info("=" * 60)
        logger.info("Validation Summary:")
        for source, report in self.quality_report.items():
            status = report.get("quality_status", "SKIPPED")
            logger.info(f"  {source}: {status}")
        logger.info("=" * 60)

        return self.quality_report


def main():
    validator = DataValidator()
    validator.validate_all()


if __name__ == "__main__":
    main()
