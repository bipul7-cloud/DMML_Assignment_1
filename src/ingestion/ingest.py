"""
RecoMart - Data Ingestion Module
Handles automated ingestion from CSV files and REST APIs with:
- Error handling and retry mechanisms
- Logging for monitoring and audit trails
- Scheduled periodic data fetching
"""

import os
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
import yaml
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "pipeline_config.yaml"
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class DataIngestor:
    """Handles data ingestion from multiple sources."""

    def __init__(self):
        self.config = load_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / "ingestion.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_file), rotation="10 MB", level="INFO")

    def ingest_csv(self, source_name, source_config):
        """Ingest data from CSV files with validation."""
        logger.info(f"[CSV Ingestion] Starting ingestion for: {source_name}")
        source_path = BASE_DIR / source_config["path"]

        if not source_path.exists():
            logger.warning(f"Source path does not exist: {source_path}")
            return None

        csv_files = list(source_path.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {source_path}")
            return None

        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
                logger.info(
                    f"  Loaded {csv_file.name}: {len(df)} rows, {len(df.columns)} columns"
                )
            except Exception as e:
                logger.error(f"  Failed to load {csv_file.name}: {e}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(
                f"[CSV Ingestion] {source_name} complete: "
                f"{len(combined)} total rows ingested"
            )
            return combined
        return None

    def ingest_api(self, source_name, source_config, max_retries=3):
        """Ingest data from REST API with retry mechanism."""
        endpoint = source_config["endpoint"]
        logger.info(f"[API Ingestion] Fetching from: {endpoint}")

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(endpoint, timeout=30)
                response.raise_for_status()
                data = response.json()

                # Save raw API response
                output_dir = BASE_DIR / source_config["path"]
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"api_{source_name}_{self.timestamp}.json"

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                logger.info(
                    f"[API Ingestion] {source_name}: {len(data)} records saved "
                    f"to {output_file.name}"
                )

                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    return df
                return pd.DataFrame(data) if data else None

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"[API Ingestion] Attempt {attempt}/{max_retries} failed: {e}"
                )
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.info(f"  Retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    logger.error(
                        f"[API Ingestion] All {max_retries} attempts failed "
                        f"for {source_name}"
                    )
                    return None

    def ingest_all(self):
        """Run ingestion for all configured data sources."""
        logger.info("=" * 60)
        logger.info(f"Starting Full Data Ingestion - {self.timestamp}")
        logger.info("=" * 60)

        results = {}
        sources = self.config.get("data_sources", {})

        for source_name, source_config in sources.items():
            source_type = source_config.get("type", "csv")
            try:
                if source_type == "csv":
                    df = self.ingest_csv(source_name, source_config)
                elif source_type == "api":
                    df = self.ingest_api(source_name, source_config)
                else:
                    logger.warning(f"Unknown source type: {source_type}")
                    continue

                if df is not None:
                    results[source_name] = {
                        "status": "SUCCESS",
                        "rows": len(df),
                        "columns": len(df.columns),
                        "timestamp": self.timestamp,
                    }
                else:
                    results[source_name] = {
                        "status": "EMPTY",
                        "timestamp": self.timestamp,
                    }
            except Exception as e:
                logger.error(f"Ingestion failed for {source_name}: {e}")
                results[source_name] = {
                    "status": "FAILED",
                    "error": str(e),
                    "timestamp": self.timestamp,
                }

        # Save ingestion report
        report_path = LOG_DIR / f"ingestion_report_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 60)
        logger.info("Ingestion Summary:")
        for name, info in results.items():
            logger.info(f"  {name}: {info['status']}")
        logger.info("=" * 60)

        return results


def main():
    ingestor = DataIngestor()
    ingestor.ingest_all()


if __name__ == "__main__":
    main()
