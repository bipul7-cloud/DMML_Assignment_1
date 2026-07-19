"""
RecoMart - Pipeline Orchestration
Automates the end-to-end data pipeline using Prefect.
DAG: ingestion → validation → preparation → transformation → feature store → model training
"""

import sys
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"

try:
    from prefect import flow, task, get_run_logger
    from prefect.tasks import task_input_hash
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(BASE_DIR))


# ===================== TASK DEFINITIONS =====================

def run_data_generation():
    """Task 1: Generate synthetic data."""
    logger.info("TASK: Data Generation")
    from src.ingestion.generate_data import main as generate_main
    generate_main()
    logger.info("Data generation completed.")
    return True


def run_data_ingestion():
    """Task 2: Ingest data from sources."""
    logger.info("TASK: Data Ingestion")
    from src.ingestion.ingest import DataIngestor
    ingestor = DataIngestor()
    result = ingestor.ingest_all()
    logger.info("Data ingestion completed.")
    return result


def run_data_validation():
    """Task 3: Validate data quality."""
    logger.info("TASK: Data Validation")
    from src.validation.validate import DataValidator
    validator = DataValidator()
    report = validator.validate_all()

    # Check if any source failed hard
    for source, info in report.items():
        status = info.get("quality_status", "UNKNOWN")
        if status == "FAIL":
            raise ValueError(f"Data validation failed for {source}")

    logger.info("Data validation completed.")
    return report


def run_data_preparation():
    """Task 4: Clean and prepare data."""
    logger.info("TASK: Data Preparation")
    from src.preparation.prepare import DataPreparator
    preparator = DataPreparator()
    datasets = preparator.prepare_all()
    logger.info("Data preparation completed.")
    return True


def run_feature_engineering():
    """Task 5: Create features."""
    logger.info("TASK: Feature Engineering")
    from src.transformation.transform import FeatureEngineer
    engineer = FeatureEngineer()
    engineer.transform_all()
    logger.info("Feature engineering completed.")
    return True


def run_feature_store():
    """Task 6: Register features in store."""
    logger.info("TASK: Feature Store")
    from src.feature_store.store import main as store_main
    store_main()
    logger.info("Feature store operations completed.")
    return True


def run_data_versioning():
    """Task 7: Version datasets."""
    logger.info("TASK: Data Versioning")
    from src.versioning.version import DataVersioner
    versioner = DataVersioner()
    versioner.version_all()
    logger.info("Data versioning completed.")
    return True


def run_model_training():
    """Task 8: Train and evaluate models."""
    logger.info("TASK: Model Training")
    from src.training.train import RecommendationTrainer
    trainer = RecommendationTrainer()
    results = trainer.train_and_evaluate()
    logger.info("Model training completed.")
    return results


# ===================== PREFECT FLOW (if available) =====================

if PREFECT_AVAILABLE:
    @task(name="generate_data", retries=2, retry_delay_seconds=10)
    def task_generate_data():
        return run_data_generation()

    @task(name="ingest_data", retries=2, retry_delay_seconds=10)
    def task_ingest_data():
        return run_data_ingestion()

    @task(name="validate_data", retries=1)
    def task_validate_data():
        return run_data_validation()

    @task(name="prepare_data", retries=1)
    def task_prepare_data():
        return run_data_preparation()

    @task(name="engineer_features", retries=1)
    def task_engineer_features():
        return run_feature_engineering()

    @task(name="feature_store_ops", retries=1)
    def task_feature_store():
        return run_feature_store()

    @task(name="version_data", retries=1)
    def task_version_data():
        return run_data_versioning()

    @task(name="train_model", retries=1)
    def task_train_model():
        return run_model_training()

    @flow(name="recomart_pipeline", log_prints=True)
    def recomart_pipeline_flow():
        """
        End-to-end RecoMart Data Pipeline DAG:

        generate_data → ingest_data → validate_data → prepare_data
        → engineer_features → feature_store → version_data → train_model
        """
        flow_logger = get_run_logger()
        flow_logger.info("Starting RecoMart Pipeline Flow")

        # Step 1: Generate synthetic data
        gen_result = task_generate_data()

        # Step 2: Ingest data
        ing_result = task_ingest_data(wait_for=[gen_result])

        # Step 3: Validate data
        val_result = task_validate_data(wait_for=[ing_result])

        # Step 4: Prepare data
        prep_result = task_prepare_data(wait_for=[val_result])

        # Step 5: Feature engineering
        feat_result = task_engineer_features(wait_for=[prep_result])

        # Step 6: Feature store
        store_result = task_feature_store(wait_for=[feat_result])

        # Step 7: Data versioning
        ver_result = task_version_data(wait_for=[store_result])

        # Step 8: Model training
        train_result = task_train_model(wait_for=[ver_result])

        flow_logger.info("RecoMart Pipeline Flow completed successfully!")
        return train_result


# ===================== SIMPLE SEQUENTIAL PIPELINE =====================

def run_sequential_pipeline():
    """Run the full pipeline sequentially (no Prefect dependency)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(str(LOG_DIR / "pipeline.log"), rotation="10 MB", level="INFO")

    logger.info("=" * 70)
    logger.info("  RecoMart End-to-End Data Pipeline")
    logger.info(f"  Started at: {timestamp}")
    logger.info("=" * 70)

    pipeline_status = {}
    steps = [
        ("1_data_generation", run_data_generation),
        ("2_data_ingestion", run_data_ingestion),
        ("3_data_validation", run_data_validation),
        ("4_data_preparation", run_data_preparation),
        ("5_feature_engineering", run_feature_engineering),
        ("6_feature_store", run_feature_store),
        ("7_data_versioning", run_data_versioning),
        ("8_model_training", run_model_training),
    ]

    for step_name, step_fn in steps:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {step_name}")
        logger.info(f"{'='*50}")
        try:
            result = step_fn()
            pipeline_status[step_name] = {
                "status": "SUCCESS",
                "timestamp": datetime.now().isoformat(),
            }
            logger.info(f"  ✓ {step_name} completed successfully")
        except Exception as e:
            pipeline_status[step_name] = {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            logger.error(f"  ✗ {step_name} FAILED: {e}")
            logger.error("Pipeline halted due to failure.")
            break

    # Save pipeline execution report
    report_path = REPORT_DIR / f"pipeline_execution_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(pipeline_status, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("  Pipeline Execution Summary")
    logger.info("=" * 70)
    for step, info in pipeline_status.items():
        status = info["status"]
        icon = "✓" if status == "SUCCESS" else "✗"
        logger.info(f"  {icon} {step}: {status}")
    logger.info("=" * 70)

    return pipeline_status


def main():
    if PREFECT_AVAILABLE and "--prefect" in sys.argv:
        logger.info("Running with Prefect orchestration...")
        recomart_pipeline_flow()
    else:
        if not PREFECT_AVAILABLE:
            logger.info("Prefect not installed. Running sequential pipeline.")
        run_sequential_pipeline()


if __name__ == "__main__":
    main()
