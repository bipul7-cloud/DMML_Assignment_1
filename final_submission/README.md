# RecoMart — End-to-End Data Management Pipeline for a Recommendation System

## Course Information

| Field | Detail |
|-------|--------|
| **Course** | Data Management for Machine Learning (AIMLCZG529 / DSECLZG529) |
| **Programme** | M.Tech AI/ML — BITS Pilani WILP |
| **Semester** | S2-25 |
| **Assignment** | Assignment I — EC1 (20 Marks) |
| **Group Number** | 43 |
| **Submission Date** | 22.07.2026 |

## Team Members — Group 43

| # | Name | BITS ID | Email |
|---|------|---------|-------|
| 1 | CHATLA VARUN | 2025AB05279 | 2025AB05279@wilp.bits-pilani.ac.in |
| 2 | AKASH KUMAR JAIN | 2025AB05278 | 2025AB05278@wilp.bits-pilani.ac.in |
| 3 | BIPUL RAJ | 2025AB05256 | 2025AB05256@wilp.bits-pilani.ac.in |
| 4 | DUGGINA MOHITHA | 2025AB05269 | 2025AB05269@wilp.bits-pilani.ac.in |
| 5 | VATTURI SAI PAVAN KUMAR | 2025AB05267 | 2025AB05267@wilp.bits-pilani.ac.in |

---

## 1. Problem Statement

RecoMart is an e-commerce startup that wants to build a **data-driven product recommendation engine** to improve customer engagement and cross-selling opportunities. The platform collects user behaviour data from multiple channels — web/mobile clickstream logs, transactional purchase history, product catalog metadata, and external API data.

The goal of this project is to design and implement a **complete, automated, modular data management pipeline** that continuously ingests raw data, validates and cleans it, engineers features, trains recommendation models, and tracks every artefact along the way — following the practices of modern ML data stacks.

---

## 2. Objectives

1. Ingest data from heterogeneous sources (CSV files + REST APIs) with error handling, retry logic, and audit logging.
2. Store raw data in a structured, partitioned data lake layout.
3. Profile and validate data quality (missing values, duplicates, schema mismatches, range violations).
4. Clean and preprocess data — handle missing values, encode categoricals, normalise numerics.
5. Engineer user-level, item-level, and user–item interaction features suitable for collaborative and content-based filtering.
6. Manage features through a versioned feature store with a metadata registry.
7. Version all datasets (raw, processed, features) with SHA-256 integrity tracking and a manifest.
8. Train and evaluate recommendation models (SVD collaborative filtering + content-based cosine similarity).
9. Track experiments, parameters, and metrics with MLflow.
10. Orchestrate the full pipeline as a DAG using Prefect (with a sequential fallback).

---

## 3. Data Sources

| Source | Type | Records | Description |
|--------|------|---------|-------------|
| **Users** | CSV (synthetic) | 1,000 | User demographics — age, gender, country, signup date, premium status |
| **Products** | CSV + JSON (API) | 500 | Product catalog — name, category, price, brand, avg rating, stock status |
| **Clickstream** | CSV (synthetic) | ~51,000 | User interaction events — views, clicks, add-to-cart, wishlist, search |
| **Transactions** | CSV (synthetic) | 10,000 | Purchase records — quantities, amounts, payment methods, statuses |
| **Ratings** | CSV (synthetic) | ~20,000 | Explicit user–item ratings (1–5 scale) with review text |
| **External API** | REST (FakeStore API) | 20 | Supplementary product data fetched from a live API |

Synthetic data is generated with intentional quality issues (missing values, duplicates, out-of-range ratings) to demonstrate the validation and cleaning pipeline.

---

## 4. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RecoMart Data Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  Data     │──▶│  Data     │──▶│  Data     │──▶│ Feature  │    │
│  │Generation │   │ Ingestion │   │Validation │   │Preparation│   │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       │              │              │               │           │
│       ▼              ▼              ▼               ▼           │
│  data/raw/      logs/         reports/         data/processed/  │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Feature   │──▶│ Feature  │──▶│  Data     │──▶│  Model   │    │
│  │Engineering│   │  Store   │   │Versioning │   │ Training │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       │              │              │               │           │
│       ▼              ▼              ▼               ▼           │
│  data/features/ feature_store/ data/versions/  models/ mlruns/  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
           Orchestrated by Prefect (src/orchestration/)
```

---

## 5. Project Structure

```
project dm4ml/
│
├── config/
│   └── pipeline_config.yaml          # Central configuration (sources, storage, model params)
│
├── src/
│   ├── ingestion/
│   │   ├── generate_data.py          # Synthetic data generator (Faker-based)
│   │   └── ingest.py                 # CSV & API ingestion with retry/logging
│   │
│   ├── validation/
│   │   └── validate.py               # Schema, missing value, duplicate, range validation
│   │
│   ├── preparation/
│   │   └── prepare.py                # Cleaning, encoding, normalisation, EDA stats
│   │
│   ├── transformation/
│   │   └── transform.py              # User, item, user-item feature engineering + SQL schema
│   │
│   ├── feature_store/
│   │   └── store.py                  # SQLite registry, versioned storage, retrieval API
│   │
│   ├── versioning/
│   │   └── version.py                # SHA-256 based dataset versioning with manifest
│   │
│   ├── training/
│   │   └── train.py                  # SVD + content-based models, evaluation, MLflow tracking
│   │
│   └── orchestration/
│       └── pipeline.py               # Prefect DAG + sequential fallback orchestrator
│
├── notebooks/
│   └── eda_analysis.ipynb            # Interactive EDA with visualisations
│
├── data/
│   ├── raw/                          # Raw ingested data (partitioned by source & date)
│   │   ├── users/
│   │   ├── products/
│   │   ├── clickstream/
│   │   ├── transactions/
│   │   └── ratings/
│   ├── validated/                    # Post-validation datasets
│   ├── processed/                    # Cleaned & encoded datasets
│   ├── features/                     # Engineered feature tables + SQL schema
│   ├── feature_store/                # Feature registry (SQLite) + versioned snapshots
│   └── versions/                     # Versioned dataset snapshots with manifest.json
│
├── models/                           # Serialised trained models (.pkl)
├── mlruns/                           # MLflow experiment tracking (SQLite backend)
├── logs/                             # Timestamped logs for every pipeline stage
├── reports/                          # Quality reports, EDA summaries, performance reports
│
├── requirements.txt                  # Python dependencies
├── SUBMISSION_GUIDE.md               # Submission requirements & instructions
├── .gitignore
└── README.md                         # This file
```

---

## 6. Methodology / Pipeline Stages

### Stage 1 — Data Generation (`src/ingestion/generate_data.py`)
- Generates realistic synthetic e-commerce data using the **Faker** library with fixed seeds for reproducibility.
- Intentionally injects ~5% missing values in user ages, ~3% in genders, ~2% missing transaction amounts, ~2% duplicate clickstream events, and ~1% out-of-range ratings — to test the validation pipeline.

### Stage 2 — Data Ingestion (`src/ingestion/ingest.py`)
- **CSV ingestion**: Reads all CSV files from each configured source directory, concatenates, and logs row/column counts.
- **API ingestion**: Fetches data from the FakeStore REST API with exponential-backoff retry (up to 3 attempts) and 30-second timeout.
- Saves a JSON ingestion report per run with per-source status, row counts, and timestamps.

### Stage 3 — Data Validation (`src/validation/validate.py`)
- **Schema checks**: Validates required columns and data types against expected schemas.
- **Missing value analysis**: Per-column and overall missing percentages.
- **Duplicate detection**: Row-level duplicate counts and percentages.
- **Range validation**: Rating values (1–5), non-negative prices, positive quantities.
- **Data profiling**: Numeric summaries, categorical value counts, memory usage.
- Generates a comprehensive JSON quality report and saves validated copies.

### Stage 4 — Data Preparation (`src/preparation/prepare.py`)
- **Missing value imputation**: Median for ages, recalculated for transaction amounts, "Unknown" for categoricals.
- **Encoding**: Label encoding for gender, product category, event type, device, payment method.
- **Normalisation**: MinMaxScaler for prices.
- **Derived features**: Account age in days, hour-of-day and day-of-week from timestamps.
- **Filtering**: Removes out-of-range ratings (< 1 or > 5), deduplicates clickstream events.
- Produces EDA summary statistics as JSON for every dataset.

### Stage 5 — Feature Engineering (`src/transformation/transform.py`)
- **User features** (15 features): total ratings, avg rating, rating std, purchase count, total spend, avg order value, click counts, view/cart counts, purchase frequency, activity score, account age, premium status, gender.
- **Item features** (13 features): rating count, avg user rating, rating variance, purchase count, revenue, view count, view-to-purchase conversion, popularity score, price, normalised price, category, stock status.
- **User–item features** (7 features): rating, interaction count, has-viewed, has-carted, has-purchased flags.
- Includes a SQL schema (`data/features/schema.sql`) documenting the feature warehouse tables.

### Stage 6 — Feature Store (`src/feature_store/store.py`)
- **SQLite metadata registry** (`data/feature_store/registry.db`) tracking feature set name, version, row/column counts, creation timestamps, and file paths.
- **Versioned offline storage**: Each registration creates a timestamped snapshot under `data/feature_store/versions/`.
- **Online store**: Latest version always available at `data/feature_store/online/` for low-latency inference retrieval.
- **Feature-level metadata**: Descriptions, data types, source tables, and transformations recorded per feature.

### Stage 7 — Data Versioning (`src/versioning/version.py`)
- Computes **SHA-256 hashes** for every dataset file to detect changes.
- Maintains a `manifest.json` with full version history — version ID, timestamp, source path, hash, and dataset statistics.
- Skips versioning if file content is unchanged (hash comparison).
- Versions all 13 datasets across raw, processed, and feature tiers.

### Stage 8 — Model Training (`src/training/train.py`)
- **SVD Collaborative Filtering**: Truncated SVD on the centered user–item rating matrix (50 latent factors). Reconstructs predicted ratings and clips to [1, 5].
- **Content-Based Filtering**: Cosine similarity on normalised item feature vectors.
- **Evaluation metrics**: Precision@K, Recall@K, NDCG@K (for K = 5, 10, 20), RMSE, MAE.
- **MLflow integration**: Logs parameters (n_factors, train/test sizes, user/item counts), all metrics, and model artefacts with unique run IDs. Uses SQLite backend.
- **Inference demo**: Generates top-5 recommendations for a sample user.

### Stage 9 — Pipeline Orchestration (`src/orchestration/pipeline.py`)
- **Prefect flow**: Defines 8 tasks with retry policies (`retries=2` for ingestion, `retries=1` for others) connected as a sequential DAG with `wait_for` dependencies.
- **Sequential fallback**: If Prefect is unavailable, runs all stages sequentially with try/except error handling, halting on failure.
- Produces a JSON pipeline execution report with per-stage status and timestamps.
- All stages produce structured logs in `logs/`.

---

## 7. Evaluation Metrics & Results

### SVD Collaborative Filtering

| Metric | Value |
|--------|-------|
| Precision@5 | 0.0017 |
| Recall@5 | 0.0050 |
| NDCG@5 | 0.0027 |
| Precision@10 | 0.0025 |
| Recall@10 | 0.0121 |
| NDCG@10 | 0.0055 |
| Precision@20 | 0.0031 |
| Recall@20 | 0.0299 |
| NDCG@20 | 0.0109 |
| RMSE | 1.4754 |
| MAE | 1.2704 |

### Content-Based Filtering

| Metric | Value |
|--------|-------|
| Precision@5 | 0.0140 |
| Recall@5 | 0.0521 |
| NDCG@5 | 0.0460 |
| Precision@10 | 0.0099 |
| Recall@10 | 0.0685 |
| NDCG@10 | 0.0522 |
| Precision@20 | 0.0074 |
| Recall@20 | 0.0942 |
| NDCG@20 | 0.0598 |

> **Note**: Metrics are low because the data is purely synthetic (random). With real user–item interactions exhibiting genuine preference patterns, collaborative filtering models perform significantly better. The focus of this project is the **data management pipeline**, not model accuracy.

---

## 8. Technology Stack

| Component | Tool/Library |
|-----------|-------------|
| Language | Python 3.11 |
| Data Processing | pandas, NumPy |
| ML / Models | scikit-learn, SciPy (SVD) |
| Data Validation | Custom validator (schema, range, duplicate checks) |
| Feature Store | Custom SQLite-based registry with versioned storage |
| Data Versioning | Custom SHA-256 manifest-based versioning |
| Experiment Tracking | MLflow (SQLite backend) |
| Pipeline Orchestration | Prefect (with sequential fallback) |
| Data Generation | Faker |
| API Ingestion | requests (with retry) |
| Logging | Loguru |
| Visualisation | Matplotlib, Seaborn |
| Configuration | YAML (PyYAML) |

---

## 9. Setup & Installation

### Prerequisites
- Python 3.10+ installed
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 10. How to Run

### Option A — Run Full Pipeline (Recommended)
Runs all 8 stages sequentially with logging and error handling:
```bash
cd "project dm4ml"
python src/orchestration/pipeline.py
```

### Option B — Run Full Pipeline with Prefect Orchestration
```bash
python src/orchestration/pipeline.py --prefect
```

### Option C — Run Individual Stages
```bash
# Step 1: Generate synthetic data
python src/ingestion/generate_data.py

# Step 2: Ingest data from sources
python src/ingestion/ingest.py

# Step 3: Validate data quality
python src/validation/validate.py

# Step 4: Clean and prepare data
python src/preparation/prepare.py

# Step 5: Engineer features
python src/transformation/transform.py

# Step 6: Register features in store
python src/feature_store/store.py

# Step 7: Version all datasets
python src/versioning/version.py

# Step 8: Train and evaluate models
python src/training/train.py
```

### Option D — Interactive EDA
Open and run `notebooks/eda_analysis.ipynb` in Jupyter or VS Code for visual exploration with charts and heatmaps.

---

## 11. Key Output Artefacts

| Artefact | Location |
|----------|----------|
| Raw datasets (CSV, JSON) | `data/raw/` |
| Validated datasets | `data/validated/` |
| Cleaned datasets | `data/processed/` |
| Engineered features | `data/features/` |
| Feature store registry | `data/feature_store/registry.db` |
| Versioned snapshots | `data/versions/` |
| Trained models (.pkl) | `models/` |
| MLflow experiments | `mlruns/` |
| Data quality reports | `reports/data_quality_report_*.json` |
| EDA summaries | `reports/eda_summary_*.json` |
| Model performance reports | `reports/model_performance_*.json` |
| Pipeline execution reports | `reports/pipeline_execution_*.json` |
| Feature metadata | `reports/feature_metadata_*.json` |
| Stage-wise logs | `logs/` |
| EDA visualisation plots | `reports/*.png` |

---

## 12. Conclusion & Future Scope

### Conclusion
This project demonstrates a fully functional, modular, end-to-end data management pipeline that takes raw multi-source e-commerce data through ingestion, validation, cleaning, feature engineering, storage, versioning, model training, and evaluation — all orchestrated as a reproducible workflow. Every stage produces auditable logs and reports.

### Future Scope
- **Real data integration**: Connect to actual e-commerce databases and streaming sources (Kafka).
- **Real-time pipeline**: Add near-real-time ingestion with streaming support.
- **Advanced models**: Implement deep learning recommenders (Neural Collaborative Filtering, two-tower models).
- **A/B testing framework**: Online evaluation of recommendation quality.
- **Cloud deployment**: Migrate to AWS S3 + Glue + SageMaker or GCP equivalents.
- **Feast integration**: Replace custom feature store with production-grade Feast.
- **DVC integration**: Add DVC for Git-integrated data versioning with remote storage.
- **CI/CD**: Automate pipeline testing and deployment with GitHub Actions.
