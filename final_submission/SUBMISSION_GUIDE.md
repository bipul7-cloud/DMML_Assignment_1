# Submission Guide — Assignment I (EC1)

## Course Details

| Field | Detail |
|-------|--------|
| **Course** | Data Management for Machine Learning (AIMLCZG529 / DSECLZG529) |
| **Programme** | M.Tech AI/ML — BITS Pilani WILP |
| **Semester** | S2-25 |
| **Assignment** | Assignment I |
| **Weightage** | 20 Marks (EC1) |
| **Group Number** | 43 |
| **Submission Deadline** | **22 July 2026** |

## Team Members — Group 43

| # | Name | BITS ID | Email |
|---|------|---------|-------|
| 1 | CHATLA VARUN | 2025AB05279 | 2025AB05279@wilp.bits-pilani.ac.in |
| 2 | AKASH KUMAR JAIN | 2025AB05278 | 2025AB05278@wilp.bits-pilani.ac.in |
| 3 | BIPUL RAJ | 2025AB05256 | 2025AB05256@wilp.bits-pilani.ac.in |
| 4 | DUGGINA MOHITHA | 2025AB05269 | 2025AB05269@wilp.bits-pilani.ac.in |
| 5 | VATTURI SAI PAVAN KUMAR | 2025AB05267 | 2025AB05267@wilp.bits-pilani.ac.in |

---

## What to Submit

You must submit a **single PDF report** containing the complete project documentation.

---

## PDF Report — Required Sections

Your PDF report **must** include all of the following:

| # | Section | What to Include |
|---|---------|-----------------|
| 1 | **Project Title** | "End-to-End Data Management Pipeline for a Recommendation System" |
| 2 | **Team Member Details** | Names, BITS IDs, and email addresses of all group members |
| 3 | **Problem Statement** | Clear definition of the business problem (product recommendation to improve conversion rate), key data sources, and expected pipeline outputs |
| 4 | **Objectives** | List of specific objectives the pipeline achieves |
| 5 | **Methodology / Pipeline** | Architecture diagram and description of the end-to-end pipeline flow (ingestion → validation → preparation → transformation → feature store → versioning → training) |
| 6 | **Implementation Details** | Description of each pipeline stage, tools used, code structure, and key design decisions |
| 7 | **Results and Output Screenshots** | Screenshots showing: pipeline execution logs, data quality reports, EDA plots, model metrics, MLflow tracking, feature store contents, versioning manifest |
| 8 | **Conclusion and Future Scope** | Summary of what was accomplished and potential improvements |
| 9 | **Google Drive Links** | See below |

---

## Google Drive Links (Include in PDF)

Your PDF must contain **two Google Drive links**:

### Link 1 — Video Walkthrough
- **Duration**: 5–10 minutes
- **Content**: Complete end-to-end demonstration of the project execution
- **Should cover**:
  - Running the full pipeline (`python src/orchestration/pipeline.py`)
  - Showing generated data in `data/raw/`
  - Showing validation/quality reports in `reports/`
  - Walking through EDA notebook (`notebooks/eda_analysis.ipynb`)
  - Showing feature store contents and versioning manifest
  - Showing model training output and MLflow tracking
  - Showing logs in `logs/`

### Link 2 — Project Deliverables (.zip)
A `.zip` file containing **all** project deliverables:

| Item | Location in Zip |
|------|-----------------|
| Source Code | `src/` (all Python modules) |
| Configuration | `config/pipeline_config.yaml` |
| Notebooks | `notebooks/eda_analysis.ipynb` |
| Datasets | `data/` (raw, processed, features) |
| Trained Models | `models/*.pkl` |
| Reports & Logs | `reports/`, `logs/` |
| MLflow Tracking | `mlruns/` |
| Feature Store | `data/feature_store/` |
| Versioning Manifest | `data/versions/manifest.json` |
| Requirements | `requirements.txt` |
| Documentation | `README.md`, `SUBMISSION_GUIDE.md` |

> **IMPORTANT**: Google Drive links must be accessible to **any BITS ID** (set sharing to "Anyone with BITS email can view").

---

## How to Prepare the Submission

### Step 1 — Run the Full Pipeline
```bash
cd "project dm4ml"
pip install -r requirements.txt
python src/orchestration/pipeline.py
```
Verify all 8 stages show **SUCCESS** in the terminal output.

### Step 2 — Run the EDA Notebook
Open `notebooks/eda_analysis.ipynb` in Jupyter or VS Code and run all cells to generate the visualisation plots in `reports/`.

### Step 3 — Take Screenshots
Capture screenshots of:
- Terminal showing successful pipeline execution (all 8 stages ✓)
- Data quality report contents
- EDA plots (user/product/rating distributions, interaction heatmap)
- Model evaluation metrics
- MLflow UI or MLflow run details
- Feature store registry contents
- Versioning manifest
- Folder structure showing the data lake layout

### Step 4 — Record the Video
- Record a 5–10 minute screen recording walking through the project
- Upload to Google Drive and set permissions to "Anyone with BITS email"

### Step 5 — Create the .zip File
```bash
# From the project root directory
# On Windows (PowerShell):
Compress-Archive -Path * -DestinationPath RecoMart_Project.zip

# On Linux/Mac:
zip -r RecoMart_Project.zip . -x "__pycache__/*" ".git/*"
```
Upload to Google Drive and set permissions.

### Step 6 — Write the PDF Report
Create a PDF with all required sections (see table above). Include the Google Drive links at the end.

### Step 7 — Submit
Submit the single PDF report through the designated submission portal before **22 July 2026**.

---

## Evaluation Rubric (20 Marks)

| Component | Description | Marks |
|-----------|-------------|-------|
| **Problem Formulation** | Clear definition of objectives, data sources, and outputs | 2 |
| **Data Pipeline Implementation** | Ingestion, storage, validation, and transformation logic | 5 |
| **Feature Store & Versioning** | Effective management of features and data lineage | 4 |
| **Model Training & Evaluation** | Performance metrics and experiment tracking | 4 |
| **Documentation & Demo** | Clarity, completeness, and demonstration quality | 5 |
| **Total** | | **20** |

---

## Checklist Before Submission

- [ ] Pipeline runs end-to-end without errors (all 8 stages SUCCESS)
- [ ] Data is generated and stored in `data/raw/` with proper folder partitioning
- [ ] Data quality report generated in `reports/`
- [ ] EDA notebook runs and produces visualisation plots
- [ ] Features engineered and stored in `data/features/` with SQL schema
- [ ] Feature store registry populated (`data/feature_store/registry.db`)
- [ ] All datasets versioned with manifest (`data/versions/manifest.json`)
- [ ] Models trained and saved in `models/`
- [ ] MLflow run tracked with metrics and parameters
- [ ] Logs present in `logs/` for every pipeline stage
- [ ] Pipeline execution report generated in `reports/`
- [ ] Video walkthrough recorded (5–10 mins) and uploaded to Google Drive
- [ ] .zip of all deliverables uploaded to Google Drive
- [ ] Both Google Drive links accessible to any BITS ID
- [ ] PDF report includes all required sections
- [ ] PDF report includes both Google Drive links
