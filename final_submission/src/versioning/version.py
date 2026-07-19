"""
RecoMart - Data Versioning Module
Manages dataset versioning using a file-based approach with metadata tracking.
Can be extended to use DVC for production workflows.
"""

import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
VERSION_DIR = DATA_DIR / "versions"
LOG_DIR = BASE_DIR / "logs"


class DataVersioner:
    """Tracks and versions datasets with metadata."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        VERSION_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(str(LOG_DIR / "versioning.log"), rotation="10 MB", level="INFO")
        self.manifest_path = VERSION_DIR / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self):
        """Load the versioning manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {"datasets": {}, "history": []}

    def _save_manifest(self):
        """Save the versioning manifest."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def _compute_hash(self, filepath):
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def version_dataset(self, name, source_path, description=""):
        """Create a versioned snapshot of a dataset."""
        source = Path(source_path)
        if not source.exists():
            logger.warning(f"Source not found: {source}")
            return None

        file_hash = self._compute_hash(source)
        version_id = f"v_{self.timestamp}"

        # Check if same content already versioned
        if name in self.manifest["datasets"]:
            latest = self.manifest["datasets"][name].get("latest_hash", "")
            if latest == file_hash:
                logger.info(f"  {name}: No changes detected, skipping versioning")
                return None

        # Create version directory and copy file
        version_path = VERSION_DIR / name / version_id
        version_path.mkdir(parents=True, exist_ok=True)
        dest = version_path / source.name
        shutil.copy2(source, dest)

        # Get file stats
        df = pd.read_csv(source)
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        }

        # Update manifest
        version_entry = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "source_path": str(source),
            "versioned_path": str(dest),
            "hash": file_hash,
            "description": description,
            "stats": stats,
        }

        if name not in self.manifest["datasets"]:
            self.manifest["datasets"][name] = {"versions": [], "latest_hash": ""}

        self.manifest["datasets"][name]["versions"].append(version_entry)
        self.manifest["datasets"][name]["latest_hash"] = file_hash
        self.manifest["datasets"][name]["latest_version"] = version_id

        self.manifest["history"].append({
            "action": "version_created",
            "dataset": name,
            "version": version_id,
            "timestamp": datetime.now().isoformat(),
        })

        self._save_manifest()
        logger.info(f"  Versioned {name} as {version_id} (hash: {file_hash[:12]}...)")
        return version_id

    def get_version(self, name, version_id=None):
        """Retrieve a specific version of a dataset."""
        if name not in self.manifest["datasets"]:
            logger.warning(f"Dataset not found: {name}")
            return None

        dataset = self.manifest["datasets"][name]
        if version_id is None:
            version_id = dataset.get("latest_version")

        for v in dataset["versions"]:
            if v["version_id"] == version_id:
                path = Path(v["versioned_path"])
                if path.exists():
                    return pd.read_csv(path)
                else:
                    logger.warning(f"Version file missing: {path}")
                    return None

        logger.warning(f"Version {version_id} not found for {name}")
        return None

    def list_versions(self, name=None):
        """List all versions, optionally filtered by dataset name."""
        if name:
            if name in self.manifest["datasets"]:
                versions = self.manifest["datasets"][name]["versions"]
                return pd.DataFrame(versions)
            return pd.DataFrame()

        all_versions = []
        for ds_name, ds_info in self.manifest["datasets"].items():
            for v in ds_info["versions"]:
                v_copy = v.copy()
                v_copy["dataset"] = ds_name
                all_versions.append(v_copy)
        return pd.DataFrame(all_versions)

    def version_all(self):
        """Version all current datasets."""
        logger.info("=" * 60)
        logger.info(f"Starting Data Versioning - {self.timestamp}")
        logger.info("=" * 60)

        # Version raw data
        raw_dir = DATA_DIR / "raw"
        for source_dir in raw_dir.iterdir():
            if source_dir.is_dir():
                csv_files = sorted(source_dir.glob("*.csv"), reverse=True)
                if csv_files:
                    self.version_dataset(
                        f"raw_{source_dir.name}",
                        csv_files[0],
                        description=f"Raw {source_dir.name} data"
                    )

        # Version processed data
        processed_dir = DATA_DIR / "processed"
        if processed_dir.exists():
            for csv_file in processed_dir.glob("*.csv"):
                self.version_dataset(
                    f"processed_{csv_file.stem}",
                    csv_file,
                    description=f"Processed {csv_file.stem} data"
                )

        # Version features
        features_dir = DATA_DIR / "features"
        if features_dir.exists():
            for csv_file in features_dir.glob("*.csv"):
                self.version_dataset(
                    f"features_{csv_file.stem}",
                    csv_file,
                    description=f"Feature set: {csv_file.stem}"
                )

        logger.info("\nVersioning Summary:")
        versions_df = self.list_versions()
        if not versions_df.empty:
            print(versions_df[["dataset", "version_id", "timestamp"]].to_string())

        logger.info("=" * 60)
        logger.info("Data versioning complete!")
        logger.info("=" * 60)


def main():
    versioner = DataVersioner()
    versioner.version_all()


if __name__ == "__main__":
    main()
