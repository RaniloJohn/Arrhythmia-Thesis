"""
MIMIC PERform AF Dataset Loader.
================================
Loads the held-out external validation dataset from Charlton et al. (2022).
Derived from the MIMIC-III Waveform Database Matched Subset, containing fingertip PPG
continuous recordings at 125 Hz with per-recording clinical AF and non-AF rhythm labels.

ROLE: HELD-OUT EXTERNAL VALIDATION ONLY.
CRITICAL ENFORCEMENT: Never enters training or threshold calibration (Settled Decision 9).
"""

import argparse
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import csv
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# Support running as a module, package, or standalone script from workspace root or 03 - ML/
try:
    from .base import PPGDatasetLoader, SubjectRecord
except (ImportError, ValueError):
    # If run directly as a script, ensure 03 - ML is on sys.path
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from training.datasets.base import PPGDatasetLoader, SubjectRecord


class MimicPerformAFLoader(PPGDatasetLoader):
    """
    Loader for the MIMIC PERform AF Dataset.

    Reads per-subject continuous CSV recordings from `03 - ML/data/mimic_perform/`.
    Each recording contains 4 channels: Time, PPG, ECG, and resp.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the loader.

        Parameters:
            data_dir: Path to MIMIC PERform AF data root. Defaults to `03 - ML/data/mimic_perform`.
        """
        if data_dir is None:
            # Locate 03 - ML/data/mimic_perform relative to repository
            possible_paths = [
                Path(__file__).resolve().parents[2] / "data" / "mimic_perform",
                Path.cwd() / "03 - ML" / "data" / "mimic_perform",
                Path.cwd() / "data" / "mimic_perform",
            ]
            self.data_dir = None
            for p in possible_paths:
                if p.exists() and len(list(p.rglob("*_data.csv"))) > 0:
                    self.data_dir = p
                    break
            if self.data_dir is None:
                self.data_dir = possible_paths[0]
        else:
            self.data_dir = Path(data_dir).resolve()

        self._csv_files: Optional[List[Path]] = None

    def _get_csv_files(self) -> List[Path]:
        """Find and validate all subject data CSV files."""
        if self._csv_files is not None:
            return self._csv_files

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"MIMIC PERform AF data directory not found at '{self.data_dir.resolve()}'.\n"
                "Expected CSV recordings from Charlton et al. (2022).\n"
                "Please download 'mimic_perform_af_csv.zip' and 'mimic_perform_non_af_csv.zip' "
                "from Zenodo (DOI: 10.5281/zenodo.15906524) and extract into this directory.\n"
                "See '03 - ML/data/README.md' for layout instructions."
            )

        csv_files = sorted(self.data_dir.rglob("*_data.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No subject CSV files (*_data.csv) found in '{self.data_dir.resolve()}'.\n"
                "Expected per-subject files such as 'mimic_perform_af_001_data.csv'.\n"
                "Please extract the dataset archives into this directory."
            )

        self._csv_files = csv_files
        return self._csv_files

    @staticmethod
    def extract_subject_id(file_path: Path) -> str:
        """Extract standardized subject identifier from CSV filename."""
        match = re.match(r"(mimic_perform_(?:non_)?af_\d+)", file_path.name)
        if not match:
            raise ValueError(f"Filename does not match expected MIMIC format: '{file_path.name}'")
        return match.group(1)

    @staticmethod
    def _parse_fix_metadata(fix_file: Path) -> Dict[str, Any]:
        """Parse metadata from corresponding _fix.txt file."""
        metadata: Dict[str, Any] = {}
        if not fix_file.exists():
            return metadata

        with open(fix_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if ":" in line_str:
                    k, v = line_str.split(":", 1)
                    metadata[k.strip().lower()] = v.strip()

        # Parse numeric sampling frequency if present
        if "sampling frequency" in metadata:
            fs_match = re.search(r"(\d+(?:\.\d+)?)", metadata["sampling frequency"])
            if fs_match:
                metadata["fs_numeric"] = float(fs_match.group(1))

        return metadata

    def _load_single_subject(self, data_file: Path) -> SubjectRecord:
        """Load a single subject CSV file into a SubjectRecord."""
        subject_id = self.extract_subject_id(data_file)
        # Determine rhythm label: 1 = AF, 0 = non-AF
        label = 0 if "_non_af_" in subject_id else 1

        # Check metadata file
        fix_file = data_file.parent / f"{subject_id}_fix.txt"
        meta = self._parse_fix_metadata(fix_file)

        # Read CSV data via built-in csv reader
        times: List[float] = []
        ppgs: List[float] = []
        with open(data_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            if "Time" not in header or "PPG" not in header:
                raise ValueError(f"File {data_file.name} missing required columns 'Time' and 'PPG'")
            t_idx = header.index("Time")
            p_idx = header.index("PPG")
            for row in reader:
                if row:
                    times.append(float(row[t_idx]))
                    ppgs.append(float(row[p_idx]))

        time_col = np.asarray(times, dtype=np.float64)
        ppg_col = np.asarray(ppgs, dtype=np.float32)

        # Calculate sampling frequency from consecutive time steps
        if len(time_col) < 2:
            raise ValueError(f"Recording in {data_file.name} contains fewer than 2 samples.")

        dt = float(time_col[1] - time_col[0])
        if dt <= 0:
            raise ValueError(f"Invalid time step dt={dt} in {data_file.name}")
        fs_calc = round(1.0 / dt, 2)

        # If metadata states sampling frequency, cross-verify
        if "fs_numeric" in meta:
            fs_meta = meta["fs_numeric"]
            if not math.isclose(fs_calc, fs_meta, rel_tol=1e-2):
                raise ValueError(
                    f"Discrepancy in {subject_id}: calculated fs={fs_calc} Hz from Time column "
                    f"differs from metadata stated fs={fs_meta} Hz in {fix_file.name}"
                )

        return SubjectRecord(
            subject_id=subject_id,
            signal=ppg_col,
            fs=fs_calc,
            label=label,
            quality=None,  # MIMIC PERform AF does not provide pre-computed per-window SQI
            source_dataset="mimic_perform",
            native_window_s=None,  # Continuous recording (~20 min), not pre-windowed
            partition="external_validation",
            is_adjudicated=False,
            metadata=meta,
        )

    def subjects(self) -> Iterator[SubjectRecord]:
        """
        Yield per-subject continuous recordings.

        Yields:
            SubjectRecord for each subject in the dataset.
        """
        for data_file in self._get_csv_files():
            yield self._load_single_subject(data_file)

    def summary(self) -> Dict[str, Any]:
        """
        Compute true dataset summary statistics from disk without assumptions.

        Returns:
            Dict of detected sample rates, counts, durations, and balance.
        """
        csv_files = self._get_csv_files()
        af_files = [f for f in csv_files if "_non_af_" not in f.name]
        non_af_files = [f for f in csv_files if "_non_af_" in f.name]

        sample_rates = set()
        total_duration_s = 0.0
        sample_counts = []

        for f in csv_files:
            # Read first few lines for dt and count total samples
            with open(f, "r", newline="", encoding="utf-8") as fp:
                reader = csv.reader(fp)
                header = next(reader)
                t_idx = header.index("Time")
                r0 = next(reader)
                r1 = next(reader)
                dt = float(r1[t_idx]) - float(r0[t_idx])
                # Count remaining lines
                line_count = 2 + sum(1 for _ in fp)

            fs = round(1.0 / dt, 2)
            sample_rates.add(fs)
            dur = (line_count - 1) * dt
            total_duration_s += dur
            sample_counts.append(line_count)

        total_subjects = len(csv_files)
        af_count = len(af_files)
        non_af_count = len(non_af_files)

        fs_str = ", ".join(f"{s:.1f} Hz" for s in sorted(sample_rates))
        mean_dur_s = total_duration_s / total_subjects if total_subjects > 0 else 0.0

        return {
            "dataset_name": "MIMIC PERform AF Dataset (Charlton et al., 2022)",
            "role": "HELD-OUT EXTERNAL VALIDATION ONLY (Never in training or threshold calibration)",
            "detected_sample_rate": fs_str,
            "native_window_s": "None (Continuous ~20 min recordings per subject)",
            "total_subjects": total_subjects,
            "af_subjects": af_count,
            "non_af_subjects": non_af_count,
            "af_class_balance": f"{af_count} AF ({af_count / total_subjects * 100:.1f}%) / {non_af_count} non-AF ({non_af_count / total_subjects * 100:.1f}%)",
            "total_duration": f"{total_duration_s:.1f} s ({total_duration_s / 3600:.2f} hours, {total_duration_s / 60:.1f} minutes)",
            "mean_duration_per_subject": f"{mean_dur_s:.1f} s ({mean_dur_s / 60:.1f} minutes)",
            "samples_per_subject": f"{min(sample_counts)} - {max(sample_counts)} points",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIC PERform AF Dataset Loader & Summary Utility"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to 03 - ML/data/mimic_perform directory",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print detected sample rate, subject count, AF balance, and duration.",
    )
    args = parser.parse_args()

    loader = MimicPerformAFLoader(data_dir=args.data_dir)
    loader.print_summary()


if __name__ == "__main__":
    main()
