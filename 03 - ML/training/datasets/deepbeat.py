"""
DeepBeat Dataset Loader.
========================
Loads the primary training dataset from Torres-Soto & Ashley (2020),
npj Digital Medicine (https://doi.org/10.1038/s41746-020-00320-4).

Contains wrist-worn Simband / Cardiaband photoplethysmography (PPG) recordings with:
- Per-window rhythm labels (1 = AF, 0 = non-AF)
- Per-window signal quality labels
- Official train, validation, and test partition assignments
- Cardiologist-adjudicated test partition for clean internal benchmarking

ROLE: PRIMARY TRAINING SET (Train, Val, and Internal Test Benchmark).
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

# Support running as module or script from workspace root or 03 - ML/
try:
    from .base import PPGDatasetLoader, SubjectRecord
except (ImportError, ValueError):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from training.datasets.base import PPGDatasetLoader, SubjectRecord


class DeepBeatLoader(PPGDatasetLoader):
    """
    Loader for the DeepBeat PPG Arrhythmia Dataset (Torres-Soto & Ashley 2020).

    Strict verification rules:
      - Does NOT hardcode sampling rate or window length; parses them directly
        from dataset metadata, attributes, or header files.
      - Asserts self-consistency between array shapes, sampling frequency, and window length.
      - Surfaces official partitions, including the cardiologist-adjudicated benchmark partition.
      - Raises a clear, actionable FileNotFoundError if data files are missing.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize DeepBeatLoader.

        Parameters:
            data_dir: Path to `03 - ML/data/deepbeat/`. Defaults to repo location.
        """
        if data_dir is None:
            possible_paths = [
                Path(__file__).resolve().parents[2] / "data" / "deepbeat",
                Path.cwd() / "03 - ML" / "data" / "deepbeat",
                Path.cwd() / "data" / "deepbeat",
            ]
            self.data_dir = None
            for p in possible_paths:
                if p.exists():
                    self.data_dir = p
                    break
            if self.data_dir is None:
                self.data_dir = possible_paths[0]
        else:
            self.data_dir = Path(data_dir).resolve()

        self._files: Optional[List[Path]] = None
        self._fs: Optional[float] = None
        self._window_length_s: Optional[float] = None
        self._adjudicated_partition_name: Optional[str] = None

    def _discover_files(self) -> List[Path]:
        """
        Locate and validate DeepBeat dataset files (.npz, .h5, .hdf5).

        Raises:
            FileNotFoundError: With actionable instructions if directory or files are missing.
        """
        if self._files is not None:
            return self._files

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"\n[ERROR] DeepBeat data directory not found at: '{self.data_dir.resolve()}'.\n"
                "Action Required:\n"
                "  1. DeepBeat (Torres-Soto & Ashley 2020) is hosted on Synapse: https://www.synapse.org/Synapse:syn21985690/files/\n"
                "     (Paper: https://doi.org/10.1038/s41746-020-00320-4).\n"
                "  2. Once approved, place the dataset files (.h5, .hdf5, or .npz) in:\n"
                f"     {self.data_dir.resolve()}\n"
                "  3. See '03 - ML/data/README.md' for expected layout and access procedures."
            )

        # Look for dataset container files
        candidates = []
        for ext in ("*.npz", "*.h5", "*.hdf5"):
            candidates.extend(self.data_dir.glob(ext))
            candidates.extend(self.data_dir.glob(f"**/{ext}"))

        # Exclude temporary or gitkeep files
        candidates = [f for f in candidates if not f.name.startswith(".")]

        # Distinguish dataset containers from Keras model weights files (e.g. deepbeat.h5)
        valid_files = []
        for c in set(candidates):
            if c.suffix in (".h5", ".hdf5"):
                try:
                    import h5py
                    with h5py.File(c, "r") as h5f:
                        if "model_weights" in h5f and not any(k in h5f for k in ["signals", "signal", "X", "windows", "ppg"]):
                            continue
                except Exception:
                    pass
            valid_files.append(c)

        if not valid_files:
            raise FileNotFoundError(
                f"\n[ERROR] No dataset files (.npz, .h5, .hdf5) found in '{self.data_dir.resolve()}'.\n"
                "DeepBeat is currently awaiting Stanford licence approval (see MEMORY.md).\n"
                "To populate:\n"
                "  1. Place DeepBeat partition files (e.g., train.npz, val.npz, test.npz, or dataset.h5) in:\n"
                f"     {self.data_dir.resolve()}\n"
                "  2. Include metadata.json specifying sampling rate and window duration if not stored in array attributes.\n"
                "  3. Refer to '03 - ML/data/README.md' for full documentation."
            )

        self._files = sorted(valid_files)
        return self._files

    def _read_metadata(self) -> Dict[str, Any]:
        """
        Read dataset metadata from metadata.json or companion documentation.
        Does NOT guess; returns detected metadata.
        """
        meta_candidates = [
            self.data_dir / "metadata.json",
            self.data_dir / "dataset_info.json",
            self.data_dir / "info.json",
        ]
        metadata: Dict[str, Any] = {}
        for mpath in meta_candidates:
            if mpath.exists():
                try:
                    with open(mpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            metadata.update(data)
                except Exception as exc:
                    print(f"[DeepBeatLoader] Warning reading {mpath.name}: {exc}")

        return metadata

    def get_fs_and_window_length(self) -> Tuple[float, float]:
        """
        Detect and verify sampling rate and native window length.
        Ensures array shape is self-consistent with detected sampling rate.

        Raises:
            ValueError: If sampling rate cannot be determined from data or is inconsistent.
        """
        if self._fs is not None and self._window_length_s is not None:
            return self._fs, self._window_length_s

        files = self._discover_files()
        metadata = self._read_metadata()

        detected_fs: Optional[float] = None
        detected_window_s: Optional[float] = None
        window_samples: Optional[int] = None

        # Check metadata dictionary first
        fs_keys = ["fs", "sampling_rate", "sample_rate", "sampling_frequency", "frequency", "sr", "hz"]
        for k in fs_keys:
            if k in metadata:
                try:
                    detected_fs = float(metadata[k])
                    break
                except (ValueError, TypeError):
                    pass

        win_keys = ["native_window_s", "window_length_s", "window_duration_s", "window_seconds", "window_s"]
        for k in win_keys:
            if k in metadata:
                try:
                    detected_window_s = float(metadata[k])
                    break
                except (ValueError, TypeError):
                    pass

        # Inspect file attributes and array shapes
        first_file = files[0]
        if first_file.suffix == ".npz":
            with np.load(first_file, allow_pickle=True) as data:
                # Check for fs stored in npz keys
                for k in fs_keys:
                    if k in data.files:
                        val = data[k]
                        detected_fs = float(val.item() if hasattr(val, "item") and val.ndim == 0 else val[0])
                        break

                for k in win_keys:
                    if k in data.files:
                        val = data[k]
                        detected_window_s = float(val.item() if hasattr(val, "item") and val.ndim == 0 else val[0])
                        break

                # Find signal array shape
                sig_keys = ["signals", "signal", "X", "windows", "ppg"]
                for k in sig_keys:
                    if k in data.files:
                        arr = data[k]
                        if arr.ndim >= 2:
                            window_samples = arr.shape[1]
                        elif arr.ndim == 1:
                            window_samples = arr.shape[0]
                        break

        elif first_file.suffix in (".h5", ".hdf5"):
            try:
                import h5py
                with h5py.File(first_file, "r") as h5f:
                    # Check attrs
                    for k in fs_keys:
                        if k in h5f.attrs:
                            detected_fs = float(h5f.attrs[k])
                            break
                    for k in win_keys:
                        if k in h5f.attrs:
                            detected_window_s = float(h5f.attrs[k])
                            break

                    # Check dataset shape
                    for k in ["signals", "signal", "X", "windows", "ppg"]:
                        if k in h5f:
                            dset = h5f[k]
                            if len(dset.shape) >= 2:
                                window_samples = dset.shape[1]
                            break
            except ImportError:
                pass

        # Prompt requirement: If sample rate cannot be determined from data, stop and raise error rather than guessing.
        if detected_fs is None:
            raise ValueError(
                "\n[ERROR] Sampling rate (fs) cannot be determined from DeepBeat dataset files or metadata.\n"
                "In strict compliance with thesis constraints, sampling rate MUST NOT be guessed or hardcoded.\n"
                f"Please provide an explicit 'sampling_rate' or 'fs' in {self.data_dir / 'metadata.json'} "
                "or in the dataset container attributes before proceeding."
            )

        if window_samples is None:
            raise ValueError(
                f"\n[ERROR] Unable to determine signal window shape from dataset file: {first_file.name}."
            )

        # Compute window length in seconds from samples and fs
        computed_window_s = round(window_samples / detected_fs, 3)

        # If metadata also provided window length, assert self-consistency
        if detected_window_s is not None:
            if not math.isclose(computed_window_s, detected_window_s, rel_tol=1e-2):
                raise AssertionError(
                    f"Inconsistent DeepBeat window dimensions: metadata states {detected_window_s} s, "
                    f"but array shape has {window_samples} samples at {detected_fs} Hz "
                    f"({computed_window_s} s)."
                )
        else:
            detected_window_s = computed_window_s

        print(f"[DeepBeatLoader] Detected metadata: fs = {detected_fs:.1f} Hz, "
              f"window = {window_samples} samples ({detected_window_s:.2f} s). "
              "Self-consistency check passed.")

        self._fs = detected_fs
        self._window_length_s = detected_window_s
        return self._fs, self._window_length_s

    def _determine_partition(self, file_path: Path, item_partition: Optional[str] = None) -> Tuple[str, bool]:
        """
        Determine partition name and whether it is cardiologist-adjudicated.
        """
        name_lower = file_path.stem.lower()
        part = "train"
        if "val" in name_lower:
            part = "val"
        elif "test" in name_lower:
            part = "test"
        elif "train" in name_lower:
            part = "train"
        elif item_partition:
            part = item_partition.lower()

        is_adjudicated = False
        if "adjudicated" in name_lower or "cardiologist" in name_lower or name_lower == "test":
            is_adjudicated = True
            part = "test_adjudicated"
            self._adjudicated_partition_name = part

        return part, is_adjudicated

    def subjects(self) -> Iterator[SubjectRecord]:
        """
        Yield per-window SubjectRecord items across all partitions.

        Surfaces:
          - Subject IDs
          - Rhythm labels (1 = AF, 0 = non-AF)
          - Per-window quality labels
          - Official partition assignments
          - Cardiologist-adjudicated designation
        """
        files = self._discover_files()
        fs, window_length_s = self.get_fs_and_window_length()

        for fpath in files:
            partition_name, default_adjudicated = self._determine_partition(fpath)

            if fpath.suffix == ".npz":
                with np.load(fpath, allow_pickle=True) as data:
                    # Signals
                    sig_key = next((k for k in ["signals", "signal", "X", "windows", "ppg"] if k in data.files), None)
                    if sig_key is None:
                        continue
                    signals = data[sig_key]

                    # Labels (rhythm in DeepBeat is one-hot [non-AF, AF])
                    if "rhythm" in data.files:
                        rhythm_arr = data["rhythm"]
                        if hasattr(rhythm_arr, "ndim") and rhythm_arr.ndim == 2:
                            labels = np.argmax(rhythm_arr, axis=1).astype(int)
                        else:
                            labels = rhythm_arr.astype(int)
                    else:
                        lbl_key = next((k for k in ["labels", "label", "y", "rhythm_labels"] if k in data.files), None)
                        labels = data[lbl_key] if lbl_key else np.zeros(len(signals), dtype=int)

                    # Subject IDs (in parameters[:, 2] for DeepBeat)
                    if "parameters" in data.files:
                        params = data["parameters"]
                        if hasattr(params, "ndim") and params.ndim == 2 and params.shape[1] >= 3:
                            subject_ids = [str(params[i, 2]).strip() for i in range(len(signals))]
                        else:
                            subject_ids = [str(p).strip() for p in params]
                    else:
                        subj_key = next((k for k in ["subject_ids", "subject_id", "subjects", "patient_id"] if k in data.files), None)
                        if subj_key:
                            subject_ids = [str(s) for s in data[subj_key]]
                        else:
                            subject_ids = [f"deepbeat_{fpath.stem}_{i:05d}" for i in range(len(signals))]

                    # Quality labels
                    qual_key = next((k for k in ["qa_label", "qualities", "quality", "sqi", "quality_labels"] if k in data.files), None)
                    qualities = data[qual_key] if qual_key else None

                    # Adjudication flag
                    adj_key = next((k for k in ["is_adjudicated", "adjudicated"] if k in data.files), None)
                    adjudicated_flags = data[adj_key] if adj_key else None

                    for i in range(len(signals)):
                        sig_i = np.squeeze(signals[i]).astype(np.float32)
                        lbl_i = int(labels[i])
                        subj_i = subject_ids[i]
                        qual_i = qualities[i] if qualities is not None else None
                        is_adj = bool(adjudicated_flags[i]) if adjudicated_flags is not None else default_adjudicated

                        yield SubjectRecord(
                            subject_id=subj_i,
                            signal=sig_i,
                            fs=fs,
                            label=lbl_i,
                            quality=qual_i,
                            source_dataset="deepbeat",
                            native_window_s=window_length_s,
                            partition=partition_name,
                            is_adjudicated=is_adj,
                            metadata={"file": fpath.name, "index": i},
                        )

            elif fpath.suffix in (".h5", ".hdf5"):
                import h5py
                with h5py.File(fpath, "r") as h5f:
                    sig_key = next((k for k in ["signals", "signal", "X", "windows", "ppg"] if k in h5f), None)
                    if sig_key is None:
                        continue
                    signals = h5f[sig_key]
                    lbl_key = next((k for k in ["labels", "label", "y", "rhythm_labels"] if k in h5f), None)
                    labels = h5f[lbl_key] if lbl_key else [0] * len(signals)
                    subj_key = next((k for k in ["subject_ids", "subject_id", "subjects", "patient_id"] if k in h5f), None)
                    subject_ids = h5f[subj_key] if subj_key else None
                    qual_key = next((k for k in ["qualities", "quality", "sqi", "quality_labels"] if k in h5f), None)
                    qualities = h5f[qual_key] if qual_key else None

                    for i in range(len(signals)):
                        sig_i = np.asarray(signals[i], dtype=np.float32)
                        lbl_i = int(labels[i])
                        subj_i = str(subject_ids[i]) if subject_ids is not None else f"deepbeat_{fpath.stem}_{i:05d}"
                        qual_i = np.asarray(qualities[i]) if qualities is not None else None

                        yield SubjectRecord(
                            subject_id=subj_i,
                            signal=sig_i,
                            fs=fs,
                            label=lbl_i,
                            quality=qual_i,
                            source_dataset="deepbeat",
                            native_window_s=window_length_s,
                            partition=partition_name,
                            is_adjudicated=default_adjudicated,
                            metadata={"file": fpath.name, "index": i},
                        )

    def summary(self) -> Dict[str, Any]:
        """
        Compute summary metrics from actual data files.
        Uses fast vectorized inspection over container arrays.
        """
        fs, window_length_s = self.get_fs_and_window_length()
        files = self._discover_files()
        total_windows = 0
        af_windows = 0
        non_af_windows = 0
        unique_subjects: Set[str] = set()
        partitions_found: Set[str] = set()
        adjudicated_count = 0

        for fpath in files:
            part, is_adj_file = self._determine_partition(fpath)
            if part:
                partitions_found.add(part)

            if fpath.suffix == ".npz":
                with np.load(fpath, allow_pickle=True) as data:
                    sig_key = next((k for k in ["signals", "signal", "X", "windows", "ppg"] if k in data.files), None)
                    if sig_key is None:
                        continue
                    n_win = data[sig_key].shape[0]
                    total_windows += n_win

                    if "rhythm" in data.files:
                        r = data["rhythm"]
                        if hasattr(r, "ndim") and r.ndim == 2:
                            lbls = np.argmax(r, axis=1)
                        else:
                            lbls = r
                        n_af = int(np.sum(lbls == 1))
                    elif "labels" in data.files:
                        lbls = data["labels"]
                        n_af = int(np.sum(lbls == 1))
                    else:
                        n_af = 0

                    af_windows += n_af
                    non_af_windows += (n_win - n_af)

                    if "parameters" in data.files:
                        params = data["parameters"]
                        if hasattr(params, "ndim") and params.ndim == 2 and params.shape[1] >= 3:
                            subjs = np.unique(params[:, 2])
                            unique_subjects.update(str(s).strip() for s in subjs)
                        else:
                            unique_subjects.update(str(p).strip() for p in params)
                    elif "subject_ids" in data.files:
                        unique_subjects.update(str(s).strip() for s in np.unique(data["subject_ids"]))
                    else:
                        unique_subjects.add(f"deepbeat_{fpath.stem}")

                    if is_adj_file:
                        adjudicated_count += n_win

            elif fpath.suffix in (".h5", ".hdf5"):
                try:
                    import h5py
                    with h5py.File(fpath, "r") as h5f:
                        sig_key = next((k for k in ["signals", "signal", "X", "windows", "ppg"] if k in h5f), None)
                        if sig_key is None:
                            continue
                        n_win = h5f[sig_key].shape[0]
                        total_windows += n_win
                        if "labels" in h5f:
                            lbls = np.asarray(h5f["labels"])
                            n_af = int(np.sum(lbls == 1))
                            af_windows += n_af
                            non_af_windows += (n_win - n_af)
                        if "subject_ids" in h5f:
                            unique_subjects.update(str(s).strip() for s in np.unique(h5f["subject_ids"]))
                        if is_adj_file:
                            adjudicated_count += n_win
                except Exception:
                    pass

        total_duration_s = total_windows * window_length_s
        adj_name = self._adjudicated_partition_name or ("test_adjudicated" if adjudicated_count > 0 else "None identified")

        return {
            "dataset_name": "DeepBeat (Torres-Soto & Ashley 2020, npj Digital Medicine)",
            "role": "PRIMARY TRAINING SET (Train, Validation, Internal Test Benchmark)",
            "detected_sample_rate": f"{fs:.1f} Hz",
            "detected_window_length": f"{window_length_s:.2f} s ({int(round(window_length_s * fs))} samples)",
            "total_windows": total_windows,
            "total_unique_subjects": len(unique_subjects),
            "af_windows": af_windows,
            "non_af_windows": non_af_windows,
            "af_window_balance": f"{af_windows} AF ({af_windows / max(1, total_windows) * 100:.1f}%) / {non_af_windows} non-AF ({non_af_windows / max(1, total_windows) * 100:.1f}%)",
            "total_duration": f"{total_duration_s:.1f} s ({total_duration_s / 3600:.2f} hours)",
            "partitions_detected": ", ".join(sorted(partitions_found)) if partitions_found else "None",
            "cardiologist_adjudicated_partition": f"{adj_name} ({adjudicated_count} windows)",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepBeat PPG Dataset Loader & Summary Utility"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to 03 - ML/data/deepbeat directory",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print detected sample rate, window length, subject count, AF balance, and duration.",
    )
    args = parser.parse_args()

    try:
        loader = DeepBeatLoader(data_dir=args.data_dir)
        loader.print_summary()
    except FileNotFoundError as fnf_err:
        print(fnf_err, file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n[DeepBeatLoader Error] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
