"""
Base Dataset Loader Interface and Common Data Structures.
=========================================================
Defines the uniform contract for all PPG arrhythmia datasets (DeepBeat, MIMIC PERform AF)
so loaders can be swapped without modifying downstream preprocessing or evaluation pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional
import numpy as np


@dataclass
class SubjectRecord:
    """
    Unified record representation for PPG dataset subjects and windows.

    Attributes:
        subject_id: Unique string identifier for the patient/subject.
        signal: 1-D float32 PPG signal array.
        fs: Sampling frequency in Hz.
        label: Rhythm classification label (1 = AF, 0 = non-AF).
        quality: Optional per-window or per-sample quality array (None if unannotated).
        source_dataset: Source dataset name ('deepbeat' or 'mimic_perform').
        native_window_s: Native window length in seconds for pre-windowed datasets
                         (e.g., DeepBeat), or None for continuous recordings (e.g., MIMIC).
        partition: Optional dataset partition ('train', 'val', 'test', 'test_adjudicated').
        is_adjudicated: Whether rhythm label is expert cardiologist-adjudicated.
        metadata: Optional dictionary containing auxiliary metadata (e.g. device, origin).
    """

    subject_id: str
    signal: np.ndarray
    fs: float
    label: int
    quality: Optional[np.ndarray] = None
    source_dataset: str = ""
    native_window_s: Optional[float] = None
    partition: Optional[str] = None
    is_adjudicated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.signal, np.ndarray):
            self.signal = np.asarray(self.signal, dtype=np.float32)
        elif self.signal.dtype != np.float32:
            self.signal = self.signal.astype(np.float32)

        if self.signal.ndim != 1:
            self.signal = self.signal.flatten()

        self.label = int(self.label)
        if self.label not in (0, 1):
            raise ValueError(f"Label must be 0 (non-AF) or 1 (AF), got {self.label}")

        self.fs = float(self.fs)
        if self.fs <= 0:
            raise ValueError(f"Sampling frequency fs must be positive, got {self.fs}")

        if self.native_window_s is not None:
            self.native_window_s = float(self.native_window_s)


class PPGDatasetLoader(ABC):
    """
    Abstract base loader for photoplethysmography (PPG) arrhythmia datasets.
    Provides a swappable interface yielding SubjectRecord items.
    """

    @abstractmethod
    def subjects(self) -> Iterator[SubjectRecord]:
        """
        Yield per-subject continuous recordings or per-window records.

        Returns:
            Iterator of SubjectRecord dataclass instances.
        """
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """
        Compute and return dataset summary statistics.

        Returns:
            Dict containing detected fs, subject count, AF/non-AF balance, total duration.
        """
        raise NotImplementedError

    def print_summary(self) -> None:
        """Print a human-readable summary table to stdout."""
        stats = self.summary()
        name = stats.get("dataset_name", self.__class__.__name__)
        print("=" * 65)
        print(f" Dataset Summary: {name}")
        print("=" * 65)
        for key, value in stats.items():
            if key == "dataset_name":
                continue
            label_str = key.replace("_", " ").title()
            print(f"  {label_str:<28}: {value}")
        print("=" * 65)
