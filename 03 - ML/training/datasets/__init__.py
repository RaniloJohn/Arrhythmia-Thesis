"""
PPG Dataset Loaders Package.
============================
Provides unified loaders for photoplethysmography (PPG) arrhythmia datasets.
"""

from .base import PPGDatasetLoader, SubjectRecord

__all__ = [
    "PPGDatasetLoader",
    "SubjectRecord",
    "DeepBeatLoader",
    "MimicPerformAFLoader",
]


def __getattr__(name: str):
    if name == "DeepBeatLoader":
        from .deepbeat import DeepBeatLoader
        return DeepBeatLoader
    if name == "MimicPerformAFLoader":
        from .mimic_perform import MimicPerformAFLoader
        return MimicPerformAFLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
