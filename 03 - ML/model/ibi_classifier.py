"""
Pure-NumPy IBI-irregularity AF classifier — edge runtime.

This is the model that measurably works on the deployment-relevant cohort: patient-disjoint
5-fold CV gives subject-level AUROC 0.97 (0.91 from rate-free irregularity features alone),
against 0.53-0.77 for the convolutional variants on identical data and protocol. It answers
RQ1 explicitly, because its inputs *are* the physiological parameters: beat-to-beat interval
dispersion, successive-difference magnitude, and interval entropy.

Twelve standardised features feed a logistic regression, so the whole classifier is a
mean vector, a scale vector, a coefficient vector and an intercept. It carries no risk of
memorising recording identity, and it runs in microseconds on the Raspberry Pi with no
dependency beyond what the DSP chain already requires.

Interpretability note for RQ3: because each coefficient attaches to a named physiological
quantity, this model is *directly* interpretable — a per-feature contribution breakdown is
available from `explain()`, which is a stronger interpretability claim than a saliency map
over a waveform. Grad-CAM on the CNN remains the waveform-localisation story.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

FEATURE_NAMES: Tuple[str, ...] = (
    "n_beats", "mean_ibi", "sd_ibi", "rmssd", "cv_ibi", "pnn50",
    "ibi_range", "ibi_iqr", "shannon_dibi", "mean_hr", "sd_hr", "median_dibi_abs",
)

# Features that encode heart RATE rather than rhythm irregularity. Reported separately so a
# reviewer can see how much of a decision rests on rate, which is a population confound in
# cohorts where AF and non-AF subjects are different people.
RATE_FEATURES: Tuple[str, ...] = ("n_beats", "mean_ibi", "mean_hr")


def ibi_features_from_peaks(peaks: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Interval statistics from detected systolic peak indices. Returns zeros when the window
    has too few usable beats to characterise rhythm, which the caller should treat as
    'insufficient quality' rather than 'non-AF'.
    """
    f = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.float64).ravel()
    if peaks.size < 4:
        return f

    ibi = np.diff(peaks) / float(fs)
    ibi = ibi[(ibi > 0.25) & (ibi < 2.5)]           # 24-240 bpm
    if ibi.size < 3:
        return f

    dibi = np.diff(ibi)
    mean_ibi = float(np.mean(ibi))
    hr = 60.0 / ibi
    hist, _ = np.histogram(dibi, bins=8, range=(-0.5, 0.5))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]

    f[:] = [
        ibi.size + 1,
        mean_ibi,
        float(np.std(ibi)),
        float(np.sqrt(np.mean(dibi ** 2))),
        float(np.std(ibi) / mean_ibi) if mean_ibi > 0 else 0.0,
        float(np.mean(np.abs(dibi) > 0.05)),
        float(np.max(ibi) - np.min(ibi)),
        float(np.subtract(*np.percentile(ibi, [75, 25]))),
        float(-np.sum(p * np.log2(p))) if p.size else 0.0,
        float(np.mean(hr)),
        float(np.std(hr)),
        float(np.median(np.abs(dibi))),
    ]
    return f


class IBIAFClassifier:
    """Standardise -> logistic regression. Loaded from the artefacts written by training."""

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.coef: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.threshold: float = 0.5
        self.consensus_k: int = 3
        self.consensus_n: int = 5
        self.model_version: str = "untrained"
        self.weights_loaded: bool = False
        self.metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------ persistence

    def load(self, path: Union[str, Path]) -> "IBIAFClassifier":
        path = Path(path)
        z = np.load(path)
        for key in ("mean", "scale", "coef"):
            if key not in z:
                raise KeyError(f"{path.name} is missing '{key}'")
        self.mean = np.asarray(z["mean"], dtype=np.float64)
        self.scale = np.asarray(z["scale"], dtype=np.float64)
        self.coef = np.asarray(z["coef"], dtype=np.float64).ravel()
        self.intercept = float(np.asarray(z["intercept"]).ravel()[0])
        if not (self.mean.size == self.scale.size == self.coef.size == len(FEATURE_NAMES)):
            raise ValueError(
                f"{path.name}: expected {len(FEATURE_NAMES)} features, got "
                f"mean={self.mean.size} scale={self.scale.size} coef={self.coef.size}")

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.threshold = float(meta.get("decision_threshold", 0.5))
            self.consensus_k = int(meta.get("consensus_k", 3))
            self.consensus_n = int(meta.get("consensus_n", 5))
            self.model_version = str(meta.get("model_version", path.stem))
            self.metrics = meta.get("metrics", {}) or {}
        else:
            self.model_version = path.stem
        self.weights_loaded = True
        return self

    # ---------------------------------------------------------------------- predict

    def _z(self, feats: np.ndarray) -> np.ndarray:
        scale = np.where(np.abs(self.scale) < 1e-12, 1.0, self.scale)
        return (np.asarray(feats, dtype=np.float64) - self.mean) / scale

    def predict_from_features(self, feats: np.ndarray) -> Tuple[float, int, float]:
        if not self.weights_loaded:
            raise RuntimeError("load() must be called before inference")
        logit = float(self.coef @ self._z(feats) + self.intercept)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))
        return prob, int(prob >= self.threshold), logit

    def predict_from_peaks(self, peaks: np.ndarray, fs: float = 100.0) -> Tuple[float, int, float]:
        return self.predict_from_features(ibi_features_from_peaks(peaks, fs))

    def explain(self, feats: np.ndarray) -> Dict[str, Any]:
        """
        Per-feature signed contribution to the logit — the interpretability output for this
        model. `rate_fraction` is the share of total absolute contribution coming from
        rate features, which flags a decision that leans on rate rather than irregularity.
        """
        z = self._z(feats)
        contrib = self.coef * z
        total = float(np.sum(np.abs(contrib)))
        rate_idx = [FEATURE_NAMES.index(n) for n in RATE_FEATURES]
        prob, af, logit = self.predict_from_features(feats)
        order = np.argsort(-np.abs(contrib))
        return {
            "probability": prob,
            "af_detected": af,
            "logit": logit,
            "contributions": {FEATURE_NAMES[i]: round(float(contrib[i]), 4) for i in order},
            "top_drivers": [FEATURE_NAMES[i] for i in order[:3]],
            "rate_fraction": round(float(np.sum(np.abs(contrib[rate_idx])) / total), 4)
            if total > 0 else 0.0,
        }

    def explain_intervals(
        self, peaks: np.ndarray, fs: float = 100.0, n_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Beat-level attribution: how much each inter-beat interval contributed to the AF
        score, plus a heat strip over the waveform for the dashboard overlay.

        The attribution is a **counterfactual**, not a gradient approximation: interval j is
        replaced by the window's median interval, the features are recomputed, and the drop
        in logit is that interval's contribution. It is therefore exactly faithful to the
        model by construction — there is nothing to validate against autograd, because no
        derivative is involved.

        This is the interpretability answer for RQ3 on this model: it names the specific
        beats whose timing drove the classification, which is the clinical question ("which
        beats were irregular?"), rather than a smeared relevance band over the waveform.

        Read contributions as a **ranking within one window, not a magnitude comparable
        across windows.** Substituting the median interval reduces measured variability in
        any window, regular or not, so a regular window also yields positive contributions.
        What distinguishes AF is the window's baseline logit (and hence `probability`); the
        per-interval values say which beats within that window mattered most.
        """
        peaks = np.asarray(peaks, dtype=np.float64).ravel()
        base_feats = ibi_features_from_peaks(peaks, fs)
        _, _, base_logit = self.predict_from_features(base_feats)

        ibi = np.diff(peaks) / float(fs)
        valid = (ibi > 0.25) & (ibi < 2.5)
        heat = np.zeros(int(n_samples), dtype=np.float64)
        intervals: List[Dict[str, Any]] = []
        if valid.sum() < 3:
            return {"logit": base_logit, "median_ibi_s": None, "intervals": intervals,
                    "top_intervals": [], "heatmap": heat, "insufficient_beats": True}

        median_ibi = float(np.median(ibi[valid]))
        for j in range(len(ibi)):
            if not valid[j]:
                continue
            counter = ibi.copy()
            counter[j] = median_ibi
            # rebuild peak positions from the counterfactual interval sequence
            cf_peaks = np.concatenate([[peaks[0]], peaks[0] + np.cumsum(counter) * fs])
            _, _, cf_logit = self.predict_from_features(ibi_features_from_peaks(cf_peaks, fs))
            contribution = base_logit - cf_logit
            intervals.append({
                "index": int(j),
                "start_sample": int(peaks[j]),
                "end_sample": int(peaks[j + 1]),
                "ibi_s": round(float(ibi[j]), 4),
                "deviation_from_median_s": round(float(ibi[j] - median_ibi), 4),
                "contribution": round(float(contribution), 4),
            })

        if intervals:
            mx = max(abs(iv["contribution"]) for iv in intervals)
            for iv in intervals:
                s = max(0, min(int(n_samples) - 1, iv["start_sample"]))
                e = max(s + 1, min(int(n_samples), iv["end_sample"]))
                heat[s:e] = max(0.0, iv["contribution"]) / mx if mx > 0 else 0.0

        ranked = sorted(intervals, key=lambda iv: -iv["contribution"])
        return {
            "logit": base_logit,
            "median_ibi_s": round(median_ibi, 4),
            "intervals": intervals,
            "top_intervals": ranked[:3],
            "heatmap": heat,
            "insufficient_beats": False,
        }

    def consensus(self, recent_flags: List[int]) -> int:
        """k-of-n temporal consensus. A single 10 s window must never raise an alarm."""
        window = list(recent_flags)[-self.consensus_n:]
        return int(sum(window) >= self.consensus_k)
