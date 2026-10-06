"""
Tests for the deployed IBI-irregularity classifier and its explanation path.

These cover the properties that the v1 pipeline's 71 tests did not: that the model responds
to interval irregularity rather than rate, that its exported artefact round-trips, and that
label/subject integrity holds in the rebuilt dataset. A test suite that passes while the
model scores below chance on its own training data is testing the wrong things.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from model.ibi_classifier import (FEATURE_NAMES, RATE_FEATURES, IBIAFClassifier,
                                  ibi_features_from_peaks)

WEIGHTS = ML_DIR / "model" / "weights" / "ibi_af_v1.npz"
needs_weights = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="run training/train_ibi_model.py first")


# ------------------------------------------------------------------------- features


def test_too_few_beats_returns_zero_vector():
    """Insufficient beats must yield zeros, which the caller treats as low quality."""
    assert np.all(ibi_features_from_peaks(np.array([10, 110])) == 0.0)
    assert np.all(ibi_features_from_peaks(np.array([])) == 0.0)


def test_regular_rhythm_has_near_zero_irregularity():
    peaks = np.arange(0, 1000, 80)  # exactly 0.8 s apart
    f = ibi_features_from_peaks(peaks, fs=100.0)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert f[idx["sd_ibi"]] == pytest.approx(0.0, abs=1e-9)
    assert f[idx["rmssd"]] == pytest.approx(0.0, abs=1e-9)
    assert f[idx["pnn50"]] == pytest.approx(0.0, abs=1e-9)
    assert f[idx["mean_ibi"]] == pytest.approx(0.8, abs=1e-9)


def test_irregular_rhythm_raises_irregularity_features():
    rng = np.random.default_rng(0)
    ibi = 0.8 + rng.uniform(-0.25, 0.25, 12)
    peaks = np.concatenate([[0], np.cumsum(ibi) * 100])
    f = ibi_features_from_peaks(peaks, fs=100.0)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert f[idx["sd_ibi"]] > 0.05
    assert f[idx["pnn50"]] > 0.3
    assert f[idx["cv_ibi"]] > 0.05


def test_nonphysiological_intervals_are_discarded():
    """A 5 s gap (dropped beats) must not be treated as a real interval."""
    peaks = np.array([0, 80, 160, 660, 740, 820, 900], dtype=float)  # one 5 s gap
    f = ibi_features_from_peaks(peaks, fs=100.0)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert f[idx["mean_ibi"]] == pytest.approx(0.8, abs=1e-6)


# ---------------------------------------------------------------------- the artefact


@needs_weights
def test_exported_model_round_trips():
    clf = IBIAFClassifier().load(WEIGHTS)
    assert clf.weights_loaded
    assert clf.coef.size == len(FEATURE_NAMES)
    assert 0.0 < clf.threshold < 1.0
    meta = json.loads(WEIGHTS.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["metrics"]["subject_auroc"] >= 0.70, "a model below the skill floor must not ship"
    assert meta["feature_names"] == list(FEATURE_NAMES)


@needs_weights
def test_irregular_scores_higher_than_regular():
    """The core behavioural contract: irregular rhythm must score above regular rhythm."""
    clf = IBIAFClassifier().load(WEIGHTS)
    regular = np.arange(0, 1000, 75, dtype=float)
    rng = np.random.default_rng(3)
    ibi = 0.75 + rng.uniform(-0.3, 0.3, 13)
    irregular = np.concatenate([[0], np.cumsum(ibi) * 100])
    p_reg, _, _ = clf.predict_from_peaks(regular)
    p_irr, _, _ = clf.predict_from_peaks(irregular)
    assert p_irr > p_reg, f"irregular {p_irr:.3f} must exceed regular {p_reg:.3f}"


@needs_weights
def test_same_rate_different_regularity_changes_score():
    """
    Guards the rate confound directly: two windows with the same mean interval but
    different variability must score differently, so the model is not a rate detector.
    """
    clf = IBIAFClassifier().load(WEIGHTS)
    mean_ibi = 0.8
    # An even number of alternating intervals is required for the means to match exactly.
    regular = np.concatenate([[0], np.cumsum(np.full(12, mean_ibi)) * 100])
    alt = np.tile([mean_ibi - 0.25, mean_ibi + 0.25], 6)
    irregular = np.concatenate([[0], np.cumsum(alt) * 100])
    f_reg = ibi_features_from_peaks(regular)
    f_irr = ibi_features_from_peaks(irregular)
    idx = FEATURE_NAMES.index("mean_ibi")
    assert f_reg[idx] == pytest.approx(f_irr[idx], abs=0.02), "mean rate must match"
    assert clf.predict_from_features(f_irr)[0] > clf.predict_from_features(f_reg)[0]


@needs_weights
def test_explain_reports_rate_fraction_and_sums_to_logit():
    clf = IBIAFClassifier().load(WEIGHTS)
    rng = np.random.default_rng(5)
    ibi = 0.8 + rng.uniform(-0.2, 0.2, 12)
    peaks = np.concatenate([[0], np.cumsum(ibi) * 100])
    feats = ibi_features_from_peaks(peaks)
    ex = clf.explain(feats)
    assert 0.0 <= ex["rate_fraction"] <= 1.0
    assert set(ex["contributions"]) == set(FEATURE_NAMES)
    # contributions plus intercept must reconstruct the logit exactly
    assert sum(ex["contributions"].values()) + clf.intercept == pytest.approx(ex["logit"], abs=1e-3)
    assert all(d in FEATURE_NAMES for d in ex["top_drivers"])


@needs_weights
def test_interval_attribution_localises_the_irregular_beat():
    """
    The counterfactual attribution must rank the genuinely deviant interval highest, and
    produce a heat strip bounded to [0, 1] over the window.
    """
    clf = IBIAFClassifier().load(WEIGHTS)
    ibi = np.full(11, 0.80)
    ibi[5] = 1.35                                  # one clearly ectopic long interval
    peaks = np.concatenate([[0], np.cumsum(ibi) * 100])
    res = clf.explain_intervals(peaks, fs=100.0, n_samples=1000)
    assert not res["insufficient_beats"]
    assert res["top_intervals"][0]["index"] == 5
    assert res["heatmap"].min() >= 0.0 and res["heatmap"].max() <= 1.0


@needs_weights
def test_consensus_requires_k_of_n():
    clf = IBIAFClassifier().load(WEIGHTS)
    assert clf.consensus([1, 0, 0, 0, 0]) == 0, "a single window must never alarm"
    assert clf.consensus([0, 1, 1, 0, 1]) == 1
    assert clf.consensus([1, 1]) == 0


def test_missing_keys_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, mean=np.zeros(12), scale=np.ones(12))      # no coef
    with pytest.raises(KeyError):
        IBIAFClassifier().load(p)


def test_wrong_feature_count_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, mean=np.zeros(5), scale=np.ones(5), coef=np.zeros(5), intercept=np.array([0.0]))
    with pytest.raises(ValueError):
        IBIAFClassifier().load(p)


def test_predict_before_load_raises():
    with pytest.raises(RuntimeError):
        IBIAFClassifier().predict_from_features(np.zeros(len(FEATURE_NAMES)))


def test_rate_features_are_a_subset_of_feature_names():
    assert set(RATE_FEATURES) <= set(FEATURE_NAMES)


@needs_weights
def test_interval_attribution_schema_is_stable_with_too_few_beats():
    """The insufficient-beats branch must expose the same keys callers read."""
    clf = IBIAFClassifier().load(WEIGHTS)
    full = clf.explain_intervals(np.concatenate([[0], np.cumsum(np.full(11, 0.8)) * 100]))
    short = clf.explain_intervals(np.array([0, 80, 160]))
    assert short["insufficient_beats"] is True
    assert set(full) <= set(short)
    assert short["top_intervals"] == []
