#!/usr/bin/env bash
# Deploy the ibi_af_v1 classifier to the live edge service. Run ON the Pi:
#   bash "/home/ranilo/Arrhythmia Thesis/03 - ML/deploy/deploy_ibi_on_pi.sh"
#
# Steps: fast-forward the checkout, prove the classifier loads and classifies
# under the Pi's own Python, run the test suite, restart the edge service, and
# confirm from the journal that it came up on ibi_af_v1 rather than falling back
# to the CNN. Stops at the first failure and leaves the running service alone
# until the pre-flight checks have passed.
set -euo pipefail

REPO="/home/ranilo/Arrhythmia Thesis"
ML="$REPO/03 - ML"
cd "$REPO"

echo "== 1/5 Update checkout"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Tracked files have local edits on the Pi; refusing to pull over them:" >&2
    git status --short --untracked-files=no >&2
    exit 1
fi
git pull --ff-only
git log --oneline -1

echo "== 2/5 Classifier pre-flight (Pi Python, NumPy only)"
cd "$ML"
python3 - <<'PY'
import time, numpy as np
from model.ibi_classifier import IBIAFClassifier
m = IBIAFClassifier().load("model/weights/ibi_af_v1.npz")
rng = np.random.default_rng(0)
regular = np.cumsum(np.full(12, 80)).astype(int)                  # 75 bpm, steady
irregular = np.cumsum(rng.integers(45, 130, size=12)).astype(int)  # AF-like spread
p_reg, _, _ = m.predict_from_peaks(regular)
p_irr, _, _ = m.predict_from_peaks(irregular)
t0 = time.perf_counter()
for _ in range(1000):
    m.predict_from_peaks(irregular)
us = (time.perf_counter() - t0) * 1e3
print(f"numpy {np.__version__} | {m.model_version} tau={m.threshold:.2f}")
print(f"P(AF) regular={p_reg:.3f} irregular={p_irr:.3f} | classifier-only {us:.1f} us/call")
assert p_irr > p_reg, "irregular rhythm did not score above regular rhythm"
PY

echo "== 3/5 Test suite"
if python3 -c "import pytest" 2>/dev/null; then
    # Parity tests need torch, which is dev-machine only (settled decision 10);
    # they skip themselves when it is absent.
    python3 -m pytest -q tests
else
    echo "pytest not installed on the Pi; skipping (install: sudo apt install python3-pytest)"
fi

echo "== 4/5 Restart edge service"
sudo systemctl restart arrhythmia-edge.service
sleep 8
systemctl is-active arrhythmia-edge.service

echo "== 5/5 Confirm the deployed classifier"
LOG="$(journalctl -u arrhythmia-edge.service --since '-30s' --no-pager)"
echo "$LOG" | grep -E "IBI classifier|WARNING|Error|Traceback" || true
if echo "$LOG" | grep -q "IBI classifier loaded: ibi_af_v1"; then
    echo "OK: edge service is running ibi_af_v1."
else
    echo "FAIL: did not see 'IBI classifier loaded: ibi_af_v1' in the journal." >&2
    exit 1
fi
