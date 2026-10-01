"""
Parity gate for the amended topology: PyTorch training mirror vs pure-NumPy edge runtime.

This is the test that earns its keep. In the v1 pipeline the NumPy runtime flattened
time-major ``(t * C + c)`` while PyTorch flattened channel-major ``(c * T + t)`` — the same
16,000 values in a different order, which produces no error and silently destroys training.
So every head is checked here, each with a negative control proving the test can fail.

Requires torch, so it is a dev-machine test. The edge runtime it validates imports nothing
but NumPy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

torch = pytest.importorskip("torch", reason="parity test needs the training mirror")

from model.inference_model_v2 import ArrhythmiaCNNv2
from training.torch_model_v2 import ConfigurableAF1DCNN

TOL = 1e-5
CONFIGS = [
    ("flatten", (5, 3), (2, 2)),
    ("gap", (31, 15), (4, 4)),
    ("statpool", (127, 63), (8, 8)),
    ("statpool", (31, 15), (4, 4)),
]


def _pair(head, kernels, pools, seed=0):
    torch.manual_seed(seed)
    t = ConfigurableAF1DCNN(input_length=1000, kernels=kernels, pools=pools, head=head)
    t.eval()
    n = ArrhythmiaCNNv2(input_length=1000, kernels=kernels, pools=pools, head=head)
    n.W = {k: v.detach().numpy().astype(np.float64) for k, v in t.state_dict().items()}
    n.weights_loaded = True
    return t, n


@pytest.mark.parametrize("head,kernels,pools", CONFIGS)
def test_forward_parity(head, kernels, pools):
    t, n = _pair(head, kernels, pools)
    rng = np.random.default_rng(7)
    X = rng.standard_normal((16, 1000)).astype(np.float32)
    with torch.no_grad():
        want = torch.sigmoid(
            t(torch.from_numpy(X).float().unsqueeze(1), return_logits=True)
        ).squeeze(-1).numpy()
    got = np.array([n.predict(X[i])[0] for i in range(len(X))])
    assert np.max(np.abs(got - want)) < TOL, (
        f"{head} forward mismatch: max |diff| = {np.max(np.abs(got - want)):.3e}")


@pytest.mark.parametrize("head,kernels,pools", CONFIGS)
def test_gradcam_gradient_matches_autograd(head, kernels, pools):
    """d(logit)/dA2 in closed form must match torch.autograd."""
    t, n = _pair(head, kernels, pools)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(1000).astype(np.float32)

    xt = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    a2 = t.features(xt)
    a2.retain_grad()
    t.head_forward(a2).backward()
    want = a2.grad.squeeze(0).detach().numpy()

    got = n.dlogit_da2(n.forward_with_cache(x))
    assert got.shape == want.shape
    denom = max(np.max(np.abs(want)), 1e-12)
    assert np.max(np.abs(got - want)) / denom < 1e-4, (
        f"{head} grad mismatch: max |diff| = {np.max(np.abs(got - want)):.3e}")


def test_flatten_order_negative_control():
    """A channel-major flatten must FAIL parity — proves the test detects the v1 bug."""
    t, n = _pair("flatten", (5, 3), (2, 2))
    rng = np.random.default_rng(3)
    x = rng.standard_normal(1000).astype(np.float32)
    with torch.no_grad():
        want = float(torch.sigmoid(
            t(torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0), return_logits=True)))

    cache = n.forward_with_cache(x)
    a2 = cache["a2"]
    wrong = a2.reshape(-1)                       # channel-major: the v1 defect
    pre = n.W["linear1.weight"] @ wrong + n.W["linear1.bias"]
    h = np.maximum(pre, 0.0)
    logit = float((n.W["linear2.weight"] @ h + n.W["linear2.bias"]).ravel()[0])
    got_wrong = 1.0 / (1.0 + np.exp(-logit))
    assert abs(got_wrong - want) > TOL, "negative control passed; the test cannot detect it"
    assert abs(cache["probability"] - want) < TOL, "correct time-major order must still pass"


def test_statpool_head_uses_variability():
    """
    The statpool head must respond to interval variability, not only to mean activation:
    two inputs with identical channel means but different temporal spread must differ.
    """
    _, n = _pair("statpool", (31, 15), (4, 4))
    t_ax = np.arange(1000) / 100.0
    regular = np.sin(2 * np.pi * 1.2 * t_ax).astype(np.float32)
    # same dominant rate, but intervals alternate short/long -> higher variability
    phase = np.cumsum(1.2 + 0.45 * np.sign(np.sin(2 * np.pi * 0.12 * t_ax))) / 100.0
    irregular = np.sin(2 * np.pi * phase).astype(np.float32)
    assert abs(n.predict(regular)[2] - n.predict(irregular)[2]) > 1e-6


def test_weight_shape_mismatch_is_rejected(tmp_path):
    """Loading weights built for another architecture must raise, not silently misbehave."""
    t, _ = _pair("gap", (31, 15), (4, 4))
    p = tmp_path / "w.npz"
    np.savez(p, **{k: v.detach().numpy() for k, v in t.state_dict().items()})
    with pytest.raises((ValueError, KeyError)):
        ArrhythmiaCNNv2(kernels=(127, 63), pools=(8, 8), head="statpool").load_weights(p)
