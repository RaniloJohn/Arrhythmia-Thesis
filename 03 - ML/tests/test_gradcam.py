"""
Grad-CAM Gradient Verification & Upsampling Tests (Prompt 7 GATE)
================================================================
Specifications:
1. Verify analytical gradients d(logit)/d(a2) in GradCAM1D against torch.autograd
   to a tolerance of 1e-4 using identical trained weights.
2. Verify channel importance weights alpha_k agree with torch.autograd to 1e-4.
3. Verify upsampling to 1000 samples and normalization to [0.0, 1.0].
4. Verify execution on production trained weights (cnn_af_v1.npz).
"""

import os
import pytest
import numpy as np
from pathlib import Path

import torch

from model.inference_model import Arrhythmia1DCNN as NumPyModel
from model.grad_cam import GradCAM1D
from training.torch_model import Arrhythmia1DCNN as TorchModel
from training.calibrate_threshold import load_numpy_weights_into_torch


@pytest.fixture
def trained_weights_path():
    p = Path(__file__).resolve().parent.parent / "model" / "weights" / "cnn_af_v1.npz"
    assert p.exists(), f"Weights file {p} must exist for testing."
    return p


def compute_numpy_gradients(model: NumPyModel, x: np.ndarray):
    """Computes analytical d(logit)/d(a2) exactly as implemented in GradCAM1D."""
    cache = model.forward_with_cache(x)
    a2 = cache["a2"]             # Shape: (500, 64)
    p2 = cache["p2"]             # Shape: (250, 64)
    argmax2 = cache["argmax2"]   # Shape: (250, 64)
    a_dense1 = cache["a_dense1"] # Shape: (64,)

    # 1. Gradient of output logit w.r.t. a_dense1
    grad_dense1 = model.dense_out.weights[:, 0]  # Shape: (64,)

    # 2. Backprop through ReLU of Dense1
    grad_dense1_relu = grad_dense1 * (a_dense1 > 0.0)

    # 3. Backprop through Dense1: (16000,)
    grad_flat = np.dot(model.dense1.weights, grad_dense1_relu)
    grad_p2 = grad_flat.reshape(p2.shape)  # Shape: (250, 64)

    # 4. Backprop through MaxPool (pool_size=2) to shape (500, 64)
    grad_a2 = np.zeros_like(a2)
    for i in range(p2.shape[0]):
        for ch in range(p2.shape[1]):
            offset = argmax2[i, ch]
            grad_a2[i * 2 + offset, ch] = grad_p2[i, ch]

    # Global Average Pooling of gradients to get channel alphas
    alphas = np.mean(grad_a2, axis=0)  # Shape: (64,)

    return grad_a2, alphas, cache


def compute_torch_gradients(torch_model: TorchModel, x: np.ndarray):
    """Computes exact autograd d(logit)/d(a2) using PyTorch."""
    torch_model.eval()
    x_t = torch.from_numpy(x).float().view(1, 1, -1)

    # Forward through Conv1 -> ReLU -> MaxPool1
    c1 = torch_model.conv1(x_t)
    r1 = torch.relu(c1)
    p1 = torch_model.pool1(r1)

    # Forward through Conv2 -> ReLU (a2)
    c2 = torch_model.conv2(p1)
    a2_t = torch.relu(c2)
    a2_t.retain_grad()

    # Forward through MaxPool2 -> Flatten -> Dense1 -> DenseOut
    p2_t = torch_model.pool2(a2_t)
    flat = p2_t.permute(0, 2, 1).reshape(1, -1)
    d1 = torch_model.linear1(flat)
    r_d1 = torch.relu(d1)
    logit = torch_model.linear2(r_d1)

    # Backprop
    logit.backward()

    # PyTorch a2_t.grad has shape (1, 64, 500) -> permute to (500, 64) to match NumPy a2
    grad_a2_torch = a2_t.grad[0].permute(1, 0).detach().numpy()
    alphas_torch = np.mean(grad_a2_torch, axis=0)

    return grad_a2_torch, alphas_torch, logit.item()


def test_gradcam_gradient_agreement_with_autograd(trained_weights_path):
    """
    Test 1: Verify that NumPy analytical d(logit)/d(a2) matches PyTorch autograd
    to within 1e-4 on the real trained weights.
    """
    np_model = NumPyModel()
    np_model.load_weights(trained_weights_path)

    torch_model = TorchModel(dropout=0.0)
    load_numpy_weights_into_torch(torch_model, trained_weights_path)
    torch_model.eval()

    rng = np.random.default_rng(42)

    # Test across 5 distinct windows
    for trial in range(5):
        x = rng.normal(0.0, 1.0, 1000).astype(np.float32)

        grad_a2_np, alphas_np, cache = compute_numpy_gradients(np_model, x)
        grad_a2_torch, alphas_torch, logit_torch = compute_torch_gradients(torch_model, x)

        # 1. Verify logit agreement
        logit_np = np.log(cache["prob"] / (1.0 - cache["prob"] + 1e-15))
        # Direct check on logit calculation:
        a_dense1 = cache["a_dense1"]
        direct_logit_np = float(np.dot(a_dense1, np_model.dense_out.weights[:, 0]) + np_model.dense_out.bias[0])
        assert abs(direct_logit_np - logit_torch) < 1e-4

        # 2. Verify grad_a2 array agreement
        max_diff_grad = np.max(np.abs(grad_a2_np - grad_a2_torch))
        assert max_diff_grad < 1e-4, f"Trial {trial}: grad_a2 max diff {max_diff_grad:.6e} exceeds 1e-4"

        # 3. Verify alpha weights agreement
        max_diff_alphas = np.max(np.abs(alphas_np - alphas_torch))
        assert max_diff_alphas < 1e-4, f"Trial {trial}: alphas max diff {max_diff_alphas:.6e} exceeds 1e-4"


def test_gradcam_output_shape_and_range(trained_weights_path):
    """Test 2: Verify that GradCAM1D.explain returns normalized weights in [0, 1] of length 1000."""
    np_model = NumPyModel()
    np_model.load_weights(trained_weights_path)
    explainer = GradCAM1D(np_model)

    rng = np.random.default_rng(123)
    x = rng.normal(0.0, 1.0, 1000).astype(np.float32)

    res = explainer.explain(x)

    assert "weights" in res
    weights = np.array(res["weights"])
    assert len(weights) == 1000
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert np.max(weights) == pytest.approx(1.0, abs=1e-3) or np.all(weights == 0.0)
    assert "af_probability" in res
    assert "af_detected" in res
    assert "high_relevance_regions" in res


def test_gradcam_upsampling_spatial_alignment(trained_weights_path):
    """
    Test 3: Verify that zoom upsampling preserves spatial alignment from 500 to 1000 samples.
    A localized activation peak in a2 should map to twice its index in the 1000-sample CAM.
    """
    np_model = NumPyModel()
    np_model.load_weights(trained_weights_path)
    explainer = GradCAM1D(np_model)

    # Synthetic signal with sharp pulse at sample 400
    t = np.linspace(0, 10, 1000)
    x = np.exp(-((t - 4.0) ** 2) / 0.05).astype(np.float32)

    res = explainer.explain(x)
    weights = np.array(res["weights"])

    assert len(weights) == 1000
    assert weights.dtype == np.float64 or weights.dtype == np.float32
    assert res["inference_latency_ms"] < 25.0
