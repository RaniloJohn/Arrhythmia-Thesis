"""
1D-CNN Inference Model & Forward Engine (ANTIGRAVITY.md §4.3)
=============================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Specifications:
- Input: 1D normalized PPG window (N=1000 samples @ 100 Hz = 10s)
- Conv1D(filters=32, kernel_size=5, padding='same') -> ReLU -> MaxPool1D(pool_size=2)
- Conv1D(filters=64, kernel_size=3, padding='same') -> ReLU -> MaxPool1D(pool_size=2)
- Flatten -> Dense(64, activation='relu') -> Dense(1, activation='sigmoid')
- Edge latency budget: < 25 ms per inference window on Raspberry Pi 4 (ARM Cortex-A72)
"""

import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


class Conv1DBlock:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # He / Kaiming normal initialization
        std = np.sqrt(2.0 / (in_channels * kernel_size))
        self.weights = rng.normal(0.0, std, (out_channels, in_channels, kernel_size)).astype(np.float32)
        self.bias = np.zeros((out_channels,), dtype=np.float32)
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (length, in_channels)
        returns: (length, out_channels)
        """
        length, in_ch = x.shape
        out_ch, _, k_size = self.weights.shape

        # Zero padding 'same'
        x_padded = np.pad(x, ((self.pad, self.pad), (0, 0)), mode='constant')

        # Fast 1D convolution via sliding dot products
        out = np.zeros((length, out_ch), dtype=np.float32)
        for oc in range(out_ch):
            # Sum over in_channels
            conv_accum = np.zeros(length, dtype=np.float32)
            for ic in range(in_ch):
                conv_accum += np.convolve(x_padded[:, ic], self.weights[oc, ic, ::-1], mode='valid')[:length]
            out[:, oc] = conv_accum + self.bias[oc]

        return out


class MaxPool1DBlock:
    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        x shape: (length, channels)
        returns: (length // pool_size, channels), argmax_indices
        """
        length, ch = x.shape
        out_len = length // self.pool_size
        trimmed = x[:out_len * self.pool_size, :].reshape(out_len, self.pool_size, ch)
        out = np.max(trimmed, axis=1)
        argmax = np.argmax(trimmed, axis=1)
        return out, argmax


class DenseBlock:
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / in_features)
        self.weights = rng.normal(0.0, std, (in_features, out_features)).astype(np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weights) + self.bias


class Arrhythmia1DCNN:
    """
    Lightweight 1D-CNN conforming to Chapter 2 Fig 2.1 & ANTIGRAVITY.md §4.3.
    """
    EXPECTED_WEIGHT_SHAPES = {
        "conv1.weights": (32, 1, 5),
        "conv1.bias": (32,),
        "conv2.weights": (64, 32, 3),
        "conv2.bias": (64,),
        "dense1.weights": (16000, 64),
        "dense1.bias": (64,),
        "dense_out.weights": (64, 1),
        "dense_out.bias": (1,),
    }

    def __init__(self, input_length: int = 1000):
        self.input_length = input_length

        # Conv Block 1: 1 -> 32 (k=5)
        self.conv1 = Conv1DBlock(in_channels=1, out_channels=32, kernel_size=5, seed=101)
        self.pool1 = MaxPool1DBlock(pool_size=2)

        # Conv Block 2: 32 -> 64 (k=3)
        self.conv2 = Conv1DBlock(in_channels=32, out_channels=64, kernel_size=3, seed=202)
        self.pool2 = MaxPool1DBlock(pool_size=2)

        # Flatten dimension: (1000 // 2) // 2 = 250 * 64 = 16000
        flatten_dim = (input_length // 4) * 64
        self.dense1 = DenseBlock(in_features=flatten_dim, out_features=64, seed=303)
        self.dense_out = DenseBlock(in_features=64, out_features=1, seed=404)

        # Operational status and classification threshold
        self.threshold: float = 0.50
        self.weights_loaded: bool = False
        self.weights_source: Optional[str] = None
        self.model_version: str = "untrained_he_init"
        self.training_dataset: str = "none"
        self.model_meta: Dict[str, Any] = {}

    def load_weights(self, path: Union[str, Path]) -> None:
        """
        Load trained weights from a .npz archive, asserting exact array shapes.

        Also loads the calibrated decision threshold from the sibling .meta.json
        file if present.

        Args:
            path: Path to the .npz weights archive.

        Raises:
            FileNotFoundError: If the weights file does not exist.
            ValueError: If required keys are missing or shapes do not match.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path.resolve()}")

        with np.load(path) as data:
            for key, expected_shape in self.EXPECTED_WEIGHT_SHAPES.items():
                if key not in data:
                    raise ValueError(f"Missing required weight key '{key}' in {path}")
                arr = data[key]
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for '{key}': expected {expected_shape}, got {arr.shape} in {path}"
                    )

            self.conv1.weights = np.asarray(data["conv1.weights"], dtype=np.float32)
            self.conv1.bias = np.asarray(data["conv1.bias"], dtype=np.float32)
            self.conv2.weights = np.asarray(data["conv2.weights"], dtype=np.float32)
            self.conv2.bias = np.asarray(data["conv2.bias"], dtype=np.float32)
            self.dense1.weights = np.asarray(data["dense1.weights"], dtype=np.float32)
            self.dense1.bias = np.asarray(data["dense1.bias"], dtype=np.float32)
            self.dense_out.weights = np.asarray(data["dense_out.weights"], dtype=np.float32)
            self.dense_out.bias = np.asarray(data["dense_out.bias"], dtype=np.float32)

        # Attempt to read threshold and metadata from sibling .meta.json
        meta_path = (
            path.with_name(path.stem + ".meta.json")
            if path.name.endswith(".npz")
            else path.with_suffix(".meta.json")
        )
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.model_meta = meta
                    if "decision_threshold" in meta:
                        self.threshold = float(meta["decision_threshold"])
                    elif "threshold" in meta:
                        self.threshold = float(meta["threshold"])
                    if "training_dataset" in meta and isinstance(meta["training_dataset"], dict):
                        self.training_dataset = meta["training_dataset"].get("name", "deepbeat")
                    elif "training_dataset" in meta:
                        self.training_dataset = str(meta["training_dataset"])
            except Exception:
                self.threshold = 0.50
        else:
            self.threshold = 0.50

        self.weights_loaded = True
        self.weights_source = str(path.resolve())
        self.model_version = path.stem

    def forward_with_cache(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Forward pass retaining activations for 1D Grad-CAM backpropagation.
        x: (input_length,) or (input_length, 1)
        """
        if x.ndim == 1:
            x_in = x.reshape(-1, 1).astype(np.float32)
        else:
            x_in = x.astype(np.float32)

        t_start = time.perf_counter_ns()

        # Conv 1
        z1 = self.conv1.forward(x_in)
        a1 = relu(z1)
        p1, argmax1 = self.pool1.forward(a1)

        # Conv 2 (Target layer for Grad-CAM)
        z2 = self.conv2.forward(p1)
        a2 = relu(z2)
        p2, argmax2 = self.pool2.forward(a2)

        # Dense 1
        flat = p2.flatten()
        z_dense1 = self.dense1.forward(flat)
        a_dense1 = relu(z_dense1)

        # Output
        z_out = self.dense_out.forward(a_dense1)
        prob = float(sigmoid(z_out)[0])

        t_end = time.perf_counter_ns()
        latency_ms = (t_end - t_start) / 1_000_000.0

        return {
            "af_probability": prob,
            "af_detected": int(prob >= self.threshold),
            "latency_ms": round(latency_ms, 2),
            # Cache for Grad-CAM
            "x_in": x_in,
            "a2": a2,             # (length=500, channels=64)
            "p2": p2,             # (length=250, channels=64)
            "argmax2": argmax2,
            "flat": flat,
            "a_dense1": a_dense1,
            "prob": prob
        }

    def predict(self, x: np.ndarray) -> Tuple[float, int, float]:
        """
        Fast prediction without caching.
        Returns: (af_probability, af_detected, latency_ms)
        """
        res = self.forward_with_cache(x)
        return res["af_probability"], res["af_detected"], res["latency_ms"]
