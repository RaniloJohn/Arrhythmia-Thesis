"""
Pure-NumPy inference runtime v2 — edge deployment target for the amended topology.

Keeps settled decision 4 intact: no torch, no TFLite, no ONNX on the Raspberry Pi. The
PyTorch model in `training/torch_model_v2.py` is the training mirror; this is the runtime,
and `tests/test_parity_v2.py` holds them to agreement.

Supports the three temporal-aggregation heads compared in the architecture study, with
analytic Grad-CAM for each:

* ``flatten``  - Chapter 2's original ``Flatten -> Dense``, time-major ``(t * C + c)``.
* ``gap``      - global average pooling over time.
* ``statpool`` - concatenated per-channel mean and standard deviation over time.

Grad-CAM gradients are derived in closed form rather than estimated. For the statistical
pooling head, with ``A`` the post-pool2 activation map of shape ``(C, T)``,
``m_c = mean_t A[c, t]`` and ``s_c = sqrt(mean_t (A[c, t] - m_c)^2)``:

    dm_c / dA[c, t] = 1 / T
    ds_c / dA[c, t] = (A[c, t] - m_c) / (T * s_c)

the second following from ``d(s^2)/dA[c, t] = (2 / T) * (A[c, t] - m_c)`` because the
mean-deviation terms sum to zero. Both are verified against ``torch.autograd``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray, padding: int) -> np.ndarray:
    """
    x: (C_in, T)  w: (C_out, C_in, K)  b: (C_out,) -> (C_out, T_out)

    Matches `torch.nn.Conv1d` with stride 1 and the given zero padding, via an explicit
    sliding-window view so the result is exact rather than approximate.
    """
    c_in, t = x.shape
    c_out, _, k = w.shape
    xp = np.pad(x, ((0, 0), (padding, padding))) if padding else x
    t_out = xp.shape[1] - k + 1
    # (C_in, T_out, K) sliding windows, then contract over (C_in, K)
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=1)  # (C_in, T_out, K)
    out = np.tensordot(w, win, axes=([1, 2], [0, 2]))              # (C_out, T_out)
    return out + b[:, None]


def maxpool1d(x: np.ndarray, pool: int) -> Tuple[np.ndarray, np.ndarray]:
    """Non-overlapping max pooling over the last axis. Returns (pooled, argmax indices)."""
    c, t = x.shape
    t_out = t // pool
    trimmed = x[:, : t_out * pool].reshape(c, t_out, pool)
    arg = np.argmax(trimmed, axis=2)
    return np.max(trimmed, axis=2), arg


class ArrhythmiaCNNv2:
    """Pure-NumPy forward pass and Grad-CAM for the amended 1D-CNN."""

    def __init__(
        self,
        input_length: int = 1000,
        kernels: Tuple[int, int] = (127, 63),
        pools: Tuple[int, int] = (8, 8),
        channels: Tuple[int, int] = (32, 64),
        head: str = "statpool",
        dense_units: int = 64,
    ) -> None:
        if head not in ("flatten", "gap", "statpool"):
            raise ValueError(f"unknown head {head!r}")
        self.input_length = input_length
        self.kernels = tuple(kernels)
        self.pools = tuple(pools)
        self.channels = tuple(channels)
        self.head = head
        self.dense_units = dense_units
        self.threshold = 0.50
        self.weights_loaded = False
        self.model_version = "untrained"
        self.W: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ persistence

    @classmethod
    def from_meta(cls, weights_path: Union[str, Path]) -> "ArrhythmiaCNNv2":
        """Construct with the architecture recorded alongside the weights, then load them."""
        weights_path = Path(weights_path)
        meta_path = weights_path.with_suffix(".meta.json")
        arch: Dict[str, Any] = {}
        if meta_path.exists():
            arch = json.loads(meta_path.read_text(encoding="utf-8")).get("architecture", {}) or {}
        model = cls(
            input_length=int(arch.get("input_length", 1000)),
            kernels=tuple(arch.get("kernels", (127, 63))),
            pools=tuple(arch.get("pools", (8, 8))),
            channels=tuple(arch.get("channels", (32, 64))),
            head=str(arch.get("head", "statpool")),
            dense_units=int(arch.get("dense_units", 64)),
        )
        model.load_weights(weights_path)
        return model

    def load_weights(self, path: Union[str, Path]) -> None:
        path = Path(path)
        z = np.load(path)
        want = {
            "conv1.weight": (self.channels[0], 1, self.kernels[0]),
            "conv1.bias": (self.channels[0],),
            "conv2.weight": (self.channels[1], self.channels[0], self.kernels[1]),
            "conv2.bias": (self.channels[1],),
            "linear2.weight": (1, self.dense_units),
            "linear2.bias": (1,),
        }
        t_out = self.input_length // self.pools[0] // self.pools[1]
        feat = {"flatten": t_out * self.channels[1], "gap": self.channels[1],
                "statpool": 2 * self.channels[1]}[self.head]
        want["linear1.weight"] = (self.dense_units, feat)
        want["linear1.bias"] = (self.dense_units,)

        for key, shape in want.items():
            if key not in z:
                raise KeyError(f"{path.name} is missing '{key}'")
            if tuple(z[key].shape) != shape:
                raise ValueError(
                    f"{path.name}: '{key}' has shape {tuple(z[key].shape)}, expected {shape}. "
                    "The weights do not match this architecture configuration."
                )
            self.W[key] = np.asarray(z[key], dtype=np.float64)

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.threshold = float(meta.get("decision_threshold", meta.get("threshold", 0.50)))
            self.model_version = str(meta.get("model_version", path.stem))
        else:
            self.model_version = path.stem
        self.weights_loaded = True

    # ---------------------------------------------------------------------- forward

    def forward_with_cache(self, x: np.ndarray) -> Dict[str, Any]:
        if not self.weights_loaded:
            raise RuntimeError("load_weights() must be called before inference")
        sig = np.asarray(x, dtype=np.float64).ravel()
        if sig.size != self.input_length:
            raise ValueError(f"expected {self.input_length} samples, got {sig.size}")

        a1 = relu(conv1d(sig[None, :], self.W["conv1.weight"], self.W["conv1.bias"],
                         self.kernels[0] // 2))
        p1, _ = maxpool1d(a1, self.pools[0])
        a2 = relu(conv1d(p1, self.W["conv2.weight"], self.W["conv2.bias"],
                         self.kernels[1] // 2))
        p2, _ = maxpool1d(a2, self.pools[1])          # (C, T) — Grad-CAM target

        c, t = p2.shape
        mean = p2.mean(axis=1)
        std = np.sqrt(np.mean((p2 - mean[:, None]) ** 2, axis=1))
        if self.head == "flatten":
            z = p2.T.reshape(-1)                      # time-major (t * C + c)
        elif self.head == "gap":
            z = mean
        else:
            z = np.concatenate([mean, std])

        pre = self.W["linear1.weight"] @ z + self.W["linear1.bias"]
        h = relu(pre)
        logit = float((self.W["linear2.weight"] @ h + self.W["linear2.bias"]).ravel()[0])
        prob = float(sigmoid(np.asarray(logit)))
        return {"a2": p2, "mean": mean, "std": std, "pre": pre, "h": h,
                "logit": logit, "probability": prob,
                "af_detected": int(prob >= self.threshold)}

    def predict(self, x: np.ndarray) -> Tuple[float, int, float]:
        c = self.forward_with_cache(x)
        return c["probability"], c["af_detected"], c["logit"]

    # --------------------------------------------------------------------- grad-cam

    def dlogit_da2(self, cache: Dict[str, Any]) -> np.ndarray:
        """Closed-form d(logit)/d(A2), shape (C, T)."""
        a2 = cache["a2"]
        c, t = a2.shape
        # back through linear2 -> relu -> linear1 to the pooled vector z
        g_h = self.W["linear2.weight"].ravel()                      # (dense_units,)
        g_pre = g_h * (cache["pre"] > 0)
        g_z = self.W["linear1.weight"].T @ g_pre                    # (feat,)

        if self.head == "flatten":
            return g_z.reshape(t, c).T
        if self.head == "gap":
            return np.repeat((g_z / t)[:, None], t, axis=1)

        g_mean, g_std = g_z[:c], g_z[c:]
        std = np.maximum(cache["std"], 1e-12)
        centred = a2 - cache["mean"][:, None]
        return g_mean[:, None] / t + g_std[:, None] * centred / (t * std[:, None])

    def grad_cam(self, x: np.ndarray, cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        1-D Grad-CAM heat strip, upsampled to the input length and normalised to [0, 1].

        alpha_c is the time-average of d(logit)/dA2[c, :] (standard Grad-CAM channel
        weighting); the map is relu(sum_c alpha_c * A2[c, :]).
        """
        cache = cache or self.forward_with_cache(x)
        a2 = cache["a2"]
        grad = self.dlogit_da2(cache)
        alpha = grad.mean(axis=1)
        cam = relu(np.tensordot(alpha, a2, axes=(0, 0)))
        if cam.max() > 0:
            cam = cam / cam.max()
        up = np.interp(
            np.linspace(0, len(cam) - 1, self.input_length),
            np.arange(len(cam)), cam,
        ) if len(cam) > 1 else np.full(self.input_length, float(cam[0]))
        return {"cam": cam, "heatmap": up, "alpha": alpha,
                "probability": cache["probability"], "af_detected": cache["af_detected"]}
