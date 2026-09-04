"""
1D Grad-CAM (Gradient-Weighted Class Activation Mapping) Explainer
==================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Specifications (ANTIGRAVITY.md §4.3 & RQ3):
- Computes analytical gradients of AF score w.r.t. final Conv1D layer activations A^k:
    alpha_k = (1 / L) * sum_i (d y^AF / d A_i^k)
    L_GradCAM^1D = ReLU(sum_k alpha_k * A^k)
- Upsamples activation vector back to the input window length (N=1000).
- Produces normalized relevance weights [0.0, 1.0] for clinician visualization overlay.
"""

import numpy as np
from typing import Dict, Any, List
from scipy.ndimage import zoom


class GradCAM1D:
    def __init__(self, model):
        self.model = model

    def explain(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Computes 1D Grad-CAM relevance vector for the input window.
        Returns: {
            'af_probability': float,
            'af_detected': int,
            'weights': List[float],  # Length N (e.g. 1000) normalized [0.0, 1.0]
            'high_relevance_regions': List[Tuple[int, int]] # Start/End indices
        }
        """
        # Run forward pass with cached activations
        cache = self.model.forward_with_cache(x)
        prob = cache["prob"]
        a2 = cache["a2"]             # Shape: (500, 64)
        p2 = cache["p2"]             # Shape: (250, 64)
        argmax2 = cache["argmax2"]   # Shape: (250, 64)
        a_dense1 = cache["a_dense1"] # Shape: (64,)

        # 1. Gradient of output logit w.r.t. a_dense1
        # dense_out: y = a_dense1 @ W_out + b_out
        grad_dense1 = self.model.dense_out.weights[:, 0]  # Shape: (64,)

        # 2. Backprop through ReLU of Dense1
        grad_dense1_relu = grad_dense1 * (a_dense1 > 0.0)

        # 3. Backprop through Dense1: (64,) @ W_dense1.T -> (16000,)
        grad_flat = np.dot(self.model.dense1.weights, grad_dense1_relu)  # Shape: (16000,)
        grad_p2 = grad_flat.reshape(p2.shape)  # Shape: (250, 64)

        # 4. Backprop through MaxPool (pool_size=2) to shape (500, 64)
        grad_a2 = np.zeros_like(a2)  # Shape: (500, 64)
        for i in range(p2.shape[0]):
            for ch in range(p2.shape[1]):
                offset = argmax2[i, ch]
                grad_a2[i * 2 + offset, ch] = grad_p2[i, ch]

        # 5. Global Average Pooling of gradients to find channel weights alpha_k
        # alpha_k = (1 / L) * sum_i (grad_a2[i, k])
        alphas = np.mean(grad_a2, axis=0)  # Shape: (64,)

        # 6. Weighted sum of feature maps: sum_k (alpha_k * A^k)
        cam = np.zeros(a2.shape[0], dtype=np.float32)  # Length: 500
        for k in range(len(alphas)):
            cam += alphas[k] * a2[:, k]

        # 7. Apply ReLU (only positive contributions to AF class)
        cam = np.maximum(0.0, cam)

        # 8. Upsample CAM from 500 back to original input length (e.g. 1000)
        target_len = self.model.input_length
        scale_factor = target_len / len(cam)
        cam_upsampled = zoom(cam, scale_factor, order=1)[:target_len]

        # 9. Normalize to [0.0, 1.0]
        max_val = np.max(cam_upsampled)
        if max_val > 1e-6:
            cam_norm = cam_upsampled / max_val
        else:
            # If uniform or non-active, default to small baseline
            cam_norm = np.zeros_like(cam_upsampled)

        # 10. Extract high relevance clusters (e.g. relevance > 0.6)
        high_threshold = 0.60
        in_region = False
        start_idx = 0
        regions = []
        for idx, val in enumerate(cam_norm):
            if val >= high_threshold and not in_region:
                in_region = True
                start_idx = idx
            elif val < high_threshold and in_region:
                in_region = False
                regions.append((start_idx, idx))
        if in_region:
            regions.append((start_idx, len(cam_norm) - 1))

        return {
            "af_probability": round(prob, 4),
            "af_detected": cache["af_detected"],
            "weights": [round(float(w), 3) for w in cam_norm],
            "high_relevance_regions": regions,
            "inference_latency_ms": cache["latency_ms"]
        }
