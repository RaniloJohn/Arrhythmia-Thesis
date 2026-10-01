"""
Configurable 1D-CNN for the AF classifier, used to justify an amendment to the Chapter 2
topology with measurements rather than assertion.

Why an amendment is needed. The Chapter 2 topology is
`Conv1D(1->32,k=5) -> ReLU -> MaxPool(2) -> Conv1D(32->64,k=3) -> ReLU -> MaxPool(2) ->
Flatten -> Dense(64) -> Sigmoid`. Two structural problems follow from that specification:

1. **Receptive field is 12 samples (0.12 s).** A typical inter-beat interval is 0.6-1.2 s
   (60-120 samples), so no convolutional feature can observe a single interval, and none
   can observe the *variability* of intervals that defines atrial fibrillation.
2. **`Flatten -> Dense(16000, 64)` is 1,024,000 parameters and position-specific.** Against
   a 35-subject cohort it memorises per-recording morphology at fixed time offsets. This is
   the mechanism behind the v1 model scoring below chance on its own training data.

The classes here keep Chapter 2's two convolutional blocks and its `Dense(64) -> Sigmoid`
classifier, and vary only (a) kernel size / pooling, which sets the receptive field, and
(b) the temporal aggregation between the convolutions and the dense layer:

* `flatten`  — Chapter 2 exactly. Retained as the control.
* `gap`      — global average pooling over time. Translation invariant, and the setting
               Grad-CAM was originally formulated for, so it also strengthens RQ3.
* `statpool` — concatenated mean **and standard deviation** over time. The standard
               deviation of a channel across the window *is* a variability measure, which
               is the quantity AF detection needs.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


class ConfigurableAF1DCNN(nn.Module):
    """Chapter 2's two conv blocks with a configurable receptive field and pooling head."""

    def __init__(
        self,
        input_length: int = 1000,
        kernels: Sequence[int] = (5, 3),
        pools: Sequence[int] = (2, 2),
        channels: Sequence[int] = (32, 64),
        head: str = "statpool",
        dense_units: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if head not in ("flatten", "gap", "statpool"):
            raise ValueError(f"unknown head {head!r}")
        self.head = head
        self.input_length = input_length
        self.kernels, self.pools, self.channels = tuple(kernels), tuple(pools), tuple(channels)

        k1, k2 = kernels
        p1, p2 = pools
        c1, c2 = channels
        self.conv1 = nn.Conv1d(1, c1, kernel_size=k1, padding=k1 // 2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(p1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(p2)

        t_out = input_length // p1 // p2
        feat = {"flatten": t_out * c2, "gap": c2, "statpool": 2 * c2}[head]
        self.feature_dim = feat
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(feat, dense_units)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(dense_units, 1)

    # --------------------------------------------------------------- introspection

    @property
    def receptive_field(self) -> int:
        """Samples of input seen by one post-pool2 activation."""
        rf, stride = 1, 1
        for k, p in zip(self.kernels, self.pools):
            rf += (k - 1) * stride
            rf += (p - 1) * stride
            stride *= p
        return rf

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --------------------------------------------------------------------- forward

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Post-pool2 activation map A2, shape (batch, channels, time). Grad-CAM target."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        return self.pool2(self.relu2(self.conv2(x)))

    def head_forward(self, a2: torch.Tensor) -> torch.Tensor:
        if self.head == "flatten":
            # time-major flatten, matching the pure-NumPy runtime's (t * C + c) order
            z = a2.permute(0, 2, 1).reshape(a2.shape[0], -1)
        elif self.head == "gap":
            z = a2.mean(dim=2)
        else:  # statpool
            z = torch.cat([a2.mean(dim=2), a2.std(dim=2, unbiased=False)], dim=1)
        z = self.relu3(self.linear1(self.dropout(z)))
        return self.linear2(z)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        logits = self.head_forward(self.features(x))
        return logits if return_logits else torch.sigmoid(logits)


# Named variants compared in the architecture study. Receptive fields are printed by
# `sweep_arch.py`; `wide_*` widens kernels and pooling so a feature spans a full beat.
VARIANTS = {
    "ch2_flatten":    dict(kernels=(5, 3),   pools=(2, 2), head="flatten"),
    "ch2_gap":        dict(kernels=(5, 3),   pools=(2, 2), head="gap"),
    "ch2_statpool":   dict(kernels=(5, 3),   pools=(2, 2), head="statpool"),
    "wide_gap":       dict(kernels=(31, 15), pools=(4, 4), head="gap"),
    "wide_statpool":  dict(kernels=(31, 15), pools=(4, 4), head="statpool"),
    "xwide_statpool": dict(kernels=(63, 31), pools=(4, 4), head="statpool"),
    # The sweep showed receptive field, not capacity, drives subject AUROC, and the trend
    # had not saturated at 1.98 s. These extend it toward the span needed to observe the
    # variability of several consecutive intervals (>= 3-4 beats).
    "xxwide_statpool": dict(kernels=(127, 63), pools=(4, 4), head="statpool"),
    "deep_statpool":   dict(kernels=(31, 15),  pools=(8, 8), head="statpool"),
    "huge_statpool":   dict(kernels=(127, 63), pools=(8, 8), head="statpool"),
    "huge_gap":        dict(kernels=(127, 63), pools=(8, 8), head="gap"),
}


def build(variant: str, input_length: int = 1000, dropout: float = 0.5) -> ConfigurableAF1DCNN:
    if variant not in VARIANTS:
        raise KeyError(f"{variant!r} not in {sorted(VARIANTS)}")
    return ConfigurableAF1DCNN(input_length=input_length, dropout=dropout, **VARIANTS[variant])
