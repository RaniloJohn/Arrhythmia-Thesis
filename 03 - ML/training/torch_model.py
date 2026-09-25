"""
PyTorch 1D-CNN Model Architecture (ANTIGRAVITY.md §4.3 & Chapter 2 Fig 2.1)
========================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Specifications:
- Input: 1D normalized PPG window (N=1000 samples @ 100 Hz = 10.0 s)
- Conv1D(in=1, out=32, kernel_size=5, padding=2) -> ReLU -> MaxPool1D(pool_size=2)
- Conv1D(in=32, out=64, kernel_size=3, padding=1) -> ReLU -> MaxPool1D(pool_size=2)
- Permute(0, 2, 1) -> Flatten (time-major indexing matching inference_model.py p2.flatten())
- Dropout(p=0.5) -> Linear(16000, 64) -> ReLU -> Linear(64, 1) -> Sigmoid
- Runtime scope: Dev-machine training only (Pi edge runtime uses pure-NumPy inference_model.py).
"""

from typing import Union
import torch
import torch.nn as nn


class Arrhythmia1DCNN(nn.Module):
    """
    PyTorch mirror of inference_model.Arrhythmia1DCNN.

    Matches Chapter 2 Fig 2.1 topology and ANTIGRAVITY.md §4.3 specifications.
    Dropout is training-only and becomes identity under model.eval(), preserving
    bit-level parity with the deployed pure-NumPy runtime.
    """

    def __init__(self, input_length: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.input_length = input_length
        self.dropout_rate = dropout

        # Conv Block 1: 1 -> 32 (k=5, padding=2 -> length 1000)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # length 1000 -> 500

        # Conv Block 2: 32 -> 64 (k=3, padding=1 -> length 500)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # length 500 -> 250

        # Flatten dimension: (input_length // 4) * 64 = 250 * 64 = 16000
        self.flatten_dim = (input_length // 4) * 64

        # Dropout before Linear(16000, 64) — active during training only
        self.dropout = nn.Dropout(p=dropout)

        # Dense Block 1: 16000 -> 64
        self.linear1 = nn.Linear(in_features=self.flatten_dim, out_features=64)
        self.relu3 = nn.ReLU()

        # Dense Block Out: 64 -> 1
        self.linear2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass through 1D-CNN.

        Args:
            x: Input tensor of shape (batch, 1000), (batch, 1, 1000), or (1000,)
            return_logits: If True, returns raw unactivated logits (for BCEWithLogitsLoss).
                           If False, returns sigmoid activation probabilities.
        Returns:
            torch.Tensor of shape (batch, 1) or (1, 1)
        """
        # Standardize input dimensions
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(1)  # (1, 1, length)
        elif x.ndim == 2:
            x = x.unsqueeze(1)               # (batch, 1, length)
        elif x.ndim != 3:
            raise ValueError(f"Expected 1D, 2D, or 3D input tensor, got shape {x.shape}")

        if x.shape[1] != 1 and x.shape[2] == 1:
            # Handle transposed input (batch, length, 1) -> (batch, 1, length)
            x = x.permute(0, 2, 1)

        # Conv Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # Shape: (batch, 32, 500)

        # Conv Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # Shape: (batch, 64, 250) [batch, channels, time]

        # CRITICAL — flatten ordering:
        # inference_model.py computes `p2.flatten()` where p2 has shape (time=250, channels=64),
        # giving C-order time-major indexing `t * 64 + c`.
        # PyTorch's tensor is (batch, channels=64, time=250), so a plain flatten gives
        # channel-major `c * 250 + t`.
        # We permute to (batch, time, channels) before flattening, i.e.:
        #   x = x.permute(0, 2, 1).reshape(batch, -1)
        # so the torch flatten order exactly matches inference_model.py's p2.flatten() row-major order.
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1).reshape(batch_size, -1)  # Shape: (batch, 16000)

        # Dense Block 1 with Dropout
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu3(x)

        # Dense Out
        logits = self.linear2(x)  # Shape: (batch, 1)

        if return_logits:
            return logits
        return torch.sigmoid(logits)
