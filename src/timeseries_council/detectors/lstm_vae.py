# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
LSTM-VAE based anomaly detector.
"""

from typing import Optional, Callable
import pandas as pd
import numpy as np

from .base import BaseDetector
from ..types import DetectionResult, Anomaly, AnomalyType, DetectionMemory
from ..logging import get_logger
from ..utils import get_device

logger = get_logger(__name__)


class LSTMVAEDetector(BaseDetector):
    """LSTM Variational Autoencoder based anomaly detector.

    Uses reconstruction error from LSTM-VAE to detect anomalies.
    Higher reconstruction error indicates potential anomaly.
    """

    def __init__(
        self,
        window_size: int = 10,
        latent_dim: int = 8,
        epochs: int = 50,
        device: str = None
    ):
        """
        Initialize LSTM-VAE detector.

        Args:
            window_size: Size of sliding window for sequences
            latent_dim: Dimension of latent space
            epochs: Number of training epochs
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
        """
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.device = get_device(device)
        self._model = None
        logger.info(f"Initialized LSTMVAEDetector (window={window_size}, latent={latent_dim}) on {self.device}")

    @property
    def name(self) -> str:
        return "LSTM-VAE"

    @property
    def description(self) -> str:
        return "LSTM Variational Autoencoder detector using reconstruction error"

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM input."""
        sequences = []
        for i in range(len(data) - self.window_size + 1):
            sequences.append(data[i:i + self.window_size])
        return np.array(sequences)

    def _build_model(self, input_dim: int):
        """Build LSTM-VAE model."""
        import torch
        import torch.nn as nn

        class LSTMVAE(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim, window_size):
                super().__init__()
                self.window_size = window_size
                self.hidden_dim = hidden_dim

                # Encoder
                self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.fc_mu = nn.Linear(hidden_dim, latent_dim)
                self.fc_var = nn.Linear(hidden_dim, latent_dim)

                # Decoder
                self.fc_decode = nn.Linear(latent_dim, hidden_dim)
                self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.output_layer = nn.Linear(hidden_dim, input_dim)

            def encode(self, x):
                _, (h, _) = self.encoder_lstm(x)
                h = h.squeeze(0)
                mu = self.fc_mu(h)
                log_var = self.fc_var(h)
                return mu, log_var

            def reparameterize(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z, seq_len):
                h = self.fc_decode(z)
                h = h.unsqueeze(1).repeat(1, seq_len, 1)
                out, _ = self.decoder_lstm(h)
                return self.output_layer(out)

            def forward(self, x):
                mu, log_var = self.encode(x)
                z = self.reparameterize(mu, log_var)
                recon = self.decode(z, x.size(1))
                return recon, mu, log_var

        return LSTMVAE(input_dim, 32, self.latent_dim, self.window_size)

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """Detect anomalies using LSTM-VAE reconstruction error."""
        error = self.validate_input(series)
        if error:
            logger.error(f"Validation failed: {error}")
            return DetectionResult(success=False, error=error)

        self._report_progress(progress_callback, "Loading PyTorch...", 0.1)

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            return DetectionResult(
                success=False,
                error="PyTorch not installed. Run: pip install torch"
            )

        try:
            self._report_progress(progress_callback, "Preparing data...", 0.15)

            # Normalize data — use baseline stats when available so the
            # reconstruction error reflects deviation from the known-normal
            # distribution, not just the current batch.
            values = series.values.reshape(-1, 1)
            use_baseline = (
                memory is not None
                and memory.baseline_stats.get("mean") is not None
                and memory.baseline_stats.get("std") is not None
                and memory.baseline_stats["std"] > 0
            )
            if use_baseline:
                mean = memory.baseline_stats["mean"]
                std = memory.baseline_stats["std"]
                logger.info(
                    f"Using baseline stats for LSTM-VAE normalization: "
                    f"mean={mean}, std={std}"
                )
            else:
                mean = values.mean()
                std = values.std()
                if std == 0:
                    std = 1
            normalized = (values - mean) / std

            # Create sequences
            sequences = self._create_sequences(normalized)
            if len(sequences) < 10:
                return DetectionResult(
                    success=False,
                    error=f"Need at least {self.window_size + 9} data points"
                )

            self._report_progress(progress_callback, "Building model...", 0.2)

            # Convert to tensors
            X = torch.FloatTensor(sequences)
            dataset = TensorDataset(X)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Build model
            model = self._build_model(1)
            device = torch.device(self.device)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            self._report_progress(progress_callback, "Training LSTM-VAE...", 0.3)

            # Training
            model.train()
            for epoch in range(self.epochs):
                for batch in loader:
                    x = batch[0].to(device)
                    optimizer.zero_grad()
                    recon, mu, log_var = model(x)

                    # Loss: reconstruction + KL divergence
                    recon_loss = nn.MSELoss()(recon, x)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = recon_loss + 0.001 * kl_loss

                    loss.backward()
                    optimizer.step()

                if epoch % 10 == 0:
                    progress = 0.3 + (0.4 * epoch / self.epochs)
                    self._report_progress(progress_callback, f"Training epoch {epoch}/{self.epochs}...", progress)

            self._report_progress(progress_callback, "Computing reconstruction errors...", 0.75)

            # Compute reconstruction errors
            model.eval()
            with torch.no_grad():
                X = X.to(device)
                recon, _, _ = model(X)
                errors = ((X - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()

            self._report_progress(progress_callback, "Finding anomalies...", 0.9)

            # Threshold based on sensitivity
            error_mean = errors.mean()
            error_std = errors.std()
            threshold = error_mean + (sensitivity * error_std)

            # Map errors back to original indices
            valid_indices = series.index[self.window_size - 1:]
            anomalies = []
            mean_val = float(series.mean())

            for i, err in enumerate(errors):
                if err > threshold:
                    idx = valid_indices[i]
                    val = float(series[idx])
                    anomaly_type = AnomalyType.SPIKE if val > mean_val else AnomalyType.DROP

                    anomalies.append(Anomaly(
                        timestamp=str(idx),
                        value=val,
                        score=float(err),
                        anomaly_type=anomaly_type
                    ))

            # Apply memory context
            anomalies = self._apply_memory(anomalies, memory)

            self._report_progress(progress_callback, "Detection complete", 1.0)

            logger.info(f"LSTM-VAE found {len(anomalies)} anomalies")

            return DetectionResult(
                success=True,
                anomaly_count=len(anomalies),
                anomalies=anomalies,
                sensitivity=sensitivity,
                mean=mean_val,
                std=float(series.std()),
                model=self.name,
                metadata={
                    "window_size": self.window_size,
                    "latent_dim": self.latent_dim,
                    "epochs": self.epochs,
                    "threshold": float(threshold),
                    "baseline_used": use_baseline,
                    "memory_applied": memory is not None,
                }
            )

        except Exception as e:
            logger.error(f"LSTM-VAE detection failed: {e}")
            return DetectionResult(success=False, error=str(e))
