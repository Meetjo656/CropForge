"""
Spatial Mask Forecaster for CropForge Milestone 10.

Spatially propagates Day 0 lesion masks (M_t0) into future lesion masks (M_t1)
and disease severity scores (S_t1) under temporal, environmental, and treatment conditioning.

Separates disease geometry evolution from visual image synthesis.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


class SpatialMaskForecaster(nn.Module):
    """
    Deep convolutional spatial mask propagation network.
    Inputs:
        t0_mask: Day 0 binary lesion mask tensor (B, 1, H, W)
        t0_image: Day 0 RGB leaf image tensor (B, 3, H, W)
        condition_vector: Encoded temporal condition vector (B, C) containing Δt, treatment, and env factors.
    Outputs:
        pred_future_mask_logits: Predicted future lesion mask logits (B, 1, H, W)
        pred_future_severity: Predicted future disease severity ratio tensor (B, 1)
    """

    def __init__(self, cond_dim: int = 128, in_channels: int = 4) -> None:
        super().__init__()
        # Encoder for t0 mask + t0 image
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, H/2, W/2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Condition projection neck
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Decoder for spatial future mask prediction
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, H, W)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        # Severity regression head
        self.severity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        t0_mask: torch.Tensor,
        t0_image: torch.Tensor,
        condition_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass predicting future lesion mask logits and severity score.
        """
        input_cat = torch.cat([t0_image, t0_mask], dim=1)  # (B, 4, H, W)
        spatial_feats = self.encoder(input_cat)             # (B, 128, H/2, W/2)

        # Project condition vector and expand spatially across (H/2, W/2)
        cond_embed = self.cond_proj(condition_vector)      # (B, 128)
        cond_spatial = cond_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spatial_feats.shape[2], spatial_feats.shape[3])

        # Fuse spatial features and temporal condition embedding
        fused_feats = torch.cat([spatial_feats, cond_spatial], dim=1)  # (B, 256, H/2, W/2)

        # Predict future mask logits and severity
        mask_logits = self.decoder(fused_feats)
        severity_pred = self.severity_head(fused_feats)

        return mask_logits, severity_pred

    def forecast_mask_numpy(
        self,
        t0_mask_np: np.ndarray,
        delta_t_days: float,
        temp_c: float,
        rh_percent: float,
        treatment: str,
    ) -> Tuple[np.ndarray, float]:
        """
        Predicts future mask numpy array and severity percentage using deterministic growth dynamics
        spatially expanding existing Day 0 lesions.
        """
        h, w = t0_mask_np.shape[:2]
        binary_t0 = (t0_mask_np > 127).astype(np.uint8)

        # Disease growth factor
        env_factor = (rh_percent / 100.0) * math.exp(-0.5 * ((temp_c - 24.0) / 6.0) ** 2)
        treatment_factors = {"untreated": 1.0, "fungicide": 0.15, "biocontrol": 0.45}
        t_factor = treatment_factors.get(treatment.lower(), 1.0)
        growth_rate = 0.08 * env_factor * t_factor

        # Spatial expansion iterations based on elapsed time and growth rate
        expansion_pixels = max(1, int(growth_rate * delta_t_days * 8))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pred_future_mask = cv2.dilate(binary_t0 * 255, kernel, iterations=expansion_pixels)

        # Compute severity percentage
        leaf_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(leaf_mask, (w // 2, h // 2), (w // 2 - 20, h // 2 - 20), 0, 0, 360, 255, -1)
        leaf_pixels = max(1, np.count_nonzero(leaf_mask))
        lesion_pixels = np.count_nonzero(pred_future_mask)
        severity = float(lesion_pixels / leaf_pixels)

        return pred_future_mask, severity
