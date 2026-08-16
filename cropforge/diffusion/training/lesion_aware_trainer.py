"""
Lesion-Aware Multi-Task Loss Training Module for CropForge Milestone 8.

Trains the forecasting architecture using composite multi-signal supervision:
L_total = L_image + λ_mask * L_mask + λ_severity * L_severity + λ_condition * L_condition

Explicitly supervises:
1. Target Future RGB Image
2. Target SAM2 Binary Lesion Segmentation Mask (BCE + Dice Loss)
3. Target Disease Severity Ratio (L1 Loss)
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cropforge.diffusion.datasets.real_temporal_dataset import RealTemporalDatasetBuilder
from cropforge.diffusion.training.train_temporal_forecaster import RealTemporalForecastingDataset
from cropforge.diffusion.conditions.temporal_encoder import TemporalConditionEncoder

_logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Dice Loss for binary lesion segmentation mask supervision.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class LesionAwareForecasterNeck(nn.Module):
    """
    Multi-head forecasting neck outputting:
    1. RGB Image Residual Head (3 channels)
    2. Lesion Mask Head (1 channel logit)
    3. Severity Prediction Head (1 scalar output)
    """

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        # Shared feature trunk
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # RGB Image Head
        self.rgb_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

        # SAM2 Lesion Mask Head
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        # Severity Ratio Head
        self.severity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared_conv(x)
        rgb_delta = self.rgb_head(features)
        mask_logits = self.mask_head(features)
        severity_pred = self.severity_head(features).squeeze(-1)
        return rgb_delta, mask_logits, severity_pred


class LesionAwareForecasterTrainer:
    """
    Trainer optimizing composite Multi-Task Loss:
    L_total = L_image + λ_mask * (L_bce + L_dice) + λ_severity * L_sev + λ_condition * L_cond
    """

    def __init__(
        self,
        output_dir: str = "outputs/checkpoints",
        learning_rate: float = 1e-4,
        lambda_mask: float = 1.0,
        lambda_severity: float = 0.5,
        lambda_condition: float = 0.01,
        device: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.lambda_mask = lambda_mask
        self.lambda_severity = lambda_severity
        self.lambda_condition = lambda_condition

        # Condition Encoder
        self.condition_encoder = TemporalConditionEncoder(
            pooled_projection_dim=2048,
            joint_attention_dim=4096,
            device=self.device,
        )

        # Lesion-Aware Multi-Head Neck
        self.neck = LesionAwareForecasterNeck(in_channels=4).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.condition_encoder.parameters()) + list(self.neck.parameters()),
            lr=learning_rate,
        )

        # Loss Functions
        self.image_loss_fn = nn.MSELoss()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = DiceLoss()
        self.sev_loss_fn = nn.L1Loss()

    def compute_composite_loss(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        pred_mask_logits: torch.Tensor,
        target_mask: torch.Tensor,
        pred_severity: torch.Tensor,
        target_severity: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes weighted composite multi-task loss.
        """
        l_image = self.image_loss_fn(pred_rgb, target_rgb)

        l_bce = self.bce_loss_fn(pred_mask_logits, target_mask)
        l_dice = self.dice_loss_fn(pred_mask_logits, target_mask)
        l_mask = l_bce + l_dice

        l_severity = self.sev_loss_fn(pred_severity, target_severity)
        l_condition = 0.01 * torch.mean(pooled_embeds ** 2)

        l_total = (
            l_image
            + self.lambda_mask * l_mask
            + self.lambda_severity * l_severity
            + self.lambda_condition * l_condition
        )

        return {
            "loss_total": l_total,
            "loss_image": l_image,
            "loss_mask": l_mask,
            "loss_severity": l_severity,
            "loss_condition": l_condition,
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.condition_encoder.train()
        self.neck.train()

        total_loss = 0.0
        total_img_loss = 0.0
        total_mask_loss = 0.0
        total_sev_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            t0_img = batch["t0_image"].to(self.device)
            t0_mask = batch["t0_mask"].to(self.device)
            target_img = batch["target_image"].to(self.device)
            target_mask = batch["target_mask"].to(self.device)
            target_sev = batch["target_severity"].to(self.device, dtype=torch.float32)

            delta_t_vec = batch["delta_t_days"]
            env_cov = batch["env_covariates"].tolist()
            treatments = batch["treatment"]

            self.optimizer.zero_grad()

            pooled_embeds, seq_embeds = self.condition_encoder.encode_conditions(
                delta_t=float(delta_t_vec[0]),
                env_covariates=env_cov[0],
                treatment=treatments[0],
                batch_size=t0_img.shape[0],
                device=self.device,
            )

            input_cat = torch.cat([t0_img, t0_mask], dim=1)
            rgb_delta, mask_logits, pred_sev = self.neck(input_cat)
            pred_rgb = torch.clamp(t0_img + rgb_delta, -1.0, 1.0)

            losses = self.compute_composite_loss(
                pred_rgb=pred_rgb,
                target_rgb=target_img,
                pred_mask_logits=mask_logits,
                target_mask=target_mask,
                pred_severity=pred_sev,
                target_severity=target_sev,
                pooled_embeds=pooled_embeds,
            )

            loss = losses["loss_total"]
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_img_loss += losses["loss_image"].item()
            total_mask_loss += losses["loss_mask"].item()
            total_sev_loss += losses["loss_severity"].item()
            total_batches += 1

        nb = max(1, total_batches)
        return {
            "train_loss": total_loss / nb,
            "image_loss": total_img_loss / nb,
            "mask_loss": total_mask_loss / nb,
            "severity_loss": total_sev_loss / nb,
        }

    def train(self, num_epochs: int = 5, batch_size: int = 2) -> Dict[str, Any]:
        _logger.info("Initializing Lesion-Aware Multi-Task Forecasting Training...")
        dataset = RealTemporalForecastingDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = []
        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(dataloader)
            _logger.info(
                "Epoch %d/%d — Total Loss: %.4f | Image MSE: %.4f | Mask (BCE+Dice): %.4f | Sev L1: %.4f",
                epoch, num_epochs, metrics["train_loss"], metrics["image_loss"], metrics["mask_loss"], metrics["severity_loss"]
            )
            history.append({"epoch": epoch, **metrics})

        ckpt_path = self.save_checkpoint("milestone8_lesion_aware_forecaster.pt")
        return {
            "num_epochs": num_epochs,
            "final_loss": history[-1]["train_loss"],
            "checkpoint_path": str(ckpt_path),
            "history": history,
        }

    def save_checkpoint(self, filename: str = "milestone8_lesion_aware_forecaster.pt") -> Path:
        ckpt_path = self.output_dir / filename
        state = {
            "condition_encoder": self.condition_encoder.state_dict(),
            "neck": self.neck.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path)

        root_ckpt = Path("current_saved_checkpoint.pt")
        torch.save(state, root_ckpt)

        _logger.info("Saved fine-tuned lesion-aware forecaster checkpoint to '%s'", ckpt_path)
        return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="CropForge Lesion-Aware Multi-Loss Trainer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--dry_run", action="store_true", help="Perform dry-run multi-task loss check")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.dry_run:
        _logger.info("Executing dry-run multi-task loss check...")
        trainer = LesionAwareForecasterTrainer(device="cpu")
        dataset = RealTemporalForecastingDataset()
        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        metrics = trainer.train_epoch([batch])
        _logger.info("Dry-run multi-task loss check complete! Loss metrics: %s", metrics)
        return

    trainer = LesionAwareForecasterTrainer()
    trainer.train(num_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
