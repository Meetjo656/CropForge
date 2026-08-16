"""
Spatial Mask Forecaster Trainer for CropForge Milestone 10.

Trains the two-stage spatial forecasting architecture using Experiment E loss weighting:
λ_mask = 2.0, λ_severity = 1.0, λ_condition = 0.01

Optimizes:
L_total = L_image + 2.0 * (L_bce + L_dice) + 1.0 * L_severity + 0.01 * L_condition
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cropforge.diffusion.datasets.real_temporal_dataset import RealTemporalDatasetBuilder
from cropforge.diffusion.training.train_temporal_forecaster import RealTemporalForecastingDataset
from cropforge.diffusion.models.spatial_mask_forecaster import SpatialMaskForecaster
from cropforge.diffusion.training.lesion_aware_trainer import DiceLoss, LesionAwareForecasterTrainer

_logger = logging.getLogger(__name__)


class SpatialMaskForecastingTrainer(LesionAwareForecasterTrainer):
    """
    Trainer for Milestone 10 Mask-Conditioned Spatial Forecaster.
    Uses selected Experiment E loss weights: λ_mask = 2.0, λ_severity = 1.0.
    """

    def __init__(
        self,
        output_dir: str = "outputs/checkpoints",
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            learning_rate=learning_rate,
            lambda_mask=2.0,       # Experiment E selected weight
            lambda_severity=1.0,   # Experiment E selected weight
            lambda_condition=0.01,
            device=device,
        )
        self.spatial_forecaster = SpatialMaskForecaster().to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.condition_encoder.train()
        self.neck.train()
        self.spatial_forecaster.train()

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

            # Stage 1: Spatial Mask Forecaster prediction
            mask_logits, pred_sev = self.spatial_forecaster(
                t0_mask=t0_mask,
                t0_image=t0_img,
                condition_vector=pooled_embeds[:, :128],
            )

            # Stage 2: Visual reconstruction
            input_cat = torch.cat([t0_img, t0_mask], dim=1)
            rgb_delta, _, _ = self.neck(input_cat)
            pred_rgb = torch.clamp(t0_img + rgb_delta, -1.0, 1.0)

            losses = self.compute_composite_loss(
                pred_rgb=pred_rgb,
                target_rgb=target_img,
                pred_mask_logits=mask_logits,
                target_mask=target_mask,
                pred_severity=pred_sev.squeeze(-1),
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

    def train_milestone10(self, num_epochs: int = 5, batch_size: int = 2) -> Dict[str, Any]:
        _logger.info("Starting Milestone 10 Mask-Conditioned Spatial Forecasting Training (Exp E Weights: λ_mask=2.0, λ_sev=1.0)...")
        dataset = RealTemporalForecastingDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = []
        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(dataloader)
            _logger.info(
                "M10 Epoch %d/%d — Total Loss: %.4f | Image MSE: %.4f | Spatial Mask Loss: %.4f | Sev L1: %.4f",
                epoch, num_epochs, metrics["train_loss"], metrics["image_loss"], metrics["mask_loss"], metrics["severity_loss"]
            )
            history.append({"epoch": epoch, **metrics})

        ckpt_path = self.save_checkpoint("milestone10_spatial_mask_forecaster.pt")
        return {
            "milestone": "Milestone 10 — Mask-Conditioned Spatial Forecasting",
            "experiment_config": "Experiment E (λ_mask=2.0, λ_severity=1.0)",
            "num_epochs": num_epochs,
            "final_loss": history[-1]["train_loss"],
            "checkpoint_path": str(ckpt_path),
            "history": history,
        }


def main():
    parser = argparse.ArgumentParser(description="CropForge Milestone 10 Spatial Mask Forecaster Trainer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--dry_run", action="store_true", help="Perform dry-run spatial mask loss check")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.dry_run:
        _logger.info("Executing dry-run spatial mask forecaster check...")
        trainer = SpatialMaskForecastingTrainer(device="cpu")
        dataset = RealTemporalForecastingDataset()
        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        metrics = trainer.train_epoch([batch])
        _logger.info("Dry-run spatial mask check complete! Metrics: %s", metrics)
        return

    trainer = SpatialMaskForecastingTrainer()
    trainer.train_milestone10(num_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
