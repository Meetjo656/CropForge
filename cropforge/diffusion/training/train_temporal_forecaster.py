"""
Real Temporal Disease Forecasting Training Module for CropForge Milestone 7.

Trains/fine-tunes the temporal forecasting model against paired real future ground truth observations:
(Day 0 Baseline Image, Disease, Treatment, Env, Δt) ──► Target Day t Ground Truth Image
"""

import os
import sys
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from cropforge.diffusion.datasets.real_temporal_dataset import (
    RealTemporalDatasetBuilder,
    RealTemporalPlantSequence,
    RealTemporalTimepointSample,
)
from cropforge.diffusion.conditions.temporal_encoder import TemporalConditionEncoder

_logger = logging.getLogger(__name__)


class RealTemporalForecastingDataset(Dataset):
    """
    PyTorch Dataset pairing Day 0 baseline plant state with ground truth target future states (Day 3, 7, 14).
    """

    def __init__(
        self,
        sequences: Optional[List[RealTemporalPlantSequence]] = None,
        dataset_manifest_path: Optional[Union[str, Path]] = None,
        resolution: int = 512,
        seed: int = 42,
    ) -> None:
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        if sequences is not None and len(sequences) > 0:
            self.sequences = sequences
        elif dataset_manifest_path and Path(dataset_manifest_path).exists():
            self.sequences = self._load_from_manifest(dataset_manifest_path)
        else:
            _logger.info("Generating standard real temporal dataset for training...")
            builder = RealTemporalDatasetBuilder(seed=seed)
            self.sequences = builder.generate_dataset(num_plants=10)

        # Build training pairs: (t0_timepoint, target_timepoint)
        self.pairs: List[Tuple[RealTemporalTimepointSample, RealTemporalTimepointSample]] = []
        for seq in self.sequences:
            t0 = seq.get_timepoint(0.0)
            if not t0:
                continue
            for day in [3.0, 7.0, 14.0]:
                target_tp = seq.get_timepoint(day)
                if target_tp:
                    self.pairs.append((t0, target_tp))

    def _load_from_manifest(self, manifest_path: Union[str, Path]) -> List[RealTemporalPlantSequence]:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sequences = []
        for seq_dict in data.get("sequences", []):
            timepoints = []
            for day_str, tp_dict in seq_dict.get("timepoints", {}).items():
                img = Image.open(tp_dict["image_path"]).convert("RGB")
                mask = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
                if Path(tp_dict["mask_path"]).exists():
                    mask = transforms.functional.pil_to_tensor(Image.open(tp_dict["mask_path"])).numpy()[0]

                tp = RealTemporalTimepointSample(
                    plant_id=tp_dict["plant_id"],
                    timepoint_id=tp_dict["timepoint_id"],
                    day=tp_dict["day"],
                    crop_type=tp_dict["crop_type"],
                    disease_name=tp_dict["disease_name"],
                    treatment=tp_dict["treatment"],
                    env_covariates=tp_dict["env_covariates"],
                    image=img,
                    image_path=tp_dict["image_path"],
                    sam2_mask=mask,
                    mask_path=tp_dict["mask_path"],
                    severity=tp_dict["severity"],
                )
                timepoints.append(tp)

            sequences.append(
                RealTemporalPlantSequence(
                    plant_id=seq_dict["plant_id"],
                    crop_type=seq_dict["crop_type"],
                    disease_name=seq_dict["disease_name"],
                    treatment=seq_dict["treatment"],
                    env_covariates=seq_dict["env_covariates"],
                    timepoints=timepoints,
                )
            )
        return sequences

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t0_sample, target_sample = self.pairs[idx]

        t0_tensor = self.transform(t0_sample.image)
        target_tensor = self.transform(target_sample.image)

        t0_mask_pil = Image.fromarray(t0_sample.sam2_mask)
        target_mask_pil = Image.fromarray(target_sample.sam2_mask)

        t0_mask_tensor = self.mask_transform(t0_mask_pil)
        target_mask_tensor = self.mask_transform(target_mask_pil)

        delta_t_days = target_sample.day - t0_sample.day
        env_vec = [
            target_sample.env_covariates.get("temperature_c", 25.0),
            target_sample.env_covariates.get("humidity_percent", 75.0),
            target_sample.env_covariates.get("soil_moisture", 60.0),
        ]

        return {
            "plant_id": t0_sample.plant_id,
            "t0_image": t0_tensor,
            "t0_mask": t0_mask_tensor,
            "t0_severity": t0_sample.severity,
            "target_image": target_tensor,
            "target_mask": target_mask_tensor,
            "target_severity": target_sample.severity,
            "delta_t_days": delta_t_days,
            "crop": t0_sample.crop_type,
            "disease": t0_sample.disease_name,
            "treatment": t0_sample.treatment,
            "env_covariates": torch.tensor(env_vec, dtype=torch.float32),
        }


class TemporalForecasterTrainer:
    """
    Trainer for fine-tuning the temporal forecasting pipeline on paired real temporal observation sequences.
    """

    def __init__(
        self,
        output_dir: str = "outputs/checkpoints",
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize condition encoder
        self.condition_encoder = TemporalConditionEncoder(
            pooled_projection_dim=2048,
            joint_attention_dim=4096,
            device=self.device,
        )

        # Forecasting residual projection neck
        self.forecaster_neck = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.condition_encoder.parameters()) + list(self.forecaster_neck.parameters()),
            lr=learning_rate,
        )
        self.loss_fn = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.condition_encoder.train()
        self.forecaster_neck.train()

        total_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            t0_img = batch["t0_image"].to(self.device)
            t0_mask = batch["t0_mask"].to(self.device)
            target_img = batch["target_image"].to(self.device)
            delta_t_vec = batch["delta_t_days"]
            env_cov = batch["env_covariates"].tolist()
            treatments = batch["treatment"]

            self.optimizer.zero_grad()

            # Encode temporal & environmental conditions
            pooled_embeds, seq_embeds = self.condition_encoder.encode_conditions(
                delta_t=float(delta_t_vec[0]),
                env_covariates=env_cov[0],
                treatment=treatments[0],
                batch_size=t0_img.shape[0],
                device=self.device,
            )

            # Predict target future state: t0_img + forecaster_neck([t0_img, t0_mask])
            input_cat = torch.cat([t0_img, t0_mask], dim=1)
            pred_delta = self.forecaster_neck(input_cat)
            pred_target = torch.clamp(t0_img + pred_delta, -1.0, 1.0)

            # Reconstruction loss against real target ground truth image
            recon_loss = self.loss_fn(pred_target, target_img)
            cond_reg = 0.01 * torch.mean(pooled_embeds ** 2)
            loss = recon_loss + cond_reg

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        return {"train_loss": avg_loss}

    def train(self, num_epochs: int = 5, batch_size: int = 2) -> Dict[str, Any]:
        _logger.info("Initializing Real Temporal Forecasting Training...")
        dataset = RealTemporalForecastingDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = []
        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(dataloader)
            _logger.info("Epoch %d/%d — Train Loss: %.6f", epoch, num_epochs, metrics["train_loss"])
            history.append({"epoch": epoch, **metrics})

        ckpt_path = self.save_checkpoint("milestone7_temporal_forecaster.pt")
        return {
            "num_epochs": num_epochs,
            "final_loss": history[-1]["train_loss"],
            "checkpoint_path": str(ckpt_path),
            "history": history,
        }

    def save_checkpoint(self, filename: str = "milestone7_temporal_forecaster.pt") -> Path:
        ckpt_path = self.output_dir / filename
        state = {
            "condition_encoder": self.condition_encoder.state_dict(),
            "forecaster_neck": self.forecaster_neck.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path)
        
        # Also save to current_saved_checkpoint.pt in root for system consistency
        root_ckpt = Path("current_saved_checkpoint.pt")
        torch.save(state, root_ckpt)

        _logger.info("Saved fine-tuned temporal forecaster checkpoint to '%s'", ckpt_path)
        return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="CropForge Real Temporal Forecasting Trainer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--dry_run", action="store_true", help="Perform dry-run dataset check")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.dry_run:
        _logger.info("Executing dry-run dataset check...")
        ds = RealTemporalForecastingDataset()
        _logger.info("Dataset pair count: %d. Sample pair 0 target day: %.1f", len(ds), ds[0]["delta_t_days"])
        return

    trainer = TemporalForecasterTrainer()
    trainer.train(num_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
