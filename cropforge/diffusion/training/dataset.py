"""
Extensible Dataset Loader and Conditioning Data Abstractions for CropForge LoRA Fine-Tuning.

Maintains strict compatibility with Milestone 2/3 dataset schemas while providing clean interfaces
for future image, mask, and multi-modal condition injection (Milestone 5).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from cropforge.diffusion.datasets.builder import DatasetBuilder
from cropforge.diffusion.prompting.prompt_builder import PromptBuilder
from cropforge.diffusion.conditions.encoder import ConditionEncoder
from cropforge.diffusion.schemas.sample_schema import DatasetSample


@dataclass
class TrainingCondition:
    """
    Extensible container for training conditioning variables.
    
    In Milestone 4: Encapsulates generated text prompts and metadata vectors.
    In Milestone 5: Will be extended with input images, SAM2 segmentation masks,
    and temporal condition embeddings.
    """
    prompt: str
    negative_prompt: str = ""
    input_image: Optional[Union[str, Path, Image.Image, torch.Tensor]] = None
    target_image: Optional[Union[str, Path, Image.Image, torch.Tensor]] = None
    segmentation_mask: Optional[Union[str, Path, Image.Image, torch.Tensor]] = None
    condition_vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConditioningEncoderInterface:
    """
    Extensible conditioning encoder interface for transforming samples to TrainingCondition objects.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.prompt_builder = PromptBuilder(config)
        self.condition_encoder = ConditionEncoder(config)

    def encode(self, sample: DatasetSample) -> TrainingCondition:
        pos_prompt, neg_prompt = self.prompt_builder.build_prompt_pair(sample)
        cond_vec = self.condition_encoder.encode_sample(sample)
        return TrainingCondition(
            prompt=pos_prompt,
            negative_prompt=neg_prompt,
            input_image=sample.input_image,
            target_image=sample.target_image,
            segmentation_mask=sample.segmentation_mask,
            condition_vector=cond_vec,
            metadata={
                "sample_id": sample.sample_id,
                "crop": sample.crop,
                "disease": sample.disease,
                "severity": sample.severity,
                "treatment": sample.treatment,
                "days_after_treatment": sample.days_after_treatment,
                "temperature": sample.temperature,
                "humidity": sample.humidity,
            },
        )


class CropForgeDiffusionDataset(Dataset):
    """
    PyTorch Dataset for CropForge SD3.5 LoRA training.
    
    Loads DatasetSample instances, builds text prompts, loads image tensors,
    and constructs TrainingCondition objects.
    """

    def __init__(
        self,
        samples: Optional[List[Union[DatasetSample, Dict[str, Any]]]] = None,
        num_synthetic_samples: int = 100,
        resolution: int = 1024,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize CropForgeDiffusionDataset.

        Args:
            samples: Pre-existing list of DatasetSample instances or dictionaries.
            num_synthetic_samples: If samples is None, generates procedurally generated samples.
            resolution: Target spatial resolution (e.g. 1024).
            config: Optional configuration dictionary.
            seed: Seed for sample generation.
        """
        self.resolution = resolution
        self.builder = DatasetBuilder(config=config)
        self.condition_encoder = ConditioningEncoderInterface(config=config)

        if samples is not None and len(samples) > 0:
            self.samples = [
                s if isinstance(s, DatasetSample) else DatasetSample(**s)
                for s in samples
            ]
        else:
            self.samples = self.builder.generate_sample_batch(count=num_synthetic_samples, seed=seed)

        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path_or_str: Optional[Union[str, Path]]) -> Image.Image:
        """Helper to load image from disk or generate synthetic color image fallback for tests/dry-run."""
        if image_path_or_str:
            p = Path(image_path_or_str)
            if p.exists() and p.is_file():
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    pass
        # Fallback synthetic image for dry-runs / testing when image files don't exist on disk
        rng_val = abs(hash(str(image_path_or_str))) % 255 if image_path_or_str else 128
        arr = np.full((self.resolution, self.resolution, 3), rng_val, dtype=np.uint8)
        return Image.fromarray(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        cond = self.condition_encoder.encode(sample)

        pil_img = self.load_target_image(sample)
        image_tensor = self.transform(pil_img)

        return {
            "pixel_values": image_tensor,
            "prompt": cond.prompt,
            "negative_prompt": cond.negative_prompt,
            "condition": cond,
            "sample_id": sample.sample_id,
        }

    def load_target_image(self, sample: DatasetSample) -> Image.Image:
        """Load target image for sample."""
        return self._load_image(sample.target_image)

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function handling TrainingCondition objects in batch."""
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        prompts = [item["prompt"] for item in batch]
        neg_prompts = [item["negative_prompt"] for item in batch]
        conditions = [item["condition"] for item in batch]
        sample_ids = [item["sample_id"] for item in batch]

        return {
            "pixel_values": pixel_values,
            "prompt": prompts,
            "negative_prompt": neg_prompts,
            "condition": conditions,
            "sample_id": sample_ids,
        }
