"""
Condition Encoder preprocessing module for CropForge Diffusion Dataset Generation.
Converts metadata attributes into normalized conditioning vectors.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
from cropforge.diffusion.configs import load_config
from cropforge.diffusion.schemas.sample_schema import DatasetSample


class ConditionEncoder:
    """
    Metadata condition encoder converting sample metadata into standardized numerical conditioning vectors.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ConditionEncoder with configuration."""
        self.config = config if config is not None else load_config()
        conditioning_cfg = self.config.get("conditioning", {})
        self.categorical_cfg: Dict[str, List[str]] = conditioning_cfg.get("categorical", {})
        self.continuous_cfg: Dict[str, Dict[str, float]] = conditioning_cfg.get("continuous", {})

    def encode_categorical(self, category_name: str, value: str) -> np.ndarray:
        """
        One-hot encode a categorical value based on configuration mappings.

        Args:
            category_name: Categorical feature key (e.g., 'crop', 'disease', 'treatment', 'severity').
            value: Categorical value string.

        Returns:
            Float32 1D numpy array representing the one-hot encoded vector.
        """
        allowed_list = self.categorical_cfg.get(category_name, [])
        vector_len = len(allowed_list) + 1  # Last index reserved for unknown/unmapped values
        one_hot = np.zeros(vector_len, dtype=np.float32)

        val_clean = value.strip().lower()
        found_idx = -1
        for idx, item in enumerate(allowed_list):
            if item.strip().lower() == val_clean:
                found_idx = idx
                break

        if found_idx >= 0:
            one_hot[found_idx] = 1.0
        else:
            one_hot[-1] = 1.0  # Unknown slot

        return one_hot

    def scale_continuous(self, feature_name: str, value: float) -> float:
        """
        Min-max scale a continuous numerical value to range [0.0, 1.0].

        Args:
            feature_name: Continuous feature key (e.g., 'days_after_treatment', 'temperature', 'humidity').
            value: Raw numerical value.

        Returns:
            Normalized float value in range [0.0, 1.0].
        """
        bounds = self.continuous_cfg.get(feature_name, {"min": 0.0, "max": 100.0})
        f_min = bounds.get("min", 0.0)
        f_max = bounds.get("max", 100.0)

        if f_max == f_min:
            return 0.0

        scaled = (float(value) - f_min) / (f_max - f_min)
        return float(np.clip(scaled, 0.0, 1.0))

    def encode_sample_dict(self, sample: Union[DatasetSample, dict]) -> Dict[str, np.ndarray]:
        """
        Encode sample into a dictionary of individual feature vectors.

        Returns:
            Dict mapping feature names to their encoded numpy arrays.
        """
        data = sample.to_dict() if isinstance(sample, DatasetSample) else sample

        encoded_dict: Dict[str, np.ndarray] = {
            "crop": self.encode_categorical("crop", str(data.get("crop", ""))),
            "disease": self.encode_categorical("disease", str(data.get("disease", ""))),
            "severity": self.encode_categorical("severity", str(data.get("severity", ""))),
            "treatment": self.encode_categorical("treatment", str(data.get("treatment", ""))),
            "days_after_treatment": np.array(
                [self.scale_continuous("days_after_treatment", float(data.get("days_after_treatment", 0)))],
                dtype=np.float32,
            ),
            "temperature": np.array(
                [self.scale_continuous("temperature", float(data.get("temperature", 0.0)))],
                dtype=np.float32,
            ),
            "humidity": np.array(
                [self.scale_continuous("humidity", float(data.get("humidity", 0.0)))],
                dtype=np.float32,
            ),
        }
        return encoded_dict

    def encode_sample(self, sample: Union[DatasetSample, dict]) -> np.ndarray:
        """
        Encode all sample features into a single unified 1D numerical conditioning vector.

        Returns:
            1D float32 numpy array concatenating categorical one-hot vectors and continuous scalars.
        """
        encoded_dict = self.encode_sample_dict(sample)
        vectors_to_concat = [
            encoded_dict["crop"],
            encoded_dict["disease"],
            encoded_dict["severity"],
            encoded_dict["treatment"],
            encoded_dict["days_after_treatment"],
            encoded_dict["temperature"],
            encoded_dict["humidity"],
        ]
        return np.concatenate(vectors_to_concat, axis=0).astype(np.float32)

    def get_vector_dimension(self) -> int:
        """Get total length/dimension of the encoded condition vector."""
        dummy_sample = {
            "crop": "",
            "disease": "",
            "severity": "",
            "treatment": "",
            "days_after_treatment": 0,
            "temperature": 0.0,
            "humidity": 0.0,
        }
        return len(self.encode_sample(dummy_sample))
