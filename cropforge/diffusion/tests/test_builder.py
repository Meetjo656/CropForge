"""
Unit tests for DatasetBuilder and MetadataManager.
"""

import json
from pathlib import Path
# pyrefly: ignore [missing-import]
import pytest
from cropforge.diffusion.datasets import DatasetBuilder, MetadataManager, ValidationError
from cropforge.diffusion.schemas import DatasetSample


@pytest.fixture
def sample_dict() -> dict:
    return {
        "sample_id": "test_sample_001",
        "crop": "Tomato",
        "disease": "Late Blight",
        "severity": "Moderate",
        "treatment": "Copper Fungicide",
        "days_after_treatment": 14,
        "temperature": 28.0,
        "humidity": 75.0,
        "input_image": "day0.png",
        "target_image": "day14.png",
        "segmentation_mask": "mask.png",
    }


def test_builder_end_to_end(sample_dict: dict, tmp_path: Path):
    builder = DatasetBuilder(base_output_dir=tmp_path)

    result = builder.build_sample(sample_dict, save_files=True, strict_validation=True)

    # Assert build result structure
    assert result.sample_id == "test_sample_001"
    assert result.validation_result.is_valid is True
    assert "Tomato" in result.prompt
    assert "Late Blight" in result.prompt
    assert len(result.condition_vector) > 0

    # Assert files saved to disk
    assert result.metadata_path is not None
    assert result.metadata_path.exists()
    assert result.prompt_path is not None
    assert result.prompt_path.exists()

    # Read back metadata JSON
    loaded_sample = MetadataManager.load_metadata(result.metadata_path)
    assert loaded_sample.sample_id == "test_sample_001"
    assert loaded_sample.crop == "Tomato"

    # Read back prompt text
    with open(result.prompt_path, "r", encoding="utf-8") as f:
        prompt_content = f.read()
    assert "POSITIVE:" in prompt_content
    assert "NEGATIVE:" in prompt_content
    assert "Tomato" in prompt_content


def test_builder_strict_validation_failure(sample_dict: dict, tmp_path: Path):
    builder = DatasetBuilder(base_output_dir=tmp_path)

    # Make sample invalid (temperature = 150)
    invalid_dict = sample_dict.copy()
    invalid_dict["temperature"] = 150.0

    with pytest.raises(ValidationError):
        builder.build_sample(invalid_dict, save_files=False, strict_validation=True)


def test_builder_non_strict_validation(sample_dict: dict, tmp_path: Path):
    builder = DatasetBuilder(base_output_dir=tmp_path)

    invalid_dict = sample_dict.copy()
    invalid_dict["temperature"] = 150.0

    # Should not raise exception when strict_validation=False
    result = builder.build_sample(invalid_dict, save_files=False, strict_validation=False)
    assert result.validation_result.is_valid is False
    assert len(result.validation_result.errors) > 0
