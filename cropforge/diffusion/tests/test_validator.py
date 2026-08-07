"""
Unit tests for DatasetValidator.
"""
# pyrefly: ignore [missing-import]
import pytest
from cropforge.diffusion.datasets import DatasetValidator, ValidationError
from cropforge.diffusion.schemas import DatasetSample


@pytest.fixture
def valid_sample() -> DatasetSample:
    return DatasetSample(
        sample_id="000001",
        crop="Tomato",
        disease="Late Blight",
        severity="Moderate",
        treatment="Copper Fungicide",
        days_after_treatment=14,
        temperature=28.0,
        humidity=75.0,
        input_image="day0.png",
        target_image="day14.png",
        segmentation_mask="mask.png",
    )


def test_validator_valid_sample(valid_sample: DatasetSample):
    validator = DatasetValidator()
    result = validator.validate(valid_sample)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validator_temperature_out_of_range(valid_sample: DatasetSample):
    validator = DatasetValidator()

    # Temperature too high (>60)
    valid_sample.temperature = 75.0
    result = validator.validate(valid_sample)
    assert result.is_valid is False
    assert any("Temperature" in err for err in result.errors)

    # Temperature negative (<0)
    valid_sample.temperature = -5.0
    result_neg = validator.validate(valid_sample)
    assert result_neg.is_valid is False


def test_validator_humidity_out_of_range(valid_sample: DatasetSample):
    validator = DatasetValidator()

    valid_sample.humidity = 120.0
    result = validator.validate(valid_sample)
    assert result.is_valid is False
    assert any("Humidity" in err for err in result.errors)


def test_validator_days_negative(valid_sample: DatasetSample):
    validator = DatasetValidator()

    valid_sample.days_after_treatment = -1
    result = validator.validate(valid_sample)
    assert result.is_valid is False
    assert any("Days after treatment" in err for err in result.errors)


def test_validator_invalid_severity(valid_sample: DatasetSample):
    validator = DatasetValidator()

    valid_sample.severity = "Extreme"
    result = validator.validate(valid_sample)
    assert result.is_valid is False
    assert any("Severity" in err for err in result.errors)


def test_validator_invalid_image_extension(valid_sample: DatasetSample):
    validator = DatasetValidator()

    valid_sample.target_image = "invalid_file.txt"
    result = validator.validate(valid_sample)
    assert result.is_valid is False
    assert any("file extension" in err for err in result.errors)


def test_validate_or_raise(valid_sample: DatasetSample):
    validator = DatasetValidator()

    # Should not raise exception for valid sample
    validator.validate_or_raise(valid_sample)

    # Should raise ValidationError for invalid sample
    valid_sample.temperature = 100.0
    with pytest.raises(ValidationError):
        validator.validate_or_raise(valid_sample)
