"""
Unit tests for ConditionEncoder module.
"""

# pyrefly: ignore [missing-import]
import pytest
import numpy as np
from cropforge.diffusion.conditions import ConditionEncoder
from cropforge.diffusion.schemas import DatasetSample


@pytest.fixture
def sample_data() -> DatasetSample:
    return DatasetSample(
        sample_id="000001",
        crop="Tomato",
        disease="Late Blight",
        severity="Moderate",
        treatment="Copper Fungicide",
        days_after_treatment=14,
        temperature=30.0,
        humidity=50.0,
        input_image="day0.png",
        target_image="day14.png",
        segmentation_mask="mask.png",
    )


def test_encoder_categorical():
    encoder = ConditionEncoder()

    # Known categorical values
    vec_tomato = encoder.encode_categorical("crop", "Tomato")
    assert isinstance(vec_tomato, np.ndarray)
    assert vec_tomato.dtype == np.float32
    assert np.sum(vec_tomato) == 1.0

    # Case insensitive matching
    vec_tomato_lower = encoder.encode_categorical("crop", "tomato")
    assert np.array_equal(vec_tomato, vec_tomato_lower)

    # Unknown categorical value -> mapped to last unknown index
    vec_unknown = encoder.encode_categorical("crop", "Dragonfruit")
    assert vec_unknown[-1] == 1.0


def test_encoder_continuous():
    encoder = ConditionEncoder()

    # Temperature min=0, max=60 -> 30 should scale to 0.5
    temp_scaled = encoder.scale_continuous("temperature", 30.0)
    assert temp_scaled == pytest.approx(0.5)

    # Humidity min=0, max=100 -> 50 should scale to 0.5
    hum_scaled = encoder.scale_continuous("humidity", 50.0)
    assert hum_scaled == pytest.approx(0.5)

    # Out of bounds continuous values should be clipped
    over_temp = encoder.scale_continuous("temperature", 100.0)
    assert over_temp == 1.0

    under_temp = encoder.scale_continuous("temperature", -10.0)
    assert under_temp == 0.0


def test_encode_sample(sample_data: DatasetSample):
    encoder = ConditionEncoder()

    vec = encoder.encode_sample(sample_data)
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.ndim == 1
    assert len(vec) == encoder.get_vector_dimension()


def test_encode_sample_dict(sample_data: DatasetSample):
    encoder = ConditionEncoder()

    enc_dict = encoder.encode_sample_dict(sample_data)
    assert "crop" in enc_dict
    assert "disease" in enc_dict
    assert "severity" in enc_dict
    assert "treatment" in enc_dict
    assert "days_after_treatment" in enc_dict
    assert "temperature" in enc_dict
    assert "humidity" in enc_dict
