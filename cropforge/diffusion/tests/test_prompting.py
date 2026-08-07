"""
Unit tests for PromptBuilder and PromptTemplateEngine.
"""
# pyrefly: ignore [missing-import]
import pytest
from cropforge.diffusion.prompting import PromptBuilder, PromptTemplateEngine
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
        temperature=28.0,
        humidity=75.0,
        input_image="day0.png",
        target_image="day14.png",
        segmentation_mask="mask.png",
    )


def test_template_rendering():
    subject = PromptTemplateEngine.render_subject("Tomato", "Late Blight", "Moderate")
    assert "Tomato" in subject
    assert "Late Blight" in subject
    assert "moderate" in subject

    healthy_subject = PromptTemplateEngine.render_subject("Tomato", "Healthy", "Healthy")
    assert "Healthy Tomato" in healthy_subject

    treatment = PromptTemplateEngine.render_treatment("Copper Fungicide", 14)
    assert "Copper Fungicide" in treatment
    assert "14 days" in treatment

    environment = PromptTemplateEngine.render_environment(28.0, 75.0)
    assert "28" in environment
    assert "75" in environment


def test_prompt_builder_positive_and_negative(sample_data: DatasetSample):
    builder = PromptBuilder()

    positive, negative = builder.build_prompt_pair(sample_data)

    # Assert positive prompt contains all required elements
    assert "Tomato" in positive
    assert "Late Blight" in positive
    assert "Copper Fungicide" in positive
    assert "28" in positive
    assert "75" in positive

    # Assert positive prompt includes photography style modifiers from config
    assert "botanical photography" in positive or "macro photography" in positive
    assert "DSLR" in positive or "macro" in positive

    # Assert negative prompt contains configured negative keywords
    assert "illustration" in negative
    assert "painting" in negative
    assert "anime" in negative
    assert "cartoon" in negative
    assert "CGI" in negative
    assert "blurry" in negative
    assert "watermark" in negative
    assert "text" in negative
    assert "logo" in negative
