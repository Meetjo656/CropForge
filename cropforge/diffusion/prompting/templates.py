"""
Prompt Template Engine for Stable Diffusion 3.5 LoRA Dataset Generation.
"""

from typing import Dict, Any, Optional, List


class PromptTemplateEngine:
    """
    Template engine for constructing structured botanical and plant pathology prompts.
    """

    DEFAULT_SUBJECT_TEMPLATE = "{crop} leaf displaying {severity} stage {disease} symptoms"
    HEALTHY_SUBJECT_TEMPLATE = "Healthy {crop} leaf exhibiting vibrant natural foliage"
    TREATMENT_TEMPLATE = "treated with {treatment}, {days_after_treatment} days post-treatment"
    NO_TREATMENT_TEMPLATE = "untreated natural disease progression"
    ENVIRONMENT_TEMPLATE = "recorded at {temperature}°C ambient temperature and {humidity}% relative humidity"
    STYLE_TEMPLATE = "{camera_style}, {lighting} lighting"

    @classmethod
    def render_subject(cls, crop: str, disease: str, severity: str) -> str:
        """Render the main subject clause."""
        if disease.strip().lower() == "healthy" or severity.strip().lower() == "healthy":
            return cls.HEALTHY_SUBJECT_TEMPLATE.format(crop=crop)
        return cls.DEFAULT_SUBJECT_TEMPLATE.format(
            crop=crop, severity=severity.lower(), disease=disease
        )

    @classmethod
    def render_treatment(cls, treatment: str, days_after_treatment: int) -> str:
        """Render treatment information clause."""
        if not treatment or treatment.strip().lower() in ("none", "n/a", "untreated"):
            return cls.NO_TREATMENT_TEMPLATE
        return cls.TREATMENT_TEMPLATE.format(
            treatment=treatment, days_after_treatment=days_after_treatment
        )

    @classmethod
    def render_environment(cls, temperature: float, humidity: float) -> str:
        """Render environmental condition clause."""
        return cls.ENVIRONMENT_TEMPLATE.format(
            temperature=round(temperature, 1), humidity=round(humidity, 1)
        )

    @classmethod
    def render_style(cls, camera_style: str, lighting: str) -> str:
        """Render camera and lighting style clause."""
        return cls.STYLE_TEMPLATE.format(
            camera_style=camera_style, lighting=lighting
        )
