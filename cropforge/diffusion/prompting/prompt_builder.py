"""
Configurable PromptBuilder for Stable Diffusion 3.5 LoRA dataset generation.
"""

from typing import Dict, Any, Optional, Tuple, List
from cropforge.diffusion.configs import load_config
from cropforge.diffusion.prompting.templates import PromptTemplateEngine
from cropforge.diffusion.schemas.sample_schema import DatasetSample


class PromptBuilder:
    """
    Constructs production-quality positive and negative prompts for diffusion model fine-tuning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PromptBuilder with configuration dictionary."""
        self.config = config if config is not None else load_config()
        self.prompting_cfg = self.config.get("prompting", {})
        self.default_camera = self.prompting_cfg.get(
            "default_camera_style", "macro botanical DSLR"
        )
        self.default_lighting = self.prompting_cfg.get(
            "default_lighting", "natural outdoor"
        )
        self.style_modifiers = self.prompting_cfg.get("style_modifiers", {})
        self.negative_keywords = self.prompting_cfg.get("negative_prompt", [])

    def build_prompt(
        self,
        crop: str,
        disease: str,
        severity: str,
        treatment: str,
        days_after_treatment: int,
        temperature: float,
        humidity: float,
        lighting: Optional[str] = None,
        camera_style: Optional[str] = None,
        include_styles: Optional[List[str]] = None,
    ) -> str:
        """
        Build a production-quality Stable Diffusion positive prompt.

        Args:
            crop: Target crop name.
            disease: Disease or condition name.
            severity: Disease severity level.
            treatment: Applied treatment.
            days_after_treatment: Days since treatment.
            temperature: Ambient temperature in Celsius.
            humidity: Ambient relative humidity percentage.
            lighting: Custom lighting style (defaults to config default).
            camera_style: Custom camera style (defaults to config default).
            include_styles: Specific style modifiers to append (e.g. ['botanical', 'macro']).

        Returns:
            A formatted positive prompt string.
        """
        effective_lighting = lighting or self.default_lighting
        effective_camera = camera_style or self.default_camera

        subject_clause = PromptTemplateEngine.render_subject(crop, disease, severity)
        treatment_clause = PromptTemplateEngine.render_treatment(treatment, days_after_treatment)
        environment_clause = PromptTemplateEngine.render_environment(temperature, humidity)
        style_clause = PromptTemplateEngine.render_style(effective_camera, effective_lighting)

        # Collect configured style modifiers (e.g., botanical, macro, pathology, scientific_realism, dslr)
        modifier_phrases: List[str] = []
        if include_styles:
            for style_key in include_styles:
                if style_key in self.style_modifiers:
                    modifier_phrases.append(self.style_modifiers[style_key])
        else:
            # Default: include standard quality modifiers from config
            for key, phrase in self.style_modifiers.items():
                modifier_phrases.append(phrase)

        prompt_parts = [
            subject_clause,
            treatment_clause,
            environment_clause,
            style_clause,
        ] + modifier_phrases

        return ", ".join(part for part in prompt_parts if part)

    def build_from_sample(self, sample: DatasetSample) -> str:
        """Build positive prompt directly from a DatasetSample instance."""
        return self.build_prompt(
            crop=sample.crop,
            disease=sample.disease,
            severity=sample.severity,
            treatment=sample.treatment,
            days_after_treatment=sample.days_after_treatment,
            temperature=sample.temperature,
            humidity=sample.humidity,
            lighting=sample.lighting,
            camera_style=sample.camera_style,
        )

    def get_negative_prompt(self) -> str:
        """Return the formatted negative prompt string."""
        return ", ".join(self.negative_keywords)

    def build_prompt_pair(self, sample: DatasetSample) -> Tuple[str, str]:
        """
        Build both positive and negative prompt strings for a given sample.

        Returns:
            Tuple of (positive_prompt, negative_prompt).
        """
        positive = self.build_from_sample(sample)
        negative = self.get_negative_prompt()
        return positive, negative
