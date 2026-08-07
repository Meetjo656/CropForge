"""
Config package for CropForge Diffusion Dataset Generation.
"""

from pathlib import Path
import yaml
from typing import Any, Dict

CONFIG_FILE_PATH = Path(__file__).parent / "dataset_config.yaml"

def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration dictionary from a YAML file."""
    path = Path(config_path) if config_path else CONFIG_FILE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
