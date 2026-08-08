"""
Unit tests for CropForge model_loader module.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cropforge.diffusion.models import model_loader
from cropforge.diffusion.models.model_loader import load_model, _MODEL_CACHE


class TestModelLoader(unittest.TestCase):

    def setUp(self):
        # Clear cache before each test
        _MODEL_CACHE.clear()

    def test_only_one_public_method(self):
        """Verify load_model is the only public export in __all__."""
        self.assertEqual(model_loader.__all__, ["load_model"])
        
        # Ensure all functions declared in model_loader are either 'load_model' or start with '_'
        for attr_name in dir(model_loader):
            if not attr_name.startswith("__") and not attr_name.startswith("_"):
                self.assertEqual(
                    attr_name,
                    "load_model",
                    f"Found unauthorized public symbol '{attr_name}' in model_loader module.",
                )

    def _create_mock_pipe(self):
        pipe = MagicMock()
        pipe.to.return_value = pipe
        return pipe

    @patch("cropforge.diffusion.models.model_loader._load_sd35_base_model")
    def test_load_model_caching(self, mock_load_base):
        mock_pipe = self._create_mock_pipe()
        mock_load_base.return_value = mock_pipe

        # First call loads the model
        pipe1 = load_model(model_id="test-sd3.5", device="cpu", torch_dtype=model_loader._torch.float32)
        self.assertEqual(mock_load_base.call_count, 1)

        # Second call with identical params should return cached instance
        pipe2 = load_model(model_id="test-sd3.5", device="cpu", torch_dtype=model_loader._torch.float32)
        self.assertEqual(mock_load_base.call_count, 1)  # Not called again
        self.assertIs(pipe1, pipe2)

    @patch("cropforge.diffusion.models.model_loader._load_sd35_base_model")
    def test_force_reload(self, mock_load_base):
        mock_pipe1 = self._create_mock_pipe()
        mock_pipe2 = self._create_mock_pipe()
        mock_load_base.side_effect = [mock_pipe1, mock_pipe2]

        pipe1 = load_model(model_id="test-sd3.5", device="cpu", torch_dtype=model_loader._torch.float32)
        pipe2 = load_model(
            model_id="test-sd3.5",
            device="cpu",
            torch_dtype=model_loader._torch.float32,
            force_reload=True,
        )

        self.assertEqual(mock_load_base.call_count, 2)
        self.assertIs(pipe1, mock_pipe1)
        self.assertIs(pipe2, mock_pipe2)

    @patch("cropforge.diffusion.models.model_loader._apply_lora")
    @patch("cropforge.diffusion.models.model_loader._load_sd35_base_model")
    def test_lora_loading_invoked(self, mock_load_base, mock_apply_lora):
        mock_pipe = self._create_mock_pipe()
        mock_load_base.return_value = mock_pipe
        mock_apply_lora.return_value = mock_pipe

        load_model(
            model_id="test-sd3.5",
            lora_path="/path/to/lora.safetensors",
            device="cpu",
            torch_dtype=model_loader._torch.float32,
        )

        mock_apply_lora.assert_called_once_with(pipe=mock_pipe, lora_path="/path/to/lora.safetensors")

    def test_internal_device_resolution(self):
        dev_cpu = model_loader._resolve_device("cpu")
        self.assertEqual(dev_cpu.type, "cpu")

        dev_explicit = model_loader._resolve_device(model_loader._torch.device("cpu"))
        self.assertEqual(dev_explicit.type, "cpu")

    def test_internal_dtype_resolution(self):
        dtype = model_loader._resolve_dtype(model_loader._torch.float16, model_loader._torch.device("cpu"))
        self.assertEqual(dtype, model_loader._torch.float16)



if __name__ == "__main__":
    unittest.main()
