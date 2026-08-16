"""
Unit tests for Milestone 12: Temporal Horizon Modeling & Recursive vs Direct Extrapolation.
"""

import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cropforge.diffusion.analysis.temporal_horizon_forecaster import RecursiveSpatialForecaster
from cropforge.diffusion.datasets.real_temporal_dataset import RealTemporalDatasetBuilder


class TestMilestone12HorizonModeling(unittest.TestCase):
    """
    Test suite verifying RecursiveSpatialForecaster multi-step rollouts and Approach A/B/C execution.
    """

    def test_recursive_forecaster_approaches(self):
        forecaster = RecursiveSpatialForecaster()
        t0_mask = np.zeros((64, 64), dtype=np.uint8)
        t0_mask[20:40, 20:40] = 255

        res_a = forecaster.forecast_approach_a_direct(t0_mask, target_horizon=14.0)
        res_b = forecaster.forecast_approach_b_twostep(t0_mask)
        res_c = forecaster.forecast_approach_c_multistep(t0_mask)

        self.assertIn("final_mask", res_a)
        self.assertIn("final_mask", res_b)
        self.assertIn("final_mask", res_c)

        self.assertEqual(res_a["final_mask"].shape, (64, 64))
        self.assertEqual(res_b["final_mask"].shape, (64, 64))
        self.assertEqual(res_c["final_mask"].shape, (64, 64))

        self.assertIn(14.0, res_a["intermediate_checkpoints"])
        self.assertIn(7.0, res_b["intermediate_checkpoints"])
        self.assertIn(14.0, res_b["intermediate_checkpoints"])
        self.assertIn(3.0, res_c["intermediate_checkpoints"])
        self.assertIn(7.0, res_c["intermediate_checkpoints"])
        self.assertIn(14.0, res_c["intermediate_checkpoints"])


if __name__ == "__main__":
    unittest.main()
