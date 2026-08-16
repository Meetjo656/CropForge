"""
Temporal Horizon Forecaster & Recursive Rollout Engine for CropForge Milestone 12.

Evaluates three distinct temporal extrapolation strategies to determine why lesion geometry collapses over long horizons:
- Approach A (Direct Extrapolation): Day 0 ─────────────────────────────► Day 14
- Approach B (Two-Step Recursive Rollout): Day 0 ──────► Day 7 ──────► Day 14
- Approach C (Multi-Step Autoregressive Rollout): Day 0 ──► Day 3 ──► Day 7 ──► Day 14
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np

from cropforge.diffusion.models.spatial_mask_forecaster import SpatialMaskForecaster

_logger = logging.getLogger(__name__)


class RecursiveSpatialForecaster:
    """
    Forecaster supporting direct vs recursive multi-step spatial mask propagation.
    """

    def __init__(self) -> None:
        self.base_forecaster = SpatialMaskForecaster()

    def forecast_step(
        self,
        mask_np: np.ndarray,
        delta_t_days: float,
        temp_c: float = 25.0,
        rh_percent: float = 75.0,
        treatment: str = "untreated",
    ) -> Tuple[np.ndarray, float]:
        """
        Single step forecast over delta_t_days.
        """
        return self.base_forecaster.forecast_mask_numpy(
            t0_mask_np=mask_np,
            delta_t_days=delta_t_days,
            temp_c=temp_c,
            rh_percent=rh_percent,
            treatment=treatment,
        )

    def forecast_approach_a_direct(
        self,
        t0_mask: np.ndarray,
        target_horizon: float = 14.0,
        temp_c: float = 25.0,
        rh_percent: float = 75.0,
        treatment: str = "untreated",
    ) -> Dict[str, Any]:
        """
        Approach A: Direct extrapolation Day 0 -> Day 14.
        """
        fut_mask, fut_sev = self.forecast_step(
            mask_np=t0_mask,
            delta_t_days=target_horizon,
            temp_c=temp_c,
            rh_percent=rh_percent,
            treatment=treatment,
        )
        return {
            "approach": "Approach A (Direct Extrapolation)",
            "final_mask": fut_mask,
            "final_severity": fut_sev,
            "intermediate_checkpoints": {14.0: (fut_mask, fut_sev)},
        }

    def forecast_approach_b_twostep(
        self,
        t0_mask: np.ndarray,
        temp_c: float = 25.0,
        rh_percent: float = 75.0,
        treatment: str = "untreated",
    ) -> Dict[str, Any]:
        """
        Approach B: Two-step recursive rollout Day 0 -> Day 7 -> Day 14.
        """
        # Step 1: Day 0 -> Day 7
        m_day7, sev_day7 = self.forecast_step(
            mask_np=t0_mask,
            delta_t_days=7.0,
            temp_c=temp_c,
            rh_percent=rh_percent,
            treatment=treatment,
        )

        # Step 2: Day 7 -> Day 14 (7 additional days)
        m_day14, sev_day14 = self.forecast_step(
            mask_np=m_day7,
            delta_t_days=7.0,
            temp_c=temp_c,
            rh_percent=rh_percent,
            treatment=treatment,
        )

        return {
            "approach": "Approach B (Two-Step Recursive Rollout)",
            "final_mask": m_day14,
            "final_severity": sev_day14,
            "intermediate_checkpoints": {
                7.0: (m_day7, sev_day7),
                14.0: (m_day14, sev_day14),
            },
        }

    def forecast_approach_c_multistep(
        self,
        t0_mask: np.ndarray,
        temp_c: float = 25.0,
        rh_percent: float = 75.0,
        treatment: str = "untreated",
    ) -> Dict[str, Any]:
        """
        Approach C: Multi-step autoregressive rollout Day 0 -> Day 3 -> Day 7 -> Day 14.
        """
        # Step 1: Day 0 -> Day 3
        m_day3, sev_day3 = self.forecast_step(
            mask_np=t0_mask,
            delta_t_days=3.0,
            temp_c=temp_c,
            rh_percent=rh_percent,
            treatment=treatment,
        )

        # Step 2: Day 3 -> Day 7 (4 additional days)
        m_day7, sev_day7 = self.forecast_step(
            mask_np=m_day3,
            delta_t_days=4.0,
            temp_c=temp_c,
            rh_percent=rh_percent,
            treatment=treatment,
        )

        # Step 3: Day 7 -> Day 14 (7 additional days)
        m_day14, sev_day14 = self.forecast_step(
            mask_np=m_day7,
            delta_t_days=7.0,
            temp_c=temp_c,
            rh_percent=rh_percent,
            treatment=treatment,
        )

        return {
            "approach": "Approach C (Multi-Step Autoregressive Rollout)",
            "final_mask": m_day14,
            "final_severity": sev_day14,
            "intermediate_checkpoints": {
                3.0: (m_day3, sev_day3),
                7.0: (m_day7, sev_day7),
                14.0: (m_day14, sev_day14),
            },
        }
