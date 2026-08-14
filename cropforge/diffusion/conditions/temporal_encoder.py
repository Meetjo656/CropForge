"""
Temporal Condition Encoder for CropForge SD3.5 Forecasting Pipeline.

Encodes continuous time horizon (delta_t), environmental covariates (temperature,
relative humidity, soil moisture), and treatment interventions into joint conditioning
embeddings compatible with SD3Transformer2DModel pooled projections and cross-attention.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """
    Compute sinusoidal positional embeddings for time horizon delta_t.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    fraction = timesteps.float().unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)
    embedding = torch.cat([torch.sin(fraction), torch.cos(fraction)], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TemporalConditionEncoder(nn.Module):
    """
    Encodes temporal, environmental, and treatment parameters for SD3.5 forecasting.
    """

    TREATMENT_MAP = {
        "untreated": 0,
        "fungicide": 1,
        "biocontrol": 2,
        "resistance_inducer": 3,
    }

    def __init__(
        self,
        pooled_projection_dim: int = 2048,
        joint_attention_dim: int = 4096,
        time_embed_dim: int = 256,
        env_dim: int = 3,
        num_treatments: int = 8,
        treatment_dim: int = 128,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.pooled_projection_dim = pooled_projection_dim
        self.joint_attention_dim = joint_attention_dim
        self.time_embed_dim = time_embed_dim

        # 1. Time horizon MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # 2. Environmental covariates projection [Temp, RH, SoilMoisture]
        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, time_embed_dim),
        )

        # 3. Treatment intervention embedding
        self.treatment_embed = nn.Embedding(num_treatments, treatment_dim)

        # 4. Joint fusion projection to SD3.5 pooled_projection_dim
        fusion_dim = time_embed_dim + time_embed_dim + treatment_dim
        self.pooled_fusion = nn.Sequential(
            nn.Linear(fusion_dim, pooled_projection_dim),
            nn.LayerNorm(pooled_projection_dim),
            nn.SiLU(),
            nn.Linear(pooled_projection_dim, pooled_projection_dim),
        )

        # 5. Sequence projection to joint_attention_dim
        self.sequence_proj = nn.Sequential(
            nn.Linear(fusion_dim, joint_attention_dim),
            nn.SiLU(),
        )

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def encode_conditions(
        self,
        delta_t: Union[float, List[float], torch.Tensor],
        env_covariates: Optional[Union[List[float], torch.Tensor]] = None,
        treatment: Optional[Union[str, List[str], int, torch.Tensor]] = None,
        batch_size: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode temporal conditions into (pooled_projections, encoder_hidden_states).

        Args:
            delta_t: Time horizon in days (e.g., 3.0, 7.0, 14.0).
            env_covariates: Environmental vector [Temp (C), RH (%), SoilMoisture (%)].
            treatment: Treatment identifier ('untreated', 'fungicide', 'biocontrol', or int).
            batch_size: Target batch size.
            device: Compute device override.
            dtype: Precision dtype override.

        Returns:
            Tuple of (pooled_projections, encoder_hidden_states):
              - pooled_projections: Shape [batch_size, pooled_projection_dim]
              - encoder_hidden_states: Shape [batch_size, 16, joint_attention_dim]
        """
        dev = device if device is not None else next(self.parameters()).device
        dt_dtype = dtype if dtype is not None else next(self.parameters()).dtype

        # Process delta_t
        if isinstance(delta_t, (float, int)):
            dt_tensor = torch.tensor([float(delta_t)] * batch_size, device=dev, dtype=torch.float32)
        elif isinstance(delta_t, list):
            dt_tensor = torch.tensor([float(x) for x in delta_t], device=dev, dtype=torch.float32)
        elif isinstance(delta_t, torch.Tensor):
            dt_tensor = delta_t.to(device=dev, dtype=torch.float32)
            if dt_tensor.ndim == 0:
                dt_tensor = dt_tensor.unsqueeze(0).expand(batch_size)
        else:
            dt_tensor = torch.zeros(batch_size, device=dev, dtype=torch.float32)

        sin_embed = get_sinusoidal_time_embedding(dt_tensor, self.time_embed_dim).to(dtype=dt_dtype)
        time_feat = self.time_mlp(sin_embed)

        # Process env_covariates
        if env_covariates is None:
            env_tensor = torch.tensor([[25.0, 75.0, 60.0]] * batch_size, device=dev, dtype=dt_dtype)
        elif isinstance(env_covariates, list):
            if isinstance(env_covariates[0], list):
                env_tensor = torch.tensor(env_covariates, device=dev, dtype=dt_dtype)
            else:
                env_tensor = torch.tensor([env_covariates] * batch_size, device=dev, dtype=dt_dtype)
        elif isinstance(env_covariates, torch.Tensor):
            env_tensor = env_covariates.to(device=dev, dtype=dt_dtype)
            if env_tensor.ndim == 1:
                env_tensor = env_tensor.unsqueeze(0).expand(batch_size, -1)
        else:
            env_tensor = torch.tensor([[25.0, 75.0, 60.0]] * batch_size, device=dev, dtype=dt_dtype)

        env_feat = self.env_proj(env_tensor)

        # Process treatment
        if treatment is None:
            treat_idx = torch.zeros(batch_size, device=dev, dtype=torch.long)
        elif isinstance(treatment, str):
            t_id = self.TREATMENT_MAP.get(treatment.lower(), 0)
            treat_idx = torch.tensor([t_id] * batch_size, device=dev, dtype=torch.long)
        elif isinstance(treatment, int):
            treat_idx = torch.tensor([treatment] * batch_size, device=dev, dtype=torch.long)
        elif isinstance(treatment, list):
            ids = [self.TREATMENT_MAP.get(t.lower(), 0) if isinstance(t, str) else int(t) for t in treatment]
            treat_idx = torch.tensor(ids, device=dev, dtype=torch.long)
        elif isinstance(treatment, torch.Tensor):
            treat_idx = treatment.to(device=dev, dtype=torch.long)
        else:
            treat_idx = torch.zeros(batch_size, device=dev, dtype=torch.long)

        treat_feat = self.treatment_embed(treat_idx).to(dtype=dt_dtype)

        # Concatenate condition features
        fused = torch.cat([time_feat, env_feat, treat_feat], dim=-1)

        # 1. Pooled projection: [batch_size, pooled_projection_dim]
        pooled_projections = self.pooled_fusion(fused)

        # 2. Encoder hidden states: [batch_size, 16, joint_attention_dim]
        seq_feat = self.sequence_proj(fused).unsqueeze(1).repeat(1, 16, 1)

        return pooled_projections, seq_feat

    def forward(
        self,
        delta_t: Union[float, List[float], torch.Tensor],
        env_covariates: Optional[Union[List[float], torch.Tensor]] = None,
        treatment: Optional[Union[str, List[str], int, torch.Tensor]] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_conditions(
            delta_t=delta_t,
            env_covariates=env_covariates,
            treatment=treatment,
            batch_size=batch_size,
        )
