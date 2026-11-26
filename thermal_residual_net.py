"""Stage 1: Physics-Informed Neural Thermal Residual Net (_free_drift upgrade)
Targets Weak Spot #1 of EmptyDayConsistentPolicy:
  TransferPolicy._features uses only 5 linear variables and completely ignores
  solar hour-of-day (hod_sin, hod_cos), occupant heat load (occ_norm), and
  nonlinear thermal lag / cold-start dynamics.

Trained purely Supervised on GPU from thermal_residuals_*.parquet with strict
causality (no future 'power' leak!) and zero RLS double-counting.
"""
import math
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from physics_foundation import EmptyDayConsistentPolicy, TransferPolicy, bounded_rls

THERMAL_FEAT_COLS = [
    'hod_sin', 'hod_cos', 'occ_norm',
    'oat_diff', 'slow_diff', 'group_diff', 'mode',
    'model_error', 'residual', 'theta_gain', 'log_area'
]
THERMAL_FEAT_DIM = len(THERMAL_FEAT_COLS)


def prepare_thermal_features_from_df(df):
    """Extract strictly causal [N, 11] float32 matrix and target [N, 1] from parquet df."""
    x = np.column_stack([
        df['hod_sin'].to_numpy(dtype=np.float32),
        df['hod_cos'].to_numpy(dtype=np.float32),
        df['occ_norm'].to_numpy(dtype=np.float32),
        np.clip(df['oat_diff'].to_numpy(dtype=np.float32), -5.0, 5.0),
        np.clip(df['slow_diff'].to_numpy(dtype=np.float32), -5.0, 5.0),
        np.clip(df['group_diff'].to_numpy(dtype=np.float32), -5.0, 5.0),
        df['mode'].to_numpy(dtype=np.float32),
        np.clip(df['model_error'].to_numpy(dtype=np.float32), 0.0, 2.0),
        np.clip(df['residual'].to_numpy(dtype=np.float32) / 2.0, -2.5, 2.5),
        np.log(np.clip(df['theta_gain'].to_numpy(dtype=np.float32), 1e-3, 10.0)),
        np.log1p(np.clip(df['zone_area'].to_numpy(dtype=np.float32), 0.0, 1e5)) / 5.0,
    ]).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
    y = np.clip(df['rls_err_K'].to_numpy(dtype=np.float32), -3.0, 3.0)[:, None]
    y = np.nan_to_num(y, nan=0.0)
    return x, y


class ThermalResidualResNet(nn.Module):
    """Per-zone Physics-Informed Residual Drift predictor (bounded output [-max_corr, max_corr])."""
    def __init__(self, in_dim=THERMAL_FEAT_DIM, hidden=192, max_corr=0.80):
        super().__init__()
        self.max_corr = float(max_corr)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.block1 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.in_proj(x)
        h = F.gelu(h + self.block1(h))
        h = F.gelu(h + self.block2(h))
        return self.max_corr * torch.tanh(self.head(h))


class ThermalAugmentedPolicy(EmptyDayConsistentPolicy):
    """EmptyDayConsistentPolicy upgraded with GPU-trained Neural Thermal Drift residual."""
    name = 'empty_day_thermal_nn'

    def __init__(self, building_meta_data, action_spec=None, parameters=None,
                 thermal_nets=None, alpha_nn=1.0, dev=None):
        self._thermal_nets = thermal_nets if isinstance(thermal_nets, (list, tuple)) else (
            [thermal_nets] if thermal_nets is not None else [])
        self._alpha_nn = float(alpha_nn)
        self._dev = dev or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(building_meta_data, action_spec, parameters)

    def reset(self):
        super().reset()
        self._cur_ctx = (0.0, 1.0, 0.0)
        self._prev_ctx = None
        self._nn_corr_log = []

    def _extract_ctx(self, obs):
        hour = self._hour(obs)
        s_idx = self.field_index.get('hod_sin_000')
        c_idx = self.field_index.get('hod_cos_000')
        h_sin = float(obs[s_idx]) if s_idx is not None else math.sin(hour * math.pi / 12.0)
        h_cos = float(obs[c_idx]) if c_idx is not None else math.cos(hour * math.pi / 12.0)
        occ_raw = float(obs[self.occupancy_index]) if self.occupancy_index is not None else 0.0
        occ_norm = math.log1p(max(occ_raw, 0.0)) / 5.0 if np.isfinite(occ_raw) else 0.0
        return (h_sin, h_cos, occ_norm)

    def _predict_nn_drift(self, temperature, outside, mode, slow, ctx):
        if not self._thermal_nets or self._alpha_nn <= 0.0 or ctx is None:
            return np.zeros(self.n_zones, dtype=float)
        h_sin, h_cos, occ_norm = ctx
        feats = self._features(temperature, outside, mode, slow)
        x_np = np.column_stack([
            np.full(self.n_zones, h_sin, dtype=np.float32),
            np.full(self.n_zones, h_cos, dtype=np.float32),
            np.full(self.n_zones, occ_norm, dtype=np.float32),
            np.clip(feats[:, 1], -5.0, 5.0).astype(np.float32),
            np.clip(feats[:, 2], -5.0, 5.0).astype(np.float32),
            np.clip(feats[:, 3], -5.0, 5.0).astype(np.float32),
            feats[:, 4].astype(np.float32),
            np.clip(self.model_error, 0.0, 2.0).astype(np.float32),
            np.clip(self.residual / 2.0, -2.5, 2.5).astype(np.float32),
            np.log(np.clip(self.theta[:, 5], 1e-3, 10.0)).astype(np.float32),
            np.log1p(np.clip(self.zone_area, 0.0, 1e5)).astype(np.float32) / 5.0,
        ])
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=5.0, neginf=-5.0)
        xt = torch.tensor(x_np, dtype=torch.float32, device=self._dev)
        with torch.no_grad():
            preds = [net(xt).squeeze(-1).cpu().numpy() for net in self._thermal_nets]
        delta_nn = self._alpha_nn * np.mean(preds, axis=0)
        return np.clip(delta_nn, -0.8, 0.8)

    def _learn(self, observed):
        delay = int(self.parameters['observation_delay'])
        if self.previous_observed is None or len(self.history) <= delay:
            return
        applied = list(self.history)[-(delay + 1)]
        old = self.previous_observed
        nn_corr = self._predict_nn_drift(old, applied['outside'], applied['mode'], applied['slow'], self._prev_ctx)

        # Causal model error update BEFORE RLS fit (including NN correction)
        features = self._features(old, applied['outside'], applied['mode'], applied['slow'])
        drift = np.einsum('ij,ij->i', self.theta[:, :5], features) + self.residual + nn_corr
        gain = self.theta[:, 5] / 1000.0
        prediction = (old + drift + gain * (applied['conductance'] * applied['supply'] + applied['reheat'])) / (
            1.0 + gain * applied['conductance'])
        error = np.where(np.isfinite(observed - prediction), np.minimum(np.abs(observed - prediction), 2.0), 2.0)
        previous_error = self.model_error.copy()

        # RLS fits strictly on residual NOT already explained by NN (zero double-counting!)
        power = (applied['conductance'] * (applied['supply'] - observed) + applied['reheat']) / 1000.0
        x = np.column_stack((features, power))
        delta = observed - old
        valid = np.isfinite(x).all(axis=1) & np.isfinite(delta) & (np.abs(delta) <= 3.0)
        lower = np.zeros_like(self.theta)
        upper = np.ones_like(self.theta) * np.array([2.0, 5.0, 2.0, 2.0, 2.0, 1.0])
        lower[:, 0] = -2.0
        lower[:, 5] = 0.5 * self._thermal_prior
        upper[:, 5] = 1.5 * self._thermal_prior
        if np.any(valid):
            fit, cov = bounded_rls(
                self.theta[valid], self.covariance[valid], x[valid],
                delta[valid] - self.residual[valid] - nn_corr[valid],
                lower[valid], upper[valid], float(self.parameters['forgetting_factor']))
            self.theta[valid], self.covariance[valid] = fit, cov
        remaining = np.clip(delta - (np.einsum('ni,ni->n', self.theta, x) + nn_corr), -5.0, 5.0)
        rate = float(self.parameters['disturbance_rate'])
        self.residual = np.clip((1.0 - rate) * self.residual + rate * remaining, -5.0, 5.0)
        self.model_error = 0.94 * previous_error + 0.06 * error

    def _free_drift(self, temperature, outside, mode):
        base_drift = super()._free_drift(temperature, outside, mode)
        nn_corr = self._predict_nn_drift(temperature, outside, mode, self.slow_temperature, self._cur_ctx)
        self._nn_corr_log.append(float(np.mean(np.abs(nn_corr))))
        return np.clip(base_drift + nn_corr, -5.0, 5.0)

    def get_action(self, obs):
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        self._cur_ctx = self._extract_ctx(obs_arr)
        act = super().get_action(obs_arr)
        self._prev_ctx = self._cur_ctx
        return act
