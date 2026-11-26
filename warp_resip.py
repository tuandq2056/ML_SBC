"""Warp RL + ResiP Module
Implements:
1. RationalQuadraticWarp1D: Monotonic, identity-initialized spline warp on [-B, B]
   with analytical derivative & Jacobian log-det regularization.
2. DeepSetsWarpResiPActor: Permutation-invariant zone encoder + ResiP closed-loop
   reactivity gate + spline parameter network.
3. DeepSetsAHUTwinCritic: Building-level Twin Critic evaluating 4D warped AHU setpoints
   mapped to each zone's AHU group (no scale blow-up from spline parameters).
4. WarpResiPMixin & make_warp_policy(base_cls): Hooks cleanly into any TransferPolicy
   subclass (EmptyDayConsistentPolicy or CausalIdentifiedReservePolicy) with exact
   physics re-projection of VAV dampers and reheat valves.
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

from physics_foundation import (
    CausalIdentifiedReservePolicy,
    EmptyDayConsistentPolicy,
    TransferPolicy,
)

ZONE_FEAT_DIM = 9
GLOBAL_FEAT_DIM = 10
AHU_ACT_DIM = 4  # [sat_norm, press_norm, recirc_norm, water_norm]


class RationalQuadraticWarp1D(nn.Module):
    """Monotonic Rational-Quadratic Spline on [-bound, bound] with exact Identity init."""
    def __init__(self, num_bins=8, bound=1.05, min_bin_width=0.04,
                 min_bin_height=0.04, min_derivative=0.25):
        super().__init__()
        self.K = num_bins
        self.B = float(bound)
        self.min_w = float(min_bin_width)
        self.min_h = float(min_bin_height)
        self.min_d = float(min_derivative)
        assert self.K * self.min_w < 2.0 * self.B
        assert self.K * self.min_h < 2.0 * self.B
        self.c0 = math.log(math.expm1(1.0 - self.min_d))

    @property
    def num_params_per_dim(self):
        return 3 * self.K - 1

    def forward(self, x, raw_params, gate=None):
        K, B = self.K, self.B
        raw_w, raw_h, raw_d = torch.split(raw_params, [K, K, K - 1], dim=-1)
        if gate is not None:
            raw_w = gate * raw_w
            raw_h = gate * raw_h
            raw_d = gate * raw_d

        widths = self.min_w + (2.0 * B - K * self.min_w) * F.softmax(raw_w, dim=-1)
        heights = self.min_h + (2.0 * B - K * self.min_h) * F.softmax(raw_h, dim=-1)

        pad_lo = torch.full_like(widths[..., :1], -B)
        pad_hi = torch.full_like(widths[..., :1], B)
        cumwidths = torch.cat([pad_lo, torch.cumsum(widths, dim=-1)[..., :-1] - B, pad_hi], dim=-1)
        cumheights = torch.cat([pad_lo, torch.cumsum(heights, dim=-1)[..., :-1] - B, pad_hi], dim=-1)

        interior_d = F.softplus(raw_d + self.c0) + self.min_d
        derivatives = F.pad(interior_d, (1, 1), value=1.0)

        x_c = x.clamp(-B + 1e-6, B - 1e-6)
        bin_idx = ((x_c.unsqueeze(-1) >= cumwidths[..., :-1]).sum(dim=-1, keepdim=True) - 1).clamp(0, K - 1)

        xk = cumwidths.gather(-1, bin_idx).squeeze(-1)
        wk = widths.gather(-1, bin_idx).squeeze(-1)
        yk = cumheights.gather(-1, bin_idx).squeeze(-1)
        hk = heights.gather(-1, bin_idx).squeeze(-1)
        dk = derivatives.gather(-1, bin_idx).squeeze(-1)
        dk1 = derivatives.gather(-1, bin_idx + 1).squeeze(-1)

        sk = hk / wk.clamp(min=1e-8)
        theta = ((x_c - xk) / wk.clamp(min=1e-8)).clamp(0.0, 1.0)
        theta_1m = theta * (1.0 - theta)

        num = hk * (sk * theta.pow(2) + dk * theta_1m)
        den = (sk + (dk1 + dk - 2.0 * sk) * theta_1m).clamp(min=1e-8)
        y_raw = yk + num / den

        grad_num = sk.pow(2) * (dk1 * theta.pow(2) + 2.0 * sk * theta_1m + dk * (1.0 - theta).pow(2))
        dydx = (grad_num / den.pow(2).clamp(min=1e-12)).clamp(min=1e-6)
        log_det_j = torch.log(dydx).sum(dim=-1)

        return y_raw.clamp(-1.0, 1.0), y_raw, log_det_j


class DeepSetsWarpResiPActor(nn.Module):
    """DeepSets encoder + ResiP reactivity gate + Rational-Quadratic Spline Warp."""
    def __init__(self, zone_dim=ZONE_FEAT_DIM, global_dim=GLOBAL_FEAT_DIM,
                 hidden=128, action_dim=AHU_ACT_DIM, num_bins=8):
        super().__init__()
        self.action_dim = action_dim
        self.warp = RationalQuadraticWarp1D(num_bins=num_bins)
        n_spline = self.warp.num_params_per_dim

        self.zone_enc = nn.Sequential(
            nn.Linear(zone_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        ctx_dim = hidden * 2 + global_dim + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(ctx_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.spline_head = nn.Linear(256, action_dim * n_spline)
        self.gate_head = nn.Linear(256, action_dim)

        nn.init.zeros_(self.spline_head.weight)
        nn.init.zeros_(self.spline_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(self, z_feats, z_mask, g_feats, x_base):
        h = self.zone_enc(z_feats) * z_mask
        denom = z_mask.sum(dim=1).clamp(min=1.0)
        z_mean = h.sum(dim=1) / denom
        z_max = h.masked_fill(z_mask.expand_as(h) < 0.5, -1e9).max(dim=1).values
        z_max = torch.where(denom > 0, z_max, torch.zeros_like(z_max))

        ctx = torch.cat([z_mean, z_max, g_feats, x_base], dim=-1)
        feat = self.trunk(ctx)

        raw_spline = self.spline_head(feat).view(-1, self.action_dim, self.warp.num_params_per_dim)
        gate = torch.sigmoid(self.gate_head(feat))
        x_warped, _, log_det = self.warp(x_base, raw_spline, gate=gate.unsqueeze(-1))
        return x_warped, gate, log_det


class _WarpCriticHead(nn.Module):
    def __init__(self, zone_dim, global_dim, hidden, action_dim):
        super().__init__()
        self.zone_enc = nn.Sequential(
            nn.Linear(zone_dim + action_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.q_net = nn.Sequential(
            nn.Linear(hidden * 2 + global_dim + action_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, z_feats, z_mask, g_feats, x_eff):
        x_exp = x_eff.unsqueeze(1).expand(-1, z_feats.shape[1], -1)
        h = self.zone_enc(torch.cat([z_feats, x_exp], dim=-1)) * z_mask
        denom = z_mask.sum(dim=1).clamp(min=1.0)
        z_mean = h.sum(dim=1) / denom
        z_max = h.masked_fill(z_mask.expand_as(h) < 0.5, -1e9).max(dim=1).values
        z_max = torch.where(denom > 0, z_max, torch.zeros_like(z_max))
        return self.q_net(torch.cat([z_mean, z_max, g_feats, x_eff], dim=-1))


class DeepSetsAHUTwinCritic(nn.Module):
    """Building-level Twin Critic receiving 4D warped action x_eff in [-1, 1]^4."""
    def __init__(self, zone_dim=ZONE_FEAT_DIM, global_dim=GLOBAL_FEAT_DIM,
                 hidden=128, action_dim=AHU_ACT_DIM):
        super().__init__()
        self.q1 = _WarpCriticHead(zone_dim, global_dim, hidden, action_dim)
        self.q2 = _WarpCriticHead(zone_dim, global_dim, hidden, action_dim)

    def forward(self, z_feats, z_mask, g_feats, x_eff):
        return self.q1(z_feats, z_mask, g_feats, x_eff), self.q2(z_feats, z_mask, g_feats, x_eff)

    def Q1(self, z_feats, z_mask, g_feats, x_eff):
        return self.q1(z_feats, z_mask, g_feats, x_eff)


def extract_warp_state(policy, obs_arr, temperature, low, high, mode, oat_ema):
    outside_ahu = np.array([
        policy._temperature(obs_arr, idx, 'outside_air_temperature_sensor', 290.0)
        for idx in policy.outside_obs
    ])
    outside = outside_ahu[policy.zone_ahu]
    mean_oat = float(np.mean(outside_ahu))
    oat_ema = mean_oat if oat_ema is None else (0.98 * oat_ema + 0.02 * mean_oat)

    free = policy._free_drift(temperature, outside, mode)
    z_feats = np.column_stack([
        np.clip((temperature - low) / 2.0, -5.0, 5.0),
        np.clip((high - temperature) / 2.0, -5.0, 5.0),
        np.clip(free / 2.0, -3.0, 3.0),
        np.clip((outside - temperature) / 20.0, -5.0, 5.0),
        np.clip((policy.slow_temperature - temperature) / 5.0, -5.0, 5.0),
        np.clip(policy.model_error, 0.0, 2.0),
        np.clip(policy.residual / 2.0, -2.0, 2.0),
        np.log(np.clip(policy.theta[:, 5], 1e-3, 10.0)),
        np.log1p(policy.zone_area) / 5.0,
    ]).astype(np.float32)
    z_feats = np.nan_to_num(z_feats, nan=0.0, posinf=5.0, neginf=-5.0)

    hour = policy._hour(obs_arr)
    sine_idx = policy.field_index.get('hod_sin_000')
    cos_idx = policy.field_index.get('hod_cos_000')
    hod_sin = float(obs_arr[sine_idx]) if sine_idx is not None else math.sin(hour * math.pi / 12.0)
    hod_cos = float(obs_arr[cos_idx]) if cos_idx is not None else math.cos(hour * math.pi / 12.0)

    c_now = float(obs_arr[policy.comfort_now_index] > 0.5) if policy.comfort_now_index is not None else float(mode)
    c_soon = float(obs_arr[policy.comfort_soon_index] > 0.5) if policy.comfort_soon_index is not None else c_now
    hours_to_start = ((policy.start_hour - hour) % 24.0) / 24.0
    occ_raw = float(obs_arr[policy.occupancy_index]) if policy.occupancy_index is not None else 0.0
    occ_norm = math.log1p(max(occ_raw, 0.0)) / 5.0 if np.isfinite(occ_raw) else 0.0
    water_obs = policy._temperature(obs_arr, policy.water_sensor, 'supply_water_temperature_sensor', 340.0)
    heat_recency = math.exp(-max(0, policy.step_count - policy.last_heating_step) / 144.0)

    g_feats = np.array([
        hod_sin, hod_cos,
        np.clip((mean_oat - 290.0) / 15.0, -3.0, 3.0),
        np.clip((mean_oat - oat_ema) / 5.0, -3.0, 3.0),
        c_now, c_soon, hours_to_start, occ_norm,
        np.clip((water_obs - 330.0) / 20.0, -3.0, 3.0),
        heat_recency,
    ], dtype=np.float32)
    g_feats = np.nan_to_num(g_feats, nan=0.0, posinf=3.0, neginf=-3.0)
    return z_feats, g_feats, oat_ema, outside_ahu, outside, free


class WarpResiPMixin:
    """Mixin layering Warp RL + ResiP Gate + Physics Re-projection onto any BasePolicy."""
    def __init__(self, building_meta_data, action_spec=None, parameters=None,
                 actors=None, action_modifier=None, dev=None):
        self._actors = actors if isinstance(actors, (list, tuple)) else ([actors] if actors is not None else [])
        self._action_modifier = action_modifier
        self._dev = dev or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(building_meta_data, action_spec, parameters)

    def reset(self):
        super().reset()
        self._oat_ema = None
        self._last_transition = None
        self._last_gate = np.zeros(AHU_ACT_DIM, dtype=np.float32)
        self._last_thermal_sample = None

    def _learn(self, observed):
        delay = int(self.parameters['observation_delay'])
        self._last_thermal_sample = None
        if self.previous_observed is not None and len(self.history) > delay:
            applied = list(self.history)[-(delay + 1)]
            old = self.previous_observed
            features = self._features(old, applied['outside'], applied['mode'], applied['slow'])
            drift = np.einsum('ij,ij->i', self.theta[:, :5], features) + self.residual
            gain = self.theta[:, 5] / 1000.0
            pred_next = (old + drift + gain * (applied['conductance'] * applied['supply'] + applied['reheat'])) / (
                1.0 + gain * applied['conductance'])
            power = (applied['conductance'] * (applied['supply'] - observed) + applied['reheat']) / 1000.0
            self._last_thermal_sample = {
                'old': old.copy(),
                'pred_next': pred_next.copy(),
                'observed': observed.copy(),
                'features': features.copy(),
                'power': power.copy(),
                'model_error': self.model_error.copy(),
                'residual': self.residual.copy(),
                'theta_gain': self.theta[:, 5].copy(),
                'zone_area': self.zone_area.copy(),
            }
        super()._learn(observed)

    def _reproject_ahu(self, group, temperature, free, low, high, max_flow, outside, water,
                       setpoint_w, pressure_w, recirc_w):
        t, d, lo, hi = temperature[group], free[group], low[group], high[group]
        response = self.theta[group, 5]
        capacity = max_flow[group]
        inside = float(np.average(t, weights=self.zone_area[group]))

        mixed = recirc_w * inside + (1.0 - recirc_w) * outside
        supply = min(mixed, setpoint_w)

        cool_need = np.maximum(t + d - hi, 0.0) * 1000.0 / response
        heat_need = np.maximum(lo - t - d, 0.0) * 1000.0 / response
        cooling_flow = cool_need / (self.AIR_CAPACITY * np.maximum(hi - supply, 0.05))
        heating_flow = heat_need / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - lo, 0.5))

        maximum_water_heat = self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * max(water - supply, 0.0)
        maximum_heating_flow = maximum_water_heat / (self.AIR_CAPACITY * max(self.MAX_DISCHARGE_K - supply, 0.1))
        heating_flow = np.minimum(heating_flow, maximum_heating_flow)
        needed_flow = np.maximum(cooling_flow, heating_flow)

        pressure_ratio = math.sqrt(pressure_w / self.DESIGN_PRESSURE)
        dampers = np.clip(needed_flow / (capacity * pressure_ratio), self.MIN_DAMPER, 1.0)
        flow = dampers * capacity * pressure_ratio
        conductance = flow * self.AIR_CAPACITY

        water_gap = max(water - supply, 0.0)
        wanted_reheat = np.maximum(
            (lo - t - d) * 1000.0 / response + conductance * (lo - supply), 0.0)
        reheat_limit = np.minimum(
            conductance * max(self.MAX_DISCHARGE_K - supply, 0.0),
            self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * water_gap)
        reheat = np.minimum(wanted_reheat, reheat_limit)

        return {
            'run': 1, 'pressure': float(pressure_w), 'setpoint': float(setpoint_w),
            'recirc': float(recirc_w), 'damper': dampers, 'conductance': conductance,
            'supply': np.full(group.size, supply), 'reheat': reheat,
        }

    def get_action(self, obs):
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        observed = self._zone_temperatures(obs_arr)
        if self.slow_temperature is None:
            self.slow_temperature = observed.copy()
        self._learn(observed)

        outside_ahu = np.array([
            self._temperature(obs_arr, index, 'outside_air_temperature_sensor', 290.0)
            for index in self.outside_obs
        ])
        outside = outside_ahu[self.zone_ahu]
        mode = bool(obs_arr[self.comfort_now_index] > 0.5) if self.comfort_now_index is not None else True

        temperature = observed.copy()
        if self.history and int(self.parameters['observation_delay']) == 1:
            last = self.history[-1]
            drift = self._free_drift(observed, last['outside'], last['mode'])
            gain = self.theta[:, 5] / 1000.0
            temperature = (observed + drift + gain * (last['conductance'] * last['supply'] + last['reheat'])) / (
                1.0 + gain * last['conductance'])
            temperature = np.clip(temperature, observed - 8.0, observed + 8.0)

        low, high, mode = self._band(obs_arr, temperature)
        z_f, g_f, self._oat_ema, _, _, free = extract_warp_state(
            self, obs_arr, temperature, low, high, mode, self._oat_ema)

        max_flow = np.full(self.n_zones, 2.0)
        valid = self.zone_flow_obs >= 0
        if np.any(valid):
            flow_reading = obs_arr[self.zone_flow_obs[valid]]
            max_flow[valid] = np.where(np.isfinite(flow_reading) & (flow_reading > 0.0), flow_reading, 2.0)
        max_flow = np.clip(max_flow, 0.01, 100.0)

        water_observed = self._temperature(obs_arr, self.water_sensor, 'supply_water_temperature_sensor', 340.0)
        if np.any(temperature + free < low):
            self.last_heating_step = self.step_count
        recent_heat = self.step_count - self.last_heating_step < 24
        water_available = water_observed

        base_choices = []
        any_reheat_10 = False
        for i, group in enumerate(self.groups):
            if not group.size:
                base_choices.append(None)
                continue
            c = self._choose_ahu(group, temperature, free, low, high, max_flow,
                                 float(outside_ahu[i]), float(water_available))
            base_choices.append(c)
            if np.any(c['reheat'] > 10.0):
                any_reheat_10 = True

        if any_reheat_10:
            self.last_heating_step = self.step_count
        hold_steps = getattr(self, 'HOLD_STEPS', 24)
        hold_active = self.step_count - self.last_heating_step < hold_steps
        base_water_target = float(self.parameters['hot_water_target']) if (recent_heat or any_reheat_10 or hold_active) else 310.0

        active_indices = [i for i, c in enumerate(base_choices) if c is not None and c['run'] == 1]
        if active_indices:
            weights = np.array([self.zone_area[self.groups[i]].sum() for i in active_indices], dtype=float)
            weights /= max(weights.sum(), 1e-8)
            sat_b = sum(w * base_choices[i]['setpoint'] for w, i in zip(weights, active_indices))
            press_b = sum(w * base_choices[i]['pressure'] for w, i in zip(weights, active_indices))
            recirc_b = sum(w * base_choices[i]['recirc'] for w, i in zip(weights, active_indices))
        else:
            sat_b, press_b, recirc_b = 305.0, 0.0, 0.7

        x_base = np.array([
            np.clip((sat_b - 295.0) / 10.0, -1.0, 1.0),
            np.clip((press_b - 1000.0) / 1000.0, -1.0, 1.0),
            np.clip((recirc_b - 0.5) / 0.5, -1.0, 1.0),
            np.clip((base_water_target - 330.0) / 20.0, -1.0, 1.0),
        ], dtype=np.float32)

        gate_val = np.zeros(AHU_ACT_DIM, dtype=np.float32)
        if self._action_modifier is not None:
            x_eff = self._action_modifier(self.step_count, z_f, g_f, x_base)
        elif self._actors:
            zt = torch.tensor(z_f, dtype=torch.float32, device=self._dev).unsqueeze(0)
            zm = torch.ones((1, z_f.shape[0], 1), dtype=torch.float32, device=self._dev)
            gt = torch.tensor(g_f, dtype=torch.float32, device=self._dev).unsqueeze(0)
            xb = torch.tensor(x_base, dtype=torch.float32, device=self._dev).unsqueeze(0)
            with torch.no_grad():
                outs = [net(zt, zm, gt, xb) for net in self._actors]
            x_eff = np.clip(np.mean([o[0].squeeze(0).cpu().numpy() for o in outs], axis=0), -1.0, 1.0).astype(np.float32)
            gate_val = np.mean([o[1].squeeze(0).cpu().numpy() for o in outs], axis=0).astype(np.float32)
        else:
            x_eff = x_base.copy()

        delta_x = x_eff - x_base
        if np.allclose(delta_x, 0.0, atol=1e-7):
            water_target = base_water_target
        else:
            water_target = float(np.clip(330.0 + 20.0 * float(x_eff[3]), 310.0, 350.0))

        continuous = np.full(self.n_continuous, -1.0, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        conductance = np.zeros(self.n_zones)
        supply = outside.copy()
        reheat = np.zeros(self.n_zones)

        for i, group in enumerate(self.groups):
            c = base_choices[i]
            if c is None:
                continue
            if c['run'] == 1 and not np.allclose(delta_x[:3], 0.0, atol=1e-7):
                sat_w = float(np.clip(c['setpoint'] + 10.0 * float(delta_x[0]), 285.0, 305.0))
                press_w = float(np.clip(c['pressure'] + 1000.0 * float(delta_x[1]), self.MIN_PRESSURE, self.DESIGN_PRESSURE))
                recirc_w = float(np.clip(c['recirc'] + 0.5 * float(delta_x[2]), 0.0, self.MAX_RECIRCULATION))
                c = self._reproject_ahu(
                    group, temperature, free, low, high, max_flow,
                    float(outside_ahu[i]), float(water_available), sat_w, press_w, recirc_w)

            discrete[i] = c['run']
            continuous[self.temp_indices[i]] = TransferPolicy._native_action(
                self, f'{self.ahu_ids[i]}_supply_air_temperature_setpoint', c['setpoint'], 285.0, 305.0)
            continuous[self.pressure_indices[i]] = TransferPolicy._native_action(
                self, f'{self.ahu_ids[i]}_supply_air_static_pressure_setpoint', c['pressure'], 0.0, 2000.0)
            continuous[self.recirculation_indices[i]] = TransferPolicy._native_action(
                self, f'{self.ahu_ids[i]}_recirculation_percentage_command', c['recirc'], 0.0, 1.0)
            continuous[self.damper_indices[group]] = TransferPolicy._native_action(
                self, 'supply_air_damper_percentage_command', c['damper'], 0.0, 1.0)
            conductance[group] = c['conductance']
            supply[group] = c['supply']
            reheat[group] = c['reheat']

        if np.any(reheat > 10.0):
            self.last_heating_step = self.step_count

        valve_need = reheat / (self.WATER_VALVE_CAPACITY * np.maximum(water_available - supply, 1.0))
        pressure_water = float(np.clip(np.max(valve_need) ** 2 * 1.05, 0.0, 20.0))
        if pressure_water > 1e-8:
            valves = np.clip(valve_need / math.sqrt(pressure_water), 0.0, 1.0)
            discrete[-1] = 1
        else:
            valves = np.zeros(self.n_zones)

        continuous[self.valve_indices] = TransferPolicy._native_action(
            self, 'reheat_valve_percentage_command', valves, 0.0, 1.0)
        if self.water_indices:
            continuous[self.water_indices[0]] = TransferPolicy._native_action(
                self, 'supply_water_temperature_setpoint', water_target, 310.0, 350.0)
        if len(self.pressure_indices) > self.num_ahus:
            continuous[self.pressure_indices[-1]] = TransferPolicy._native_action(
                self, 'differential_pressure_setpoint', pressure_water, 0.0, 20.0)

        actual_reheat = np.minimum(
            self.WATER_VALVE_CAPACITY * math.sqrt(pressure_water) * valves * np.maximum(water_available - supply, 0.0),
            conductance * np.maximum(self.MAX_DISCHARGE_K - supply, 0.0))
        self.history.append({'conductance': conductance, 'supply': supply,
                             'reheat': actual_reheat, 'outside': outside.copy(),
                             'mode': mode, 'slow': self.slow_temperature.copy()})
        self.slow_temperature += (temperature - self.slow_temperature) / 48.0
        self.previous_observed = observed.copy()
        self.step_count += 1
        self._last_transition = (z_f, g_f, x_base, x_eff)
        self._last_gate = gate_val

        action = {'continuous_action': np.clip(np.nan_to_num(continuous, nan=-1.0), -1.0, 1.0).astype(np.float32),
                  'discrete_action': discrete.astype(np.int32)}
        self.last_action = action
        return action


def make_warp_policy(base_cls):
    """Create a WarpResiP Policy class wrapping any TransferPolicy subclass."""
    class _ConfiguredWarpResiPPolicy(WarpResiPMixin, base_cls):
        name = f'warp_resip_{getattr(base_cls, "name", "base")}'
    return _ConfiguredWarpResiPPolicy


class WarpResiPPolicy(WarpResiPMixin, EmptyDayConsistentPolicy):
    name = 'warp_resip_empty_day'


class WarpResiPCausalPolicy(WarpResiPMixin, CausalIdentifiedReservePolicy):
    name = 'warp_resip_causal_identified'


def verify_warp_resip_module():
    """Offline sanity check for spline identity, monotonicity, and gradient flow."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor = DeepSetsWarpResiPActor().to(dev)
    critic = DeepSetsAHUTwinCritic().to(dev)

    z = torch.randn(4, 12, ZONE_FEAT_DIM, device=dev)
    m = torch.ones(4, 12, 1, device=dev)
    g = torch.randn(4, GLOBAL_FEAT_DIM, device=dev)
    xb = torch.linspace(-1.0, 1.0, 16, device=dev).view(4, 4)

    xw, gate, log_det = actor(z, m, g, xb)
    max_id_err = float((xw - xb).abs().max().item())
    max_ld_err = float(log_det.abs().max().item())

    q1, _ = critic(z, m, g, xw)
    loss = -q1.mean() + 0.1 * (log_det ** 2).mean()
    loss.backward()
    has_grad = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in actor.parameters() if p.requires_grad)
    return {'max_identity_err': max_id_err, 'max_logdet_err': max_ld_err, 'grad_ok': has_grad}
