"""Physics Foundation: TransferPolicy -> EmptyDayConsistentPolicy
Attribution: Daniel Wu, jazivxt (Facilities V42), penguin069 (Deep Setback)
Source: https://www.kaggle.com/code/penguin069/smart-buildings-empty-day-consistent-model
No neural weights. Only metadata and observations enter decisions.
"""
import math
import re
from collections import deque
import numpy as np


class TransferPolicy:
    'A transferable controller implementing the official two-method API.'
    name = 'adaptive_physics'
    DEFAULTS = {
        'comfort_margin': 0.12, 'precondition_hours': 1.0, 'recovery_steps': 1.5,
        'forgetting_factor': 0.997, 'model_residual_gain': 0.16, 'comfort_cost': 65000.0,
        'gas_cost_ratio': 0.30, 'hot_water_target': 345.0, 'standby_minimum': 282.0,
        'standby_maximum': 307.0, 'observation_delay': 1, 'response_scale': 0.75,
        'disturbance_rate': 0.35,
    }
    AIR_CAPACITY = 1.2 * 1006.0
    WATER_VALVE_CAPACITY = 4180.0 * 1000.0 * 0.108 / 3600.0
    DESIGN_PRESSURE = 2000.0
    MIN_PRESSURE = 200.0
    MIN_DAMPER = 0.10
    MAX_RECIRCULATION = 0.70
    MAX_DISCHARGE_K = 308.15

    def __init__(self, building_meta_data, action_spec=None, parameters=None):
        self.metadata = building_meta_data
        self.action_spec = action_spec
        self.parameters = dict(self.DEFAULTS)
        if parameters:
            self.parameters.update(parameters)
        self.n_continuous = int(building_meta_data['n_continuous_action'])
        self.n_discrete = int(building_meta_data['n_discrete_action'])
        self.field_names = [str(x) for x in building_meta_data['observation_field_names']]
        self.field_index = {name: i for i, name in enumerate(self.field_names)}
        self.obs_norm = building_meta_data.get('observation_normalization_metadata', {})
        self.action_norm = building_meta_data.get('action_normalization_metadata', {})
        self.temp_indices = list(building_meta_data['temp_indices'])
        self.pressure_indices = list(building_meta_data['pressure_indices'])
        self.recirculation_indices = list(building_meta_data['recirculation_indices'])
        self.water_indices = list(building_meta_data['water_indices'])
        self.num_ahus = len(self.temp_indices)
        zone_actions = building_meta_data['zone_action_indices']
        self.zone_ids = sorted(zone_actions, key=self._natural_key)
        self.n_zones = len(self.zone_ids)
        self.zone_number = {name: i for i, name in enumerate(self.zone_ids)}
        self.zone_obs = np.array([
            self.field_index.get(f'{z}_zone_air_temperature_sensor', -1) for z in self.zone_ids
        ], dtype=int)
        self.zone_low_obs = np.array([
            self.field_index.get(f'{z}_zone_air_heating_temperature_setpoint_sensor', -1) for z in self.zone_ids
        ], dtype=int)
        self.zone_high_obs = np.array([
            self.field_index.get(f'{z}_zone_air_cooling_temperature_setpoint_sensor', -1) for z in self.zone_ids
        ], dtype=int)
        self.zone_flow_obs = np.array([
            self.field_index.get(f'{z}_supply_air_flowrate_setpoint', -1) for z in self.zone_ids
        ], dtype=int)
        self.damper_indices = np.array([
            zone_actions[z]['supply_air_damper_percentage_command'] for z in self.zone_ids
        ], dtype=int)
        self.valve_indices = np.array([
            zone_actions[z]['reheat_valve_percentage_command'] for z in self.zone_ids
        ], dtype=int)
        assignments = building_meta_data.get('ahu_zone_assignments', building_meta_data.get('aha_zone_assignments', {}))
        self.ahu_ids = sorted(assignments, key=self._natural_key)
        self.groups = []
        self.zone_ahu = np.zeros(self.n_zones, dtype=int)
        self.outside_obs = []
        for i, ahu in enumerate(self.ahu_ids):
            group = np.array([
                self.zone_number[str(room).replace('room_', 'VAV_')]
                for room in assignments[ahu]
                if str(room).replace('room_', 'VAV_') in self.zone_number
            ], dtype=int)
            self.groups.append(group)
            self.zone_ahu[group] = i
            self.outside_obs.append(self._find_field(
                f'ahs_{ahu}_outside_air_temperature_sensor',
                f'{ahu}_outside_air_temperature_sensor',
                suffix='outside_air_temperature_sensor'))
        self.water_sensor = self._find_field(suffix='supply_water_temperature_sensor')
        self.comfort_now_index = self.field_index.get('comfort_mode_now')
        self.comfort_soon_index = self.field_index.get('comfort_mode_soon')
        self.occupancy_index = self.field_index.get('num_occupants')
        self.comfort_band = tuple(map(float, building_meta_data['comfort_temp_window']))
        self.eco_band = tuple(map(float, building_meta_data['eco_temp_window']))
        self.start_hour = float(building_meta_data.get('morning_start_hour', 6))
        self.end_hour = float(building_meta_data.get('evening_start_hour', 19))
        self.zone_area, self.zone_ua = self._geometry_priors(building_meta_data.get('zone_map'))
        effective_capacity = 3800.0 * self.zone_area + 21000.0 * np.sqrt(self.zone_area)
        self.initial_gain = np.clip(300000.0 / effective_capacity, 0.08, 2.5)
        self.reset()

    @staticmethod
    def _natural_key(value):
        return tuple(int(x) if x.isdigit() else x for x in re.split(r'(\d+)', str(value)))

    def _find_field(self, *exact, suffix=''):
        for name in exact:
            if name in self.field_index:
                return self.field_index[name]
        if suffix:
            for name, index in self.field_index.items():
                if name.endswith(suffix):
                    return index
        return None

    def _geometry_priors(self, zone_map):
        areas = np.full(self.n_zones, 55.0)
        ua = np.full(self.n_zones, 18.0)
        if zone_map is None:
            return areas, ua
        grid = np.asarray(zone_map)
        if grid.ndim != 2 or grid.size > 5000000:
            return areas, ua
        height, width = grid.shape
        inside = (grid == 0).ravel()
        visited = np.zeros(grid.size, dtype=bool)
        flat = grid.ravel()
        counts, external_edges = [], []
        for start in np.flatnonzero(inside):
            if visited[start]:
                continue
            stack = [int(start)]
            visited[start] = True
            count, edge = 0, 0
            while stack:
                p = stack.pop()
                count += 1
                row, col = divmod(p, width)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    q = nr * width + nc
                    if inside[q] and not visited[q]:
                        visited[q] = True
                        stack.append(q)
                    elif flat[q] == 2:
                        edge += 1
                    elif flat[q] == 1:
                        rr, cc = row + 2 * dr, col + 2 * dc
                        if 0 <= rr < height and 0 <= cc < width and grid[rr, cc] == 2:
                            edge += 1
            counts.append(count)
            external_edges.append(edge)
        if len(counts) != self.n_zones:
            return areas, ua
        for j, name in enumerate(self.zone_ids):
            match = re.search(r'(\d+)$', name)
            label = int(match.group(1)) - 1 if match else j
            if 0 <= label < len(counts):
                areas[j] = max(1.0, 0.04 * counts[label])
                ua[j] = max(0.5, external_edges[label] * 0.2 * 3.0 * 1.3)
        return areas, ua

    def reset(self):
        self.step_count = 0
        self.history = deque(maxlen=3)
        self.previous_observed = None
        self.slow_temperature = None
        self.residual = np.zeros(self.n_zones)
        self.model_error = np.full(self.n_zones, 0.08)
        self.theta = np.zeros((self.n_zones, 6))
        self.theta[:, 0] = self.initial_gain * self.zone_area * 2.0 / 1000.0
        self.theta[:, 1] = self.initial_gain * self.zone_ua * 20.0 / 1000.0
        self.theta[:, 2] = 0.18
        self.theta[:, 3] = 0.08
        self.theta[:, 4] = self.initial_gain * self.zone_area * 8.0 / 1000.0
        self.theta[:, 5] = self.initial_gain * self.parameters['response_scale']
        self.theta[:, :2] *= self.parameters['response_scale']
        self.theta[:, 4] *= self.parameters['response_scale']
        prior = np.array([0.12, 0.20, 0.15, 0.15, 0.16, 0.8])
        self.covariance = np.repeat(np.diag(prior)[None, :, :], self.n_zones, axis=0)
        self.last_heating_step = -1000
        self.last_action = None

    def _temperature(self, obs, index, kind, fallback):
        if index is None:
            return fallback
        value = float(obs[index])
        if 150.0 < value < 450.0:
            return value
        norm = self.obs_norm.get(kind, {})
        mean = float(norm.get('sample_mean', fallback))
        std = float(norm.get('sample_std', math.sqrt(norm.get('sample_variance', 1.0))))
        value = value * std + mean
        return value if np.isfinite(value) and 150.0 < value < 450.0 else fallback

    def _zone_temperatures(self, obs):
        norm = self.obs_norm.get('zone_air_temperature_sensor', {})
        mean = float(norm.get('sample_mean', 295.0))
        std = float(norm.get('sample_std', math.sqrt(norm.get('sample_variance', 25.0))))
        raw = obs[self.zone_obs].astype(float)
        value = np.where((raw > 150.0) & (raw < 450.0), raw, raw * std + mean)
        fallback = self.previous_observed if self.previous_observed is not None else mean
        return np.where(np.isfinite(value) & (value > 150.0) & (value < 450.0), value, fallback)

    def _hour(self, obs):
        sine = self.field_index.get('hod_sin_000')
        cosine = self.field_index.get('hod_cos_000')
        if sine is None or cosine is None:
            return (self.start_hour + self.step_count / 12.0) % 24.0
        return (math.atan2(float(obs[sine]), float(obs[cosine])) % (2.0 * math.pi)) * 12.0 / math.pi

    def _band(self, obs, temperature):
        now = bool(obs[self.comfort_now_index] > 0.5) if self.comfort_now_index is not None else (
            self.start_hour <= self._hour(obs) < self.end_hour)
        soon = bool(obs[self.comfort_soon_index] > 0.5) if self.comfort_soon_index is not None else now
        default = self.comfort_band if now else self.eco_band
        low = np.full(self.n_zones, default[0])
        high = np.full(self.n_zones, default[1])
        valid = (self.zone_low_obs >= 0) & (self.zone_high_obs >= 0)
        if np.any(valid):
            sensor_low = obs[self.zone_low_obs[valid]].astype(float)
            sensor_high = obs[self.zone_high_obs[valid]].astype(float)
            good = (sensor_low > 150.0) & (sensor_high < 450.0) & (sensor_high > sensor_low)
            indices = np.flatnonzero(valid)[good]
            low[indices] = sensor_low[good]
            high[indices] = sensor_high[good]
        if now:
            low = np.maximum(low, self.comfort_band[0])
            high = np.minimum(high, self.comfort_band[1])
        elif soon:
            hour = self._hour(obs)
            remaining = max(0.0, (self.start_hour - hour) % 24.0)
            progress = np.clip(1.0 - remaining / max(0.1, self.parameters['precondition_hours']), 0.0, 1.0)
            low = self.eco_band[0] + progress * (self.comfort_band[0] - self.eco_band[0]) + np.zeros(self.n_zones)
            high = self.eco_band[1] + progress * (self.comfort_band[1] - self.eco_band[1]) + np.zeros(self.n_zones)
        else:
            low[:] = self.parameters['standby_minimum']
            high[:] = self.parameters['standby_maximum']
        margin = self.parameters['comfort_margin'] if (now or soon) else 0.0
        low += margin
        high -= margin
        recovery = max(1.0, self.parameters['recovery_steps'])
        low = np.minimum(low, temperature + np.maximum((low - temperature) / recovery, 0.0))
        high = np.maximum(high, temperature - np.maximum((temperature - high) / recovery, 0.0))
        return low, high, now

    def _group_temperature(self, temperature):
        means = np.zeros(self.n_zones)
        for group in self.groups:
            if group.size:
                means[group] = np.average(temperature[group], weights=self.zone_area[group])
        return means

    def _features(self, temperature, outside, mode, slow):
        return np.column_stack((
            np.ones(self.n_zones),
            np.clip((outside - temperature) / 20.0, -5.0, 5.0),
            np.clip((slow - temperature) / 5.0, -5.0, 5.0),
            np.clip((self._group_temperature(temperature) - temperature) / 5.0, -5.0, 5.0),
            np.full(self.n_zones, float(mode)),
        ))

    def _learn(self, observed):
        delay = int(self.parameters['observation_delay'])
        if self.previous_observed is None or len(self.history) <= delay:
            return
        applied = list(self.history)[-(delay + 1)]
        old = self.previous_observed
        power = (applied['conductance'] * (applied['supply'] - observed) + applied['reheat']) / 1000.0
        features = self._features(old, applied['outside'], applied['mode'], applied['slow'])
        x = np.column_stack((features, power))
        delta = np.clip(observed - old, -15.0, 15.0)
        prediction = np.einsum('ij,ij->i', self.theta, x)
        remaining = delta - prediction
        rate = float(self.parameters['disturbance_rate'])
        self.model_error = 0.94 * self.model_error + 0.06 * np.minimum(np.abs(remaining - self.residual), 2.0)
        self.residual = np.clip((1.0 - rate) * self.residual + rate * remaining, -5.0, 5.0)

    def _free_drift(self, temperature, outside, mode):
        features = self._features(temperature, outside, mode, self.slow_temperature)
        return np.clip(np.einsum('ij,ij->i', self.theta[:, :5], features) + self.residual, -5.0, 5.0)

    def _native_action(self, kind, value, low, high):
        spec = self.action_norm.get(kind)
        if spec is None:
            spec = next((v for k, v in self.action_norm.items() if str(k).endswith(kind)), None)
        if spec:
            low = float(spec.get('min_native_value', low))
            high = float(spec.get('max_native_value', high))
            out_low = float(spec.get('min_normalized_value', -1.0))
            out_high = float(spec.get('max_normalized_value', 1.0))
        else:
            out_low, out_high = -1.0, 1.0
        return np.clip(out_low + (np.asarray(value) - low) * (out_high - out_low) / max(high - low, 1e-12), out_low, out_high)

    def _choose_ahu(self, group, temperature, free, low, high, max_flow, outside, water):
        t, d, lo, hi = temperature[group], free[group], low[group], high[group]
        response = self.theta[group, 5]
        capacity = max_flow[group]
        inside = float(np.average(t, weights=self.zone_area[group]))
        targets = np.array([285.0, 287.0, 289.0, 291.0, 293.0, 295.0, 297.0, 299.0, 302.0, 305.0])
        targets = np.concatenate((targets, [float(np.clip(np.min(hi) - 1.5, 285.0, 305.0))]))
        setpoint = np.repeat(targets, 2)
        recirc = np.full_like(setpoint, self.MAX_RECIRCULATION)
        if outside < inside - 0.5:
            recirc[1::2] = np.clip((targets - outside) / max(inside - outside, 1e-6), 0.0, self.MAX_RECIRCULATION)
        mixed = recirc * inside + (1.0 - recirc) * outside
        supply = np.minimum(mixed, setpoint)
        sa = supply[:, None]
        cool_need = np.maximum(t + d - hi, 0.0) * 1000.0 / response
        heat_need = np.maximum(lo - t - d, 0.0) * 1000.0 / response
        cooling_flow = cool_need[None, :] / (self.AIR_CAPACITY * np.maximum(hi[None, :] - sa, 0.05))
        heating_flow = heat_need[None, :] / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - lo[None, :], 0.5))
        maximum_water_heat = self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * np.maximum(water - sa, 0.0)
        maximum_heating_flow = maximum_water_heat / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - sa, 0.1))
        heating_flow = np.minimum(heating_flow, maximum_heating_flow)
        needed_flow = np.maximum(cooling_flow, heating_flow)
        fraction = np.max(needed_flow / capacity[None, :], axis=1)
        pressure = np.clip(self.DESIGN_PRESSURE * fraction ** 2, self.MIN_PRESSURE, self.DESIGN_PRESSURE)
        pressure_ratio = np.sqrt(pressure / self.DESIGN_PRESSURE)
        dampers = np.clip(needed_flow / (capacity[None, :] * pressure_ratio[:, None]), self.MIN_DAMPER, 1.0)
        flow = dampers * capacity[None, :] * pressure_ratio[:, None]
        conductance = flow * self.AIR_CAPACITY
        water_gap = np.maximum(water - sa, 0.0)
        wanted_reheat = np.maximum(
            (lo[None, :] - t[None, :] - d[None, :]) * 1000.0 / response[None, :] + conductance * (lo[None, :] - sa), 0.0)
        reheat_limit = np.minimum(
            conductance * np.maximum(self.MAX_DISCHARGE_K - sa, 0.0),
            self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * water_gap)
        reheat = np.minimum(wanted_reheat, reheat_limit)
        predicted = (t[None, :] + d[None, :] + response[None, :] * (conductance * sa + reheat) / 1000.0) / (
            1.0 + response[None, :] * conductance / 1000.0)
        violation = np.maximum(lo[None, :] - predicted, 0.0) + np.maximum(predicted - hi[None, :], 0.0)
        thermal_penalty = np.sum(violation + 2.0 * violation ** 2, axis=1)
        total_flow = flow.sum(axis=1)
        fan = total_flow * pressure * (2.0 - recirc) / 0.9
        chiller = total_flow * self.AIR_CAPACITY * np.maximum(mixed - supply, 0.0) / 3.5
        gas = reheat.sum(axis=1)
        objective = fan + chiller + self.parameters['gas_cost_ratio'] * gas
        objective += self.parameters['comfort_cost'] * thermal_penalty
        off_prediction = t + d
        off_violation = np.maximum(lo - off_prediction, 0.0) + np.maximum(off_prediction - hi, 0.0)
        off_objective = self.parameters['comfort_cost'] * np.sum(off_violation + 2.0 * off_violation ** 2)
        best = int(np.argmin(objective))
        if off_objective <= objective[best]:
            return {'run': 0, 'pressure': 0.0, 'setpoint': 305.0, 'recirc': 0.7,
                    'damper': np.full(group.size, self.MIN_DAMPER), 'conductance': np.zeros(group.size),
                    'supply': np.full(group.size, mixed[best]), 'reheat': np.zeros(group.size)}
        return {'run': 1, 'pressure': float(pressure[best]), 'setpoint': float(setpoint[best]),
                'recirc': float(recirc[best]), 'damper': dampers[best],
                'conductance': conductance[best], 'supply': np.full(group.size, supply[best]),
                'reheat': reheat[best]}

    def get_action(self, obs):
        obs = np.asarray(obs, dtype=float).reshape(-1)
        observed = self._zone_temperatures(obs)
        if self.slow_temperature is None:
            self.slow_temperature = observed.copy()
        self._learn(observed)
        outside_ahu = np.array([
            self._temperature(obs, index, 'outside_air_temperature_sensor', 290.0)
            for index in self.outside_obs
        ])
        outside = outside_ahu[self.zone_ahu]
        mode = bool(obs[self.comfort_now_index] > 0.5) if self.comfort_now_index is not None else True
        temperature = observed.copy()
        if self.history and int(self.parameters['observation_delay']) == 1:
            last = self.history[-1]
            drift = self._free_drift(observed, last['outside'], last['mode'])
            gain = self.theta[:, 5] / 1000.0
            temperature = (observed + drift + gain * (last['conductance'] * last['supply'] + last['reheat'])) / (
                1.0 + gain * last['conductance'])
            temperature = np.clip(temperature, observed - 8.0, observed + 8.0)
        low, high, mode = self._band(obs, temperature)
        free = self._free_drift(temperature, outside, mode)
        max_flow = np.full(self.n_zones, 2.0)
        valid = self.zone_flow_obs >= 0
        if np.any(valid):
            flow_reading = obs[self.zone_flow_obs[valid]]
            max_flow[valid] = np.where(np.isfinite(flow_reading) & (flow_reading > 0.0), flow_reading, 2.0)
        max_flow = np.clip(max_flow, 0.01, 100.0)
        water_observed = self._temperature(obs, self.water_sensor, 'supply_water_temperature_sensor', 340.0)
        heat_demand = np.any(temperature + free < low)
        if heat_demand:
            self.last_heating_step = self.step_count
        recent_heat = self.step_count - self.last_heating_step < 24
        water_target = self.parameters['hot_water_target'] if recent_heat else 310.0
        water_available = water_observed
        continuous = np.full(self.n_continuous, -1.0, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        conductance = np.zeros(self.n_zones)
        supply = outside.copy()
        reheat = np.zeros(self.n_zones)
        for i, group in enumerate(self.groups):
            if not group.size:
                continue
            choice = self._choose_ahu(group, temperature, free, low, high, max_flow,
                                      float(outside_ahu[i]), float(water_available))
            discrete[i] = choice['run']
            continuous[self.temp_indices[i]] = self._native_action(
                f'{self.ahu_ids[i]}_supply_air_temperature_setpoint', choice['setpoint'], 285.0, 305.0)
            continuous[self.pressure_indices[i]] = self._native_action(
                f'{self.ahu_ids[i]}_supply_air_static_pressure_setpoint', choice['pressure'], 0.0, 2000.0)
            continuous[self.recirculation_indices[i]] = self._native_action(
                f'{self.ahu_ids[i]}_recirculation_percentage_command', choice['recirc'], 0.0, 1.0)
            continuous[self.damper_indices[group]] = self._native_action(
                'supply_air_damper_percentage_command', choice['damper'], 0.0, 1.0)
            conductance[group] = choice['conductance']
            supply[group] = choice['supply']
            reheat[group] = choice['reheat']
        if np.any(reheat > 10.0):
            self.last_heating_step = self.step_count
            water_target = self.parameters['hot_water_target']
        valve_need = reheat / (self.WATER_VALVE_CAPACITY * np.maximum(water_available - supply, 1.0))
        pressure_water = float(np.clip(np.max(valve_need) ** 2 * 1.05, 0.0, 20.0))
        if pressure_water > 1e-8:
            valves = np.clip(valve_need / math.sqrt(pressure_water), 0.0, 1.0)
            discrete[-1] = 1
        else:
            valves = np.zeros(self.n_zones)
        continuous[self.valve_indices] = self._native_action('reheat_valve_percentage_command', valves, 0.0, 1.0)
        if self.water_indices:
            continuous[self.water_indices[0]] = self._native_action(
                'supply_water_temperature_setpoint', water_target, 310.0, 350.0)
        if len(self.pressure_indices) > self.num_ahus:
            continuous[self.pressure_indices[-1]] = self._native_action(
                'differential_pressure_setpoint', pressure_water, 0.0, 20.0)
        actual_reheat = np.minimum(
            self.WATER_VALVE_CAPACITY * math.sqrt(pressure_water) * valves * np.maximum(water_available - supply, 0.0),
            conductance * np.maximum(self.MAX_DISCHARGE_K - supply, 0.0))
        self.history.append({'conductance': conductance, 'supply': supply,
                             'reheat': actual_reheat, 'outside': outside.copy(),
                             'mode': mode, 'slow': self.slow_temperature.copy()})
        self.slow_temperature += (temperature - self.slow_temperature) / 48.0
        self.previous_observed = observed.copy()
        self.step_count += 1
        action = {'continuous_action': np.clip(np.nan_to_num(continuous, nan=-1.0), -1.0, 1.0).astype(np.float32),
                  'discrete_action': discrete.astype(np.int32)}
        self.last_action = action
        return action


class GuardedJointThermalPolicy(TransferPolicy):
    name = 'facilities_v40_joint_thermal'
    def _zone_temperatures(self, obs):
        value = super()._zone_temperatures(obs)
        previous = self.previous_observed if self.previous_observed is not None else np.mean(self.comfort_band)
        return np.where(np.isfinite(value) & (value >= 150.) & (value <= 400.), value, previous)


def bounded_rls(theta, covariance, x, target, lower, upper, forgetting=.997):
    px = np.einsum('nij,nj->ni', covariance, x)
    denominator = forgetting + np.einsum('ni,ni->n', x, px)
    gain = px / np.maximum(denominator[:, None], 1e-9)
    innovation = np.clip(target - np.einsum('ni,ni->n', theta, x), -2., 2.)
    fitted = np.clip(theta + gain * innovation[:, None], lower, upper)
    updated = (covariance - np.einsum('ni,nj->nij', gain, px)) / forgetting
    updated = .5 * (updated + updated.transpose(0, 2, 1))
    return fitted, updated


class IdentifiedJointThermalPolicy(GuardedJointThermalPolicy):
    name = 'facilities_v41_identified_thermal'
    def reset(self):
        super().reset()
        self._thermal_prior = self.theta[:, 5].copy()
    def _learn(self, observed):
        delay = int(self.parameters['observation_delay'])
        if self.previous_observed is None or len(self.history) <= delay:
            return
        applied = list(self.history)[-(delay + 1)]
        power = (applied['conductance'] * (applied['supply'] - observed) + applied['reheat']) / 1000.
        features = self._features(self.previous_observed, applied['outside'], applied['mode'], applied['slow'])
        x = np.column_stack((features, power))
        delta = observed - self.previous_observed
        valid = np.isfinite(x).all(axis=1) & np.isfinite(delta) & (np.abs(delta) <= 3.)
        lower = np.zeros_like(self.theta)
        upper = np.ones_like(self.theta) * np.array([2., 5., 2., 2., 2., 1.])
        lower[:, 0] = -2.
        lower[:, 5] = .5 * self._thermal_prior
        upper[:, 5] = 1.5 * self._thermal_prior
        if np.any(valid):
            fit, cov = bounded_rls(self.theta[valid], self.covariance[valid], x[valid],
                delta[valid] - self.residual[valid], lower[valid], upper[valid],
                float(self.parameters['forgetting_factor']))
            self.theta[valid], self.covariance[valid] = fit, cov
        remaining = np.clip(delta - np.einsum('ni,ni->n', self.theta, x), -5., 5.)
        self.model_error = .94 * self.model_error + .06 * np.minimum(np.abs(remaining - self.residual), 2.)
        rate = float(self.parameters['disturbance_rate'])
        self.residual = np.clip((1 - rate) * self.residual + rate * remaining, -5., 5.)


class ErrorReserveJointThermalPolicy(GuardedJointThermalPolicy):
    name = 'facilities_v42_error_reserve_thermal'
    def _band(self, obs, temperature):
        low, high, mode = super()._band(obs, temperature)
        if mode:
            extra = np.minimum(np.clip(.5 * (self.model_error - .08), 0., .25), .25 * np.maximum(high - low, 0.))
            low, high = low + extra, high - extra
        return low, high, mode


class IdentifiedReservePolicy(IdentifiedJointThermalPolicy, ErrorReserveJointThermalPolicy):
    name = 'identified_reserve'


class CausalIdentifiedReservePolicy(IdentifiedReservePolicy):
    name = 'causal_identified'
    def _learn(self, observed):
        delay = int(self.parameters['observation_delay'])
        if self.previous_observed is None or len(self.history) <= delay:
            return
        applied = list(self.history)[-(delay + 1)]
        old = self.previous_observed
        features = self._features(old, applied['outside'], applied['mode'], applied['slow'])
        drift = np.einsum('ij,ij->i', self.theta[:, :5], features) + self.residual
        gain = self.theta[:, 5] / 1000.0
        prediction = (old + drift + gain * (applied['conductance'] * applied['supply'] + applied['reheat'])) / (
            1.0 + gain * applied['conductance'])
        error = np.where(np.isfinite(observed - prediction), np.minimum(np.abs(observed - prediction), 2.0), 2.0)
        previous_error = self.model_error.copy()
        super()._learn(observed)
        self.model_error = 0.94 * previous_error + 0.06 * error


class WarmStandbyPolicy(CausalIdentifiedReservePolicy):
    name = 'causal_warm_standby'
    HOLD_STEPS = 36 * 12
    def _native_action(self, kind, value, low, high):
        if (kind == 'supply_water_temperature_setpoint'
                and self.step_count - self.last_heating_step < self.HOLD_STEPS):
            value = self.parameters['hot_water_target']
        return super()._native_action(kind, value, low, high)


class WeekendWaterHoldPolicy(WarmStandbyPolicy):
    name = 'warm_hold72'
    HOLD_STEPS = 72 * 12


class DeepFastRecoveryPolicy(WeekendWaterHoldPolicy):
    name = 'hold72_deep_fast_recovery'
    DEFAULTS = dict(WeekendWaterHoldPolicy.DEFAULTS, standby_minimum=274.0, recovery_steps=1.0)


class EmptyDayConsistentPolicy(DeepFastRecoveryPolicy):
    name = 'deep_empty_day_consistent_model'
    EMPTY_CONFIRMATION_HOURS = 2.0
    def reset(self):
        super().reset()
        self.empty_day_relaxed_steps = 0
    def _band(self, obs, temperature):
        low, high, schedule_mode = super()._band(obs, temperature)
        if not schedule_mode or self.occupancy_index is None:
            return low, high, schedule_mode
        occupants = float(obs[self.occupancy_index])
        elapsed = (self._hour(obs) - self.start_hour) % 24.0
        scheduled_duration = (self.end_hour - self.start_hour) % 24.0
        if (not np.isfinite(occupants) or occupants != 0.0
                or not self.EMPTY_CONFIRMATION_HOURS <= elapsed < scheduled_duration):
            return low, high, schedule_mode
        self.empty_day_relaxed_steps += 1
        return (np.full(self.n_zones, self.parameters['standby_minimum']),
                np.full(self.n_zones, self.parameters['standby_maximum']), schedule_mode)
