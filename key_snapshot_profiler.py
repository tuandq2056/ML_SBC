"""Key-Position Counterfactual Snapshot Profiler.

Captures exact simulator state snapshots at pivotal operational moments across a 72h
building run (morning warm-up, peak cooling, mild shoulder economizer, comfort crisis,
empty-day relaxation, 72h weekend boiler hold) and evaluates counterfactual decision
arms from the exact same physical state + RNG seed to reveal AT WHICH POSITION,
WHICH DECISION IS STRONGEST.
"""
import copy
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from physics_foundation import EmptyDayConsistentPolicy
from warp_resip import AHU_ACT_DIM, WarpResiPPolicy

try:
    from thermal_residual_net import ThermalAugmentedPolicy
except Exception:
    ThermalAugmentedPolicy = None

try:
    from smart_buildings.smart_control.simulator.stochastic_convection_simulator import _seed_kernel_rng
except Exception:
    def _seed_kernel_rng(seed):
        return None


KEY_POSITIONS = [
    'POS_1_MORNING_WARMUP',
    'POS_2_PEAK_COOLING',
    'POS_3_MILD_SHOULDER',
    'POS_4_COMFORT_CRISIS',
    'POS_5_EMPTY_DAY_START',
    'POS_6_WEEKEND_WATER_HOLD',
]


def _make_modifier(kind):
    def _mod(step_cnt, z_f, g_f, xb):
        x = xb.copy()
        ahu_active = x[1] > -0.99
        if kind == 'sat_-2K' and ahu_active:
            x[0] = float(np.clip(x[0] - 0.20, -1.0, 1.0))
        elif kind == 'sat_+2K' and ahu_active:
            x[0] = float(np.clip(x[0] + 0.20, -1.0, 1.0))
        elif kind == 'press_-25%' and ahu_active:
            x[1] = float(np.clip(x[1] - 0.25, -1.0, 1.0))
        elif kind == 'press_+25%' and ahu_active:
            x[1] = float(np.clip(x[1] + 0.25, -1.0, 1.0))
        elif kind == 'recirc_eco' and ahu_active:
            x[2] = float(np.clip(x[2] - 0.40, -1.0, 1.0))
        elif kind == 'recirc_max' and ahu_active:
            x[2] = 0.40  # recirc = 0.70
        elif kind == 'water_325K':
            x[3] = -0.25
        elif kind == 'water_310K':
            x[3] = -1.00
        return x
    return _mod


def build_default_arms(thermal_nets=None, alpha_nn=0.5, iql_actors=None, dev=None):
    """Build standard physical perturbation arms plus optional trained Stage 1 / Stage 2 models."""
    arms = {
        'base': {'type': 'warp', 'modifier': None, 'actors': []},
        'sat_-2K': {'type': 'warp', 'modifier': _make_modifier('sat_-2K'), 'actors': []},
        'sat_+2K': {'type': 'warp', 'modifier': _make_modifier('sat_+2K'), 'actors': []},
        'press_-25%': {'type': 'warp', 'modifier': _make_modifier('press_-25%'), 'actors': []},
        'press_+25%': {'type': 'warp', 'modifier': _make_modifier('press_+25%'), 'actors': []},
        'recirc_eco': {'type': 'warp', 'modifier': _make_modifier('recirc_eco'), 'actors': []},
        'recirc_max': {'type': 'warp', 'modifier': _make_modifier('recirc_max'), 'actors': []},
        'water_325K': {'type': 'warp', 'modifier': _make_modifier('water_325K'), 'actors': []},
        'water_310K': {'type': 'warp', 'modifier': _make_modifier('water_310K'), 'actors': []},
    }
    if thermal_nets and ThermalAugmentedPolicy is not None:
        arms['stage1_thermal'] = {'type': 'thermal', 'nets': thermal_nets, 'alpha_nn': alpha_nn, 'dev': dev}
    if iql_actors:
        arms['stage2_iql_shield'] = {'type': 'warp', 'modifier': None, 'actors': iql_actors, 'dev': dev}
    return arms


def _copy_policy_state(src, dst, obs_arr):
    for attr in [
        'step_count', 'last_heating_step', 'empty_day_relaxed_steps', '_oat_ema',
    ]:
        if hasattr(src, attr):
            setattr(dst, attr, copy.deepcopy(getattr(src, attr)))
    for arr_attr in [
        'previous_observed', 'slow_temperature', 'residual', 'model_error',
        'theta', 'covariance', '_prev_delta_x',
    ]:
        if hasattr(src, arr_attr) and hasattr(dst, arr_attr):
            val = getattr(src, arr_attr)
            setattr(dst, arr_attr, None if val is None else np.array(val, copy=True))
    dst.history = copy.deepcopy(src.history)
    dst.last_action = copy.deepcopy(src.last_action)
    if hasattr(dst, '_extract_ctx'):
        dst._cur_ctx = dst._extract_ctx(obs_arr)
        dst._prev_ctx = dst._cur_ctx


def _step_components(env):
    try:
        rr = env.reward_function.compute_reward(env.building.reward_info)
        return (
            float(rr.normalized_productivity_regret),
            float(rr.normalized_energy_cost),
            float(rr.normalized_carbon_emission),
        )
    except Exception:
        return 0.0, 0.0, 0.0


def _instantiate_arm_policy(arm_spec, meta, action_spec, src_policy, obs_arr):
    if arm_spec['type'] == 'thermal' and ThermalAugmentedPolicy is not None:
        pol = ThermalAugmentedPolicy(
            meta, action_spec, thermal_nets=arm_spec['nets'],
            alpha_nn=arm_spec['alpha_nn'], dev=arm_spec.get('dev'),
        )
    else:
        pol = WarpResiPPolicy(
            meta, action_spec,
            actors=arm_spec.get('actors', []),
            action_modifier=arm_spec.get('modifier', None),
            dev=arm_spec.get('dev'),
            comfort_shield=True,
        )
    _copy_policy_state(src_policy, pol, obs_arr)
    return pol


def _run_branch_from_snapshot(snap, arm_spec, horizon, load_env_fn, create_meta_fn, sc_path, branch_seed):
    random.setstate(snap['py_rng'])
    np.random.set_state(snap['np_rng'])
    _seed_kernel_rng(branch_seed)

    clone_mode = 'deepcopy'
    try:
        env_b = copy.deepcopy(snap['env'])
        obs_b = np.array(snap['obs'], copy=True)
    except Exception:
        clone_mode = 'replay'
        env_b = load_env_fn(sc_path / 'config.gin')
        obs_b, _ = env_b.reset()
        for act in snap['action_history']:
            obs_b, _, _, _, _ = env_b.step(act)
        obs_b = np.asarray(obs_b, dtype=float).reshape(-1)
        _seed_kernel_rng(branch_seed)

    meta_b = create_meta_fn(env_b, sc_path)
    pol_b = _instantiate_arm_policy(arm_spec, meta_b, env_b.action_spec(), snap['policy'], obs_b)

    tot_r, tot_reg, tot_en, tot_cb = 0.0, 0.0, 0.0, 0.0
    for _ in range(horizon):
        act = pol_b.get_action(obs_b)
        obs_b, r, _, _, _ = env_b.step(act)
        obs_b = np.asarray(obs_b, dtype=float).reshape(-1)
        reg, en, cb = _step_components(env_b)
        tot_r += float(r)
        tot_reg += reg
        tot_en += en
        tot_cb += cb
    return {
        'total_rew': tot_r,
        'regret': tot_reg,
        'energy': tot_en,
        'carbon': tot_cb,
        'clone_mode': clone_mode,
    }


def collect_scenario_snapshots(sc_path, load_env_fn, create_meta_fn, steps=864, horizon=24, warmup=24):
    """Roll out Base once cleanly and snapshot env+policy state at up to 6 key positions."""
    env = load_env_fn(sc_path / 'config.gin')
    obs, _ = env.reset()
    meta = create_meta_fn(env, sc_path)
    pol = WarpResiPPolicy(meta, env.action_spec(), actors=[])

    captured = {}
    action_history = []
    base_rew_trace = []
    prev_ahu_run = False

    for st in range(steps):
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        hour = pol._hour(obs_arr)
        act_pre = pol.get_action(obs_arr)
        _, g_f, xb, _ = pol._last_transition
        ahu_run = bool(np.any(act_pre['discrete_action'][:-1] == 1))
        water_target = 330.0 + 20.0 * float(xb[3])

        obs_next, r, _, _, _ = env.step(act_pre)
        reg, _, _ = _step_components(env)
        base_rew_trace.append(float(r))
        action_history.append(copy.deepcopy(act_pre))

        next_st = st + 1
        if warmup <= next_st <= steps - horizon - 1:
            obs_n_arr = np.asarray(obs_next, dtype=float).reshape(-1)
            next_hour = pol._hour(obs_n_arr)
            c_now_n = bool(obs_n_arr[pol.comfort_now_index] > 0.5) if pol.comfort_now_index is not None else bool(g_f[4] > 0.5)
            c_soon_n = bool(obs_n_arr[pol.comfort_soon_index] > 0.5) if pol.comfort_soon_index is not None else bool(g_f[5] > 0.5)
            oat_K_n = float(np.mean([
                pol._temperature(obs_n_arr, idx, 'outside_air_temperature_sensor', 290.0)
                for idx in pol.outside_obs
            ]))
            oat_C_n = oat_K_n - 273.15
            occ_raw_n = float(obs_n_arr[pol.occupancy_index]) if pol.occupancy_index is not None else 0.0
            occ_norm_n = math.log1p(max(occ_raw_n, 0.0)) / 5.0 if np.isfinite(occ_raw_n) else 0.0
            mean_T_n = float(np.mean(pol._zone_temperatures(obs_n_arr)))

            triggers = []
            if 'POS_1_MORNING_WARMUP' not in captured and (
                (not c_now_n and c_soon_n) or ((not prev_ahu_run) and ahu_run and 4.0 <= hour <= 10.5)
            ):
                triggers.append('POS_1_MORNING_WARMUP')
            if 'POS_2_PEAK_COOLING' not in captured and (
                c_now_n and occ_norm_n > 0.05 and ahu_run and 11.0 <= next_hour <= 16.5 and oat_C_n >= 18.0
            ):
                triggers.append('POS_2_PEAK_COOLING')
            if 'POS_3_MILD_SHOULDER' not in captured and (
                c_now_n and ahu_run and (mean_T_n - oat_K_n >= 1.5)
            ):
                triggers.append('POS_3_MILD_SHOULDER')
            if 'POS_4_COMFORT_CRISIS' not in captured and reg > 5e-4:
                triggers.append('POS_4_COMFORT_CRISIS')
            if 'POS_5_EMPTY_DAY_START' not in captured and getattr(pol, 'empty_day_relaxed_steps', 0) == 1:
                triggers.append('POS_5_EMPTY_DAY_START')
            if 'POS_6_WEEKEND_WATER_HOLD' not in captured and (
                (not ahu_run) and water_target >= 345.0 and (st - pol.last_heating_step >= 72)
            ):
                triggers.append('POS_6_WEEKEND_WATER_HOLD')

            if triggers:
                snap_obj = {
                    'step': next_st,
                    'hour': float(next_hour),
                    'oat_C': float(oat_C_n),
                    'occ_norm': float(occ_norm_n),
                    'obs': np.array(obs_n_arr, copy=True),
                    'policy': copy.deepcopy(pol),
                    'env': None,
                    'action_history': copy.deepcopy(action_history),
                    'py_rng': random.getstate(),
                    'np_rng': np.random.get_state(),
                }
                try:
                    snap_obj['env'] = copy.deepcopy(env)
                except Exception:
                    snap_obj['env'] = None
                for pos in triggers:
                    captured[pos] = snap_obj

        prev_ahu_run = ahu_run
        obs = obs_next

    for pos, sn in captured.items():
        s0 = sn['step']
        sn['uninterrupted_base_rew'] = float(sum(base_rew_trace[s0: s0 + horizon]))
    return captured


def profile_key_snapshots(scenario_paths, load_env_fn, create_meta_fn,
                          steps=864, horizon=24, arms=None, arm_builder=None,
                          out_dir=None, seed=42):
    """Run branching counterfactual audit across all captured key positions."""
    if arms is None and arm_builder is None:
        arms = build_default_arms()

    rows = []
    parity_errs = []
    clone_modes = set()
    all_arm_names = []

    for sc_idx, sc_path in enumerate(scenario_paths):
        bldg = sc_path.parts[-2]
        sc_key = f'{bldg}/{sc_path.parts[-1]}'
        sc_arms = arm_builder(bldg) if arm_builder is not None else arms
        for k in sc_arms:
            if k not in all_arm_names:
                all_arm_names.append(k)

        captured = collect_scenario_snapshots(
            sc_path, load_env_fn, create_meta_fn, steps=steps, horizon=horizon
        )
        print(f'[{sc_idx+1}/{len(scenario_paths)}] {sc_key}: captured {len(captured)}/{len(KEY_POSITIONS)} key positions {list(captured.keys())}')

        for pos, snap in captured.items():
            b_seed = seed * 1000 + sc_idx * 31 + snap['step']
            base_res = _run_branch_from_snapshot(
                snap, sc_arms['base'], horizon, load_env_fn, create_meta_fn, sc_path, b_seed
            )
            clone_modes.add(base_res['clone_mode'])
            parity_errs.append(abs(base_res['total_rew'] - snap['uninterrupted_base_rew']))

            for arm_name, arm_spec in sc_arms.items():
                res = base_res if arm_name == 'base' else _run_branch_from_snapshot(
                    snap, arm_spec, horizon, load_env_fn, create_meta_fn, sc_path, b_seed
                )
                rows.append({
                    'building': bldg,
                    'scenario': sc_key,
                    'position': pos,
                    'step': snap['step'],
                    'hour': round(snap['hour'], 2),
                    'oat_C': round(snap['oat_C'], 2),
                    'occ_norm': round(snap['occ_norm'], 3),
                    'arm': arm_name,
                    'arm_rew': res['total_rew'],
                    'base_rew': base_res['total_rew'],
                    'gain_total': res['total_rew'] - base_res['total_rew'],
                    'comfort_saved': base_res['regret'] - res['regret'],
                    'energy_saved': base_res['energy'] - res['energy'],
                    'carbon_saved': base_res['carbon'] - res['carbon'],
                })

    df_detail = pd.DataFrame(rows)
    if df_detail.empty:
        print('Khong thu thap duoc snapshot nao!')
        return df_detail, pd.DataFrame()

    max_parity = float(max(parity_errs)) if parity_errs else 0.0
    print(f'\nSnapshot Fidelity Check ({clone_modes}): max |cloned_base - orig_base| = {max_parity:.6f}')

    mat_gain = df_detail.pivot_table(index='position', columns='arm', values='gain_total', aggfunc='mean')
    arm_cols = [c for c in all_arm_names if c in mat_gain.columns and c != 'base']
    mat_gain = mat_gain.reindex(index=[p for p in KEY_POSITIONS if p in mat_gain.index], columns=arm_cols)

    df_summary = df_detail[df_detail['arm'] != 'base'].groupby(['position', 'arm']).agg(
        n=('gain_total', 'count'),
        gain_mean=('gain_total', 'mean'),
        win_pct=('gain_total', lambda s: float((s > 1e-6).mean() * 100.0)),
        comfort_saved=('comfort_saved', 'mean'),
        energy_saved=('energy_saved', 'mean'),
        carbon_saved=('carbon_saved', 'mean'),
    ).reset_index()

    print('\n=== MA TRAN QUYET DINH TAI CAC VI TRI QUAN TRONG (Mean 2h Gain vs Base) ===')
    print(mat_gain.round(5).to_string())

    print('\n=== TOP QUYET DINH MANH NHAT CHO TUNG VI TRI ===')
    for pos in mat_gain.index:
        sub = df_summary[df_summary['position'] == pos].sort_values('gain_mean', ascending=False)
        b = sub.iloc[0]
        w = sub.iloc[-1]
        print(
            f'  {pos:25s} (n={int(b["n"])}): MANH NHAT = {b["arm"]:16s} '
            f'(gain={b["gain_mean"]:+.5f}, win={b["win_pct"]:5.1f}%, cmf={b["comfort_saved"]:+.5f}, eng={b["energy_saved"]:+.5f}) '
            f'| TE NHAT = {w["arm"]} ({w["gain_mean"]:+.5f})'
        )

    if out_dir is not None:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        df_detail.to_csv(out_p / 'key_snapshots_detail.csv', index=False)
        mat_gain.to_csv(out_p / 'key_snapshots_matrix.csv')
        df_summary.to_csv(out_p / 'key_snapshots_summary.csv', index=False)

        fig, ax = plt.subplots(figsize=(max(9, len(arm_cols) * 1.15), 5.2))
        vals = mat_gain.fillna(0.0).to_numpy()
        vmax = max(1e-4, float(np.percentile(np.abs(vals), 95)))
        im = ax.imshow(vals, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(arm_cols)))
        ax.set_xticklabels(arm_cols, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(np.arange(len(mat_gain.index)))
        ax.set_yticklabels(mat_gain.index, fontsize=9)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = mat_gain.iloc[i, j]
                txt = 'nan' if pd.isna(v) else f'{v:+.4f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=8, color='black')
        plt.colorbar(im, ax=ax, label='Mean 2h Reward Gain vs Base (>0 = Thang Base)')
        ax.set_title('Key-Position x Decision Gain Matrix (Exact State Branching)')
        plt.tight_layout()
        plt.savefig(out_p / 'key_snapshots_heatmap.png', dpi=150)
        plt.show()

    return df_detail, mat_gain
