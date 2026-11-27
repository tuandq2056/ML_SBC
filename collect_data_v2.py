"""SOTA Data Collection & Quality Audit Engine (v2) — Track Data Only (Model-Neutral).

Addresses 6 empirical/SOTA gaps against Wave 0 (864s single-policy):
1. Multi-Policy Coverage (arXiv:2408.07986): anchor EmptyDay + Orthogonal Explore + PIDWaterReset.
2. Orthogonal Single-Axis Excitation: decouples SAT/Press/Recirc/Water (low VIF / collinearity).
3. POMDP History & Geometry Enrichment: ep_id, step_idx, dow_sin/cos, zone_ua, dT_30m, dOAT_30m, ahu_spread.
4. Full-Week Temporal Realism (default STEPS=2016 = 7 days vs 864 = 3 days from Sun 06:00).
5. BOPTEST KPI Decomposition: separate Comfort Regret, Energy Cost, and Carbon Emission per step.
6. Automated SOTA Data Audit: Temporal drift (0..864 vs 864..2016), Regret Ratio, Violation %, Excitation VIF.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from physics_foundation import EmptyDayConsistentPolicy
from warp_resip import AHU_ACT_DIM, ZONE_FEAT_DIM, WarpResiPPolicy


def _step_kpis(env):
    """Read BOPTEST-style KPI decomposition without mutating simulator accumulators."""
    try:
        rr = env.reward_function.compute_reward(env.building.reward_info)
        return (
            float(rr.normalized_productivity_regret),
            float(rr.normalized_energy_cost),
            float(rr.normalized_carbon_emission),
        )
    except Exception:
        return 0.0, 0.0, 0.0


def _extract_dow(policy, obs_arr):
    s_idx = policy.field_index.get('dow_sin_000')
    c_idx = policy.field_index.get('dow_cos_000')
    if s_idx is not None and c_idx is not None:
        return float(obs_arr[s_idx]), float(obs_arr[c_idx])
    # Fallback cyclic week (period=7*24h=168h, step_count in 5-min ticks)
    ang = 2.0 * math.pi * ((policy.step_count / 12.0) % 168.0) / 168.0
    return math.sin(ang), math.cos(ang)


def _extract_4d_from_action(policy, act, fallback_xb):
    """Project any policy's hybrid action dict into normalized 4D [SAT, Press, Recirc, Water]."""
    cont = np.asarray(act['continuous_action'], dtype=np.float32)
    disc = np.asarray(act['discrete_action'], dtype=np.int32)
    active = [i for i in range(policy.num_ahus) if disc[i] == 1]
    if active:
        weights = np.array([policy.zone_area[policy.groups[i]].sum() for i in active], dtype=np.float64)
        weights /= max(weights.sum(), 1e-8)
        sat_n = float(sum(w * cont[policy.temp_indices[i]] for w, i in zip(weights, active)))
        prs_n = float(sum(w * cont[policy.pressure_indices[i]] for w, i in zip(weights, active)))
        rec_n = float(sum(w * cont[policy.recirculation_indices[i]] for w, i in zip(weights, active)))
    else:
        sat_n, prs_n, rec_n = float(fallback_xb[0]), float(fallback_xb[1]), float(fallback_xb[2])
    wat_n = float(cont[policy.water_indices[0]]) if policy.water_indices else float(fallback_xb[3])
    return np.clip(np.array([sat_n, prs_n, rec_n, wat_n], dtype=np.float32), -1.0, 1.0)


def _ortho_modifier_factory(block_steps=6, rng=None):
    """Orthogonal single-axis perturbation so excitation VIF stays near 1.0."""
    rng = rng or np.random.default_rng(42)
    state = {'axis': 0, 'delta': 0.0, 'tag': 'ortho_sat'}
    axes = ['ortho_sat', 'ortho_press', 'ortho_recirc', 'ortho_water']

    def _mod(step_cnt, z_f, g_f, xb):
        if step_cnt % block_steps == 0:
            ax = int(rng.integers(0, 4))
            state['axis'] = ax
            state['tag'] = axes[ax]
            if ax == 0:
                state['delta'] = float(rng.uniform(-0.25, 0.25))
            elif ax == 1:
                state['delta'] = float(rng.uniform(-0.35, 0.25))
            elif ax == 2:
                state['delta'] = float(rng.uniform(-0.45, 0.30))
            else:
                # Wider downward water exploration during idle/low-heat, gentle during active
                ahu_on = xb[1] > -0.99
                state['delta'] = float(rng.uniform(-0.25, 0.05) if ahu_on else rng.uniform(-0.80, 0.05))
        x = xb.copy()
        ax = state['axis']
        if ax < 3 and xb[1] <= -0.99:
            return x  # AHU off -> keep base AHU values untouched
        x[ax] = float(np.clip(x[ax] + state['delta'], -1.0, 1.0))
        return x

    return _mod, state


def _rollout_episode_v2(sc_path, bldg, sc_key, steps, mode, load_env_fn, create_meta_fn,
                        pid_cls=None, block_steps=6, seed=42):
    """Run one scenario episode under mode in {'anchor', 'ortho_explore', 'pid_water'}."""
    env = load_env_fn(sc_path / 'config.gin')
    obs, _ = env.reset()
    meta = create_meta_fn(env, sc_path)
    rng = np.random.default_rng(seed)

    if mode == 'anchor':
        pol = WarpResiPPolicy(meta, env.action_spec(), actors=[])
        ctrl_pol = None
        state_ref = {'tag': 'anchor_empty_day'}
    elif mode == 'ortho_explore':
        mod_fn, state_ref = _ortho_modifier_factory(block_steps=block_steps, rng=rng)
        pol = WarpResiPPolicy(meta, env.action_spec(), action_modifier=mod_fn)
        ctrl_pol = None
    elif mode == 'pid_water':
        assert pid_cls is not None
        pol = WarpResiPPolicy(meta, env.action_spec(), actors=[])
        ctrl_pol = pid_cls(meta, env.action_spec())
        state_ref = {'tag': 'pid_water'}
    else:
        raise ValueError(f'Unknown mode: {mode}')

    zone_ua_norm = (np.log1p(np.clip(pol.zone_ua, 0.0, 1e5)) / 4.0).astype(np.float32)

    z_l, g_l, xb_l, xe_l = [], [], [], []
    z_ex_l, g_ex_l = [], []
    r_l, reg_l, en_l, cb_l = [], [], [], []
    tags_l, th_rows = [], []

    temp_hist, oat_hist = [], []
    prev_gf, prev_gex, prev_zex = None, None, None

    for st in range(steps):
        obs_arr = np.asarray(obs, dtype=float).reshape(-1)
        act_base_ref = pol.get_action(obs_arr)
        zf, gf, xb, xe = pol._last_transition

        if ctrl_pol is not None:
            act = ctrl_pol.get_action(obs_arr)
            xe = _extract_4d_from_action(pol, act, xb)
        else:
            act = act_base_ref

        temps = pol._zone_temperatures(obs_arr)
        oat_c = float(gf[2]) * 15.0 + 290.0 - 273.15
        temp_hist.append(temps.copy())
        oat_hist.append(oat_c)

        dT_30m = ((temps - temp_hist[max(0, st - 6)]) / 2.0).astype(np.float32)
        d_oat_30m = float((oat_c - oat_hist[max(0, st - 6)]) / 5.0)
        dow_sin, dow_cos = _extract_dow(pol, obs_arr)

        spread = np.zeros(pol.n_zones, dtype=np.float32)
        for grp in pol.groups:
            if grp.size > 1:
                spread[grp] = float(np.max(temps[grp]) - np.min(temps[grp])) / 2.0
        z_ex = np.column_stack([zone_ua_norm, np.clip(dT_30m, -5.0, 5.0), np.clip(spread, 0.0, 5.0)]).astype(np.float32)

        # Thermal residual capture strictly on physics-consistent EmptyDay/Ortho trajectories
        if ctrl_pol is None and pol._last_thermal_sample is not None and prev_gf is not None:
            ts = pol._last_thermal_sample
            err = ts['observed'] - ts['pred_next']
            nz = len(err)
            for j in range(nz):
                th_rows.append([
                    bldg, sc_key, state_ref['tag'], st - 1, j,
                    float(prev_gf[0]), float(prev_gf[1]), float(prev_gf[7]),
                    float(ts['features'][j, 1]), float(ts['features'][j, 2]),
                    float(ts['features'][j, 3]), float(ts['features'][j, 4]),
                    float(ts['power'][j]), float(ts['model_error'][j]), float(ts['residual'][j]),
                    float(ts['theta_gain'][j]), float(ts['zone_area'][j]),
                    float(prev_zex[j, 0]), float(prev_zex[j, 1]), float(prev_zex[j, 2]),
                    float(prev_gex[0]), float(prev_gex[1]), float(prev_gex[2]),
                    float(ts['old'][j]), float(ts['pred_next'][j]), float(ts['observed'][j]), float(err[j]),
                ])

        obs_next, rew, _, _, _ = env.step(act)
        reg, en, cb = _step_kpis(env)
        viol_flag = float(reg > 1e-5)
        g_ex = np.array([dow_sin, dow_cos, np.clip(d_oat_30m, -3.0, 3.0), viol_flag], dtype=np.float32)

        z_l.append(zf); g_l.append(gf); xb_l.append(xb); xe_l.append(xe)
        z_ex_l.append(z_ex); g_ex_l.append(g_ex)
        r_l.append(float(rew)); reg_l.append(reg); en_l.append(en); cb_l.append(cb)
        tags_l.append(state_ref['tag'])

        prev_gf, prev_gex, prev_zex = gf, g_ex, z_ex
        obs = obs_next

    return {
        'bldg': bldg, 'sc_key': sc_key, 'mode': mode,
        'z_l': z_l, 'g_l': g_l, 'xb_l': xb_l, 'xe_l': xe_l,
        'z_ex_l': z_ex_l, 'g_ex_l': g_ex_l,
        'r_l': r_l, 'reg_l': reg_l, 'en_l': en_l, 'cb_l': cb_l,
        'tags_l': tags_l, 'th_rows': th_rows,
    }


def _compute_vif(X):
    """Variance Inflation Factor per column on standardized active delta actions + OAT."""
    if X.shape[0] < 30:
        return [np.nan] * X.shape[1]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    vifs = []
    for j in range(X.shape[1]):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        others = np.column_stack([others, np.ones(len(y))])
        coef, _, _, _ = np.linalg.lstsq(others, y, rcond=None)
        pred = others @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        vifs.append(1.0 / max(1.0 - r2, 1e-6))
    return vifs


def audit_dataset_v2(BUF, df_audit, df_th, out_dir=None):
    """Run SOTA Data Quality Audit (Temporal Drift, Regret Ratio, Violation %, Excitation VIF, BOPTEST KPI)."""
    print('=' * 92)
    print('SOTA DATA QUALITY AUDIT REPORT (V2)')
    print('=' * 92)

    # 1. Temporal Realism: 0..864 (3d) vs 864..end (rest of week)
    if df_audit['step'].max() >= 900:
        df_audit['window_split'] = np.where(df_audit['step'] < 864, 'Day0_3 (0-864)', 'Day3_7 (864+)')
        t_cmp = df_audit[df_audit['policy_tag'] == 'anchor_empty_day'].groupby('window_split').agg(
            steps=('reward_1step', 'count'),
            active_pct=('is_active', lambda s: round(float(s.mean() * 100), 2)),
            occ_pct=('occ_norm', lambda s: round(float((s > 0.01).mean() * 100), 2)),
            rew_per_step=('reward_1step', lambda s: round(float(s.mean()), 5)),
            comfort_regret=('comfort_regret', lambda s: round(float(s.mean()), 5)),
            energy_cost=('energy_cost', lambda s: round(float(s.mean()), 5)),
            viol_pct=('comfort_viol_flag', lambda s: round(float(s.mean() * 100), 2)),
        )
        print('\n1. KIEM DINH PHAN PHOI THOI GIAN (3 NGAY DAU vs CAC NGAY TIEP THEO - ANCHOR):')
        print(t_cmp.to_string())

    # 2. Coverage & Regret Ratio (arXiv:2408.07986)
    print('\n2. REGRET RATIO (delta_tau), TY LE VI PHAM & DO PHU HANH DONG THEO POLICY:')
    pol_stats = df_audit.groupby('policy_tag').agg(
        samples=('reward_1step', 'count'),
        active_pct=('is_active', lambda s: round(float(s.mean() * 100), 2)),
        rew_n_mean=('reward_nstep', lambda s: round(float(s.mean()), 5)),
        anc_n_mean=('reward_anchor', lambda s: round(float(s.mean()), 5)),
        win_vs_anchor_pct=('regret_vs_anchor', lambda s: round(float((s > 0).mean() * 100), 2)),
        viol_pct=('comfort_viol_flag', lambda s: round(float(s.mean() * 100), 2)),
        sat_std=('eff_sat', lambda s: round(float(s.std()), 3)),
        press_std=('eff_press', lambda s: round(float(s.std()), 3)),
        recirc_std=('eff_recirc', lambda s: round(float(s.std()), 3)),
        water_std=('eff_water', lambda s: round(float(s.std()), 3)),
    )
    pol_stats['regret_ratio_pct'] = ((pol_stats['rew_n_mean'] - pol_stats['anc_n_mean']) / (pol_stats['anc_n_mean'].abs() + 1e-6) * 100).round(2)
    print(pol_stats.to_string())

    # 3. Persistence of Excitation & VIF (when AHU is actively modulating)
    ortho_act = df_audit[df_audit['policy_tag'].str.startswith('ortho_') & (df_audit['base_press'] > -0.99)].copy()
    vifs = {}
    if len(ortho_act) >= 50:
        d_mat = np.column_stack([
            ortho_act['eff_sat'] - ortho_act['base_sat'],
            ortho_act['eff_press'] - ortho_act['base_press'],
            ortho_act['eff_recirc'] - ortho_act['base_recirc'],
            ortho_act['eff_water'] - ortho_act['base_water'],
            ortho_act['oat_celsius'],
        ])
        vif_vals = _compute_vif(d_mat)
        cols = ['d_sat', 'd_press', 'd_recirc', 'd_water', 'oat_C']
        vifs = {c: round(float(v), 3) for c, v in zip(cols, vif_vals)}
        print('\n3. KIEM TRA KICH THICH DOC LAP (Excitation VIF < 5.0 = Khong da cong tuyen):')
        print('   VIF:', vifs)

    # 4. BOPTEST KPI Breakdown by Climate Station (Anchor)
    anc = df_audit[df_audit['policy_tag'] == 'anchor_empty_day'].copy()
    anc['station'] = anc['scenario'].str.split('/').str[1].str.rsplit('_', n=1).str[0]
    clim = anc.groupby('station').agg(
        rew_per_step=('reward_1step', lambda s: round(float(s.mean()), 5)),
        comfort_regret=('comfort_regret', lambda s: round(float(s.mean()), 5)),
        energy_cost=('energy_cost', lambda s: round(float(s.mean()), 5)),
        carbon_emission=('carbon_emission', lambda s: round(float(s.mean()), 5)),
        viol_pct=('comfort_viol_flag', lambda s: round(float(s.mean() * 100), 2)),
    )
    print('\n4. PHAN RA BOPTEST KPI CUA ANCHOR THEO KHI HAU:')
    print(clim.to_string())

    audit_summary = {
        'n_rl_samples': int(len(df_audit)),
        'n_thermal_samples': int(len(df_th)),
        'vifs': vifs,
    }

    if out_dir is not None:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / 'data_audit_v2_metrics.json').write_text(json.dumps(audit_summary, indent=2))

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        # Panel 1: Hour-of-week reward comparison
        anc_hr = anc.copy()
        anc_hr['hr'] = (anc_hr['step'] / 12.0).astype(int)
        hr_s = anc_hr.groupby('hr')[['comfort_regret', 'energy_cost', 'carbon_emission']].mean()
        axes[0, 0].plot(hr_s.index, hr_s['comfort_regret'], label='Comfort Regret')
        axes[0, 0].plot(hr_s.index, hr_s['energy_cost'], label='Energy Cost')
        axes[0, 0].plot(hr_s.index, hr_s['carbon_emission'], label='Carbon')
        axes[0, 0].axvline(72, color='red', linestyle='--', alpha=0.6, label='864-step cutoff (72h)')
        axes[0, 0].set_title('BOPTEST KPI Breakdown across Full Horizon (Hours)')
        axes[0, 0].legend(fontsize=8)

        # Panel 2: Action spread across policies (Coverage check)
        pol_stds = pol_stats[['sat_std', 'press_std', 'recirc_std', 'water_std']]
        pol_stds.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Action Std by Policy (State-Action Coverage)')
        axes[0, 1].tick_params(axis='x', rotation=25)

        # Panel 3: RLS tail > 0.3K by Hour-of-Day
        if len(df_th) > 0:
            th_tmp = df_th.copy()
            th_tmp['hr24'] = ((np.arctan2(th_tmp['hod_sin'], th_tmp['hod_cos']) % (2 * np.pi)) * 12.0 / np.pi).astype(int)
            th_h = th_tmp.groupby('hr24')['rls_err_K'].agg(mae=lambda s: s.abs().mean(), gt03=lambda s: (s.abs() > 0.3).mean() * 100)
            axes[1, 0].bar(th_h.index, th_h['gt03'], color='tab:orange')
            axes[1, 0].set_title('% RLS Thermal Error > 0.3K by Hour of Day')
            axes[1, 0].set_xlabel('Hour (0-23)')

        # Panel 4: Climate KPI breakdown
        clim[['comfort_regret', 'energy_cost', 'carbon_emission']].plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Mean Per-Step BOPTEST KPI by Climate Station')
        axes[1, 1].tick_params(axis='x', rotation=15)

        plt.tight_layout()
        plt.savefig(out_p / 'data_audit_v2_report.png', dpi=150)
        plt.show()

    return audit_summary


def run_collect_and_audit_v2(
    scenario_paths,
    load_env_fn,
    create_meta_fn,
    pid_cls=None,
    out_dir=None,
    steps=2016,
    n_step=6,
    gamma=0.96,
    block_steps=6,
    include_pid_water=True,
    seed=42,
):
    """Collect multi-policy, orthogonal-excitation, KPI-decomposed v2 dataset & run SOTA audit."""
    out_p = Path(out_dir) if out_dir is not None else Path.cwd()
    out_p.mkdir(parents=True, exist_ok=True)
    tag = f'v2_s{steps}_n{len(scenario_paths)}'

    raw_eps, anchor_scores, thermal_all = [], {}, []
    anchor_rn_map = {}
    max_zones = 0
    ep_counter = 0

    modes = ['anchor', 'ortho_explore']
    if include_pid_water and pid_cls is not None:
        modes.append('pid_water')

    for sc_idx, sc_path in enumerate(tqdm(scenario_paths, desc=f'Collecting v2 ({steps} steps)')):
        bldg = sc_path.parts[-2]
        sc_key = f'{bldg}/{sc_path.parts[-1]}'
        for m_idx, mode in enumerate(modes):
            ep_data = _rollout_episode_v2(
                sc_path, bldg, sc_key, steps, mode, load_env_fn, create_meta_fn,
                pid_cls=pid_cls, block_steps=block_steps, seed=seed + sc_idx * 17 + m_idx,
            )
            ep_data['ep_id'] = ep_counter
            ep_counter += 1
            raw_eps.append(ep_data)
            thermal_all.extend(ep_data['th_rows'])
            max_zones = max(max_zones, ep_data['z_l'][0].shape[0])
            if mode == 'anchor':
                anchor_scores[sc_key] = float(sum(ep_data['r_l']))
                rl = ep_data['r_l']
                anchor_rn_map[sc_key] = [
                    sum((gamma ** k) * rl[t + k] for k in range(n_step))
                    for t in range(len(rl) - n_step)
                ]

    Z_c, M_c, G_c, XB_c, XE_c = [], [], [], [], []
    Z_ex, G_ex = [], []
    R_n, R_anc, R_cmf, R_en, R_cb = [], [], [], [], []
    Z_n, M_n, G_n, XB_n = [], [], [], []
    B_id, EP_id, ST_idx, ANC, ACT, TAGS = [], [], [], [], [], []
    pq_rows = []

    for ep in raw_eps:
        bldg, sc_key, mode = ep['bldg'], ep['sc_key'], ep['mode']
        T = len(ep['r_l'])
        nz = ep['z_l'][0].shape[0]
        mask = np.zeros((max_zones, 1), dtype=np.float32)
        mask[:nz, 0] = 1.0
        is_anc = int(mode == 'anchor')

        zl, gl, xbl, xel = ep['z_l'], ep['g_l'], ep['xb_l'], ep['xe_l']
        zexl, gexl = ep['z_ex_l'], ep['g_ex_l']
        rl, regl, enl, cbl, tagsl = ep['r_l'], ep['reg_l'], ep['en_l'], ep['cb_l'], ep['tags_l']

        for t in range(T - n_step):
            rn = sum((gamma ** k) * rl[t + k] for k in range(n_step))
            rcmf = sum((gamma ** k) * regl[t + k] for k in range(n_step))
            ren = sum((gamma ** k) * enl[t + k] for k in range(n_step))
            rcb = sum((gamma ** k) * cbl[t + k] for k in range(n_step))
            ra = anchor_rn_map[sc_key][t]
            is_act = int(xbl[t][1] > -0.99 or xel[t][1] > -0.99 or abs(rl[t]) > 1e-5 or abs(rn) > 1e-5)

            zc = np.zeros((max_zones, ZONE_FEAT_DIM), dtype=np.float32); zc[:nz] = zl[t]
            zn = np.zeros((max_zones, ZONE_FEAT_DIM), dtype=np.float32); zn[:nz] = zl[t + n_step]
            ze = np.zeros((max_zones, 3), dtype=np.float32); ze[:nz] = zexl[t]

            Z_c.append(zc); M_c.append(mask); G_c.append(gl[t]); XB_c.append(xbl[t]); XE_c.append(xel[t])
            Z_ex.append(ze); G_ex.append(gexl[t])
            R_n.append(rn); R_anc.append(ra); R_cmf.append(rcmf); R_en.append(ren); R_cb.append(rcb)
            Z_n.append(zn); M_n.append(mask); G_n.append(gl[t + n_step]); XB_n.append(xbl[t + n_step])
            B_id.append(bldg); EP_id.append(ep['ep_id']); ST_idx.append(t)
            ANC.append(is_anc); ACT.append(is_act); TAGS.append(tagsl[t])

            oat_c = (float(gl[t][2]) * 15.0 + 290.0) - 273.15
            pq_rows.append([
                bldg, sc_key, ep['ep_id'], tagsl[t], is_anc, is_act, t,
                float(rl[t]), float(rn), float(ra), float(rn - ra),
                float(regl[t]), float(enl[t]), float(cbl[t]), float(gexl[t][3]),
                float(xbl[t][0]), float(xel[t][0]), float(xbl[t][1]), float(xel[t][1]),
                float(xbl[t][2]), float(xel[t][2]), float(xbl[t][3]), float(xel[t][3]),
                oat_c, float(gl[t][7]), float(gl[t][9]),
                float(gexl[t][0]), float(gexl[t][1]), float(gexl[t][2]), float(np.mean(zexl[t][:, 2])),
            ])

    BUF = {
        'z_cur': np.array(Z_c, dtype=np.float32), 'm_cur': np.array(M_c, dtype=np.float32),
        'g_cur': np.array(G_c, dtype=np.float32), 'xb_cur': np.array(XB_c, dtype=np.float32),
        'xe_cur': np.array(XE_c, dtype=np.float32), 'r_n': np.array(R_n, dtype=np.float32)[:, None],
        'r_anc': np.array(R_anc, dtype=np.float32)[:, None],
        'z_nxt': np.array(Z_n, dtype=np.float32), 'm_nxt': np.array(M_n, dtype=np.float32),
        'g_nxt': np.array(G_n, dtype=np.float32), 'xb_nxt': np.array(XB_n, dtype=np.float32),
        'bldg': np.array(B_id), 'is_anchor': np.array(ANC, dtype=np.int32), 'is_active': np.array(ACT, dtype=np.int32),
        # v2 SOTA additions
        'ep_id': np.array(EP_id, dtype=np.int32), 'step_idx': np.array(ST_idx, dtype=np.int32),
        'policy_tag': np.array(TAGS),
        'z_extra_cur': np.array(Z_ex, dtype=np.float32), 'g_extra_cur': np.array(G_ex, dtype=np.float32),
        'r_comfort_n': np.array(R_cmf, dtype=np.float32)[:, None],
        'r_energy_n': np.array(R_en, dtype=np.float32)[:, None],
        'r_carbon_n': np.array(R_cb, dtype=np.float32)[:, None],
    }

    npz_path = out_p / f'warp_replay_{tag}.npz'
    pq_path = out_p / f'warp_transitions_{tag}.parquet'
    th_path = out_p / f'thermal_residuals_{tag}.parquet'
    bs_path = out_p / f'base_scores_{tag}.json'

    np.savez_compressed(npz_path, **BUF)
    bs_path.write_text(json.dumps(anchor_scores, indent=2))

    df_audit = pd.DataFrame(pq_rows, columns=[
        'building', 'scenario', 'ep_id', 'policy_tag', 'is_anchor', 'is_active', 'step',
        'reward_1step', 'reward_nstep', 'reward_anchor', 'regret_vs_anchor',
        'comfort_regret', 'energy_cost', 'carbon_emission', 'comfort_viol_flag',
        'base_sat', 'eff_sat', 'base_press', 'eff_press', 'base_recirc', 'eff_recirc', 'base_water', 'eff_water',
        'oat_celsius', 'occ_norm', 'heat_recency',
        'dow_sin', 'dow_cos', 'd_oat_30m', 'mean_ahu_spread',
    ])
    df_audit.to_parquet(pq_path, engine='pyarrow', index=False)

    df_th = pd.DataFrame(thermal_all, columns=[
        'building', 'scenario', 'policy_tag', 'step', 'zone_idx',
        'hod_sin', 'hod_cos', 'occ_norm', 'oat_diff', 'slow_diff', 'group_diff', 'mode',
        'power', 'model_error', 'residual', 'theta_gain', 'zone_area',
        'zone_ua_norm', 'dT_30m', 'temp_spread_ahu', 'dow_sin', 'dow_cos', 'd_oat_30m',
        'old_temp', 'rls_pred', 'true_temp', 'rls_err_K',
    ])
    df_th.to_parquet(th_path, engine='pyarrow', index=False)

    audit_summary = audit_dataset_v2(BUF, df_audit, df_th, out_dir=out_p)
    print(f'\nDa luu bo Data v2: {npz_path.name} ({len(df_audit)} RL, {len(df_th)} Thermal)')
    return BUF, df_audit, df_th, anchor_scores, audit_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect and audit v2 Smart Buildings dataset')
    parser.add_argument('--steps', type=int, default=2016)
    parser.add_argument('--max-scenarios', type=int, default=0, help='0 = all 30 scenarios')
    parser.add_argument('--out-dir', type=str, default='.')
    parser.add_argument('--no-pid', action='store_true')
    args = parser.parse_args()

    from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data
    from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy

    train_dir = Path('smart_buildings/training_scenarios')
    scs = sorted([p.parent for p in train_dir.glob('*/*/config.gin')])
    if args.max_scenarios > 0:
        scs = scs[:args.max_scenarios]

    run_collect_and_audit_v2(
        scs, load_hybrid_environment, create_meta_fn=create_building_meta_data,
        pid_cls=None if args.no_pid else PIDWaterResetPolicy,
        out_dir=args.out_dir, steps=args.steps, include_pid_water=not args.no_pid,
    )
