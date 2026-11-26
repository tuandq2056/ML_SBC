"""Building Trend Profiler & Mathematically Balanced 3-Fold Partitioner.
Runs directly on cached .npz / .parquet / base_scores.json in ~2 seconds
with ZERO simulator overhead.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _all_partitions(items, k=3):
    """Generate unique k-way balanced-size partitions of buildings."""
    items = list(items)
    n = len(items)
    if n == 6 and k == 3:
        def _pairs(rem):
            if len(rem) <= 2:
                yield (tuple(rem),)
                return
            first = rem[0]
            for j in range(1, len(rem)):
                pair = (first, rem[j])
                rest = [rem[i] for i in range(1, len(rem)) if i != j]
                for sub in _pairs(rest):
                    yield (pair,) + sub
        yield from _pairs(items)
        return
    folds = [[] for _ in range(k)]
    for idx, item in enumerate(items):
        folds[idx % k].append(item)
    yield tuple(tuple(f) for f in folds)


def profile_buildings(BUF, df_audit, BASE_SCORES, out_dir=None, n_folds=3):
    """Analyze physical/control/reward trends per building and find optimal balanced folds."""
    b_arr = BUF['bldg']
    anc_mask = BUF['is_anchor'] == 1
    exp_mask = BUF['is_anchor'] == 0
    buildings = sorted(set(b_arr))

    bldg_scores = {b: [] for b in buildings}
    for sc_key, val in BASE_SCORES.items():
        b = sc_key.split('/')[0]
        if b in bldg_scores:
            bldg_scores[b].append(float(val))

    rows = []
    for b in buildings:
        bm_anc = anc_mask & (b_arr == b)
        bm_exp = exp_mask & (b_arr == b)
        bm_act = bm_anc & (BUF['is_active'] == 1)

        z = BUF['z_cur'][bm_anc]
        m = BUF['m_cur'][bm_anc]
        g = BUF['g_cur'][bm_anc]
        xb = BUF['xb_cur'][bm_anc]

        valid_0 = m[0, :, 0] > 0.5
        nz = int(valid_0.sum())
        valid_z = m[:, :, 0] > 0.5
        areas = np.expm1(z[0, valid_0, 8] * 5.0)

        dist_low = z[:, :, 0][valid_z] * 2.0
        dist_high = z[:, :, 1][valid_z] * 2.0
        free_drift = z[:, :, 2][valid_z] * 2.0
        me_vals = z[:, :, 5][valid_z]
        res_vals = z[:, :, 6][valid_z] * 2.0

        oat_c = (g[:, 2] * 15.0 + 290.0) - 273.15
        occ_pct = float((g[:, 7] > 0.01).mean() * 100.0)
        act_pct = float(BUF['is_active'][bm_anc].mean() * 100.0)
        water_hot_pct = float((xb[:, 3] > 0.70).mean() * 100.0)

        if bm_act.sum() > 0:
            xb_act = BUF['xb_cur'][bm_act]
            sat_act_C = float((xb_act[:, 0].mean() * 10.0 + 295.0) - 273.15)
            press_act_Pa = float(xb_act[:, 1].mean() * 1000.0 + 1000.0)
            recirc_act = float(xb_act[:, 2].mean() * 0.5 + 0.5)
        else:
            sat_act_C, press_act_Pa, recirc_act = np.nan, np.nan, np.nan

        bm_exp_act = bm_exp & (BUF['is_active'] == 1)
        if bm_exp_act.sum() > 20:
            adv = BUF['r_n'][bm_exp_act, 0] - BUF['r_anc'][bm_exp_act, 0]
            dx = BUF['xe_cur'][bm_exp_act] - BUF['xb_cur'][bm_exp_act]
            win_rate_act = float((adv > 0).mean() * 100.0)
            corrs = []
            for j in range(4):
                if np.std(dx[:, j]) > 1e-6 and np.std(adv) > 1e-8:
                    c = float(np.nan_to_num(np.corrcoef(dx[:, j], adv)[0, 1], nan=0.0))
                else:
                    c = 0.0
                corrs.append(c)
        else:
            win_rate_act = 0.0
            corrs = [0.0, 0.0, 0.0, 0.0]

        sc_vals = bldg_scores[b]
        rows.append({
            'building': b,
            'n_scenarios': len(sc_vals),
            'n_zones': nz,
            'area_m2': round(float(areas.sum()), 0),
            'base_mean': float(np.mean(sc_vals)) if sc_vals else 0.0,
            'base_std': float(np.std(sc_vals)) if sc_vals else 0.0,
            'active_pct': act_pct,
            'occ_pct': occ_pct,
            'water345_pct': water_hot_pct,
            'oat_mean_C': float(oat_c.mean()),
            'oat_min_C': float(oat_c.min()),
            'oat_max_C': float(oat_c.max()),
            'cold_viol_pct': float((dist_low < 0.0).mean() * 100.0),
            'hot_viol_pct': float((dist_high < 0.0).mean() * 100.0),
            'free_drift_C': float(free_drift.mean()),
            'rls_err_mean': float(me_vals.mean()),
            'rls_res_std': float(res_vals.std()),
            'sat_act_C': sat_act_C,
            'press_act_Pa': press_act_Pa,
            'recirc_act': recirc_act,
            'exp_win_act_pct': win_rate_act,
            'corr_sat': corrs[0],
            'corr_press': corrs[1],
            'corr_recirc': corrs[2],
            'corr_water': corrs[3],
        })

    df_prof = pd.DataFrame(rows)

    print('=' * 96)
    print('1. HO SO VAT LY & QUY MO TUNG TOA NHA (BASE ANCHOR):')
    print('=' * 96)
    cols1 = ['building', 'n_zones', 'area_m2', 'base_mean', 'base_std', 'active_pct', 'occ_pct', 'water345_pct', 'oat_mean_C']
    print(df_prof[cols1].round(2).to_string(index=False))

    print('\n' + '=' * 96)
    print('2. CHE DO VAN HANH, LOI NHIET & DO NHAY THAM SO KHI ACTIVE:')
    print('=' * 96)
    cols2 = ['building', 'cold_viol_pct', 'hot_viol_pct', 'rls_err_mean', 'sat_act_C', 'press_act_Pa', 'exp_win_act_pct', 'corr_sat', 'corr_press', 'corr_recirc', 'corr_water']
    print(df_prof[cols2].round(3).to_string(index=False))

    strat_cols = ['base_mean', 'active_pct', 'water345_pct', 'oat_mean_C', 'rls_err_mean']
    feat_mat = df_prof[strat_cols].to_numpy(dtype=float)
    feat_z = (feat_mat - feat_mat.mean(axis=0)) / (feat_mat.std(axis=0) + 1e-8)
    weights = np.array([2.0, 1.5, 1.0, 1.0, 1.0])
    feat_w = feat_z * weights
    global_mean = feat_w.mean(axis=0)

    b_to_idx = {b: i for i, b in enumerate(buildings)}
    best_part, best_score = None, float('inf')
    for part in _all_partitions(buildings, k=n_folds):
        max_dev = 0.0
        for grp in part:
            idx = [b_to_idx[b] for b in grp]
            val_dev = float(np.linalg.norm(feat_w[idx].mean(axis=0) - global_mean))
            max_dev = max(max_dev, val_dev)
        if max_dev < best_score:
            best_score = max_dev
            best_part = part

    print('\n' + '=' * 96)
    print(f'3. PHAN HOACH {n_folds}-FOLD TOI UU DA TIEU CHI (Max Standardized Deviation = {best_score:.3f}):')
    print('=' * 96)
    fold_check = []
    for f_idx, val_grp in enumerate(best_part):
        tr_grp = [b for b in buildings if b not in val_grp]
        v_df = df_prof[df_prof['building'].isin(val_grp)]
        t_df = df_prof[df_prof['building'].isin(tr_grp)]
        fold_check.append({
            'fold': f_idx,
            'val_buildings': ', '.join(val_grp),
            'val_base': round(float(v_df['base_mean'].mean()), 3),
            'tr_base': round(float(t_df['base_mean'].mean()), 3),
            'val_active%': round(float(v_df['active_pct'].mean()), 1),
            'tr_active%': round(float(t_df['active_pct'].mean()), 1),
            'val_water345%': round(float(v_df['water345_pct'].mean()), 1),
            'tr_water345%': round(float(t_df['water345_pct'].mean()), 1),
            'val_oat_C': round(float(v_df['oat_mean_C'].mean()), 1),
            'tr_oat_C': round(float(t_df['oat_mean_C'].mean()), 1),
        })
    print(pd.DataFrame(fold_check).to_string(index=False))

    if df_audit is not None:
        anc_df = df_audit[df_audit['is_anchor'] == 1].copy()
        anc_df['hour_elapsed'] = anc_df['step'] / 12.0

        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        for b in buildings:
            sub = anc_df[anc_df['building'] == b].groupby('step')
            hrs = sub['hour_elapsed'].first()
            axes[0, 0].plot(hrs, sub['reward_1step'].mean().rolling(12, min_periods=1).mean(), label=b)
            axes[0, 1].plot(hrs, (sub['is_active'].mean() * 100.0).rolling(12, min_periods=1).mean(), label=b)
            axes[1, 0].plot(hrs, sub['oat_celsius'].mean().rolling(12, min_periods=1).mean(), label=b)

        axes[0, 0].set_title('Reward 1-step (Hourly Rolling Mean, 72h)')
        axes[0, 0].set_xlabel('Hour'); axes[0, 0].set_ylabel('Reward'); axes[0, 0].legend(fontsize=8)
        axes[0, 1].set_title('AHU Active % across 72h (Day 1 vs Day 2 vs Day 3)')
        axes[0, 1].set_xlabel('Hour'); axes[0, 1].set_ylabel('Active %'); axes[0, 1].legend(fontsize=8)
        axes[1, 0].set_title('Outdoor Air Temperature Profile (C)')
        axes[1, 0].set_xlabel('Hour'); axes[1, 0].set_ylabel('Temp (C)'); axes[1, 0].legend(fontsize=8)

        x = np.arange(len(buildings))
        w = 0.2
        axes[1, 1].bar(x - 1.5*w, df_prof['corr_sat'], width=w, label='SAT')
        axes[1, 1].bar(x - 0.5*w, df_prof['corr_press'], width=w, label='Pressure')
        axes[1, 1].bar(x + 0.5*w, df_prof['corr_recirc'], width=w, label='Recirc')
        axes[1, 1].bar(x + 1.5*w, df_prof['corr_water'], width=w, label='Water')
        axes[1, 1].axhline(0.0, color='black', linewidth=0.8)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([b.replace('building_train_', 'B') for b in buildings])
        axes[1, 1].set_title('Corr(Delta_Action, Adv) during Active Explore')
        axes[1, 1].legend(fontsize=8)

        plt.tight_layout()
        if out_dir is not None:
            plt.savefig(Path(out_dir) / 'building_trends_profile.png', dpi=150)
        plt.show()

    return df_prof, best_part


def load_and_profile(search_dirs=('/kaggle/working', '/kaggle/input', '.'), n_folds=3):
    """Auto-locate cached .npz/.parquet/.json files and run profile_buildings."""
    npz_file, pq_file, json_file = None, None, None
    for d in search_dirs:
        p = Path(d)
        if not p.exists():
            continue
        if npz_file is None:
            hits = sorted(list(p.rglob('warp_replay_*.npz')))
            if hits: npz_file = hits[-1]
        if pq_file is None:
            hits = sorted(list(p.rglob('warp_transitions_*.parquet')))
            if hits: pq_file = hits[-1]
        if json_file is None:
            hits = sorted(list(p.rglob('base_scores_*.json')))
            if hits: json_file = hits[-1]
    assert npz_file is not None and json_file is not None, 'Khong tim thay file cache .npz hoac base_scores.json!'
    data = np.load(npz_file, allow_pickle=False)
    BUF = {k: data[k] for k in data.files}
    BASE_SCORES = json.loads(json_file.read_text())
    df_audit = pd.read_parquet(pq_file) if pq_file is not None else None
    out_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
    return profile_buildings(BUF, df_audit, BASE_SCORES, out_dir=out_dir, n_folds=n_folds)
