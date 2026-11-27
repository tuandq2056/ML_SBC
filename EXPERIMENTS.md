# ML_SBC: Nhat ky Thuc nghiem & Quyet dinh Huong di (Experiment Log)

File nay luu lich su tung dot chay (Wave) tren Kaggle, so sanh truc tiep voi moc **Base 864-step Mean (`-5.0692`)**, va ghi ro ket luan de ca nhom cung theo doi va chot huong tiep theo.

Chi tiet ly thuyet, kien truc va quy trinh push file xem tai [`WORKFLOW.md`](WORKFLOW.md).

---

## Bang Tong hop Cac Moc & Dot Chay

| Wave | Mo ta & Phuong an | Notebook / Code | Validation Split | Mean 864-step (OOF / Eval) | Chenh lech vs Base | Kaggle LB | Trang thai & Quyet dinh |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline** | `EmptyDayConsistentPolicy` (Vat ly goc) | `physics_foundation.py` | Full 30 scenarios | `-5.0692` | `+0.0000` | `~0.010` | Moc chuan bat buoc phai vuot |
| **Legacy v1** | Deterministic `TD3+BC` + `symlog` + `Q1` (khong Shield) | `train_and_evaluate.ipynb` (cu) | 5-Fold `GroupKFold` | `-19.8996` | `-14.8304` | - | **Da loai bo:** Fold 3 `building_train_4` rot `-81.3099` do OOD & `symlog` |
| **Wave 0** | Thu thap 864 steps + Building Profiler + Balanced 3-Fold | `collect_and_profile.ipynb` | Balanced 3-Fold | `-5.0692` | `0.0000` | - | **Hoan tat:** Da chot 3-Fold, luu `experiment_logs/wave_0_profile/` |
| **Snapshots** | 6 Vi tri ban le x Quyet dinh vat ly (Re nhanh `H=24` buoc) | `profile_key_snapshots.ipynb` | Exact Snapshot Branching | `water_310K` manh nhat | `+0.0024` den `+0.1425` | - | **Hoan tat:** Giam nhiet nuoc (`water_310K`/`325K`) thang o ca 4/4 vi tri active; luu `experiment_logs/snapshots/` |
| **Wave 1** | **Stage 1:** `ThermalResidualResNet` (`_free_drift`, `alpha_nn=0.5`) | `train_stage1_thermal.ipynb` | Balanced 3-Fold | **`-4.9062`** | **`+0.1630`** | **`0.00997`** *(Rank #308)* | **Dat moc vuot Base o 6/6 toa:** Da submit LB & dung `ThermalAugmentedPolicy` lam Base cho Wave 3 |
| **Wave 2** | **Stage 2:** Stochastic `IQL-AWR` + `CAPS` + `Safety Shield` (`WATER_ONLY=False`) | `train_and_evaluate.ipynb` | Balanced 3-Fold | `-5.0699` | `-0.0007` | *(Chua submit)* | **AHU warp khong vuot Base:** Fold 0,2 giu Identity (`0.0000`), Fold 1 `-0.0020` -> thu Wave 2b `TARGET_WATER_ONLY=True` |
| **Wave 2b** | **meoluoi3 Ablation (8 runs):** `WATER_ONLY`, `No Shield`, `No CAPS`, **3-Paper Upgrade (`Soft-Margin 0.05 / 0.15` + Dodeja + DAWN)**, `Linear vs Spline` | Kaggle GPU T4 (`meoluoi3`) | Wave 0 3-Fold (`[1,4]\|[2,7]\|[5,8]`) | **`-5.0615`** *(Best Safe)* / **`-5.0511`** *(No Shield)* | **`+0.0077`** / **`+0.0181`** | *(Running `Run 4a`)* | **Hoan tat 8-run Ablation:** `Spline > Linear`, khep kin Pareto 4 diem (`1e-6 -> 0.05 -> 0.15 -> inf`), xac nhan DAWN (`critic_sens`, `val_diff`) & CAPS |
| **Wave 3** | **Stage 1 (`ThermalResidualResNet`) + Stage 2 (`run4a` 3-Paper Upgrade, MRO Parity `0.00000000`)** | `run6b_wave3_stage1_plus_stage2.ipynb` (`meoluoi3`) | Wave 0 3-Fold (`[1,4]\|[2,7]\|[5,8]`) | **`-4.8954`** | **`+0.1738`** *(+0.0108 vs S1)* | *(Running `Run 6b`)* | **Ky luc moi toan team (thang Base 6/6 toa):** `MRO diff=0.00000000`, dang chay rerun Kaggle LB |

---

## Chi tiet tung Dot (Dien theo mau sau moi lan chay Kaggle)

### Wave 0: CPU Data Collection & Building Trend Profiling (`collect_and_profile.ipynb`)
- **Trang thai:** Hoan tat (`51,480` RL samples; `1,392,130` Zone thermal samples; Base Mean `-5.0692`).
- **Thu muc log:** `experiment_logs/wave_0_profile/` (`base_scores_empty_day_s864_n30.json`, `balanced_3fold_empty_day_s864_n30.json`, `building_trends_profile.png`)
- **Phan hoach 3-Fold ban dau cua Wave 0** (Max Standardized Deviation = `1.374`):
  - `Fold 0 = ["building_train_1", "building_train_4"]` (`val_base = -5.039`, `tr_base = -5.084`)
  - `Fold 1 = ["building_train_2", "building_train_7"]` (`val_base = -5.672`, `tr_base = -4.768`)
  - `Fold 2 = ["building_train_5", "building_train_8"]` (`val_base = -4.497`, `tr_base = -5.355`)
- **Cap nhat chuan hoa 3-Fold can bang them vat lieu (`ext_cond`, `ext_dens`) & `rls` cho cac Wave tiep theo** (`balanced_3fold_empty_day_s864_n30.json`, Max Deviation = `1.322`, Base spread giam tu `1.175` xuong `0.283`):
  - `Fold 0 = ["building_train_1", "building_train_7"]` (`val_base = -5.252`, `ext_dens = 1191.0`, `cond = 0.380`)
  - `Fold 1 = ["building_train_2", "building_train_5"]` (`val_base = -4.970`, `ext_dens = 882.4`, `cond = 0.321` — dung lam Fast 1-Split Holdout `Train=1,4,7,8 / Test=2,5`)
  - `Fold 2 = ["building_train_4", "building_train_8"]` (`val_base = -4.986`, `ext_dens = 806.9`, `cond = 0.226`)
- **Bang `th_summary` (Sai so RLS):**
  - `building_train_1` (`168,090`): MAE `0.0146` K, RMSE `0.0852` K, `>0.3K`: `0.58%`
  - `building_train_2` (`366,350`): MAE `0.0245` K, RMSE `0.1791` K, `>0.3K`: `1.22%`
  - `building_train_4` (`232,740`): MAE `0.0120` K, RMSE `0.0756` K, `>0.3K`: `0.36%`
  - `building_train_5` (`340,490`): MAE `0.0116` K, RMSE `0.0475` K, `>0.3K`: `0.29%`
  - `building_train_7` (`155,160`): MAE `0.0147` K, RMSE `0.0780` K, `>0.3K`: `0.64%`
  - `building_train_8` (`129,300`): MAE `0.0248` K, RMSE `0.1590` K, `>0.3K`: `1.23%`
- **Nhan xet tu `building_trends_profile.png`:**
  - `corr_water` duong nhat quan o ca 6 toa (`+0.025` den `+0.219`, manh nhat o `building_train_8` `+0.219`, `building_train_5` `+0.124`, `building_train_7` `+0.095`) va `corr_recirc` duong nhe o ca 6 toa (`+0.030` den `+0.104`), trong khi `corr_press` am nhe (`-0.019` den `-0.138`) va `corr_sat` phan hoa theo toa (`-0.060` den `+0.106`).
  - Sai so RLS tap trung cao hon o `building_train_2` va `building_train_8` (~`1.22–1.23%` buoc lech `>0.3K`), la hai toa nang nhat va co du dia cho Stage 1 / Stage 2.

---

### Snapshots: Key-Position Counterfactual Snapshot Profiler (`2026-11-27`)
- **Moi truong chay:** Kaggle CPU (`SCENARIOS_PER_BLDG = 2`, `12` kich ban, `STEPS_SCAN = 864`, `HORIZON = 24`; fix loi `SuperLU` deepcopy trong `key_snapshot_profiler.py`).
- **File dinh kem:** `experiment_logs/snapshots/` (`key_snapshots_matrix.csv`, `key_snapshots_summary.csv`, `key_snapshots_detail.csv`, `key_snapshots_heatmap.png`)
- **Ket qua noi bat (Mean 2h Gain vs Base):**
  - `POS_1_MORNING_WARMUP` (`n=12`): `water_310K` **`+0.0464`** (win `33.3%`), `water_325K` `+0.0318` (`33.3%`), `press_-25%` `+0.0122` (win `58.3%`); `recirc_eco` `-0.0633`, `sat_-2K` `-0.0461`.
  - `POS_2_PEAK_COOLING` (`n=11`): `water_310K` / `water_325K` **`+0.0041`** (win `27.3%`), `press_-25%` `+0.0000` (`9.1%`); `press_+25%` `-0.0373`, `recirc_eco` `-0.0253`.
  - `POS_3_MILD_SHOULDER` (`n=4`): `water_310K` **`+0.1425`** (win **`100.0%`**), `water_325K` `+0.0933` (`100.0%`), `sat_+2K` `+0.0003` (`25.0%`); `recirc_eco` `-0.1053`, `sat_-2K` `-0.0608`.
  - `POS_5_EMPTY_DAY_START` (`n=1`): tat ca arm vat ly deu `0.0000`.
  - `POS_6_WEEKEND_WATER_HOLD` (`n=4`): `water_310K` / `water_325K` **`+0.0024`** (win **`100.0%`**), cac arm AHU `0.0000`.
- **Ket luan:** Giam nhiet nuoc nong (`water_310K`/`325K`) la quyet dinh thang ro ret nhat nho tiet kiem energy & carbon, trong khi ha ap suat nhe (`press_-25%`) chi co loi o `POS_1_MORNING_WARMUP`.

---

### Wave 1: Stage 1 Physics-Informed Thermal Residual Net (`2026-11-27`)
- **Moi truong chay:** Kaggle GPU (`train_stage1_thermal.ipynb`)
- **Cau hinh chinh:** `EPOCHS = 20`, `BATCH_SIZE = 2048`, `LR = 1e-3`, `best_alpha = 0.5` (`stage1_config.json`)
- **File dinh kem:** `experiment_logs/wave_1_thermal/` (`oof_stage1_thermal.csv`, `stage1_thermal_curves.png`, `stage1_config.json`)
- **Ket qua chi tiet (3-Fold OOF, 864 steps, 30/30 kich ban):**
  - Fold 0 (`["building_train_1", "building_train_7"]`): Base `-5.2525` -> ThermalNN `-5.2096` (**Diff `+0.0429`**) — `B1: +0.0395`, `B7: +0.0462`
  - Fold 1 (`["building_train_2", "building_train_5"]`): Base `-4.9695` -> ThermalNN `-4.8118` (**Diff `+0.1578`**) — `B2: +0.2762`, `B5: +0.0394`
  - Fold 2 (`["building_train_4", "building_train_8"]`): Base `-4.9857` -> ThermalNN `-4.6974` (**Diff `+0.2883`**) — `B4: +0.1178`, `B8: +0.4588`
  - **OOF Mean:** Base `-5.0692` -> ThermalNN **`-4.9062`** (**Diff `+0.1630`**)
- **Phan tich:** Ca `6/6` toa nha unseen deu tang diem ro ret; hai toa nang nhat va sai so RLS cao nhat (`building_train_8` `+0.4588` va `building_train_2` `+0.2762`) huong loi manh nhat khi bo sung `hod_sin/cos` + `occ_norm` vao `_free_drift`.
- **De xuat:** Submit Stage 1 (`ThermalAugmentedPolicy`) len Kaggle LB, va thu nghiem dung `ThermalAugmentedPolicy` lam Base cho Stage 2 / Wave 3.

---

### Wave 2: Stage 2 Stochastic IQL-AWR + CAPS + Safety Shield (`2026-11-27`)
- **Moi truong chay:** Local RTX 4060 (`train_and_evaluate.ipynb`)
- **Cau hinh chinh:** `BASE_MODE='empty_day'`, `COMFORT_SHIELD=True`, `TARGET_WATER_ONLY=False`, `TRAIN_ITERS=3000`, `BATCH_SIZE=256`, `IQL_TAU=0.70`, `BETA_AWR=3.0`, `EPI_PENALTY=0.50`
- **File dinh kem:** `experiment_logs/wave_2_iql_warp/` (`oof_stage2_iql_warp.csv`, `oof_stage2_iql_warp.png`, `stage2_iql_curves.png`, `stage2_config.json`)
- **Ket qua chi tiet (3-Fold OOF, 864 steps, 30/30 kich ban):**
  - Fold 0 (`["building_train_1", "building_train_7"]`): Base `-5.2525` -> Warp-IQL `-5.2525` (`Diff +0.0000`, `shield_rate = 0.00%`, `BestTrainedPilotDiff = -0.0011`)
  - Fold 1 (`["building_train_2", "building_train_5"]`): Base `-4.9695` -> Warp-IQL `-4.9716` (`Diff -0.0020`, `shield_rate = 4.32%`, `BestTrainedPilotDiff = +0.0022`) — `B2: -0.0033`, `B5: -0.0008`
  - Fold 2 (`["building_train_4", "building_train_8"]`): Base `-4.9857` -> Warp-IQL `-4.9857` (`Diff +0.0000`, `shield_rate = 0.00%`, `BestTrainedPilotDiff = -0.0039`)
  - **OOF Mean:** Base `-5.0692` -> Warp-IQL `-5.0699` (**Diff `-0.0007`**)
  - **Mean 4D ResiP Gate `[SAT, Press, Recirc, Water]`:** `[0.0541, 0.0790, 0.0751, 0.0781]`
- **Phan tich:** Cơ che `SavedSafeDiff` da giu Identity an toan o Fold 0 va Fold 2 khi `BestTrainedPilotDiff <= 0`; Fold 1 tang nhe tren 1 kich ban pilot nhung giam `-0.0020` tren full 10 kich ban. Khop voi ket qua `Snapshots`, bo giai AHU vat ly `x_base[:3]` da gan toi uu cuc bo, du dia chinh nam o `water_target`.
- **De xuat Wave tiep theo:** Bat **`TARGET_WATER_ONLY = True`** (Wave 2b) hoac ghep chong **`ThermalAugmentedPolicy` + `TARGET_WATER_ONLY = True`** (Wave 3) va chay lai `profile_key_snapshots.ipynb` voi checkpoint Stage 1/2.

---

### Wave 2b: meoluoi3 Ablation — Residual RL Combination, Safety Shield & 3-Paper Upgrades (`2026-11-27`)
- **Nguoi chay / Moi truong:** `meoluoi3` (`@brea_d`) — Kaggle GPU Tesla T4 (`SEED = 42`, `BASE_MODE = 'empty_day'`, `STEPS_EVAL = 864`, 30/30 kich ban).
- **Luu y phan hoach:** Toan bo 8 phien chay tren **Wave 0 3-Fold (`[1,4] | [2,7] | [5,8]`)** (cung cache `warp_replay_empty_day_s864_n30.npz`, moc Base `-5.0692`) de dam bao so sanh cong bang 100% giua cac ablation.
- **Thu muc log:** `experiment_logs/wave_2b_meoluoi3_ablation/` (`33` file nhe `.csv`, `.json`, `.png`, tong ~`1.65 MB`).
- **Bang ket qua 8 phien Ablation (30 kich ban, 864 steps):**

| Phien (`folder`) | Toan tu / Cau hinh chinh | OOF Mean | Diff vs Base | Worst Scenario | Shield Rate | Nhan xet chinh |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `run5b_linear_residual` | **Linear (`a_base + Δa`)** + Soft-Shield `0.05` + Actor LN | `-5.0735` | **`-0.0043`** | `-0.0415` | `0.02%` | **Te nhat:** Cong tuyen tinh kem on dinh, hong Fold 2 (`B8 -0.0247`) |
| `run5a_softshield_actorLN` | Spline Warp + Soft-Shield `0.05` + **CO Actor LN** | `-5.0692` | `+0.0001` | `-0.0210` | `0.90%` | Kiem chung DAWN (2026): giu Actor LN lam Fold 0 ket Identity |
| `run4b_3papers_water_only` | Spline Warp (3 Papers) + **`WATER_ONLY=True`** | `-5.0685` | `+0.0007` | **`0.0000`** | `0.00%` | **0/30 kich ban am**, thang ca `6/6` toa nha |
| `run6a_pareto015` | Spline Warp (Full 3 Papers) + **Soft-Shield `0.15`** + Water Reach-Avoid + **DAWN Diagnostics** | `-5.0651` | `+0.0041` | `-0.0206` | `0.41%` | Khep kin Pareto (`1e-6 -> 0.05 -> 0.15 -> inf`): `0.15` long qua (`B4/denver_2: -0.0206`), khang dinh `0.05` la dinh toi uu; `critic_sens=0.68-1.22`, `val_diff=+0.0061..+0.0112` |
| `run2_water_only` | Spline Warp + Shield cung + **`WATER_ONLY=True`** | `-5.0643` | **`+0.0050`** | `-0.0044` | `0.00%` | Chinh chon loc nuoc nong an toan (`B8/fairbanks_8: +0.1002`) |
| `run3b_fork` | Spline Warp + No Shield + **tat CAPS (`0.0`)** | `-5.0639` | `+0.0053` | `-0.0158`* | `0.00%` | Tat CAPS lam tut diem >3x so voi `run3_no_shield` -> CAPS bat buoc |
| `run4a_3papers_full4d` *(Best Safe)* | **Spline Warp + Soft-Shield `0.05` + go tran Water (Dodeja) + BO Actor LN (DAWN) + Multi-Weather Pilot** | **`-5.0615`** | **`+0.0077`** | **`-0.0097`** | `0.73%` | **Can bang Safety-Optimality tot nhat:** giam lo worst-case **8.7x** so voi `run3`, dao nguoc `B1/fairbanks_1` tu `-0.0843 -> +0.0524` *(dang submit Kaggle LB)* |
| `run3_no_shield` *(Max Mean)* | Spline Warp + **`COMFORT_SHIELD=False`** + CAPS | **`-5.0511`** | **`+0.0181`** | **`-0.0843`** | `0.00%` | Cao nhat (`B2/fairbanks_2: +0.6163`) nhung rui ro duoi cao 8.7x |

- **Ket luan & De xuat Wave 3:**
  1. **Spline Warp vuot troi Cong tuyen tinh** (`+0.0077` / `+0.0001` vs `-0.0043`).
  2. **Duong cong Pareto 4 diem (Galesloot 2026):** `1e-6` (`-0.0016`) -> **`0.05` (`+0.0077`, worst `-0.0097`)** -> `0.15` (`+0.0041`, worst `-0.0206`) -> `inf` (`+0.0181`, worst `-0.0843`). Khang dinh **`0.05` la interior optimum** thuc su!
  3. **DAWN (Ma et al., 2026):** Bo `LayerNorm` tren Actor day Full 4D tu `+0.0001` len **`+0.0077`**; log `dawn_diagnostics.json` (`run6a`) xac nhan `critic_sens` on dinh (`0.68–1.22`) va `val_diff` duong bi chan (`+0.0061..+0.0112`) xuyen suot ca 3 fold.

---

### Wave 3: Stage 1 Thermal + Stage 2 Brea_D 3-Paper Combination (`2026-11-28`)
- **Nguoi chay / Moi truong:** `meoluoi3` (`@brea_d`) — Kaggle GPU Tesla T4 (`run6b_wave3_stage1_plus_stage2.ipynb`, `SEED = 42`, `STEPS_EVAL = 864`, 30/30 kich ban).
- **Cau hinh chinh:** Ghep truc tiep **Stage 1 (`ThermalResidualResNet`, `alpha_nn = 0.5`, 20 epochs/fold)** lam Base cho **Stage 2 (`run4a`: Soft-Margin Shield `0.05` + Dodeja Unclamped Water & Test-time Ensemble Variance Gating + DAWN bo Actor LN + Multi-Weather Pilot)** sau khi va loi MRO `_cur_ctx`/`_prev_ctx` trong `WarpResiPMixin`.
- **File dinh kem:** `experiment_logs/wave_3_stage1_plus_stage2/` (`oof_stage2_iql_warp.csv`, `oof_stage2_iql_warp.png`, `stage2_iql_curves.png`, `stage2_config.json`).
- **Kiem dinh toan hoc (Wave 3 MRO Zero-Diff Sanity):** `S1 = -0.474906 | S1+IdWarp = -0.474906 | diff = 0.00000000` (khop tuyet doi 100%).
- **Ket qua chi tiet (3-Fold OOF, Wave 0 split `[1,4]|[2,7]|[5,8]`, 30/30 kich ban):**
  - Fold 0 (`["building_train_1", "building_train_4"]`): Base `-5.0390` -> S1+S2 **`-4.9637`** (`Diff +0.0753`, `B1: +0.0438`, `B4: +0.1067`)
  - Fold 1 (`["building_train_2", "building_train_7"]`): Base `-5.6720` -> S1+S2 **`-5.4756`** (`Diff +0.1964`, `B2: +0.3288`, `B7: +0.0640`; Safe Fallback giu nguyen S1 khong tut diem)
  - Fold 2 (`["building_train_5", "building_train_8"]`): Base `-4.4967` -> S1+S2 **`-4.2470`** (`Diff +0.2497`, `B5: +0.0430`, `B8: +0.4565`)
  - **OOF Mean:** Base `-5.0692` -> S1+S2 **`-4.8954`** (**Diff `+0.1738` vs Base**, vuot Stage 1 don le `-4.9062` them **`+0.0108`**!)
  - **Kich ban noi bat:** `B8/fairbanks_8 (+1.3819)`, `B2/fairbanks_2 (+1.1600)`, `B8/phoenix_4 (+0.5423)`, `B4/denver_2 (+0.5142)`; `shield_rate = 0.11%`; thang Base o **ca 6/6 toa nha**.
- **Phan tich & Huong tiep theo:**
  - Co che **Safe Identity Fallback** hoat dong chinh xac o Fold 1 khi tap replay Stage 2 cu (`EmptyDay`) co *Offline Distribution Shift* so voi quy dao Stage 1.
  - Da submit song song **`submit_run4a_inference_only` (V4)** va **`submit_run6b_wave3_inference_only` (V1)** len Kaggle Hidden Test (dang chay rerun). De day Stage 2 cong them nhieu hon nua tren Stage 1 o Wave tiep theo, can thu thap replay data moi truc tiep tu chinh `ThermalAugmentedPolicy` (`Stage 1`) thay vi dung lai replay cua `EmptyDay`.

---

### Mau dien cho Wave moi (Copy block nay khi push dot moi)

```markdown
### Wave X: <Ten thi nghiem> (<Ngay YYYY-MM-DD>)
- **Nguoi chay / Commit:** `<git commit hash>`
- **Cau hinh chinh:** `BASE_MODE=...`, `alpha_nn=...` hoac `TARGET_WATER_ONLY=...`, `COMFORT_SHIELD=...`
- **File dinh kem:** `experiment_logs/wave_X_.../` (`*.csv`, `*.json`, `*.png`)
- **Ket qua chi tiet (3-Fold OOF, 864 steps):**
  - Fold 0 (`[...]`) : Base `...` -> Model `...` (Diff `...`)
  - Fold 1 (`[...]`) : Base `...` -> Model `...` (Diff `...`)
  - Fold 2 (`[...]`) : Base `...` -> Model `...` (Diff `...`)
  - **OOF Mean:** Base `-5.0692` -> Model `...` (**Diff `...`**)
  - Chi so phu: `nn_mae` (Stage 1) hoac `shield_rate` & `Mean 4D Gate` (Stage 2), `Kaggle LB = ...` (neu co)
- **Phan tich nguyen nhan:** <Tai sao tang / tai sao giam o fold nao?>
- **De xuat huong cho Wave tiep theo:** <Giu nguyen / giam alpha_nn / bat TARGET_WATER_ONLY / ghep Stage 1 + Stage 2>
```
