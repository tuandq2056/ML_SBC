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
| **Snapshots** | 6 Vi tri ban le x Quyet dinh vat ly + Model (Re nhanh `H=24` buoc) | `profile_key_snapshots.ipynb` | Exact Snapshot Branching | *(Chua chay)* | *(Ma tran 6x10)* | - | Lap ban do vi tri nao quyet dinh nao manh (`key_snapshots_matrix.csv`, `key_snapshots_heatmap.png`) |
| **Wave 1** | **Stage 1:** `ThermalResidualResNet` (`_free_drift`) | `train_stage1_thermal.ipynb` | Balanced 3-Fold | *(Chua chay)* | *(Cho so lieu)* | *(Chua submit)* | **Uu tien so 1:** Sua sai so RLS thieu `hod_sin/cos` & `occ_norm` |
| **Wave 2** | **Stage 2:** Stochastic `IQL-AWR` + `CAPS` + `Safety Shield` | `train_and_evaluate.ipynb` | Balanced 3-Fold | *(Chua chay)* | *(Cho so lieu)* | *(Chua submit)* | **Uu tien so 2:** Chay doi chung hoac bat `TARGET_WATER_ONLY=True` neu AHU bi shield |
| **Wave 3** | Ghep Stage 1 + Stage 2 hoac tinh chinh theo phan tich Wave 1-2 | *(Tuy ket qua Wave 1-2)* | Balanced 3-Fold | *(Chua chay)* | *(Cho so lieu)* | *(Chua submit)* | Se mo ta chi tiet sau khi co log cua Wave 1 & Wave 2 |

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
