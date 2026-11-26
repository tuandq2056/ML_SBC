# ML_SBC: Nhat ky Thuc nghiem & Quyet dinh Huong di (Experiment Log)

File nay luu lich su tung dot chay (Wave) tren Kaggle, so sanh truc tiep voi moc **Base 864-step Mean (`-5.0692`)**, va ghi ro ket luan de ca nhom cung theo doi va chot huong tiep theo.

Chi tiet ly thuyet, kien truc va quy trinh push file xem tai [`WORKFLOW.md`](WORKFLOW.md).

---

## Bang Tong hop Cac Moc & Dot Chay

| Wave | Mo ta & Phuong an | Notebook / Code | Validation Split | Mean 864-step (OOF / Eval) | Chenh lech vs Base | Kaggle LB | Trang thai & Quyet dinh |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline** | `EmptyDayConsistentPolicy` (Vat ly goc) | `physics_foundation.py` | Full 30 scenarios | `-5.0692` | `+0.0000` | `~0.010` | Moc chuan bat buoc phai vuot |
| **Legacy v1** | Deterministic `TD3+BC` + `symlog` + `Q1` (khong Shield) | `train_and_evaluate.ipynb` (cu) | 5-Fold `GroupKFold` | `-19.8996` | `-14.8304` | - | **Da loai bo:** Fold 3 `building_train_4` rot `-81.3099` do OOD & `symlog` |
| **Wave 0** | Thu thap 864 steps + Building Profiler + Balanced 3-Fold | `collect_and_profile.ipynb` | Balanced 3-Fold | *(Cho chay xong)* | `0.0000` | - | **Dang chay CPU:** Cho lay `balanced_3fold_*.json`, `th_summary`, `building_trends_profile.png` |
| **Snapshots** | 6 Vi tri ban le x Quyet dinh vat ly + Model (Re nhanh `H=24` buoc) | `profile_key_snapshots.ipynb` | Exact Snapshot Branching | *(Chua chay)* | *(Ma tran 6x10)* | - | Lap ban do vi tri nao quyet dinh nao manh (`key_snapshots_matrix.csv`, `key_snapshots_heatmap.png`) |
| **Wave 1** | **Stage 1:** `ThermalResidualResNet` (`_free_drift`) | `train_stage1_thermal.ipynb` | Balanced 3-Fold | *(Chua chay)* | *(Cho so lieu)* | *(Chua submit)* | **Uu tien so 1:** Sua sai so RLS thieu `hod_sin/cos` & `occ_norm` |
| **Wave 2** | **Stage 2:** Stochastic `IQL-AWR` + `CAPS` + `Safety Shield` | `train_and_evaluate.ipynb` | Balanced 3-Fold | *(Chua chay)* | *(Cho so lieu)* | *(Chua submit)* | **Uu tien so 2:** Chay doi chung hoac bat `TARGET_WATER_ONLY=True` neu AHU bi shield |
| **Wave 3** | Ghep Stage 1 + Stage 2 hoac tinh chinh theo phan tich Wave 1-2 | *(Tuy ket qua Wave 1-2)* | Balanced 3-Fold | *(Chua chay)* | *(Cho so lieu)* | *(Chua submit)* | Se mo ta chi tiet sau khi co log cua Wave 1 & Wave 2 |

---

## Chi tiet tung Dot (Dien theo mau sau moi lan chay Kaggle)

### Wave 0: CPU Data Collection & Building Trend Profiling (`collect_and_profile.ipynb`)
- **Trang thai:** Dang cho ket qua tu Kaggle CPU.
- **Thu muc log du kien:** `experiment_logs/wave_0_profile/`
- **So lieu can cap nhat khi xong:**
  - Phan hoach 3-Fold can bang chot duoc tu `balanced_3fold_*.json`: `Fold 0 = [...]`, `Fold 1 = [...]`, `Fold 2 = [...]`
  - Bang `th_summary` (RLS MAE/RMSE theo toa nha, dac biet `rls_mae_K` va `err_gt_03K_pct`)
  - Nhan xet tu `building_trends_profile.png` (chieu `corr_sat`, `corr_press`, `corr_recirc`, `corr_water` co tin hieu ro hay nhieu?)

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
