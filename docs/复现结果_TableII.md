# 复现结果 · Table II 对照数据（五集 · 全帧序列 · τ₂=0.6）

> 生成时间：2026-09-30 23:58 ｜ 由 `scripts/merge_repro_results.py` 从各集 `class_summary.csv` 合并

| 数据集 | 类别 | Dice 均值 | 标准差 | 病例数 | 有效帧 |
|--------|------|----------|--------|--------|--------|
| ACDC | RV cavity | **0.5764** | 0.1900 | 21 | 2247 |
| ACDC | myocardium | **0.5182** | 0.1920 | 21 | 3158 |
| ACDC | LV cavity | **0.7774** | 0.1229 | 21 | 3118 |
| Montgomery-County-CXR-Set | left lung | **0.6913** | 0.0362 | 32 | 4504 |
| Montgomery-County-CXR-Set | right lung | **0.6738** | 0.0444 | 32 | 4504 |
| CAMUS | Left Ventricular Myocardium | **0.0018** | 0.0044 | 24 | 8509 |
| CAMUS | Left Ventricle | **0.4115** | 0.0912 | 24 | 8509 |
| CAMUS | Left Atrium | **0.6852** | 0.0717 | 24 | 8509 |
| PolypGen2021_MultiCenterData_v3 | polyp | **0.6349** | 0.1507 | 19 | 7327 |
| Amos | spleen | **0.5714** | 0.2364 | 16 | 3816 |
| Amos | right kidney | **0.4204** | 0.2885 | 23 | 4594 |
| Amos | left kidney | **0.6080** | 0.2494 | 21 | 4860 |
| Amos | liver | **0.6325** | 0.1889 | 21 | 6293 |
| Amos | stomach | **0.4318** | 0.3093 | 20 | 4236 |

**总体**：类别宏平均 **0.5453**；帧加权平均 **0.5174**（合计 74184 帧）。
