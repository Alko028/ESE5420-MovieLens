# 论文统一结果 — 完整固定协议复跑

Full fixed-protocol reproducibility run after an earlier test evaluation. The same test set was already observed; this is not an additional independent test. No tuning, feature changes, threshold reselection, or best-of-runs selection used test results.

| Method | Precision % [95% CI] | Recall % [95% CI] | AP [95% CI] |
|---|---:|---:|---:|
| Movie-rate floor | 73.67 [71.78, 75.61] | 51.08 [49.34, 52.94] | 0.7261 [0.7061, 0.7469] |
| LR | 80.92 [79.53, 82.37] | 50.75 [48.04, 53.37] | 0.8027 [0.7886, 0.8171] |
| kNN (511) | 81.35 [79.92, 82.81] | 50.70 [48.00, 53.26] | 0.8011 [0.7865, 0.8160] |
| Original GNN | 77.04 [75.56, 78.64] | 49.23 [46.27, 52.00] | 0.7690 [0.7531, 0.7848] |
| LR + genre | 80.93 [79.58, 82.43] | 50.60 [47.91, 53.13] | 0.8039 [0.7898, 0.8186] |
| kNN + genre (127) | 78.73 [77.28, 80.19] | 50.85 [48.08, 53.63] | 0.7771 [0.7619, 0.7936] |
| GNN wide64 | 77.32 [75.85, 78.85] | 49.11 [46.19, 51.87] | 0.7697 [0.7539, 0.7855] |
| GNN wide128 | 77.09 [75.63, 78.62] | 48.90 [45.98, 51.63] | 0.7702 [0.7543, 0.7860] |
| GNN deep2x32 | 78.22 [76.79, 79.68] | 50.90 [48.19, 53.48] | 0.7791 [0.7631, 0.7949] |

Positive share: 0.548485; all-positive recall=1 and constant-score AP=positive share.

主检验为原 GNN−LR 的 Precision 配对差值。其余为补充/genre/探索比较，区间未经多重比较校正。

论文统一使用本目录这一轮结果。正式 kNN 为扩网格后 training CV 选出的511版；旧127版只是开发阶段复现，不能混入主表。两层 GNN 存在 CUDA 稀疏反向的浮点训练波动；本轮保留新训练结果，不按旧新表现择优。详细检查见上一级 `full_rerun_work_log.md`。

图在 `figures/`（五组PDF＋PNG），LaTeX/CSV表在 `tables/`，英文图注见 `captions.md`，插入示例见 `latex_usage.tex`，完整精度见 `paper_results.json`。`figure_table_preview.pdf` 是10页图表排版画廊，四表两次编译无警告并已渲染检查；它不是提交用draft.pdf。

## 本次完成

九模型保存预测的独立 sklearn 点指标、全部区间/配对差值分位数与各前20次用户bootstrap核验通过；生成五组PDF/PNG和统一CSV/LaTeX表。图表生成未读取预处理或重新推理。

## 整体剩余

把新结果、方法、Related Work、Limitations、复现链接和AI致谢整合到NeurIPS论文，编译检查并提交新版draft.pdf；后续视频。

## 下一任务及标准

授权后更新论文；数字与paper_results.json一致，引用完整、图注/单位/阈值和复跑身份清楚，编译PDF可读且满足课程格式。

## 2026-11-26 paper/publication status

The latest NeurIPS paper is ../draft.tex and ../draft.pdf; all results are already integrated. Full current reproduction instructions are in ../reproduce_final_experiments.md. Earlier pending-paper notes above describe the historical generation stage, not current publication status.
