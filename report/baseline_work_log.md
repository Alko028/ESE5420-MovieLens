# Baseline 实现与运行记录

日期：2026-11-08。对应 11 月 9 日 checkpoint。

## 本次完成、整体剩余、下一任务

- **完成**：公共评价、frequency floor、训练集内部 5-fold CV 的 L2 Logistic Regression、2,000 次配对用户 bootstrap、提交 TXT/PNG，以及复现验证。
- **整体剩余**：确认代码发布与 Canvas 提交；之后 kNN、early draft、限时 GNN、类型消融和冻结后的最终测试。
- **下一任务**：核对提交文件及代码链接，提交 checkpoint；再进入 kNN 和 11 月 16 日初稿。

## 1. 为什么更改调参方式

教授 proposal feedback 原先建议 validation AP 选择 C。用户新提供的 Canvas checkpoint 说明明确要求 K-fold CV 调参，因此采用更具体的当前要求：只在 training targets 内进行 5-fold CV，验证集不参与 C 选择。外层 history/train/validation/test 和预处理产物完全不变。

每折都从 `X_raw` 开始，使用 `Pipeline(StandardScaler, LogisticRegression)` 在折内训练部分拟合 scaler。不能直接把全训练集预先标准化的 `X` 拿去 CV，因为其均值和标准差已包含该折留出数据。最终选 C 后才在全部 training targets 重拟合 scaler 和模型。

History 是独立的已观察信息，九特征、邻域和图均来自 history，未使用任何 CV 留出目标的标签，因此可在各折复用。分层折划分针对训练目标记录；同一用户可能跨折，因为目标是已有历史用户的偏好补全，不是冷启动泛化。用户内相关性在结果 bootstrap 中按用户处理。

## 2. 代码和参数

| 文件 | 工作内容 |
|---|---|
| `data training/common/evaluation.py` | 分数并列处理、precision/recall/AP、阈值搜索、用户 bootstrap |
| `data training/common/baseline_run.py` | 结果保存、运行配置、输入和代码哈希 |
| `data training/movie_high_rating_rate.py` | 读取既有 frequency_score，评价无参数参考模型 |
| `data training/logistic_regression.py` | 5-fold CV、C 选择、最终拟合、portable model.npz 保存 |
| `data training/configs/baseline.json` | 所有实验参数，训练前写定 |
| `data training/build_checkpoint.py` | 根据保存结果生成 TXT、图、表及配对差值 |
| `data training/tests/test_evaluation.py` | 评价与 CV 防泄漏测试 |

- LR：L2、lbfgs、max_iter=3000、tol=1e-6，默认截距、无类别重加权。
- C 候选：0.001、0.01、0.1、1、10、100。按五折 AP 的算术均值选择；完全并列时取较小 C。
- StratifiedKFold：5 折，shuffle=True，seed=42。保存每个 training row_id 的 fold。
- 已选择 C=0.1，平均 CV AP=0.8002026967。全部 30 个 CV 拟合和最终拟合均收敛；最终 25 次迭代。
- 执行设备：CPU；这些小规模 LR 实验不需要 CUDA。

## 3. 阈值和误差区间

- 使用验证集分数的各个不同取值作为候选，预测规则 `score >= threshold`，相同分数不会拆分处理。
- 在 recall ≥ 0.5 下最大化 precision；完全并列时优先较高 recall，再优先较高 threshold。
- AP 使用按分数分组后的非插值 average precision，不是梯形积分 PR-AUC。
- 指标为 pooled rating-level 指标，不是先算每个用户再平均。
- Bootstrap：943 位验证用户，每次等概率有放回抽取 943 位，保留每位用户的全部记录及重复次数；2,000 次，seed=20261108。
- 两个模型使用同一用户集合、种子和抽样数，形成配对比较；用 2.5%/97.5% 分位数构造区间。
- 每次重采样固定模型、C 和阈值，不重拟合或重新调参。无正样本或无正预测的未定义指标会记录并排除，不能假装为 0；本次每个指标的 2,000 个重采样均有效。
- 验证集参与了阈值选择，所以 precision 区间是描述性、条件性的，不包含选择偏差和训练不确定性。AP 不依赖阈值，C 则在独立的训练 CV 中选择。

## 4. 实际验证结果

| 模型 | 阈值 | Precision [95% CI] | Recall [95% CI] | AP [95% CI] |
|---|---:|---|---|---|
| Movie high-rating rate | 0.651515 | 0.7399 [0.7185, 0.7599] | 0.5075 [0.4881, 0.5275] | 0.7338 [0.7128, 0.7538] |
| L2 Logistic Regression | 0.699734 | 0.8193 [0.8040, 0.8339] | 0.5016 [0.4750, 0.5253] | 0.8067 [0.7914, 0.8203] |

LR 相对 reference 的配对 ΔPrecision=0.07942，95% CI [0.06073, 0.09945]；ΔAP=0.07286，95% CI [0.05782, 0.08809]。这是验证集上的 baseline 比较，不是 GNN 主问题的答案。

## 5. 输出和复现证据

各方法的 `data training/outputs/<method>/` 保存：

- `result.json`：指标、CI、阈值、参数、时间、版本、代码和输入哈希。
- `validation_predictions.csv`：row_id、user_id、movie_id、y、score、prediction；只留本地。
- `bootstrap.npz`：2,000 组指标及用户集合；只留本地。
- `validation_pr_curve.csv`：阈值及 PR 曲线。
- LR 额外保存 `cv_folds.csv`、`cv_summary.csv`、`cv_assignments.csv` 和 `model.npz`。model.npz 包含 scaler、系数、截距及类别，不依赖 pickle。

验证：14 项自动测试全部通过；评价结果与 sklearn 对照通过；用户加权 bootstrap 与显式复制用户记录一致；每折 scaler 确实只看到该折训练行。保存模型重建预测与 sklearn 输出在 1e-12 容差内一致。

两个模型又在临时输出目录完整复现一次，指标、阈值、全部 bootstrap 数组完全一致。该次实际数据读取被拦截检查，只打开 `train.npz` 和 `validation.npz`，未读取测试输入、测试标签或图。原始正式结果没有被覆盖。

提交材料：`report/checkpoint_submission/baseline.txt` 与 `baseline.png`。TXT 按 Canvas 九个栏目排列，正文 370 词；仅使用 first name。PNG 展示验证 AP 和 precision 的用户 bootstrap 区间，已进行视觉检查。

## 6. 公开仓库边界

用户授权仓库：https://github.com/Alko028/ESE5420-MovieLens 。只发布源代码、说明、合成测试及汇总结果。原始数据、用户级派生数组、预测、划分记录、模型和本地运行日志均不提交。复现者自行从 GroupLens 下载数据，按 README 条款使用并致谢。

完整本地操作指南见 [reproduction_guide.md](reproduction_guide.md)。
