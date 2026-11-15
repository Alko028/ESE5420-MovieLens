# 2026-11-14 — 数据对齐与三次完整复跑审计

## 本次完成

结论：当前 frequency floor、L2 Logistic Regression、kNN 的开发实验符合当前 scope 的数据与训练协议。三个模型各完整复跑三次，共九次；未更改划分、随机种子、参数网格或原始结果。固定种子重复验证可复现性，不等于三份独立实验，也不用于缩小置信区间。

可复现脚本：`data training/audit_reproducibility.py`。
机器检查证据：`data training/outputs/audit_2026-11-14_verified/audit.json`；其下 `repeat_1`、`repeat_2`、`repeat_3` 保存每次完整产物。

| 检查项目 | 执行与结果 |
|---|---|
| 数据来源 | 原始 u.data、u.item、u.genre、u.info 的 SHA-256 与预处理 manifest 完全一致 |
| 固定划分 | 三个划分文件的哈希一致；四个原始分区不重叠、合计 100,000 行；没有重新划分 |
| 特征来源 | 从 61,175 条 observed history 重建历史状态和九特征；训练、验证 X_raw 与保存数据逐元素相同 |
| 样本资格与标签 | 按历史中电影是否出现重新检查合格样本；行号、用户 ID、电影 ID 和 rating>=4 标签逐项一致 |
| LR / kNN 输入 | 同一 19,585×9 训练矩阵、同一 9,575×9 验证矩阵、相同标签和特征顺序 |
| Frequency floor | 同一 history、同一验证样本；仅使用电影高分比例，不拟合训练目标，不能描述成与 LR 一样的九特征学习模型 |
| 训练内 CV | 相同 stratified 5-fold、seed=42；保存的每行 fold ID 完全相同；按平均 AP 选超参数 |
| 折内标准化 | 在真实运行中拦截 StandardScaler.fit，逐元素核对 X 和 y 为对应训练折；LR 每轮 31 次、kNN 每轮 51 次，三轮合计 246 次全部通过 |
| 最终拟合 | 每轮最后一次拟合使用全部 19,585 个训练目标；验证集未用于拟合或选择 C、k、weights |
| 结果复现 | 三次复跑的 CSV、模型数组、2,000 次 bootstrap 数组与原结果完全一致；指标、区间、阈值和选中参数也一致 |
| 独立指标计算 | 使用 sklearn 独立计算 precision、recall、AP、PR 阈值选择；每次运行另重算前 20 次用户 bootstrap，数值容差 1e-14 内一致 |
| 区间核验 | 从每次保存的全部 2,000 次 bootstrap 重新计算 2.5%/97.5% 分位数，与报告区间一致 |
| 数据访问 | 审计进程禁止打开真实 test_inputs.npz 和 sealed/test_labels.npz；实际预处理产物访问只有 train.npz、validation.npz、manifest.json |
| 自动测试 | `python -m unittest discover -s "data training/tests" -v`：17 项通过，包括目标评分扰动不改变输入、折内标准化与 bootstrap 对照 |

原始 u.data 为检查文件身份而全文件计算哈希；特征重建只解析 history 和开发目标的评分，未计算测试标签统计、预测或测试性能。测试相关的单元测试只使用临时合成数据。

## 复核后的结果

| 模型 | 选中设置 | Validation precision | Recall | AP |
|---|---|---:|---:|---:|
| Movie high-rating rate | 无拟合超参数 | 73.99% | 50.75% | 0.7338 |
| L2 Logistic Regression | C=0.1 | 81.93% | 50.16% | 0.8067 |
| kNN classifier | k=127、distance、Euclidean | 82.30% | 50.65% | 0.8046 |

kNN−LR 的 precision 差为 +0.37 个百分点，配对 95% 用户 bootstrap 区间为 [-0.33,+1.06] 个百分点。区间跨 0，当前不能认定 kNN 优于 LR。AP 差值区间同样跨 0。

比较对齐的是数据、折分、调参指标、阈值规则、评估样本和用户重采样；各模型不必使用相同数值的阈值。二者都满足验证 recall>=0.5，但实际 recall 并非严格相等。

## 发现、修正与边界

- 修正 `project_proposal.md` 划分表的旧文字：超参数由训练集内 CV 选择，验证集只选择操作阈值并进行开发评价。实现原本已遵守这一规则，本次没有更改训练算法或结果。
- 审计脚本首次试运行将 indices.npz 中的 eligible/excluded 派生索引也当作原始分区，触发总数检查失败；修正为仅检查 history/train/validation/test 四个分区后完成九次运行。这是新增审计脚本的问题，未修改固定划分或模型代码。首次空输出目录不作为通过证据。
- kNN 的 k=127 在预定网格上界，只能称为该网格内最佳；本次不据验证结果扩大网格。LR 比较 6 个候选、kNN 比较 10 个候选，报告应公开搜索预算，不能声称计算预算相等。
- 九个历史特征含 history-only 的电影邻域统计。这与监督 kNN classifier 的 k 是不同概念；LR 和 kNN 都能使用同一邻域信息。
- 该任务是已有用户、历史中出现的电影上的随机评分补全，不是新用户泛化或严格时间预测。每折共享独立 observed history 符合当前协议；CV 留出目标评分不进入历史特征。
- 验证集既选阈值又报告表现；bootstrap 固定模型及阈值，不涵盖训练、调参和阈值选择的不确定性。当前结果不能冒充最终测试证据。
- 本次未改动原有模型结果及用户编辑的 baseline.txt，未推送 GitHub。

## 复现操作

在项目根目录执行以下命令；输出目录必须尚不存在，以免覆盖历史证据：

```powershell
python "data training/audit_reproducibility.py" --repeats 3 --output-dir "data training/outputs/audit_my_repeat"
python -m unittest discover -s "data training/tests" -v
```

脚本遇到任一不一致会报错，不会自动改数据或挑选结果。全部通过后生成 audit.json。输出含本地训练样本或用户级预测，不应公开上传。

## 整体剩余

尚需 early draft、GNN 限时可行性决策、类型消融、冻结方法后的最终测试和最终论文/视频。当前完成的是开发实验复核，并非整个 project 完成。

## 下一任务

按当前进度准备 11 月 16 日 NeurIPS early draft：正文最多 6 页，纳入真实 baseline/kNN 结果、误差区间、训练内 CV 方法和待做实验；验收标准为 draft.pdf 对齐课程栏目并清楚区分已完成与计划工作。本次没有自动开始论文生成或新模型训练。
