# kNN 模型搭建、训练与验证记录

日期：2026-11-14。

## 本次完成、整体剩余、下一任务

- **本次完成**：实现 `data training/knn.py`、独立参数文件、CV 调参、验证阈值选择、2,000 次用户 bootstrap、模型保存/恢复校验，以及与 LR 和 frequency floor 的配对比较。17 项自动测试通过，真实运行独立复现一致。
- **整体剩余**：11 月 16 日 NeurIPS early draft；限时 GNN、类型消融、最终方案冻结后的测试与后续论文/视频。现有 Canvas 作业是否已提交仍须用户确认。
- **下一任务**：将已有 LR/kNN 结果写入 early draft，明确 planned 实验；按 11 月 16 日节点决定 GNN 是否能稳定训练。当前没有提前读取最终测试标签。

## 1. 模型究竟做什么

每个样本仍是一个用户–电影组合，输入固定为九个历史特征，输出是评分 >=4 的概率分数。kNN 在**九维标准化特征空间**寻找最近的 k 条训练目标记录，对它们的 0/1 标签投票。

这与预处理中每部电影的最多20个 cosine 邻居是两件事：20 用于构建输入特征；本轮 kNN 的 k 用于在训练样本中查找近邻。没有新增特征或重新划分数据。

uniform 为高分邻居数量/k；distance 按距离倒数加权。遇到距离恰好为0的邻居时使用 sklearn 的规则：仅由零距离邻居投票，避免无穷权重。kNN 不通过梯度下降学习权重，fit 主要保存训练例及其标签。

## 2. 固定协议与配置

- 输入：19,585 条 training targets、9,575 条 validation；9 个 history-only 特征。
- 最终 test 仍封存；本轮数据加载审计只允许 `train.npz`、`validation.npz`。
- 训练目标内 5-fold StratifiedKFold，shuffle=True、seed=42，与 LR 的每条 row_id 的折归属完全相同。
- 每折从 X_raw 开始，单独拟合 StandardScaler；没有拿全训练集标准化后的 X 直接做 CV。
- 网格：k=[5,15,31,63,127]，weights=[uniform,distance]，10个组合、50次折内拟合。
- 距离：Euclidean；算法：brute；CPU，n_jobs=1，底层线程限制4。这里的数据规模无需CUDA，也没有安装新的GPU依赖。
- 选择指标：五折 AP 均值；完全并列时优先较小k，再优先uniform。
- 最终选中 **k=127，distance**，平均 CV AP=**0.8015997648**；随后只用全部训练目标重拟合 scaler 和模型。
- k=127位于预设网格上界。这意味着只确认它是本轮网格内最佳，没有声称全局最佳；本轮未看验证结果后再扩展搜索。
- 本轮kNN的10个候选组合与LR的6个候选并非完全相同搜索预算，应在论文中如实交代。

## 3. 验证结果

统一规则：在 validation 上选择满足 recall>=0.5 时 precision 最高的阈值。bootstrap 为2,000次按用户有放回抽样，seed=20261108，与已有模型配对；固定模型和阈值，不重训。

| 模型 | Precision [95% CI] | Recall | AP [95% CI] |
|---|---|---:|---|
| Movie high-rating rate | 0.7399 [0.7185,0.7599] | 0.5075 | 0.7338 [0.7128,0.7538] |
| L2 Logistic Regression | 0.8193 [0.8040,0.8339] | 0.5016 | 0.8067 [0.7914,0.8203] |
| kNN | **0.8230 [0.8084,0.8367]** | **0.5065** | **0.8046 [0.7898,0.8181]** |

kNN 阈值=0.6868557900；recall 95%区间=[0.4804,0.5306]；所有指标的2,000个bootstrap样本均有效。首轮 CV+最终fit约7秒，仅作为本机运行记录，不作严格速度benchmark。

### 配对差值：kNN − Logistic Regression

- ΔPrecision=**+0.37个百分点**，95%区间 **[-0.33,+1.06]个百分点**。
- ΔAP=**-0.00207**，95%区间 **[-0.00714,+0.00282]**。
- 两个区间均跨0：当前验证证据不足以判定kNN优于或劣于LR；这不是统计等效检验，不能据此证明两者完全相同。
- 相对frequency floor，kNN的ΔPrecision=+8.31个百分点，95%区间[+6.47,+10.30]，支持相对简单参考的提升。

这些是验证集结果。阈值在验证集上挑选，区间未包含训练或选择的不确定性，不可解释成最终测试结论。

## 4. 文件和输出

- `data training/knn.py`：CV、最终fit、保存模型、验证评估。
- `data training/configs/knn.json`：事先固定的近邻网格和计算参数。
- `data training/compare_knn.py`：核验数据哈希、样本顺序、标签、CV折和bootstrap配置一致后计算配对差值。
- `data training/tests/test_knn.py`：投票手算/零距离行为、逐折标准化、与LR相同折划分、不合法k测试。
- `data training/outputs/knn/result.json`：参数、指标、区间、阈值、运行时间和输入/代码哈希。
- 同目录 `cv_summary.csv`、`cv_folds.csv`、`cv_assignments.csv`：完整CV结果及本地折索引。
- 同目录 `model.npz`：scaler、标准化训练特征和标签、选中k/weights/metric。包含用户级训练信息，仅保留本地。
- 同目录 `validation_predictions.csv`、`bootstrap.npz`、`validation_pr_curve.csv`：本地预测与重采样。
- 同目录 `access_audit.json`、`verification.json`：实际读取边界及复现验证记录。
- `report/tables/knn_comparison.csv`、`knn_paired_comparison.json`：三模型汇总和配对比较，可用于early draft。

## 5. 验证和复现

17项自动测试全部通过。新增测试核对 uniform/distance 的手算概率、零距离规则、每次scaler仅接触折内训练行、折归属与LR一致，并拒绝超过折内训练数量的k。

实际运行对照已有LR结果确认：输入产物哈希一致、9,575条验证记录/标签逐行一致、训练CV折逐行一致、bootstrap用户集合/顺序/种子一致。模型从model.npz恢复后，预测与原模型在1e-14容差内一致。

另在临时目录重跑完整kNN CV与评价，选中参数、CV AP、验证指标、区间、阈值及2,000组bootstrap数组完全一致。读取审计只出现train/validation，未读取最终测试文件。复跑是复现核验，不是另选随机种子挑最好结果。

在当前项目根目录运行：

```powershell
python -m unittest discover -s "data training/tests" -v
# 首轮结果已存在，复现请使用新的目录：
python "data training/knn.py" --output-dir "data training/outputs/knn_repeat1"
# 该命令比较三个正式模型目录中的结果，不自动选用 repeat1：
python "data training/compare_knn.py"
```

全新环境首次运行kNN时可直接 `python "data training/knn.py"`。它要求已有固定预处理输入；比较脚本还要求已有LR和frequency结果。不要覆盖正式划分，不从多个重复运行中挑分数。

本次仅完成本地开发，未推送GitHub，未修改此前checkpoint提交TXT/PNG，也未提交Canvas。
