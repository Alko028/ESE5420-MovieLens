# MovieLens 100K 数据预处理工作记录

日期：2026-11-06。此文保留当时预处理阶段的状态；2026-11-08 的模型实现见 [baseline 工作记录](baseline_work_log.md)。LR 的 CV 从 X_raw 开始逐折标准化，不使用全训练集预先标准化的 X 进行 CV。用途：后续训练和 baseline checkpoint 的实现参考；不是已完成的实验结果报告。

## 任务状态与下一步

- **本次完成**：共享预处理代码、固定划分、九特征、标准化、图输入、统一加载接口，以及 9 项自动测试和真实数据核验。
- **距离整体目标尚缺**：公共评价、frequency floor 与 Logistic Regression 结果、用户 bootstrap、checkpoint 说明；后续还有 kNN、限时 GNN、类型消融、最终测试、论文和视频。完整 checklist 见 [项目进度](project_progress.md)。
- **下一项任务**：实现公共评价并跑通 movie high-rating rate 参考模型，保存验证指标、阈值和区间；保持测试集封存。完成后更新工作记录及项目进度，再推进 Logistic Regression。

## 1. 本次完成范围

- [x] 重新核对本地 `Lab3/ml-100k`，读取官方数据格式及元信息。
- [x] 按教授反馈后的 scope，实现一组用户内部随机划分和固定测试集。
- [x] 实现九个 history-only 特征、frequency floor 分数和训练集标准化。
- [x] 生成与这些特征共用历史来源的电影图和用户历史矩阵，供后续 GNN 使用。
- [x] 保存索引、排除数量、参数、版本和 SHA-256 校验信息。
- [x] 实现统一加载接口，默认不暴露测试输入或测试标签。
- [x] 完成 9 项自动测试和本地真实数据接口核验。
- [ ] 模型训练、C/k 等调参、阈值选择、用户 bootstrap 和性能比较尚未开始。

执行依据：当前 `project_proposal.md`，而不是旧 PDF 中的五划分方案或早期评分回归方案。

## 2. 原始数据接口

| 文件 | 读取方式 | 使用字段 |
|---|---|---|
| `u.data` | 制表符分隔，四列整数 | user ID、movie ID、1–5 评分；timestamp 仅校验，随后丢弃 |
| `u.item` | Latin-1 编码，竖线分隔，24 列 | 第 1 列 movie ID；最后 19 列类型标记 |
| `u.genre` | `name|index` | 索引 0–18 的类型名称，包含 unknown |
| `u.info` | 官方用户、电影、评分数量 | 校验文件规模 |

核实规模：100,000 条评分，943 位用户，1,682 部电影。未加载 `u.user` 人口统计信息，不使用官方的 u1–u5 划分，因为它们不符合当前四分区方案。

原始用户和电影 ID 是 **1-based**；所有图、矩阵和 tensor 索引是 **0-based**。转换固定为 `index = official_id - 1`。电影类型按显式 movie ID 对齐，不依赖 u.item 的行顺序。

`row_id` 是 u.data 的原始零基行号，每个输出样本保留该值。解析会拒绝重复用户–电影对、非法评分、非法 ID、错误列数、缺失电影和非法类型标记。

## 3. 数据划分规则

参数位于 `data training/configs/preprocessing.json`：

- 固定测试随机种子：20261106。
- 开发划分随机种子：42。
- 每位用户先按 movie ID 排序，再用独立的用户级随机生成器选测试样本。
- 测试种子与开发种子分离，未来只改变开发种子不会改变测试集成员。
- 每位用户有 N 条评分时，test 与 validation 各取 `floor(N / 10)` 条，train 取 `floor(N / 5)` 条，history 接收剩余全部评分。
- 因逐用户整数取整，总体比例不恰好等于 60/20/10/10；不丢弃剩余记录，也不按标签分层挑选。
- 划分只依赖用户–电影 ID，不依赖评分或时间。
- 随机划分用于偏好补全，允许较晚历史评分帮助预测较早的留出评分，不作未来预测解释。

### 本次实际数量

| 分区 | 原始数量 | 模型可用数量 | 因电影没有 history 被排除 |
|---|---:|---:|---:|
| History | 61,175 | 61,175 | 0 |
| Training targets | 19,633 | 19,585 | 48 |
| Validation | 9,596 | 9,575 | 21 |
| Sealed test | 9,596 | 9,570 | 26 |

History 覆盖 1,607 部电影。其余电影的目标对在所有方法中统一排除；不将它们的目标评分移入 history 来补齐。训练、验证、测试均使用同一合格性规则。

测试集这里只记录样本数量与基于 history 的可用性，没有计算其高分比例、模型指标或阈值。

## 4. 九个特征的定义与顺序

令 H 为 observed history，R 为用户×电影历史评分矩阵；缺失位置填 0，同时另存观察掩码 M，避免把缺失当成负类。高分指示为历史评分 ≥ 4。

| 列号（0-based） | 名称 | 定义 |
|---|---|---|
| 0 | `user_mean_rating` | 用户在 H 中的平均评分 |
| 1 | `user_high_rating_rate` | 用户在 H 中评分 ≥ 4 的比例 |
| 2 | `user_rating_count` | 用户在 H 中的评分数量 |
| 3 | `movie_mean_rating` | 电影在 H 中的平均评分 |
| 4 | `movie_high_rating_rate` | 电影在 H 中评分 ≥ 4 的比例；也是 frequency floor |
| 5 | `movie_rating_count` | 电影在 H 中的评分数量 |
| 6 | `neighbor_weighted_mean_rating` | 该用户已评分邻居的相似度加权平均评分 |
| 7 | `neighbor_weighted_high_rating_rate` | 该用户已评分邻居的相似度加权高分比例 |
| 8 | `neighbor_observed_count` | 用户在目标电影邻域内实际有历史评分的电影数量 |

### 邻域公式和固定选择

- 相似度使用 H 中两个电影列向量的 cosine：`s(i,j) = dot(R[:,i], R[:,j]) / (norm(R[:,i]) * norm(R[:,j]))`。这是零填充的完整历史向量 cosine，不是仅对共同评分用户单独归一化的 cosine。
- 无历史电影的相似度为 0。排除 self-loop，只保留正相似度邻居，每部电影最多 20 个；相似度相同时优先较小 movie ID。
- top-20 对每个目标电影独立选取，因此最终邻域图是有向图，最多 20 条入边。图不自动对称化。
- 对目标用户 u 和电影 i，S 为 i 的邻域与 u 的已观察电影集合的交集。
- 列 6 为 `sum(s(i,j) * R[u,j], j in S) / sum(s(i,j), j in S)`。
- 列 7 为 `sum(s(i,j) * 1[R[u,j]>=4], j in S) / sum(s(i,j), j in S)`。
- 列 8 为 S 的大小。若 S 为空，列 6、7 分别回退到用户历史平均评分和高分比例，列 8 为 0。
- 本次零邻居样本数：训练 1,244，验证 637，测试输入 552；这是 history-only 的输入诊断，不涉及测试结果。

这些细节是对 scope 中未展开公式的明确实现选择，已在训练之前固定；没有通过验证性能或测试结果挑选。九维输入不加入 movie ID、user ID、timestamp 或类型标记。

### 标准化

仅使用 **19,585 个合格 training targets 的原始输入** 拟合逐列均值及总体标准差（ddof=0）。标准差为 0 时替换为 1。保存 scaler，并对 train、validation、test inputs 应用同一变换。

`X_raw` 是原始九维特征，`X` 是 float32 标准化特征。构建原始特征的历史统计只来自 H；标准化统计来自训练输入。这两个拟合层次不能混淆。

## 5. 代码文件与职责

| 文件（相对 data training） | 主要内容 |
|---|---|
| `prepare_data.py` | 命令行入口；默认路径相对于脚本，不依赖终端当前目录 |
| `configs/preprocessing.json` | 种子、邻居数、标签与相似度规则 |
| `common/raw_data.py` | `MovieLens` 数据结构、官方文件解析和一致性检查 |
| `common/splitting.py` | `make_splits()`；固定 test 与单开发划分 |
| `common/features.py` | `build_history()`、`make_features()`、`fit_scaler()`、`transform()` |
| `common/preprocessing.py` | 统一运行、保存产物、manifest 与文件校验信息 |
| `common/dataset.py` | `load_development()`、`load_graph()`、显式 `load_final_test()` |
| `verify_data.py` | 对真实开发数据检查 ID、标签、标准化及 100 个邻域特征 |
| `tests/test_preprocessing.py` | 合成数据的泄漏、划分、公式、接口与完整性测试 |
| `requirements.txt` | 本次验证的 NumPy/SciPy 版本 |

`movie_high_rating_rate.py`、`logistic_regression.py`、`knn.py` 和 `gnn.py` 仍为空白，未开始训练。

## 6. 输出文件与统一接口

### 划分记录：data training/splits/baseline_v1/

- `assignments.csv`：每条原始记录的 row_id、user_id、movie_id、partition、eligible；不含评分和时间戳。
- `per_user_counts.csv`：各用户四分区的整数数量。
- `indices.npz`：各分区完整 row_id，以及各目标分区 eligible/excluded 的 row_id。

### 模型输入：data training/outputs/preprocessed/

| 文件 | 内容 |
|---|---|
| `train.npz` | 19,585 行九维特征、目标标签、ID 和 frequency_score |
| `validation.npz` | 9,575 行，同一字段结构 |
| `test_inputs.npz` | 9,570 行输入；**没有 y** |
| `sealed/test_labels.npz` | 单独保存最终测试 row_id 和 y |
| `scaler.npz` | 九列 mean、scale、feature_names |
| `graph.npz` | 历史评分、观察掩码、电影边、类型元信息和 ID 对应关系 |
| `manifest.json` | 完成标志、配置、版本、源文件/代码/产物哈希、样本数量及规则 |
| `verification.json` | 实际数据验证结果；可用 verify_data.py 重新生成 |

每个 tabular batch 的统一字段：

| 字段 | 形状/类型 | 用途 |
|---|---|---|
| `row_id` | [N], int64 | 跨模型对齐原始记录 |
| `user_id`, `movie_id` | [N], int32 | 原始 1-based ID；user_id 用于配对用户 bootstrap |
| `user_index`, `movie_index` | [N], int32 | 0-based 矩阵索引；转 PyTorch gather 索引时使用 long |
| `X_raw`, `X` | [N, 9], float32 | 原始和标准化特征 |
| `frequency_score` | [N], float32 | 原始电影高分比例，即 X_raw 的第 4 列 |
| `y` | [N], int8 | 训练/验证的 0–1 标签；test inputs 不含此字段 |
| `feature_names` | [9], Unicode | 列顺序检查 |

### GNN 接口

- `history_rating`：[943, 1682] float32，只有 H 的评分，未观察位置为 0。
- `history_mask`：同形状 bool，区分未知与已观察。
- `edge_index`：[2, 32132] int64，第一行 source neighbor，第二行 target movie；节点索引为 movie ID − 1。
- `edge_weight`：[32132] float32，原始正 cosine 权重，未进行行归一化；没有 self-loop。未来 GNN 如需归一化或 self-loop，须在模型配置中显式记录。
- 图的生成与 tabular 邻域使用同一个 adjacency，不加入验证/测试边或类型权重。
- `movie_genres`：[1682, 19]，按官方类型顺序；仅用于后续补充实验，不自动附加到 ratings-only 输入。
- GNN 可为每位用户构造节点信号 `[history_rating[u] / 5, history_mask[u]]`，在各目标 movie_index 取输出；标签仍来自同一 train/validation batch。
- 用户类型偏好和 57 个额外类型特征本次尚未生成，避免扩大 baseline checkpoint 范围。

### 训练脚本调用示例（尚未训练）

在 `data training` 下的模型脚本中：

```python
from common import load_development, load_graph

train, validation = load_development()

# Frequency floor: already-computed history-only score.
train_scores = train["frequency_score"]
validation_scores = validation["frequency_score"]

# Logistic Regression / kNN: same rows, features and labels.
X_train, y_train = train["X"], train["y"]
X_validation, y_validation = validation["X"], validation["y"]

# GNN: same target rows and labels; graph supplies history-only node inputs.
graph = load_graph()
user_indices = train["user_index"]
movie_indices = train["movie_index"]
```

NumPy 产物不绑定 CPU/GPU。未来安装并验证 PyTorch/CUDA 后，可在 GNN 代码中将浮点数组转为 float32 tensor、索引转为 long，并移动到同一 device。本次环境未安装 PyTorch，未运行 CUDA；当前预处理使用 CPU。

默认开发接口不读取 test inputs 或 test labels。`load_final_test()` 默认抛出异常，只有显式 `protocol_frozen=True` 才读取最终测试数据；实际实验须等模型、超参数、阈值和分析协议冻结。文件没有加密，不能把这个门槛描述成系统安全隔离。

## 7. 验证记录

执行：`python -m unittest discover -s "data training/tests" -v`，**9 项通过**：

1. 手算的 cosine 邻域和九特征一致。
2. 无邻居回退正确；history 中的目标对被拒绝。
3. 标准化仅按训练输入拟合，常数列处理正确。
4. 邻居相似度相同的排序规则确定。
5. 开发种子变化不影响固定测试集；输入行顺序变化不改变用户–电影分区成员。
6. 改变所有目标评分及时间戳后，所有特征、图和 scaler 完全不变。
7. 开发加载器不读取测试文件，模型输入字段对齐；最终测试门槛仅在合成样本上验证。
8. 重复构建得到相同数组；产物被修改时哈希检查拒绝加载。
9. 非法评分被拒绝，包括可能发生小整数溢出的异常值。

真实数据验证：原始行号和 ID 对齐；train/validation 标签与评分 ≥ 4 对齐；目标对不在 history mask；九维特征全部有限；训练 X 各列均值约 0、标准差约 1；独立重算 100 条验证样本邻域公式通过。默认测试加载被阻止，未读取封存测试标签进行评价。

这验证的是数据接口和预处理，不是四种训练方法已经训练成功，也不是最终模型性能证据。

## 8. 运行、复现和限制

从项目根目录：

```powershell
python "data training/prepare_data.py"
python -m unittest discover -s "data training/tests" -v
python "data training/verify_data.py"
```

默认产物已生成，重复运行 prepare 会拒绝覆盖。训练代码应直接读取现有数据，不要重新生成划分。需要独立复现时可显式指定新的 `--output-dir` 和 `--split-dir`，使用相同配置验证数组相等；不同种子只可用于事先决定的扩展实验。

本次版本：Python 3.13.9，NumPy 2.3.5，SciPy 1.16.3。保存 manifest 的原因是跨库版本可能存在数值差异，不能仅凭随机种子承诺任意环境逐字节一致。

所有原始和衍生的用户级数据只保留在本地，不公开再分发；论文仅使用汇总统计与图表。下一步是训练 frequency floor 和 L2 Logistic Regression、用验证 AP 选择 C，再做阈值选择和用户 bootstrap，仍保持测试集封存。
