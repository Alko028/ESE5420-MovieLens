# MovieLens 数据处理与 baseline 复现操作指南

日期：2026-11-08。适用于 Windows PowerShell。已从公开仓库全新 clone 验证完整流程，结果与正式版本一致。推荐在**全新目录**运行一次完整流程，避免覆盖当前项目的正式划分和结果。

## 0. 准备环境

需要 Git 和 Python；本次验证环境是 Python 3.13.9。打开 PowerShell，先检查：

```powershell
git --version
python --version
```

模型和预处理均使用 CPU，不需要安装 CUDA 或 PyTorch。Python 小版本、数值库或平台不同可能导致极小数值差异，不能保证不同环境逐字节相同。

## 1. 创建独立副本并安装依赖

下面在 Documents 中创建新目录；如果这个目录已经存在，请换一个新名字，不要删旧结果。

```powershell
Set-Location ([Environment]::GetFolderPath('MyDocuments'))
git clone https://github.com/Alko028/ESE5420-MovieLens.git ESE5420-reproduction
Set-Location ESE5420-reproduction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r "data training/requirements.txt"
```

后面的命令全部使用 `.venv` 中的 Python，不需要激活环境或修改 PowerShell 执行策略。

## 2. 下载数据

从 GroupLens 官方地址下载并解压，不从其他人的仓库复制数据：

```powershell
New-Item -ItemType Directory -Path Lab3 -Force | Out-Null
Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-100k.zip" -OutFile "ml-100k.zip"
Expand-Archive -LiteralPath "ml-100k.zip" -DestinationPath "Lab3"
Test-Path "Lab3/ml-100k/u.data"
```

最后一行应输出 `True`。需要的结构：

```text
ESE5420-reproduction/
  data training/
  report/
  Lab3/
    ml-100k/
      u.data
      u.item
      u.genre
      u.info
      README
```

也可以从你现有的项目复制整个 `Lab3/ml-100k` 到这个位置。遵守 README 条款，不把数据上传 GitHub。

## 3. 先运行自动测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s "data training/tests" -v
```

预期：**14 tests，OK**。测试使用合成小数据，不读取真实测试标签，也不训练 GNN。

## 4. 生成预处理数据

```powershell
.\.venv\Scripts\python.exe "data training/prepare_data.py"
.\.venv\Scripts\python.exe "data training/verify_data.py"
```

核对数字：

| 内容 | 预期 |
|---|---:|
| 原始评分 | 100,000 |
| History | 61,175 |
| Training 原始 / 合格 | 19,633 / 19,585 |
| Validation 原始 / 合格 | 9,596 / 9,575 |
| Test 原始 / 合格 | 9,596 / 9,570 |
| 输入维度 | 9 |
| 图边数 | 32,132 |

`data training/splits/baseline_v1/` 保存划分；`data training/outputs/preprocessed/` 保存模型输入、scaler、graph 和 manifest。目标电影在 history 中没出现时，统一从所有模型的合格目标中排除。

测试标签虽然必须由预处理保存，但训练入口不会读取。不要自己打开它来挑模型或阈值，也不要调用 `load_final_test(protocol_frozen=True)`。

## 5. 运行两个 baseline

```powershell
.\.venv\Scripts\python.exe "data training/movie_high_rating_rate.py"
.\.venv\Scripts\python.exe "data training/logistic_regression.py"
```

第一条不需要拟合模型，直接使用 history 的电影高分比例。第二条先在 training targets 内进行 5-fold CV，再重拟合选中的 LR；每折只用该折训练数据拟合 scaler。

不要自行修改随机种子、候选 C 或阈值规则来挑更好结果。本轮固定：测试 seed 20261106，开发与 CV seed 42，bootstrap seed 20261108；六个 C 候选为 0.001、0.01、0.1、1、10、100。

核对结果（数值容许末尾浮点差异）：

| 内容 | Movie frequency | Logistic Regression |
|---|---:|---:|
| Selected C | 不适用 | 0.1 |
| Validation threshold | 0.651515 | 0.699734 |
| Precision | 0.739884 | 0.819303 |
| Recall | 0.507457 | 0.501605 |
| AP | 0.733828 | 0.806693 |

每个模型目录里的 `result.json` 是完整结果。`validation_predictions.csv` 用于本地排查和配对，不能上传公开仓库；`bootstrap.npz` 保存 2,000 组重采样结果。LR 的 `cv_summary.csv` 保存各个 C 的 CV 表，`model.npz` 保存模型和 scaler。

## 6. 生成可读报告和提交文件

```powershell
.\.venv\Scripts\python.exe "data training/build_checkpoint.py"
```

查看：

- `report/checkpoint_submission/baseline.txt`：必交文件，370 词。
- `report/checkpoint_submission/baseline.png`：可选误差区间图。
- `report/baseline_checkpoint.md`：本地指标说明。
- `report/tables/baseline_results.csv`：汇总表。
- `report/tables/paired_baseline_differences.json`：LR 与 reference 的配对差值。

生成脚本只整理已有结果，不重新训练、不访问最终测试集。提交前自己再读一遍 TXT，确认名字、组号和代码链接。

## 7. 已跑过一次，再运行怎么办

为了保护划分和结果，预处理和模型脚本会拒绝覆盖已有输出。这不是训练失败。

- **想完整重跑**：使用另一个新 clone 目录，重复以上步骤。
- **只想用同一固定数据复现某个模型**：指定新的输出目录，例如：

```powershell
.\.venv\Scripts\python.exe "data training/logistic_regression.py" --output-dir "data training/outputs/logistic_regression_repeat1"
```

此命令仍读取原来的固定预处理数据，不重新划分。`build_checkpoint.py` 默认读取原始正式模型目录，不会自动切换到 repeat1；可以直接比较两个 `result.json`。不要拿多次运行的最高分代替正式结果。

## 8. 常见问题

- **找不到 u.data**：检查解压后是否多套一层文件夹，准确路径必须是 `Lab3/ml-100k/u.data`。
- **找不到 numpy/sklearn**：确认命令用的是 `.venv/Scripts/python.exe`，且 requirements 安装成功。
- **Refusing to overwrite**：目录已有结果；按上一节选择新目录，不删除或覆盖封存划分。
- **Artifact hash mismatch**：处理后的文件被改过或混用了不同运行的文件。保留现场，在新 clone 中完整重跑，不关闭检查。
- **结果略不同**：先核对 Python、requirements、种子、C 网格和原始数据来源；明显不同则核查 manifest，不先改参数追分。
- **Recall 区间低于 0.5**：正常。约束针对原验证集选阈值时的点估计，不保证每次 bootstrap 或未来测试都 ≥ 0.5。

## 本次完成、整体剩余、下一任务

本指南覆盖当前已完成的预处理与两个 baseline。整体项目仍有 kNN、限时 GNN、类型消融和最终测试。你的下一步是按步骤在独立目录跑一遍、核对数字，再使用已检查的 TXT/PNG 提交 checkpoint。
