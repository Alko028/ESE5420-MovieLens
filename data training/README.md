# MovieLens 100K shared preprocessing

预处理、movie_high_rating_rate 和 Logistic Regression 已完成；kNN 已完成，GNN 尚未实现。LR 按 Canvas checkpoint 要求在训练集内部做 5-fold CV，并在各折单独拟合 scaler。

完整工作记录、特征公式、输出字段和模型使用示例见 [预处理工作记录](../report/preprocessing_work_log.md)。

从项目根目录执行：

```powershell
python "data training/prepare_data.py"
python -m unittest discover -s "data training/tests" -v
python "data training/verify_data.py"
```

第一次预处理已经完成。为保护固定划分，`prepare_data.py` 不覆盖现有输出；以后训练直接通过 `common.load_development()` 读取，不重新划分。

- 参数：`configs/preprocessing.json`
- 固定划分：`splits/baseline_v1/`
- 模型输入：`outputs/preprocessed/`
- 运行与审计信息：`outputs/preprocessed/manifest.json`
- 真实数据验证结果：`outputs/preprocessed/verification.json`
- 环境：Python 3.13.9、NumPy 2.3.5、SciPy 1.16.3；依赖记录在 `requirements.txt`。

预处理使用 CPU。产物采用 NumPy 数组，不绑定训练设备；后续 GNN 可以在自己的训练脚本中转为 PyTorch tensor 并移动到 CUDA。

输出包含用户级数据，仅用于本地实验，不随论文再分发。测试标签在单独的 `sealed/` 目录中，默认开发加载器不会读取；这属于工作流隔离，不是加密或系统权限隔离。

完整运行步骤见 [复现指南](../report/reproduction_guide.md)，当前结果见 [baseline 工作记录](../report/baseline_work_log.md)。

2026-11-14：本地kNN实现与结果见 [kNN工作记录](../report/knn_work_log.md)。共17项测试通过。复现命令：`python "data training/knn.py" --output-dir "data training/outputs/knn_repeat1"`；对比正式结果：`python "data training/compare_knn.py"`。

完整对齐审计：`python "data training/audit_reproducibility.py" --output-dir "data training/outputs/audit_my_repeat"`。默认每个模型复跑3次，核对真实拟合数据、固定CV、指标与全部bootstrap数组，禁止访问真实测试产物。输出目录须不存在。详细证据与限制见 [复跑审计记录](../report/reproducibility_audit.md)。
