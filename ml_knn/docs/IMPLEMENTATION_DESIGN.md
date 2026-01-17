# KNN 实现设计文档 (IMPLEMENTATION_DESIGN.md)

---

## 1. 与 ALGO_SPEC 对齐摘要

本设计严格基于 `ALGO_SPEC.md` (2026-01-18) 作为唯一真实来源。

### DoD 覆盖清单（逐条对齐）

| DoD 要求 | 设计对应 | 状态 |
|---------|---------|------|
| **实验完整性：6 个实验全部实现** | `src/experiments/` 下 6 个实验模块 | ✅ |
| **特性点可见：K 值/距离/标准化** | 实验 A1+C1+A2+A3 输出可视化 | ✅ |
| **默认数据可跑通（synthetic）** | `src/data/data_loader.py` 默认 synthetic 模式 | ✅ |
| **用户数据支持（CSV/Excel）** | `data_loader.py` 支持 `data_path` 参数 | ✅ |
| **至少 1 指标** | 所有实验输出 accuracy/f1_score 到 `outputs/metrics/` | ✅ |
| **至少 1 图** | 所有实验输出图表到 `outputs/figures/` | ✅ |
| **至少 1 对照** | 实验 B1 对照 3 个基线模型 | ✅ |
| **最小可运行路径（CPU/2GB/5min）** | 纯 NumPy+sklearn 实现，无 GPU 依赖 | ✅ |
| **单命令运行** | `python main.py` 默认运行所有实验 | ✅ |

### 规格书关键参数引用

- **complexity_level:** Standard
- **default_data_mode:** synthetic
- **experiments:** 6 个（A1, A2, B1, C1, C2, A3）
- **metrics:** accuracy, f1_score, confusion_matrix, train_time, predict_time
- **figures:** 8+ 张（K 值曲线、决策边界 x3、对比图 x2、混淆矩阵 x2）
- **outputs_dir:** `outputs/` (metrics/ + figures/ + logs/)
- **runtime_requirements:** 1 core CPU, <2GB memory, <5min, <=1000 samples

---

## 2. 项目落地形式选择

### 选择：Standard 级别项目结构

**理由：**

1. **ALGO_SPEC 明确指定 complexity_level = Standard**：需要 6 个实验 + 丰富可视化，单文件无法清晰组织。
2. **实验独立性**：6 个实验各有独立逻辑（K 值敏感性、距离对比、基线对照、决策边界、最近邻、标准化），需要模块化设计。
3. **可维护性**：分离数据加载、核心算法、实验逻辑、可视化、评估，便于后续扩展和调试。
4. **不需要 Project 级别**：无需分布式计算、复杂工程架构、多环境部署，Standard 足够。

### 不选择 Minimal 的原因
- Minimal（单/少文件）无法充分展示 KNN 核心特性（K 值敏感性、决策边界复杂度变化）。
- 6 个实验 + 8+ 张图的输出需求，单文件会导致代码臃肿、难以维护。

---

## 3. 目录结构草案（Project Tree Draft）

```
ml_knn/
├── PLAN.md                          # Phase 1 计划文档
├── ALGO_SPEC.md                     # Phase 2 算法规格书
├── IMPLEMENTATION_DESIGN.md         # Phase 3 实现设计文档（本文件）
├── README.md                        # 项目说明与快速开始
├── requirements.txt                 # Python 依赖
├── config.yaml                      # 实验配置文件（可选）
├── main.py                          # 主入口：运行所有实验
│
├── src/                             # 源代码目录
│   ├── __init__.py
│   ├── core/                        # 核心算法模块
│   │   ├── __init__.py
│   │   └── knn_classifier.py       # KNN 分类器实现
│   ├── data/                        # 数据加载与预处理
│   │   ├── __init__.py
│   │   ├── data_loader.py          # 数据加载（synthetic/CSV/Excel）
│   │   └── preprocessing.py        # 预处理（标准化/缺失值/编码）
│   ├── experiments/                 # 实验模块（6 个实验）
│   │   ├── __init__.py
│   │   ├── exp_a1_k_sensitivity.py         # A1: K 值敏感性分析
│   │   ├── exp_a2_distance_comparison.py   # A2: 距离度量对比
│   │   ├── exp_b1_baseline_comparison.py   # B1: 基线模型对照
│   │   ├── exp_c1_decision_boundary.py     # C1: 决策边界可视化
│   │   ├── exp_c2_nearest_neighbors.py     # C2: 最近邻样本可视化
│   │   └── exp_a3_scaling_impact.py        # A3: 特征标准化影响
│   ├── evaluation/                  # 评估模块
│   │   ├── __init__.py
│   │   └── metrics.py              # 指标计算（accuracy/f1/confusion_matrix）
│   └── visualization/               # 可视化模块
│       ├── __init__.py
│       ├── plot_curves.py          # 曲线图（K 值 vs 准确率）
│       ├── plot_decision_boundary.py  # 决策边界图
│       └── plot_comparison.py      # 对比图（柱状图/热力图/ROC）
│
├── data/                            # 用户数据目录（可选）
│   ├── .gitkeep                    # 保持目录存在
│   └── README.md                   # 数据格式说明
│
├── outputs/                         # 输出目录（自动生成）
│   ├── metrics/                    # 指标结果（CSV）
│   │   ├── k_sensitivity.csv
│   │   ├── distance_comparison.csv
│   │   ├── baseline_comparison.csv
│   │   └── scaling_impact.csv
│   ├── figures/                    # 图表（PNG）
│   │   ├── k_vs_accuracy.png
│   │   ├── decision_boundary_k1.png
│   │   ├── decision_boundary_k5.png
│   │   ├── decision_boundary_k20.png
│   │   ├── distance_comparison.png
│   │   ├── confusion_matrix_*.png
│   │   ├── baseline_comparison.png
│   │   ├── nearest_neighbors_*.png
│   │   └── scaling_comparison.png
│   └── logs/                       # 运行日志
│       └── experiment_log.txt
│
└── tests/                           # 单元测试（可选）
    ├── __init__.py
    ├── test_knn_classifier.py      # 测试 KNN 分类器
    └── test_data_loader.py         # 测试数据加载
```

---

## 4. 文件职责（File Responsibilities）

### 4.1 主入口


**`main.py`**
- **职责：** 
  - 解析命令行参数（data_path, target_col, experiment, seed）
  - 初始化输出目录（outputs/metrics, outputs/figures, outputs/logs）
  - 按顺序调用 6 个实验模块
  - 记录总运行时间和日志
- **输入：** CLI 参数
- **输出：** 控制台日志 + `outputs/logs/experiment_log.txt`

### 4.2 核心算法模块

**`src/core/knn_classifier.py`**
- **职责：** 
  - 实现 KNN 分类器类 `KNNClassifier`
  - 支持多种距离度量（euclidean, manhattan, minkowski）
  - 支持 K 值配置
  - 提供 `fit()` 和 `predict()` 方法
  - 提供 `get_neighbors()` 方法（用于可视化最近邻）
- **核心方法：**
  - `__init__(k, metric, p)`: 初始化
  - `fit(X_train, y_train)`: 存储训练数据
  - `predict(X_test)`: 预测测试样本
  - `_compute_distance(x1, x2)`: 计算距离
  - `get_neighbors(x, k)`: 获取 K 个最近邻（返回索引和距离）

### 4.3 数据加载与预处理

**`src/data/data_loader.py`**
- **职责：** 
  - 加载数据（synthetic/CSV/Excel/公开数据集）
  - 自动识别目标列（target/label 或用户指定）
  - 划分训练/测试集
- **核心函数：**
  - `load_data(mode='synthetic', data_path=None, target_col='target', test_size=0.3, random_state=42)`
  - `generate_synthetic_classification(n_samples, n_features, n_classes, random_state)`
  - `load_csv(path, target_col)`
  - `load_excel(path, target_col, sheet_name=0)`
  - `load_public_dataset(name)`: 加载 iris/wine/diabetes

**`src/data/preprocessing.py`**
- **职责：** 
  - 特征标准化（StandardScaler, MinMaxScaler, None）
  - 缺失值处理（均值填充/中位数填充/删除）
  - 类别特征编码（One-Hot, Label Encoding）
- **核心函数：**
  - `scale_features(X_train, X_test, method='standard')`
  - `handle_missing_values(X, strategy='mean')`
  - `encode_categorical(X, method='onehot')`

### 4.4 实验模块（6 个独立实验）

**`src/experiments/exp_a1_k_sensitivity.py`**
- **职责：** 实验 A1 - K 值敏感性分析
- **输入：** 训练/测试数据
- **输出：** 
  - `outputs/metrics/k_sensitivity.csv`（K 值、训练准确率、测试准确率）
  - `outputs/figures/k_vs_accuracy.png`（曲线图）
  - `outputs/figures/decision_boundary_k{1,5,20}.png`（决策边界）
- **核心逻辑：** 
  - 遍历 K ∈ {1, 3, 5, 10, 20, 50}
  - 训练 KNN 并计算训练/测试准确率
  - 对 K=1,5,20 生成二维决策边界图

**`src/experiments/exp_a2_distance_comparison.py`**
- **职责：** 实验 A2 - 距离度量对比
- **输入：** 训练/测试数据
- **输出：** 
  - `outputs/metrics/distance_comparison.csv`（距离度量、准确率、F1-score）
  - `outputs/figures/distance_comparison.png`（柱状图）
  - `outputs/figures/confusion_matrix_{metric}.png`（混淆矩阵）
- **核心逻辑：** 
  - 遍历距离度量 ∈ {euclidean, manhattan, minkowski(p=3)}
  - 固定 K=5，对比不同距离度量的效果

**`src/experiments/exp_b1_baseline_comparison.py`**
- **职责：** 实验 B1 - 基线模型对照
- **输入：** 训练/测试数据
- **输出：** 
  - `outputs/metrics/baseline_comparison.csv`（模型、准确率、训练时间、预测时间）
  - `outputs/figures/baseline_comparison.png`（柱状图）
  - `outputs/figures/roc_curve_comparison.png`（ROC 曲线）
- **核心逻辑：** 
  - 训练 KNN + Dummy Classifier + Logistic Regression + Decision Tree
  - 对比准确率、训练时间、预测时间
  - 绘制 ROC 曲线（多分类 OvR）

**`src/experiments/exp_c1_decision_boundary.py`**
- **职责：** 实验 C1 - 决策边界可视化
- **输入：** 二维合成数据（2 特征，3 类别）
- **输出：** 
  - `outputs/figures/decision_boundary_k{1,5,20}.png`（决策边界等高线图）
- **核心逻辑：** 
  - 生成二维非线性可分数据
  - 对 K=1,5,20 绘制决策边界等高线图
  - 叠加训练样本散点图

**`src/experiments/exp_c2_nearest_neighbors.py`**
- **职责：** 实验 C2 - 最近邻样本可视化
- **输入：** 训练/测试数据（二维）
- **输出：** 
  - `outputs/figures/nearest_neighbors_sample{1,2,3}.png`（最近邻散点图）
- **核心逻辑：** 
  - 选择 3 个测试样本
  - 对每个样本，获取 K=5 个最近邻
  - 绘制散点图，标注邻居类别和距离

**`src/experiments/exp_a3_scaling_impact.py`**
- **职责：** 实验 A3 - 特征标准化影响
- **输入：** 训练/测试数据
- **输出：** 
  - `outputs/metrics/scaling_impact.csv`（标准化方法、准确率）
  - `outputs/figures/scaling_comparison.png`（柱状图）
  - `outputs/figures/confusion_matrix_{scaling}.png`（混淆矩阵）
- **核心逻辑：** 
  - 对比无标准化、StandardScaler、MinMaxScaler
  - 固定 K=5，对比准确率和混淆矩阵

### 4.5 评估模块

**`src/evaluation/metrics.py`**
- **职责：** 
  - 计算分类指标（accuracy, f1_score, confusion_matrix）
  - 计算时间指标（train_time, predict_time）
  - 保存指标到 CSV
- **核心函数：**
  - `compute_accuracy(y_true, y_pred)`
  - `compute_f1_score(y_true, y_pred, average='weighted')`
  - `compute_confusion_matrix(y_true, y_pred)`
  - `save_metrics_to_csv(metrics_dict, output_path)`

### 4.6 可视化模块

**`src/visualization/plot_curves.py`**
- **职责：** 绘制曲线图（K 值 vs 准确率）
- **核心函数：**
  - `plot_k_vs_accuracy(k_values, train_acc, test_acc, output_path)`

**`src/visualization/plot_decision_boundary.py`**
- **职责：** 绘制决策边界等高线图
- **核心函数：**
  - `plot_decision_boundary(model, X, y, k, output_path)`

**`src/visualization/plot_comparison.py`**
- **职责：** 绘制对比图（柱状图、热力图、ROC 曲线）
- **核心函数：**
  - `plot_bar_comparison(data_dict, ylabel, title, output_path)`
  - `plot_confusion_matrix(cm, labels, output_path)`
  - `plot_roc_curves(models_dict, X_test, y_test, output_path)`

---

## 5. CLI 设计

### 5.1 默认行为（无参数）

```bash
python main.py
```

**行为：**
- 使用 synthetic 数据（`make_classification`，500 样本，10 特征，3 类别）
- 运行所有 6 个实验（A1, A2, B1, C1, C2, A3）
- 输出所有指标和图表到 `outputs/`
- 打印运行日志到控制台和 `outputs/logs/experiment_log.txt`

### 5.2 支持的 CLI 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_path` | str | None | 用户数据路径（CSV/Excel），若指定则不使用 synthetic |
| `--target_col` | str | 'target' | 目标列名（仅当 data_path 指定时有效） |
| `--experiment` | str | 'all' | 指定运行的实验（'all' 或 'A1'/'A2'/'B1'/'C1'/'C2'/'A3'） |
| `--seed` | int | 42 | 随机种子（用于 synthetic 数据生成和数据划分） |
| `--test_size` | float | 0.3 | 测试集比例 |
| `--k_default` | int | 5 | 默认 K 值（用于非 K 值敏感性实验） |

### 5.3 使用示例

**示例 1：默认运行（synthetic 数据，所有实验）**
```bash
python main.py
```

**示例 2：使用用户 CSV 数据**
```bash
python main.py --data_path data/input.csv --target_col label
```

**示例 3：只运行 K 值敏感性实验**
```bash
python main.py --experiment A1
```

**示例 4：指定随机种子**
```bash
python main.py --seed 123
```

---

## 6. 输出物规范

### 6.1 输出目录结构

```
outputs/
├── metrics/                    # 指标结果（CSV 格式）
│   ├── k_sensitivity.csv
│   ├── distance_comparison.csv
│   ├── baseline_comparison.csv
│   └── scaling_impact.csv
├── figures/                    # 图表（PNG 格式，300 DPI）
│   ├── k_vs_accuracy.png
│   ├── decision_boundary_k1.png
│   ├── decision_boundary_k5.png
│   ├── decision_boundary_k20.png
│   ├── distance_comparison.png
│   ├── confusion_matrix_euclidean.png
│   ├── confusion_matrix_manhattan.png
│   ├── confusion_matrix_minkowski.png
│   ├── baseline_comparison.png
│   ├── roc_curve_comparison.png
│   ├── nearest_neighbors_sample1.png
│   ├── nearest_neighbors_sample2.png
│   ├── nearest_neighbors_sample3.png
│   ├── scaling_comparison.png
│   ├── confusion_matrix_no_scaling.png
│   ├── confusion_matrix_standard.png
│   └── confusion_matrix_minmax.png
└── logs/                       # 运行日志
    └── experiment_log.txt
```

### 6.2 指标文件格式（CSV）

**`k_sensitivity.csv`**
```csv
k,train_accuracy,test_accuracy
1,1.0000,0.8533
3,0.9714,0.8800
5,0.9429,0.9000
10,0.9143,0.8867
20,0.8714,0.8667
50,0.8286,0.8400
```

**`distance_comparison.csv`**
```csv
metric,test_accuracy,f1_score
euclidean,0.9000,0.8985
manhattan,0.8867,0.8850
minkowski,0.8933,0.8920
```

**`baseline_comparison.csv`**
```csv
model,test_accuracy,train_time_sec,predict_time_sec
KNN,0.9000,0.0012,0.0345
Dummy,0.3333,0.0001,0.0002
LogisticRegression,0.8667,0.0234,0.0003
DecisionTree,0.8533,0.0156,0.0002
```

**`scaling_impact.csv`**
```csv
scaling_method,test_accuracy
none,0.6533
standard,0.9000
minmax,0.8967
```

### 6.3 图表规范

- **格式：** PNG
- **分辨率：** 300 DPI（适合论文/报告）
- **尺寸：** 10x6 英寸（曲线图）、8x6 英寸（柱状图）、6x6 英寸（混淆矩阵）
- **字体：** Arial, 12pt（标题）、10pt（坐标轴）
- **颜色：** 使用 seaborn 默认调色板（colorblind-friendly）

---

## 7. 依赖原则

### 7.1 核心依赖（必须）

```
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
pandas>=1.3.0
seaborn>=0.11.0
openpyxl>=3.0.0  # 用于读取 Excel
```

### 7.2 依赖原则

1. **不使用 PyTorch/TensorFlow**：KNN 是传统 ML 算法，无需深度学习框架。
2. **纯 CPU 实现**：所有计算基于 NumPy 和 scikit-learn，无 GPU 依赖。
3. **最小化依赖**：仅使用必要的科学计算和可视化库。
4. **版本兼容性**：支持 Python 3.8+。

### 7.3 可选依赖

```
pytest>=6.0.0  # 用于单元测试（可选）
```

---

## 8. 验证计划

### 8.1 冒烟测试（Smoke Test）

**目标：** 确保基本功能可运行，无崩溃。

**测试步骤：**
1. 安装依赖：`pip install -r requirements.txt`
2. 运行默认命令：`python main.py`
3. 检查输出目录：
   - `outputs/metrics/` 包含 4 个 CSV 文件
   - `outputs/figures/` 包含至少 8 张 PNG 图片
   - `outputs/logs/experiment_log.txt` 存在且无错误日志
4. 检查运行时间：< 5 分钟
5. 检查内存占用：< 2GB

**预期结果：**
- 所有实验成功运行
- 无 Python 异常或错误
- 输出文件完整

### 8.2 DoD 验证清单

| DoD 项 | 验证方法 | 通过标准 |
|--------|----------|----------|
| **6 个实验全部实现** | 检查 `src/experiments/` 下 6 个 .py 文件 | 文件存在且可导入 |
| **K 值特性可见** | 查看 `k_vs_accuracy.png` 和 `decision_boundary_k*.png` | 曲线呈现过拟合→最优→欠拟合趋势 |
| **距离度量特性可见** | 查看 `distance_comparison.png` | 不同距离度量准确率有差异 |
| **标准化特性可见** | 查看 `scaling_comparison.png` | 无标准化准确率显著低于标准化 |
| **默认 synthetic 可跑通** | 运行 `python main.py` | 无错误，输出完整 |
| **用户数据支持** | 运行 `python main.py --data_path data/test.csv` | 成功加载并运行 |
| **至少 1 指标** | 检查 `outputs/metrics/*.csv` | 至少包含 accuracy |
| **至少 1 图** | 检查 `outputs/figures/*.png` | 至少 8 张图 |
| **至少 1 对照** | 查看 `baseline_comparison.csv` | KNN vs 3 个基线模型 |
| **CPU 可运行** | 在无 GPU 环境运行 | 成功运行 |
| **< 5 分钟** | 计时 `python main.py` | 总时间 < 5 分钟 |
| **< 2GB 内存** | 监控内存占用 | 峰值 < 2GB |

### 8.3 单元测试（可选）

**测试文件：** `tests/test_knn_classifier.py`

**测试用例：**
- `test_knn_fit()`: 测试 fit 方法是否正确存储训练数据
- `test_knn_predict()`: 测试 predict 方法是否返回正确形状
- `test_distance_euclidean()`: 测试欧氏距离计算
- `test_distance_manhattan()`: 测试曼哈顿距离计算
- `test_get_neighbors()`: 测试最近邻获取

**运行命令：**
```bash
pytest tests/
```

---

## 9. 实现优先级

### Phase 4 实现顺序（建议）

1. **基础设施（优先级 P0）**
   - `main.py`（CLI 框架）
   - `src/data/data_loader.py`（synthetic 数据生成）
   - `outputs/` 目录初始化

2. **核心算法（优先级 P0）**
   - `src/core/knn_classifier.py`（KNN 分类器）
   - `src/data/preprocessing.py`（标准化）

3. **评估与可视化基础（优先级 P0）**
   - `src/evaluation/metrics.py`（accuracy 计算）
   - `src/visualization/plot_curves.py`（基础曲线图）

4. **实验实现（优先级 P1）**
   - `src/experiments/exp_a1_k_sensitivity.py`（最核心实验）
   - `src/experiments/exp_c1_decision_boundary.py`（最直观可视化）
   - `src/experiments/exp_a3_scaling_impact.py`（验证预处理必要性）

5. **剩余实验（优先级 P2）**
   - `src/experiments/exp_a2_distance_comparison.py`
   - `src/experiments/exp_b1_baseline_comparison.py`
   - `src/experiments/exp_c2_nearest_neighbors.py`

6. **完善功能（优先级 P3）**
   - 用户数据加载（CSV/Excel）
   - 完整可视化（混淆矩阵、ROC 曲线）
   - README.md 和文档

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **决策边界可视化仅支持 2D** | 高维数据无法可视化 | 实验 C1 使用 PCA 降维到 2D，或仅在 2D synthetic 数据上演示 |
| **大数据集运行时间超限** | 超过 5 分钟 | 限制 synthetic 数据规模 <= 1000 样本 |
| **用户数据格式不规范** | 加载失败 | 提供详细错误提示和数据格式说明（data/README.md） |
| **K 值过大导致内存溢出** | 程序崩溃 | 限制 K 值上限 <= min(50, n_samples//2) |

---

**设计文档完成日期：** 2026-01-18
