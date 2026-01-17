# KNN 算法规格书 (ALGO_SPEC.md)

---

## 1. 基本信息

- **算法名称（原始输入）：** KNN
- **规范化名称：** knn
- **分类：** 机器学习 (ML)
- **分类理由：** KNN (K-Nearest Neighbors) 是经典的基于实例的机器学习算法，使用距离度量进行分类/回归，不涉及神经网络训练，属于传统机器学习范畴。
- **本规格书日期：** 2026-01-18

---

## 2. 算法定位（What / When）

### 任务类型
- 分类（Classification）
- 回归（Regression）

### 最典型应用场景
**多类别分类问题**：根据样本特征将其归类到 K 个最近邻居的多数类别中，例如鸢尾花品种分类、手写数字识别。

### 适用条件
1. **小到中等规模数据集**：KNN 是懒惰学习算法，预测时需遍历所有训练样本，数据量过大会导致预测缓慢。
2. **特征空间有意义的距离度量**：特征之间的距离能够反映样本相似性（需要特征标准化）。
3. **类别边界不规则**：KNN 能够拟合复杂的非线性决策边界，适合局部模式明显的数据。

### 局限
1. **计算成本高**：预测时需要计算与所有训练样本的距离，时间复杂度 O(n)，不适合大规模数据。
2. **对特征尺度敏感**：不同量纲的特征会导致距离计算偏差，必须进行标准化。
3. **维度灾难**：高维空间中距离度量失效，所有点距离趋于相等，分类效果下降。

---

## 3. 数据契约（Data Contract）

### 输入数据形态
- **主要形态：** tabular（表格数据）
- **特征类型：** 数值型特征（连续或离散）

### 是否监督学习
- **是**，需要标签数据进行训练。
- **标签提供方式：** 
  - 分类任务：类别标签（整数或字符串）
  - 回归任务：连续数值目标

### 默认数据策略（强制）
- **默认使用 synthetic（合成数据）**：
  - 使用 `sklearn.datasets.make_classification` 生成多类别分类数据
  - 使用 `sklearn.datasets.make_regression` 生成回归数据
  - 使用 `sklearn.datasets.make_blobs` 生成聚类分布数据
- **若必须用公开数据**：
  - 鸢尾花数据集（Iris）：`sklearn.datasets.load_iris()`，150 样本，4 特征，3 类别
  - 葡萄酒数据集（Wine）：`sklearn.datasets.load_wine()`，178 样本，13 特征，3 类别
  - 波士顿房价（回归）：`sklearn.datasets.load_diabetes()`，442 样本，10 特征

### 用户数据接口（支持但非必须）
- **CSV 格式：** `data/input.csv`
- **Excel 格式：** `data/input.xlsx`（默认读取第一个 sheet）
- **目标列名：** 默认为 `target` 或 `label`，若不存在需通过 `target_col` 参数指定
- **特征列：** 除目标列外的所有数值列

### 必要预处理与原因
1. **特征标准化（StandardScaler 或 MinMaxScaler）**：
   - **原因：** KNN 基于距离计算，不同量纲的特征会导致距离被大数值特征主导，必须标准化到相同尺度。
2. **缺失值处理（均值/中位数填充或删除）**：
   - **原因：** 距离计算无法处理缺失值，必须填充或删除含缺失值的样本。
3. **类别特征编码（One-Hot 或 Label Encoding）**：
   - **原因：** KNN 只能处理数值特征，类别特征需转换为数值表示。

---

## 4. 核心价值点与标志性特性（Signature Value）

### 核心价值点
1. **非参数模型，无需训练**：KNN 不需要学习参数，直接存储训练数据，预测时实时计算，适合动态更新数据场景。
2. **局部决策边界灵活**：能够拟合复杂的非线性决策边界，对局部模式敏感。
3. **直观可解释**：预测结果基于最近邻样本的多数投票，易于理解和解释。

### 必须通过代码"看得见"的特性点
1. **K 值对决策边界的影响**：
   - 小 K 值（如 K=1）：决策边界复杂，容易过拟合，对噪声敏感。
   - 大 K 值（如 K=50）：决策边界平滑，容易欠拟合，分类过于保守。
   - **可视化：** 绘制不同 K 值下的二维决策边界图，展示边界复杂度变化。

2. **距离度量方式的影响**：
   - 欧氏距离（Euclidean）：适合各向同性数据。
   - 曼哈顿距离（Manhattan）：适合高维稀疏数据。
   - **对照实验：** 在同一数据集上对比不同距离度量的分类准确率。

3. **特征标准化的必要性**：
   - 未标准化：大数值特征主导距离计算，分类效果差。
   - 标准化后：所有特征权重平等，分类效果显著提升。
   - **对照实验：** 对比标准化前后的分类准确率和混淆矩阵。

---

## 5. 实验设计（>=3，且覆盖三类）

### 实验 A1：K 值敏感性分析（特性实验）
- **目的：** 验证 K 值对模型复杂度和泛化能力的影响。
- **变量：** K ∈ {1, 3, 5, 10, 20, 50}
- **指标：** 训练集准确率、测试集准确率
- **可视化输出：** 
  - K 值 vs 准确率曲线图（训练集和测试集双曲线）
  - K=1, K=5, K=20 的二维决策边界对比图
- **预期现象：** 
  - K=1 时训练集准确率 100%，测试集准确率较低（过拟合）
  - K 增大时训练集准确率下降，测试集准确率先升后降（存在最优 K 值）
  - 决策边界从复杂锯齿状变为平滑区域

### 实验 A2：距离度量对比（特性实验）
- **目的：** 验证不同距离度量对分类效果的影响。
- **变量：** 距离度量 ∈ {euclidean, manhattan, minkowski(p=3)}
- **指标：** 测试集准确率、F1-score
- **可视化输出：** 
  - 不同距离度量的准确率柱状图
  - 不同距离度量的混淆矩阵热力图
- **预期现象：** 
  - 欧氏距离在各向同性数据上表现最好
  - 曼哈顿距离在高维数据上可能更稳定

### 实验 B1：与基线模型对照（对照实验）
- **目的：** 验证 KNN 相对于简单基线的优势。
- **对照组：** 
  - Baseline 1：随机猜测（Dummy Classifier - stratified）
  - Baseline 2：逻辑回归（Logistic Regression）
  - Baseline 3：决策树（Decision Tree）
- **指标：** 测试集准确率、训练时间、预测时间
- **可视化输出：** 
  - 模型对比柱状图（准确率、训练时间、预测时间）
  - ROC 曲线对比（多分类 OvR）
- **预期现象：** 
  - KNN 准确率优于随机猜测和简单线性模型
  - KNN 预测时间显著高于参数模型（逻辑回归、决策树）

### 实验 C1：决策边界可视化（可视化/解释输出）
- **目的：** 直观展示 KNN 的局部决策机制。
- **变量：** 二维合成数据（2 特征，3 类别，非线性可分）
- **指标：** 无（纯可视化）
- **可视化输出：** 
  - 二维特征空间的决策边界等高线图
  - 训练样本散点图叠加在决策边界上
  - 不同 K 值（K=1, 5, 20）的决策边界对比
- **预期现象：** 
  - K=1 时决策边界呈现 Voronoi 图形态，每个训练样本周围形成独立区域
  - K 增大时决策边界变得平滑，局部细节减少

### 实验 C2：最近邻样本可视化（可视化/解释输出）
- **目的：** 展示 KNN 预测的可解释性（基于哪些邻居做出决策）。
- **变量：** 选择 3 个测试样本
- **指标：** 无（纯可视化）
- **可视化输出：** 
  - 对每个测试样本，绘制其 K 个最近邻训练样本的散点图
  - 标注邻居的类别和距离
  - 显示最终预测类别和投票分布
- **预期现象：** 
  - 测试样本的预测类别与多数邻居类别一致
  - 距离越近的邻居对预测影响越大（可选：距离加权）

### 实验 A3：特征标准化影响（特性实验）
- **目的：** 验证特征标准化对 KNN 的必要性。
- **变量：** 标准化方式 ∈ {无标准化, StandardScaler, MinMaxScaler}
- **指标：** 测试集准确率
- **可视化输出：** 
  - 标准化前后的准确率对比柱状图
  - 标准化前后的混淆矩阵对比
- **预期现象：** 
  - 未标准化时，大数值特征主导距离计算，准确率显著下降
  - StandardScaler 和 MinMaxScaler 效果接近，均显著优于未标准化

---

## 6. 输出物与验收标准（DoD）

### 最小验收标准
- **至少 1 指标：** 测试集准确率（Accuracy）
- **至少 1 图：** K 值 vs 准确率曲线图
- **至少 1 对照：** KNN vs 随机猜测基线

### 完整验收标准
- **实验完整性：** 上述 6 个实验（A1, A2, B1, C1, C2, A3）全部实现并输出结果
- **特性点可见：** 
  - K 值对决策边界的影响（实验 A1 + C1）
  - 距离度量的影响（实验 A2）
  - 特征标准化的必要性（实验 A3）
- **默认数据可跑通：** 使用 synthetic 数据（`make_classification`）能够完整运行所有实验
- **用户数据支持：** 支持读取 CSV/Excel 格式的用户数据，自动识别目标列
- **最小可运行路径：** 
  - 单核 CPU 可运行
  - 内存 < 2GB
  - 运行时间 < 5 分钟（所有实验）
  - 数据规模：训练集 <= 1000 样本

### 输出文件清单
```
outputs/
├── metrics/
│   ├── k_sensitivity.csv          # K 值敏感性分析结果
│   ├── distance_comparison.csv    # 距离度量对比结果
│   ├── baseline_comparison.csv    # 基线模型对比结果
│   └── scaling_impact.csv         # 标准化影响结果
├── figures/
│   ├── k_vs_accuracy.png          # K 值 vs 准确率曲线
│   ├── decision_boundary_k1.png   # K=1 决策边界
│   ├── decision_boundary_k5.png   # K=5 决策边界
│   ├── decision_boundary_k20.png  # K=20 决策边界
│   ├── distance_comparison.png    # 距离度量对比柱状图
│   ├── confusion_matrix_*.png     # 各实验混淆矩阵
│   ├── baseline_comparison.png    # 基线模型对比图
│   ├── nearest_neighbors_*.png    # 最近邻样本可视化
│   └── scaling_comparison.png     # 标准化影响对比图
└── logs/
    └── experiment_log.txt         # 实验运行日志
```

---

## 7. 复杂度等级与实现建议

### 复杂度等级
**Standard**

### 理由
- **算法本身简单**：KNN 核心逻辑仅需距离计算 + 排序 + 投票，无需复杂数学推导。
- **实验设计丰富**：需要 6 个实验覆盖特性分析、对照实验、可视化，需要一定工程量。
- **可视化要求高**：决策边界、最近邻样本、对比图表等需要精心设计。
- **不适合 Minimal**：Minimal 无法充分展示 KNN 的核心特性（K 值敏感性、决策边界）。
- **不需要 Project**：无需复杂工程架构、分布式计算或深度优化。

### 最小实现边界（Standard 级别）
即使是 Standard 级别，也必须保证闭环：
1. **核心算法实现**：
   - KNN 分类器（支持多类别）
   - 距离计算（欧氏、曼哈顿、闵可夫斯基）
   - K 值选择（交叉验证）
2. **数据处理**：
   - 合成数据生成
   - 特征标准化
   - 训练/测试集划分
3. **实验执行**：
   - 6 个实验全部实现
   - 自动保存结果到 `outputs/`
4. **可视化**：
   - 至少 8 张图（K 值曲线、决策边界 x3、对比图 x2、混淆矩阵 x2）
5. **运行入口**：
   - 单命令运行所有实验：`python main.py`
   - 支持配置文件指定参数

---

## 8. 结构化规格块（YAML）

```yaml
algo: knn
category: ml
task_type: 
  - classification
  - regression
input_type: tabular
complexity_level: Standard
default_data_mode: synthetic
user_data_supported: true
target_col_default: target

data_generation:
  synthetic:
    classification: sklearn.datasets.make_classification
    regression: sklearn.datasets.make_regression
    blobs: sklearn.datasets.make_blobs
  public_datasets:
    - iris: sklearn.datasets.load_iris
    - wine: sklearn.datasets.load_wine
    - diabetes: sklearn.datasets.load_diabetes

preprocessing:
  - feature_scaling:
      methods: [StandardScaler, MinMaxScaler]
      required: true
      reason: "KNN 基于距离计算，不同量纲特征会主导距离"
  - missing_values:
      methods: [mean_imputation, median_imputation, drop]
      required: true
      reason: "距离计算无法处理缺失值"
  - categorical_encoding:
      methods: [OneHot, LabelEncoding]
      required: false
      reason: "仅当存在类别特征时需要"

experiments:
  - id: A1
    name: k_sensitivity
    type: feature
    purpose: "验证 K 值对模型复杂度和泛化能力的影响"
    variables:
      k_values: [1, 3, 5, 10, 20, 50]
    metrics: [train_accuracy, test_accuracy]
    figures: 
      - k_vs_accuracy_curve
      - decision_boundary_k1
      - decision_boundary_k5
      - decision_boundary_k20
    
  - id: A2
    name: distance_comparison
    type: feature
    purpose: "验证不同距离度量对分类效果的影响"
    variables:
      distance_metrics: [euclidean, manhattan, minkowski]
    metrics: [test_accuracy, f1_score]
    figures:
      - distance_comparison_bar
      - confusion_matrix_per_metric
    
  - id: B1
    name: baseline_comparison
    type: contrast
    purpose: "验证 KNN 相对于简单基线的优势"
    baselines:
      - dummy_classifier
      - logistic_regression
      - decision_tree
    metrics: [test_accuracy, train_time, predict_time]
    figures:
      - model_comparison_bar
      - roc_curve_comparison
    
  - id: C1
    name: decision_boundary_visualization
    type: visualization
    purpose: "直观展示 KNN 的局部决策机制"
    variables:
      k_values: [1, 5, 20]
    figures:
      - decision_boundary_contour
      - training_samples_scatter
    
  - id: C2
    name: nearest_neighbors_visualization
    type: visualization
    purpose: "展示 KNN 预测的可解释性"
    variables:
      test_samples: 3
    figures:
      - nearest_neighbors_scatter_per_sample
    
  - id: A3
    name: scaling_impact
    type: feature
    purpose: "验证特征标准化对 KNN 的必要性"
    variables:
      scaling_methods: [none, StandardScaler, MinMaxScaler]
    metrics: [test_accuracy]
    figures:
      - scaling_comparison_bar
      - confusion_matrix_comparison

metrics:
  - accuracy
  - f1_score
  - confusion_matrix
  - train_time
  - predict_time

figures:
  - k_vs_accuracy_curve
  - decision_boundary_contour
  - distance_comparison_bar
  - baseline_comparison_bar
  - roc_curve_comparison
  - nearest_neighbors_scatter
  - confusion_matrix_heatmap
  - scaling_comparison_bar

outputs_dir: outputs/
outputs_structure:
  metrics: outputs/metrics/
  figures: outputs/figures/
  logs: outputs/logs/

runtime_requirements:
  cpu: 1_core
  memory: 2GB
  time: 5_minutes
  max_samples: 1000

dependencies:
  - numpy
  - scikit-learn
  - matplotlib
  - pandas
  - seaborn
```

---

## 附录：数学原理简述（供实现参考）

### KNN 分类算法流程
1. **输入：** 训练集 D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}，测试样本 x_test，超参数 K
2. **计算距离：** 计算 x_test 与所有训练样本的距离 d(x_test, xᵢ)
3. **选择邻居：** 选择距离最小的 K 个训练样本
4. **投票决策：** 
   - 分类：返回 K 个邻居中出现次数最多的类别
   - 回归：返回 K 个邻居目标值的平均值

### 距离度量公式
- **欧氏距离：** d(x, y) = √(Σ(xᵢ - yᵢ)²)
- **曼哈顿距离：** d(x, y) = Σ|xᵢ - yᵢ|
- **闵可夫斯基距离：** d(x, y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)

### K 值选择策略
- **交叉验证：** 在验证集上尝试不同 K 值，选择准确率最高的 K
- **经验规则：** K = √n（n 为训练样本数）
- **奇数优先：** 二分类问题中使用奇数 K 避免平票

---

**规格书完成日期：** 2026-01-18
