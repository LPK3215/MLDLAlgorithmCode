# KNN 算法案例项目

## 项目简介

本项目实现了 K-Nearest Neighbors (KNN) 算法的完整案例演示，包含 6 个实验，展示 KNN 的核心特性：
- K 值对决策边界的影响
- 距离度量方式的影响
- 特征标准化的必要性
- 与基线模型的对比
- 决策边界可视化
- 最近邻样本可解释性

## 快速开始

### 0. 激活虚拟环境（必须）

**重要：本项目依赖于 `mldl_algo311` 虚拟环境，所有操作必须在该环境下进行。**

```bash
# 激活虚拟环境
conda activate mldl_algo311

# 进入项目根目录
cd MLDLAlgorithmCode

# 进入 KNN 子项目目录
cd ml_knn
```

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行默认演示（使用合成数据）

**方法 1：使用快速启动脚本（推荐）**

```bash
# Windows 用户
run.bat

# 脚本会自动：
# 1. 激活虚拟环境 mldl_algo311
# 2. 检查并安装依赖
# 3. 运行项目
```

**方法 2：手动运行**

```bash
python main.py
```

这将运行所有 6 个实验，输出结果到 `outputs/` 目录。

### 3. 使用自定义数据

```bash
python main.py --data_path data/input.csv --target_col label
```

## CLI 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_path` | str | None | 用户数据路径（CSV/Excel） |
| `--target_col` | str | 'target' | 目标列名 |
| `--experiment` | str | 'all' | 指定实验（'all' 或 'A1'/'A2'/'B1'/'C1'/'C2'/'A3'） |
| `--seed` | int | 42 | 随机种子 |
| `--test_size` | float | 0.3 | 测试集比例 |
| `--k_default` | int | 5 | 默认 K 值 |

## 输出结果

运行后，结果将保存在 `outputs/` 目录：

```
outputs/
├── metrics/          # 指标结果（CSV）
├── figures/          # 图表（PNG）
└── logs/             # 运行日志
```

## 实验列表

- **A1**: K 值敏感性分析
- **A2**: 距离度量对比
- **B1**: 基线模型对照
- **C1**: 决策边界可视化
- **C2**: 最近邻样本可视化
- **A3**: 特征标准化影响

## 项目结构

```
ml_knn/
├── main.py                      # 主入口
├── src/
│   ├── core/                    # KNN 分类器
│   ├── data/                    # 数据加载 + 预处理
│   ├── experiments/             # 6 个实验模块
│   ├── evaluation/              # 指标计算
│   └── visualization/           # 可视化
├── data/                        # 用户数据（可选）
└── outputs/                     # 输出结果
```

## 系统要求

- **虚拟环境：** mldl_algo311（必须）
- Python 3.8+
- CPU 单核可运行
- 内存 < 2GB
- 运行时间 < 5 分钟

## 环境配置

本项目是 `MLDLAlgorithmCode` 总仓库的子项目，依赖于统一的虚拟环境 `mldl_algo311`。

**首次使用前，请确保已激活虚拟环境：**

```bash
conda activate mldl_algo311
cd MLDLAlgorithmCode/ml_knn
```

## 许可证

MIT License
