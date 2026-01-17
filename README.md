# MLDLAlgorithmCode

> 机器学习与深度学习核心算法实现项目  
> 系统化实现 30 个核心算法，规范化开发流程，可复现实验结果

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 目录

- [项目简介](#项目简介)
- [核心算法清单](#核心算法清单)
- [技术栈](#技术栈)
- [环境配置](#环境配置)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [开发流程](#开发流程)
- [项目特点](#项目特点)
- [项目状态](#项目状态)
- [文档导航](#文档导航)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目简介

本项目旨在**系统化地实现和演示 30 个机器学习和深度学习核心算法**，采用规范化的开发流程，确保每个算法都具有：

- 📚 **完整的文档**：算法规格书、实现设计、快速开始指南
- 💻 **清晰的实现**：模块化代码、中文注释、易于理解
- 🔬 **可复现的实验**：至少 3 个实验，验证算法核心特性
- 📊 **可视化结果**：图表展示、指标输出、结果分析

### 🎯 项目目标

1. **学习目标**：深入理解算法原理和实现细节
2. **实践目标**：掌握从零实现算法的能力
3. **工程目标**：培养规范化的项目开发习惯
4. **应用目标**：能够将算法应用到实际问题

---

## 核心算法清单

> 详细算法清单请查看：[📚 1-核心算法清单.md](docs/1-核心算法清单.md)

### 机器学习算法（17 个）

| 序号 | 算法名称 | 英文名称 | 状态 |
|------|---------|---------|------|
| 1 | 线性回归 | Linear Regression | ⏳ 待开发 |
| 2 | 逻辑回归 | Logistic Regression | ⏳ 待开发 |
| 3 | 感知机 | Perceptron | ⏳ 待开发 |
| 4 | 朴素贝叶斯 | Naive Bayes | ⏳ 待开发 |
| 5 | K近邻 | K-Nearest Neighbors | ✅ 已完成 |
| 6 | 决策树 | Decision Tree | ⏳ 待开发 |
| 7 | 随机森林 | Random Forest | ⏳ 待开发 |
| 8 | Boosting | AdaBoost/GBDT/XGBoost | ⏳ 待开发 |
| 9 | 支持向量机 | Support Vector Machine | ⏳ 待开发 |
| 10 | K-Means聚类 | K-Means Clustering | ⏳ 待开发 |
| 11 | 层次聚类 | Hierarchical Clustering | ⏳ 待开发 |
| 12 | DBSCAN | DBSCAN | ⏳ 待开发 |
| 13 | 高斯混合模型 | Gaussian Mixture Model | ⏳ 待开发 |
| 14 | 主成分分析 | Principal Component Analysis | ⏳ 待开发 |
| 15 | 线性判别分析 | Linear Discriminant Analysis | ⏳ 待开发 |
| 16 | 隐马尔可夫模型 | Hidden Markov Model | ⏳ 待开发 |
| 17 | 条件随机场 | Conditional Random Field | ⏳ 待开发 |

### 深度学习算法（13 个）

| 序号 | 算法名称 | 英文名称 | 状态 |
|------|---------|---------|------|
| 1 | 多层感知机 | Multi-Layer Perceptron | ⏳ 待开发 |
| 2 | LeNet | LeNet | ⏳ 待开发 |
| 3 | AlexNet | AlexNet | ⏳ 待开发 |
| 4 | VGG | VGG | ⏳ 待开发 |
| 5 | ResNet | ResNet | ⏳ 待开发 |
| 6 | Vanilla RNN | Vanilla RNN | ⏳ 待开发 |
| 7 | LSTM | Long Short-Term Memory | ⏳ 待开发 |
| 8 | GRU | Gated Recurrent Unit | ⏳ 待开发 |
| 9 | 注意力机制与Transformer | Attention & Transformer | ⏳ 待开发 |
| 10 | 自编码器 | AutoEncoder | ⏳ 待开发 |
| 11 | 变分自编码器 | Variational AutoEncoder | ⏳ 待开发 |
| 12 | 生成对抗网络 | Generative Adversarial Network | ⏳ 待开发 |
| 13 | 图神经网络 | Graph Neural Network | ⏳ 待开发 |
| 14 | 深度强化学习 | Deep Reinforcement Learning | ⏳ 待开发 |

**进度统计**：✅ 已完成 1/30 | ⏳ 待开发 29/30 | 完成率：3.3%

---

## 技术栈

### 核心技术

| 类别 | 技术 | 版本 | 说明 |
|------|------|------|------|
| 编程语言 | Python | 3.11 | 主要开发语言 |
| 深度学习框架 | PyTorch | 2.5.1 | 支持 CUDA 12.1 |
| 机器学习库 | scikit-learn | latest | 传统机器学习算法 |
| 数值计算 | numpy | latest | 数组和矩阵运算 |
| 数据处理 | pandas | latest | 数据加载和处理 |
| 科学计算 | scipy | latest | 科学计算工具 |
| 可视化 | matplotlib | latest | 图表绘制 |
| 进度条 | tqdm | latest | 进度显示 |
| 配置管理 | pyyaml | latest | YAML 配置文件 |

### 开发工具

- **环境管理**：Conda
- **版本控制**：Git
- **代码编辑**：任意 Python IDE（推荐 VS Code / PyCharm）
- **文档格式**：Markdown

---

## 环境配置

> 详细环境配置请查看：[🔧 2-虚拟环境搭建.md](docs/2-虚拟环境搭建.md)
- **其他工具**: tqdm, pyyaml

## 环境配置

### 📌 虚拟环境统一说明

**本项目所有子项目共享统一的虚拟环境：`mldl_algo311`**

**重要：每次使用本项目或任何子项目前，必须先激活虚拟环境！**

### 🔧 环境配置步骤

#### 1. 创建虚拟环境（首次使用）

```bash
# 创建 Python 3.11 虚拟环境
conda create -n mldl_algo311 python=3.11 -y

# 激活虚拟环境
conda activate mldl_algo311
```

#### 2. 激活虚拟环境（每次使用前）

```bash
conda activate mldl_algo311
```

#### 3. 安装全局依赖（可选）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas matplotlib scikit-learn tqdm pyyaml
```

### 🔍 验证环境

```bash
# 查看当前激活的环境
conda info --envs

# 检查 Python 版本
python --version  # 应该显示 Python 3.11.x

# 查看已安装的包
pip list
```

### ⚠️ 常见问题

**Q1: 忘记激活虚拟环境会怎样？**

如果未激活虚拟环境直接运行子项目，可能会遇到：
- `ModuleNotFoundError`：缺少依赖包
- 版本冲突：使用了系统全局 Python 环境
- 运行失败：依赖包版本不匹配

**解决方法：** 始终先执行 `conda activate mldl_algo311`

**Q2: 如何退出虚拟环境？**

```bash
conda deactivate
```

**Q3: 如何删除虚拟环境？**

```bash
conda remove -n mldl_algo311 --all
```

## 项目结构

```
MLDLAlgorithmCode/
├── README.md                              # 📖 项目说明文档
├── .gitignore                             # Git 忽略配置
├── docs/                                  # 📚 项目文档目录
│   ├── 1-核心算法清单.md                  # 30 个算法详细清单
│   ├── 2-虚拟环境搭建.md                  # 环境配置详细说明
│   ├── 3-提示词-生成算法项目.md           # AI 生成项目工作流程
│   └── 4-提示词-学习算法项目.md           # AI 学习项目助手
├── ml_knn/                                # ✅ KNN 算法子项目（示例）
│   ├── docs/                             # 子项目文档
│   │   ├── PLAN.md                       # 项目计划
│   │   ├── ALGO_SPEC.md                  # 算法规格书
│   │   ├── IMPLEMENTATION_DESIGN.md      # 实现设计文档
│   │   └── QUICK_START.md                # 快速开始指南
│   ├── src/                              # 源代码
│   │   ├── core/                         # 核心算法实现
│   │   ├── data/                         # 数据加载和预处理
│   │   ├── experiments/                  # 实验模块
│   │   ├── evaluation/                   # 评估指标
│   │   └── visualization/                # 可视化模块
│   ├── data/                             # 数据目录
│   ├── outputs/                          # 输出结果
│   │   ├── figures/                      # 图表
│   │   ├── metrics/                      # 指标
│   │   └── logs/                         # 日志
│   ├── README.md                         # 子项目说明
│   ├── requirements.txt                  # 依赖包
│   ├── main.py                           # 主入口
│   └── run.bat                           # 快速启动脚本
└── [其他算法]/                            # ⏳ 其他算法子项目（待创建）
```

---

## 快速开始

### 方式一：使用已有项目（以 KNN 为例）

```powershell
# Step 1: 激活虚拟环境（必须）
conda activate mldl_algo311

# Step 2: 进入项目目录
cd MLDLAlgorithmCode/ml_knn

# Step 3: 安装依赖
pip install -r requirements.txt

# Step 4: 运行项目
python main.py

# 或使用快速启动脚本（Windows）
run.bat
```

### 方式二：创建新的算法项目

```powershell
# Step 1: 激活虚拟环境
conda activate mldl_algo311

# Step 2: 使用 AI 助手生成项目
# 复制 docs/3-提示词-生成算法项目.md 中的提示词给 AI
# AI 会引导你完成 Phase 0-5 的项目创建流程
```

### 方式三：学习已有项目

```powershell
# 使用 AI 助手学习项目
# 复制 docs/4-提示词-学习算法项目.md 中的提示词给 AI
# AI 会详细讲解项目的原理、代码和实验
```

---

## 开发流程

> 详细开发流程请查看：[🚀 3-提示词-生成算法项目.md](docs/3-提示词-生成算法项目.md)

每个算法的实现遵循**规范化的五阶段流程**：

### Phase 0：启动与输入收集
- 📋 输出计划表
- 🤔 询问算法名称
- 🏷️ 自动判断 ML/DL 分类

### Phase 1：创建子项目文件夹
- 📁 创建项目目录结构
- 📝 生成 PLAN.md（项目计划）

### Phase 2：生成算法规格书
- 📚 分析算法原理和应用场景
- 🔬 设计实验方案（≥3个）
- 📄 输出 ALGO_SPEC.md

### Phase 3：生成实现设计文档
- 🏗️ 设计代码结构
- 📐 规划模块职责
- 📄 输出 IMPLEMENTATION_DESIGN.md

### Phase 4：生成代码实现
- 💻 实现核心算法
- 🔬 实现所有实验
- 📊 生成可视化和指标
- 📝 添加中文注释
- 📄 生成 README.md 和 QUICK_START.md

### Phase 5：验证与修复
- ✅ 15 项验收清单检查
- 🐛 修复问题
- 📊 输出运行命令和结果
- 🔄 自动更新根项目 README.md

---

## 项目特点

### 🎯 核心特点

| 特点 | 说明 |
|------|------|
| 📋 **规范化流程** | 统一的 Phase 0-5 开发流程，确保质量和可追溯性 |
| 🎲 **合成数据优先** | 默认使用自动生成的数据，无需准备真实数据集 |
| 🔬 **实验驱动** | 每个算法至少 3 个实验（特性/对照/可视化） |
| 🔄 **可复现性** | 固定随机种子，确保结果一致 |
| 💻 **低资源友好** | 提供 CPU 可运行的实现方案 |
| 📝 **中文注释** | 所有代码都有详细的中文注释 |
| 📚 **完整文档** | 每个项目都有完整的文档体系 |
| 🤖 **AI 辅助** | 提供 AI 提示词，辅助项目生成和学习 |

### 🌟 项目亮点

1. **从零实现**：不依赖现成库，深入理解算法原理
2. **模块化设计**：清晰的代码结构，易于理解和扩展
3. **实验验证**：通过实验验证算法的核心特性
4. **可视化展示**：图表直观展示算法行为
5. **工程化实践**：培养规范化的项目开发习惯

---

## 项目状态

### 📊 已完成的算法

| 算法 | 目录 | 分类 | 状态 | 实验数 | 说明 |
|------|------|------|------|--------|------|
| K-Nearest Neighbors | [`ml_knn/`](ml_knn/) | ML | ✅ 完成 | 6 | K 值敏感性、距离度量、基线对照等 |

### 🚧 进行中的算法

暂无

### ⏳ 待开发的算法

查看 [核心算法清单](#核心算法清单) 了解所有待开发算法

---

## 文档导航

### 📚 核心文档

| 文档 | 说明 | 链接 |
|------|------|------|
| 核心算法清单 | 30 个算法的详细清单和分类 | [📚 查看](docs/1-核心算法清单.md) |
| 虚拟环境搭建 | 环境配置的详细步骤和常见问题 | [🔧 查看](docs/2-虚拟环境搭建.md) |
| 生成算法项目 | 使用 AI 生成新算法项目的提示词 | [🚀 查看](docs/3-提示词-生成算法项目.md) |
| 学习算法项目 | 使用 AI 学习已有项目的提示词 | [🎓 查看](docs/4-提示词-学习算法项目.md) |

### 🎯 子项目文档

| 项目 | 说明 | 链接 |
|------|------|------|
| KNN 算法 | K 近邻算法完整实现 | [📖 查看](ml_knn/README.md) |

---

## 贡献指南

### 🤝 如何贡献

欢迎贡献新的算法实现！请遵循以下步骤：

1. **Fork 本项目**
2. **创建新分支**：`git checkout -b feature/new-algorithm`
3. **使用提示词生成项目**：参考 `docs/3-提示词-生成算法项目.md`
4. **完成 Phase 0-5**：确保通过所有验收清单
5. **提交代码**：`git commit -m "Add: 新算法名称"`
6. **推送分支**：`git push origin feature/new-algorithm`
7. **创建 Pull Request**

### ✅ 代码规范

- 遵循 Phase 0-5 开发流程
- 所有代码必须有中文注释
- 至少包含 3 个实验
- 通过 15 项验收清单
- 使用 `>=` 版本规范（requirements.txt）

### 📝 文档规范

- README.md 在子项目根目录
- 过程文档（PLAN/ALGO_SPEC/IMPLEMENTATION_DESIGN/QUICK_START）在 docs/ 目录
- 所有文档使用 Markdown 格式
- 包含完整的环境配置说明

---

## 许可证

本项目采用 MIT 许可证，仅供学习和研究使用。

---

## 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 📧 Email：[你的邮箱]
- 💬 Issues：[GitHub Issues](https://github.com/你的用户名/MLDLAlgorithmCode/issues)

---

## 致谢

感谢所有为本项目做出贡献的开发者！

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个 Star！⭐**

Made with ❤️ by [你的名字]

</div>
