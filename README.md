# MLDLAlgorithmCode

机器学习与深度学习核心算法实现项目

## 项目简介

本项目旨在系统化地实现和演示30个机器学习和深度学习核心算法，采用规范化的开发流程，确保每个算法都具有完整的文档、清晰的实现和可复现的实验结果。

## 核心算法清单

### 机器学习算法（17个）

1. 线性回归（Linear Regression）
2. 逻辑回归（Logistic Regression）
3. 感知机（Perceptron）
4. 朴素贝叶斯（Naive Bayes）
5. K近邻（KNN）
6. 决策树（Decision Tree）
7. 随机森林（Random Forest）
8. Boosting（AdaBoost/GBDT/XGBoost）
9. 支持向量机（SVM）
10. K-Means聚类
11. 层次聚类（Hierarchical Clustering）
12. DBSCAN
13. 高斯混合模型（GMM）
14. 主成分分析（PCA）
15. 线性判别分析（LDA）
16. 隐马尔可夫模型（HMM）
17. 条件随机场（CRF）

### 深度学习算法（13个）

1. 多层感知机（MLP）
2. LeNet
3. AlexNet
4. VGG
5. ResNet
6. Vanilla RNN
7. LSTM
8. GRU
9. 注意力机制与Transformer
10. 自编码器（AutoEncoder）
11. 变分自编码器（VAE）
12. 生成对抗网络（GAN）
13. 图神经网络（GNN）
14. 深度强化学习（DQN/Policy Gradient/Actor-Critic/PPO）

## 技术栈

- **Python**: 3.11
- **深度学习框架**: PyTorch 2.5.1（支持CUDA 12.1）
- **机器学习库**: scikit-learn
- **数据处理**: numpy, pandas, scipy
- **可视化**: matplotlib
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
├── README.md                     # 项目说明（包含环境配置）
├── docs/                         # 项目文档
│   ├── 1-核心算法.md             # 算法清单
│   ├── 2-提示词-给出算法名称生成对应代码.md  # 工作流程
│   └── 3-虚拟环境搭建.md         # 环境配置详细说明
├── ml_knn/                       # KNN 算法子项目 ✅
│   ├── docs/                    # 子项目文档
│   │   ├── PLAN.md              # 项目计划
│   │   ├── ALGO_SPEC.md         # 算法规格书
│   │   ├── IMPLEMENTATION_DESIGN.md  # 实现设计文档
│   │   └── QUICK_START.md       # 快速开始
│   ├── README.md                # 子项目说明
│   ├── requirements.txt         # 子项目依赖
│   ├── main.py                  # 主入口
│   ├── src/                     # 源代码
│   ├── data/                    # 数据目录
│   └── outputs/                 # 输出结果
└── [其他算法]/                   # 其他算法子项目（待创建）
```

## 开发流程

每个算法的实现遵循以下五阶段流程：

1. **Phase 0**: 启动与输入收集
2. **Phase 1**: 创建子项目文件夹结构
3. **Phase 2**: 生成算法规格书（ALGO_SPEC.md）
4. **Phase 3**: 生成实现设计文档（IMPLEMENTATION_DESIGN.md）
5. **Phase 4**: 生成代码实现
6. **Phase 5**: 验证与修复

## 项目特点

- ✅ **规范化流程**: 统一的开发流程，确保质量和可追溯性
- ✅ **合成数据优先**: 默认使用自动生成的数据，无需准备真实数据集
- ✅ **实验驱动**: 每个算法至少包含3个实验（特性实验、对照实验、可视化实验）
- ✅ **可复现性**: 固定随机种子，确保结果一致
- ✅ **低资源友好**: 提供CPU可运行的实现方案

## 使用说明

### 快速开始

```powershell
# Step 0: 激活虚拟环境（必须）
conda activate mldl_algo311

# Step 1: 进入项目根目录
cd MLDLAlgorithmCode

# Step 2: 进入子项目目录（以 ml_knn 为例）
cd ml_knn

# Step 3: 安装子项目依赖
pip install -r requirements.txt

# Step 4: 运行子项目
python main.py
```

详细的工作流程和使用说明请参考：
- [工作流程文档](docs/2-提示词-给出算法名称生成对应代码.md)
- [虚拟环境详细说明](docs/3-虚拟环境搭建.md)

## 项目状态

### 已完成的算法

| 算法 | 目录 | 分类 | 状态 | 实验数 | 说明 |
|------|------|------|------|--------|------|
| K-Nearest Neighbors | `ml_knn/` | ML | ✅ 完成 | 6 | K 值敏感性、距离度量、基线对照等 |

### 进行中的算法

🚧 项目处于初始化阶段，更多算法实现正在逐步进行中...

### 快速访问

- [KNN 算法项目](ml_knn/README.md)
- [工作流程文档](docs/2-提示词-给出算法名称生成对应代码.md)

## License

本项目仅供学习和研究使用。
