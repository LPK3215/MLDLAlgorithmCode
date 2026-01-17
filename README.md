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

### 创建虚拟环境

```powershell
conda create -n mldl_algo311 python=3.11 -y
conda activate mldl_algo311
```

### 安装依赖

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas matplotlib scikit-learn tqdm pyyaml
```

## 项目结构

```
MLDLAlgorithmCode/
├── docs/                         # 项目文档
│   ├── 1-核心算法.md             # 算法清单
│   ├── 2-提示词-给出算法名称生成对应代码.md  # 工作流程
│   └── 3-虚拟环境搭建.md         # 环境配置
├── [算法名称]/                   # 各算法子项目（待创建）
│   ├── ALGO_SPEC.md             # 算法规格书
│   ├── IMPLEMENTATION_DESIGN.md # 实现设计文档
│   ├── src/                     # 源代码
│   ├── experiments/             # 实验脚本
│   └── results/                 # 实验结果
└── README.md                    # 项目说明
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

详细的工作流程和使用说明请参考 `docs/2-提示词-给出算法名称生成对应代码.md`

## 项目状态

🚧 项目处于初始化阶段，算法实现正在逐步进行中...

## License

本项目仅供学习和研究使用。
