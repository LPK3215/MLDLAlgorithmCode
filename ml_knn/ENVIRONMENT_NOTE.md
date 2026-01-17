# 虚拟环境依赖说明

## ⚠️ 重要提示

**本项目（ml_knn）是 `MLDLAlgorithmCode` 总仓库的子项目，依赖于统一的虚拟环境 `mldl_algo311`。**

## 为什么需要虚拟环境？

1. **依赖隔离**：避免与系统 Python 环境或其他项目冲突
2. **版本管理**：确保所有子项目使用相同的依赖版本
3. **可复现性**：保证在不同机器上运行结果一致
4. **便于管理**：统一管理所有算法项目的依赖

## 虚拟环境信息

- **环境名称：** `mldl_algo311`
- **Python 版本：** 3.11
- **适用范围：** MLDLAlgorithmCode 下所有子项目

## 使用流程

### 每次使用前（必须）

```bash
# 激活虚拟环境
conda activate mldl_algo311

# 进入项目目录
cd MLDLAlgorithmCode/ml_knn
```

### 首次使用

```bash
# 1. 激活虚拟环境
conda activate mldl_algo311

# 2. 进入项目目录
cd MLDLAlgorithmCode/ml_knn

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行项目
python main.py
```

### 后续使用

```bash
# 1. 激活虚拟环境
conda activate mldl_algo311

# 2. 进入项目目录
cd MLDLAlgorithmCode/ml_knn

# 3. 直接运行（依赖已安装）
python main.py
```

## 快速启动（Windows）

我们提供了快速启动脚本 `run.bat`，会自动完成环境激活和依赖检查：

```bash
# 双击运行或在命令行执行
run.bat
```

## 常见问题

### Q: 忘记激活虚拟环境会怎样？

**A:** 会出现以下错误：
```
ModuleNotFoundError: No module named 'sklearn'
ModuleNotFoundError: No module named 'seaborn'
```

**解决方法：** 执行 `conda activate mldl_algo311`

### Q: 如何确认虚拟环境已激活？

**A:** 命令行提示符前会显示 `(mldl_algo311)`：
```
(mldl_algo311) D:\MLDLAlgorithmCode\ml_knn>
```

### Q: 如何查看虚拟环境中已安装的包？

```bash
conda activate mldl_algo311
pip list
```

### Q: 如何退出虚拟环境？

```bash
conda deactivate
```

## 依赖包清单

本项目依赖以下包（已在 requirements.txt 中定义）：

- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0
- seaborn >= 0.11.0
- openpyxl >= 3.0.0

## 相关文档

- [总仓库环境配置说明](../ENVIRONMENT_SETUP.md)
- [总仓库 README](../README.md)
- [本项目 README](README.md)

---

**最后更新：** 2026-01-18
