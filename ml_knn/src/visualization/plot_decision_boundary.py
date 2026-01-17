"""
决策边界可视化模块
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_decision_boundary(model, X, y, k, output_path, resolution=0.1):
    """
    绘制决策边界
    
    Parameters:
    -----------
    model : KNNClassifier
        训练好的 KNN 模型
    X : np.ndarray, shape (n_samples, 2)
        二维特征数据
    y : np.ndarray
        标签
    k : int
        K 值
    output_path : str
        输出路径
    resolution : float
        网格分辨率
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    
    # 绘制训练样本
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k', 
                         cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Class')
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(f'Decision Boundary (K={k})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure saved to: {output_path}")
