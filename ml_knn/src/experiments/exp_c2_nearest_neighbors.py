"""
实验 C2：最近邻样本可视化
展示 KNN 预测的可解释性
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from src.core.knn_classifier import KNNClassifier
from src.data.data_loader import generate_synthetic_2d
from sklearn.model_selection import train_test_split


def run_experiment_c2(output_dir='outputs', k=5):
    """
    运行实验 C2：最近邻样本可视化
    
    Parameters:
    -----------
    output_dir : str
        输出目录
    k : int
        K 值
    """
    print("\n" + "="*60)
    print("实验 C2：最近邻样本可视化")
    print("="*60)
    
    # 生成二维合成数据
    print("\n生成二维合成数据...")
    X, y = generate_synthetic_2d(n_samples=300, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 训练模型
    model = KNNClassifier(k=k, metric='euclidean')
    model.fit(X_train, y_train)
    
    # 选择 3 个测试样本
    test_indices = [0, 15, 30]
    
    # 为每个测试样本绘制最近邻
    for idx, test_idx in enumerate(test_indices, 1):
        print(f"\n绘制测试样本 {idx} 的最近邻...")
        
        test_sample = X_test[test_idx]
        true_label = y_test[test_idx]
        
        # 获取最近邻
        neighbors = model.get_neighbors(test_sample, k=k)
        
        # 预测
        pred_label = model.predict([test_sample])[0]
        
        # 绘制
        plt.figure(figsize=(10, 8))
        
        # 绘制所有训练样本（浅色）
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                   s=30, alpha=0.3, cmap='viridis', edgecolors='none')
        
        # 绘制最近邻（高亮）
        neighbor_features = neighbors['features']
        neighbor_labels = neighbors['labels']
        plt.scatter(neighbor_features[:, 0], neighbor_features[:, 1], 
                   c=neighbor_labels, s=150, alpha=0.8, cmap='viridis',
                   edgecolors='black', linewidths=2, marker='o',
                   label='K Nearest Neighbors')
        
        # 绘制测试样本（红色星形）
        plt.scatter(test_sample[0], test_sample[1], 
                   c='red', s=300, marker='*', edgecolors='black',
                   linewidths=2, label=f'Test Sample (True: {true_label}, Pred: {pred_label})')
        
        # 绘制连线
        for i in range(k):
            plt.plot([test_sample[0], neighbor_features[i, 0]],
                    [test_sample[1], neighbor_features[i, 1]],
                    'k--', alpha=0.3, linewidth=1)
        
        # 添加距离标注
        for i in range(k):
            mid_x = (test_sample[0] + neighbor_features[i, 0]) / 2
            mid_y = (test_sample[1] + neighbor_features[i, 1]) / 2
            dist = neighbors['distances'][i]
            plt.text(mid_x, mid_y, f'{dist:.2f}', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.title(f'Nearest Neighbors Visualization (Sample {idx}, K={k})',
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(f'{output_dir}/figures', exist_ok=True)
        output_path = f'{output_dir}/figures/nearest_neighbors_sample{idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure saved to: {output_path}")
        
        # 打印投票分布
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        print(f"  投票分布: {dict(zip(unique, counts))}")
        print(f"  真实标签: {true_label}, 预测标签: {pred_label}")
    
    print("\n✓ 实验 C2 完成！")
