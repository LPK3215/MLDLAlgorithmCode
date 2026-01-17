"""
曲线图绘制模块
"""
import matplotlib.pyplot as plt
import os


def plot_k_vs_accuracy(k_values, train_acc, test_acc, output_path):
    """
    绘制 K 值 vs 准确率曲线
    
    Parameters:
    -----------
    k_values : list
        K 值列表
    train_acc : list
        训练集准确率
    test_acc : list
        测试集准确率
    output_path : str
        输出路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_acc, marker='o', label='Train Accuracy', linewidth=2)
    plt.plot(k_values, test_acc, marker='s', label='Test Accuracy', linewidth=2)
    plt.xlabel('K Value', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('K Value vs Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure saved to: {output_path}")
