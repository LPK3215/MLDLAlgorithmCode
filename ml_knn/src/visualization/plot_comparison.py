"""
对比图绘制模块
柱状图、混淆矩阵、ROC 曲线
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os


def plot_bar_comparison(data_dict, ylabel, title, output_path, xlabel='Method'):
    """
    绘制柱状对比图
    
    Parameters:
    -----------
    data_dict : dict
        数据字典 {name: value}
    ylabel : str
        Y 轴标签
    title : str
        图表标题
    output_path : str
        输出路径
    xlabel : str
        X 轴标签
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    names = list(data_dict.keys())
    values = list(data_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color='steelblue', alpha=0.8, edgecolor='black')
    
    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure saved to: {output_path}")


def plot_confusion_matrix(cm, labels, output_path, title='Confusion Matrix'):
    """
    绘制混淆矩阵热力图
    
    Parameters:
    -----------
    cm : np.ndarray
        混淆矩阵
    labels : list
        类别标签
    output_path : str
        输出路径
    title : str
        图表标题
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure saved to: {output_path}")


def plot_roc_curves(models_dict, X_test, y_test, output_path):
    """
    绘制 ROC 曲线对比（多分类 OvR）
    
    Parameters:
    -----------
    models_dict : dict
        模型字典 {name: model}
    X_test : np.ndarray
        测试特征
    y_test : np.ndarray
        测试标签
    output_path : str
        输出路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取类别数
    n_classes = len(np.unique(y_test))
    
    # 二值化标签（OvR）
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models_dict.items():
        # 预测概率（如果模型支持）
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
        else:
            # 对于不支持 predict_proba 的模型，使用 decision_function 或跳过
            continue
        
        # 计算每个类别的 ROC 曲线
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 计算 micro-average ROC 曲线
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # 绘制 micro-average ROC 曲线
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'{model_name} (AUC = {roc_auc["micro"]:.2f})',
                linewidth=2)
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison (Micro-Average)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure saved to: {output_path}")


def plot_grouped_bar_comparison(data_df, output_path, title='Model Comparison'):
    """
    绘制分组柱状图（用于多指标对比）
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        数据框，包含多个指标列
    output_path : str
        输出路径
    title : str
        图表标题
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data_df.plot(kind='bar', figsize=(12, 6), width=0.8, edgecolor='black')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure saved to: {output_path}")
