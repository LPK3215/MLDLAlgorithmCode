"""
评估指标模块
计算准确率、F1-score、混淆矩阵等
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os


def compute_accuracy(y_true, y_pred):
    """计算准确率"""
    return accuracy_score(y_true, y_pred)


def compute_f1_score(y_true, y_pred, average='weighted'):
    """计算 F1-score"""
    return f1_score(y_true, y_pred, average=average)


def compute_confusion_matrix(y_true, y_pred):
    """计算混淆矩阵"""
    return confusion_matrix(y_true, y_pred)


def save_metrics_to_csv(metrics_dict, output_path):
    """
    保存指标到 CSV
    
    Parameters:
    -----------
    metrics_dict : dict or list of dict
        指标字典
    output_path : str
        输出路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换为 DataFrame
    if isinstance(metrics_dict, dict):
        df = pd.DataFrame([metrics_dict])
    else:
        df = pd.DataFrame(metrics_dict)
    
    # 保存到 CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Metrics saved to: {output_path}")
