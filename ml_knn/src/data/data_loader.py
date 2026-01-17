"""
数据加载模块
支持 synthetic（合成数据）、CSV、Excel、公开数据集
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs, load_iris, load_wine, load_diabetes
from sklearn.model_selection import train_test_split


def load_data(mode='synthetic', data_path=None, target_col='target', 
              test_size=0.3, random_state=42, n_samples=500, n_features=10, n_classes=3):
    """
    加载数据
    
    Parameters:
    -----------
    mode : str
        数据模式 ('synthetic', 'csv', 'excel', 'iris', 'wine', 'diabetes')
    data_path : str, optional
        用户数据路径（CSV/Excel）
    target_col : str
        目标列名
    test_size : float
        测试集比例
    random_state : int
        随机种子
    n_samples : int
        合成数据样本数
    n_features : int
        合成数据特征数
    n_classes : int
        合成数据类别数
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : np.ndarray
        训练/测试数据
    """
    if mode == 'synthetic':
        X, y = generate_synthetic_classification(n_samples, n_features, n_classes, random_state)
    elif mode == 'csv':
        if data_path is None:
            raise ValueError("data_path must be provided for CSV mode")
        X, y = load_csv(data_path, target_col)
    elif mode == 'excel':
        if data_path is None:
            raise ValueError("data_path must be provided for Excel mode")
        X, y = load_excel(data_path, target_col)
    elif mode in ['iris', 'wine', 'diabetes']:
        X, y = load_public_dataset(mode)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def generate_synthetic_classification(n_samples=500, n_features=10, n_classes=3, random_state=42):
    """
    生成合成分类数据
    
    Parameters:
    -----------
    n_samples : int
        样本数
    n_features : int
        特征数
    n_classes : int
        类别数
    random_state : int
        随机种子
    
    Returns:
    --------
    X, y : np.ndarray
        特征和标签
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(0, n_features // 4),
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
        flip_y=0.1  # 添加 10% 噪声
    )
    return X, y


def generate_synthetic_2d(n_samples=300, n_classes=3, random_state=42):
    """
    生成二维合成数据（用于决策边界可视化）
    
    Parameters:
    -----------
    n_samples : int
        样本数
    n_classes : int
        类别数
    random_state : int
        随机种子
    
    Returns:
    --------
    X, y : np.ndarray
        特征和标签
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=n_classes,
        cluster_std=1.0,
        random_state=random_state
    )
    return X, y


def load_csv(path, target_col='target'):
    """
    加载 CSV 数据
    
    Parameters:
    -----------
    path : str
        CSV 文件路径
    target_col : str
        目标列名
    
    Returns:
    --------
    X, y : np.ndarray
        特征和标签
    """
    df = pd.read_csv(path)
    
    # 检查目标列是否存在
    if target_col not in df.columns:
        # 尝试常见的目标列名
        for col in ['target', 'label', 'class', 'y']:
            if col in df.columns:
                target_col = col
                break
        else:
            raise ValueError(f"Target column '{target_col}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    # 分离特征和标签
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    
    return X, y


def load_excel(path, target_col='target', sheet_name=0):
    """
    加载 Excel 数据
    
    Parameters:
    -----------
    path : str
        Excel 文件路径
    target_col : str
        目标列名
    sheet_name : int or str
        Sheet 名称或索引
    
    Returns:
    --------
    X, y : np.ndarray
        特征和标签
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    
    # 检查目标列是否存在
    if target_col not in df.columns:
        # 尝试常见的目标列名
        for col in ['target', 'label', 'class', 'y']:
            if col in df.columns:
                target_col = col
                break
        else:
            raise ValueError(f"Target column '{target_col}' not found in Excel. Available columns: {df.columns.tolist()}")
    
    # 分离特征和标签
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    
    return X, y


def load_public_dataset(name):
    """
    加载公开数据集
    
    Parameters:
    -----------
    name : str
        数据集名称 ('iris', 'wine', 'diabetes')
    
    Returns:
    --------
    X, y : np.ndarray
        特征和标签
    """
    if name == 'iris':
        data = load_iris()
    elif name == 'wine':
        data = load_wine()
    elif name == 'diabetes':
        data = load_diabetes()
    else:
        raise ValueError(f"Unknown public dataset: {name}")
    
    return data.data, data.target
