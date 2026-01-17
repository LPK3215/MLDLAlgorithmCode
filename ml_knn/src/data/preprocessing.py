"""
数据预处理模块
特征标准化、缺失值处理、类别编码
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_features(X_train, X_test, method='standard'):
    """
    特征标准化
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        训练/测试特征
    method : str
        标准化方法 ('standard', 'minmax', 'none')
    
    Returns:
    --------
    X_train_scaled, X_test_scaled : np.ndarray
        标准化后的特征
    scaler : object
        标准化器对象
    """
    if method == 'none':
        return X_train, X_test, None
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def handle_missing_values(X, strategy='mean'):
    """
    处理缺失值
    
    Parameters:
    -----------
    X : np.ndarray
        特征矩阵
    strategy : str
        填充策略 ('mean', 'median', 'drop')
    
    Returns:
    --------
    X_filled : np.ndarray
        填充后的特征矩阵
    """
    if not np.isnan(X).any():
        return X
    
    if strategy == 'mean':
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X_filled = X.copy()
        X_filled[inds] = np.take(col_mean, inds[1])
        return X_filled
    elif strategy == 'median':
        col_median = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X_filled = X.copy()
        X_filled[inds] = np.take(col_median, inds[1])
        return X_filled
    elif strategy == 'drop':
        return X[~np.isnan(X).any(axis=1)]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
