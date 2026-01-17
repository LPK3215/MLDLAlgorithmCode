"""
KNN 分类器实现
支持多种距离度量和 K 值配置
"""
import numpy as np
from collections import Counter


class KNNClassifier:
    """K-Nearest Neighbors 分类器"""
    
    def __init__(self, k=5, metric='euclidean', p=2):
        """
        初始化 KNN 分类器
        
        Parameters:
        -----------
        k : int
            最近邻数量
        metric : str
            距离度量方式 ('euclidean', 'manhattan', 'minkowski')
        p : int
            闵可夫斯基距离的参数（仅当 metric='minkowski' 时有效）
        """
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        """
        训练模型（存储训练数据）
        
        Parameters:
        -----------
        X_train : np.ndarray, shape (n_samples, n_features)
            训练特征
        y_train : np.ndarray, shape (n_samples,)
            训练标签
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self
    
    def predict(self, X_test):
        """
        预测测试样本
        
        Parameters:
        -----------
        X_test : np.ndarray, shape (n_samples, n_features)
            测试特征
        
        Returns:
        --------
        y_pred : np.ndarray, shape (n_samples,)
            预测标签
        """
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """预测单个样本"""
        # 计算距离
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        
        # 获取 K 个最近邻的索引
        k_indices = np.argsort(distances)[:self.k]
        
        # 获取 K 个最近邻的标签
        k_nearest_labels = self.y_train[k_indices]
        
        # 多数投票
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def _compute_distance(self, x1, x2):
        """
        计算两个样本之间的距离
        
        Parameters:
        -----------
        x1, x2 : np.ndarray
            两个样本向量
        
        Returns:
        --------
        distance : float
            距离值
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            return np.power(np.sum(np.abs(x1 - x2) ** self.p), 1 / self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def get_neighbors(self, x, k=None):
        """
        获取样本 x 的 K 个最近邻
        
        Parameters:
        -----------
        x : np.ndarray
            查询样本
        k : int, optional
            最近邻数量，默认使用 self.k
        
        Returns:
        --------
        neighbors : dict
            包含 'indices', 'distances', 'labels' 的字典
        """
        if k is None:
            k = self.k
        
        # 计算距离
        distances = np.array([self._compute_distance(x, x_train) for x_train in self.X_train])
        
        # 获取 K 个最近邻的索引
        k_indices = np.argsort(distances)[:k]
        
        return {
            'indices': k_indices,
            'distances': distances[k_indices],
            'labels': self.y_train[k_indices],
            'features': self.X_train[k_indices]
        }
