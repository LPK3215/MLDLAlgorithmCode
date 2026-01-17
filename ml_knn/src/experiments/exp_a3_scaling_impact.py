"""
实验 A3：特征标准化影响
验证特征标准化对 KNN 的必要性
"""
import numpy as np
from src.core.knn_classifier import KNNClassifier
from src.data.preprocessing import scale_features
from src.evaluation.metrics import compute_accuracy, compute_confusion_matrix, save_metrics_to_csv
from src.visualization.plot_comparison import plot_bar_comparison, plot_confusion_matrix


def run_experiment_a3(X_train, X_test, y_train, y_test, output_dir='outputs', k=5):
    """
    运行实验 A3：特征标准化影响
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        训练/测试特征（未标准化）
    y_train, y_test : np.ndarray
        训练/测试标签
    output_dir : str
        输出目录
    k : int
        K 值
    """
    print("\n" + "="*60)
    print("实验 A3：特征标准化影响")
    print("="*60)
    
    # 标准化方法列表
    scaling_methods = ['none', 'standard', 'minmax']
    
    # 存储结果
    results = []
    accuracy_dict = {}
    
    # 遍历标准化方法
    for method in scaling_methods:
        print(f"\n测试标准化方法: {method}...")
        
        # 标准化
        if method == 'none':
            X_train_scaled = X_train
            X_test_scaled = X_test
        else:
            X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test, method=method)
        
        # 训练模型
        model = KNNClassifier(k=k, metric='euclidean')
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        
        # 计算准确率
        acc = compute_accuracy(y_test, y_pred)
        cm = compute_confusion_matrix(y_test, y_pred)
        
        accuracy_dict[method] = acc
        
        results.append({
            'scaling_method': method,
            'test_accuracy': acc
        })
        
        print(f"  测试准确率: {acc:.4f}")
        
        # 绘制混淆矩阵
        labels = np.unique(y_test)
        plot_confusion_matrix(cm, labels,
                            f'{output_dir}/figures/confusion_matrix_{method}.png',
                            title=f'Confusion Matrix ({method} scaling)')
    
    # 保存指标
    save_metrics_to_csv(results, f'{output_dir}/metrics/scaling_impact.csv')
    
    # 绘制准确率对比柱状图
    plot_bar_comparison(accuracy_dict, 'Test Accuracy', 'Feature Scaling Impact',
                       f'{output_dir}/figures/scaling_comparison.png',
                       xlabel='Scaling Method')
    
    print("\n✓ 实验 A3 完成！")
