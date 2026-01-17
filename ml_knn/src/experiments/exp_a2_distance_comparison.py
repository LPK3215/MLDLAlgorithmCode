"""
实验 A2：距离度量对比
验证不同距离度量对分类效果的影响
"""
import numpy as np
from src.core.knn_classifier import KNNClassifier
from src.evaluation.metrics import compute_accuracy, compute_f1_score, compute_confusion_matrix, save_metrics_to_csv
from src.visualization.plot_comparison import plot_bar_comparison, plot_confusion_matrix


def run_experiment_a2(X_train, X_test, y_train, y_test, output_dir='outputs', k=5):
    """
    运行实验 A2：距离度量对比
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        训练/测试特征
    y_train, y_test : np.ndarray
        训练/测试标签
    output_dir : str
        输出目录
    k : int
        K 值
    """
    print("\n" + "="*60)
    print("实验 A2：距离度量对比")
    print("="*60)
    
    # 距离度量列表
    metrics = ['euclidean', 'manhattan', 'minkowski']
    
    # 存储结果
    results = []
    accuracy_dict = {}
    
    # 遍历距离度量
    for metric in metrics:
        print(f"\n测试距离度量: {metric}...")
        
        # 训练模型
        if metric == 'minkowski':
            model = KNNClassifier(k=k, metric=metric, p=3)
        else:
            model = KNNClassifier(k=k, metric=metric)
        
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        acc = compute_accuracy(y_test, y_pred)
        f1 = compute_f1_score(y_test, y_pred)
        cm = compute_confusion_matrix(y_test, y_pred)
        
        accuracy_dict[metric] = acc
        
        results.append({
            'metric': metric,
            'test_accuracy': acc,
            'f1_score': f1
        })
        
        print(f"  测试准确率: {acc:.4f}")
        print(f"  F1-score: {f1:.4f}")
        
        # 绘制混淆矩阵
        labels = np.unique(y_test)
        plot_confusion_matrix(cm, labels,
                            f'{output_dir}/figures/confusion_matrix_{metric}.png',
                            title=f'Confusion Matrix ({metric})')
    
    # 保存指标
    save_metrics_to_csv(results, f'{output_dir}/metrics/distance_comparison.csv')
    
    # 绘制准确率对比柱状图
    plot_bar_comparison(accuracy_dict, 'Test Accuracy', 'Distance Metric Comparison',
                       f'{output_dir}/figures/distance_comparison.png',
                       xlabel='Distance Metric')
    
    print("\n✓ 实验 A2 完成！")
