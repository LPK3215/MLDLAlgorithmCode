"""
实验 A1：K 值敏感性分析
验证 K 值对模型复杂度和泛化能力的影响
"""
import numpy as np
from src.core.knn_classifier import KNNClassifier
from src.evaluation.metrics import compute_accuracy, save_metrics_to_csv
from src.visualization.plot_curves import plot_k_vs_accuracy
from src.visualization.plot_decision_boundary import plot_decision_boundary
from src.data.data_loader import generate_synthetic_2d
from sklearn.model_selection import train_test_split


def run_experiment_a1(X_train, X_test, y_train, y_test, output_dir='outputs'):
    """
    运行实验 A1：K 值敏感性分析
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        训练/测试特征
    y_train, y_test : np.ndarray
        训练/测试标签
    output_dir : str
        输出目录
    """
    print("\n" + "="*60)
    print("实验 A1：K 值敏感性分析")
    print("="*60)
    
    # K 值列表
    k_values = [1, 3, 5, 10, 20, 50]
    
    # 存储结果
    results = []
    train_accuracies = []
    test_accuracies = []
    
    # 遍历 K 值
    for k in k_values:
        print(f"\n测试 K={k}...")
        
        # 训练模型
        model = KNNClassifier(k=k, metric='euclidean')
        model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 计算准确率
        train_acc = compute_accuracy(y_train, y_train_pred)
        test_acc = compute_accuracy(y_test, y_test_pred)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        results.append({
            'k': k,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        })
        
        print(f"  训练准确率: {train_acc:.4f}")
        print(f"  测试准确率: {test_acc:.4f}")
    
    # 保存指标
    save_metrics_to_csv(results, f'{output_dir}/metrics/k_sensitivity.csv')
    
    # 绘制 K 值 vs 准确率曲线
    plot_k_vs_accuracy(k_values, train_accuracies, test_accuracies,
                      f'{output_dir}/figures/k_vs_accuracy.png')
    
    # 绘制决策边界（使用二维数据）
    print("\n生成决策边界可视化（使用二维数据）...")
    X_2d, y_2d = generate_synthetic_2d(n_samples=300, n_classes=3, random_state=42)
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
        X_2d, y_2d, test_size=0.3, random_state=42
    )
    
    for k in [1, 5, 20]:
        print(f"  绘制 K={k} 决策边界...")
        model_2d = KNNClassifier(k=k, metric='euclidean')
        model_2d.fit(X_train_2d, y_train_2d)
        plot_decision_boundary(model_2d, X_train_2d, y_train_2d, k,
                             f'{output_dir}/figures/decision_boundary_k{k}.png')
    
    print("\n✓ 实验 A1 完成！")
