"""
实验 B1：基线模型对照
验证 KNN 相对于简单基线的优势
"""
import numpy as np
import time
from src.core.knn_classifier import KNNClassifier
from src.evaluation.metrics import compute_accuracy, save_metrics_to_csv
from src.visualization.plot_comparison import plot_roc_curves
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import os


def run_experiment_b1(X_train, X_test, y_train, y_test, output_dir='outputs', k=5):
    """
    运行实验 B1：基线模型对照
    
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
    print("实验 B1：基线模型对照")
    print("="*60)
    
    # 定义模型
    models = {
        'KNN': KNNClassifier(k=k, metric='euclidean'),
        'Dummy': DummyClassifier(strategy='stratified', random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42)
    }
    
    # 存储结果
    results = []
    models_for_roc = {}
    
    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\n测试模型: {model_name}...")
        
        # 训练时间
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 预测时间
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # 计算准确率
        acc = compute_accuracy(y_test, y_pred)
        
        results.append({
            'model': model_name,
            'test_accuracy': acc,
            'train_time_sec': train_time,
            'predict_time_sec': predict_time
        })
        
        print(f"  测试准确率: {acc:.4f}")
        print(f"  训练时间: {train_time:.4f} 秒")
        print(f"  预测时间: {predict_time:.4f} 秒")
        
        # 保存支持 predict_proba 的模型用于 ROC 曲线
        if hasattr(model, 'predict_proba'):
            models_for_roc[model_name] = model
    
    # 保存指标
    save_metrics_to_csv(results, f'{output_dir}/metrics/baseline_comparison.csv')
    
    # 绘制对比柱状图
    df = pd.DataFrame(results)
    df.set_index('model', inplace=True)
    
    # 准确率对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 准确率
    df['test_accuracy'].plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xticklabels(df.index, rotation=45, ha='right')
    
    # 训练时间
    df['train_time_sec'].plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_ylabel('Train Time (sec)', fontsize=12)
    axes[1].set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticklabels(df.index, rotation=45, ha='right')
    
    # 预测时间
    df['predict_time_sec'].plot(kind='bar', ax=axes[2], color='lightgreen', edgecolor='black')
    axes[2].set_ylabel('Predict Time (sec)', fontsize=12)
    axes[2].set_title('Prediction Time Comparison', fontsize=13, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_xticklabels(df.index, rotation=45, ha='right')
    
    plt.tight_layout()
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    plt.savefig(f'{output_dir}/figures/baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure saved to: {output_dir}/figures/baseline_comparison.png")
    
    # 绘制 ROC 曲线
    if len(models_for_roc) > 0:
        plot_roc_curves(models_for_roc, X_test, y_test,
                       f'{output_dir}/figures/roc_curve_comparison.png')
    
    print("\n✓ 实验 B1 完成！")
