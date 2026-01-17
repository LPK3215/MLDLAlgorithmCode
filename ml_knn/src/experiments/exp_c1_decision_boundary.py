"""
实验 C1：决策边界可视化
直观展示 KNN 的局部决策机制
"""
from src.core.knn_classifier import KNNClassifier
from src.visualization.plot_decision_boundary import plot_decision_boundary
from src.data.data_loader import generate_synthetic_2d
from sklearn.model_selection import train_test_split


def run_experiment_c1(output_dir='outputs'):
    """
    运行实验 C1：决策边界可视化
    
    Parameters:
    -----------
    output_dir : str
        输出目录
    """
    print("\n" + "="*60)
    print("实验 C1：决策边界可视化")
    print("="*60)
    
    # 生成二维合成数据
    print("\n生成二维合成数据...")
    X, y = generate_synthetic_2d(n_samples=300, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # K 值列表
    k_values = [1, 5, 20]
    
    # 绘制不同 K 值的决策边界
    for k in k_values:
        print(f"\n绘制 K={k} 决策边界...")
        
        # 训练模型
        model = KNNClassifier(k=k, metric='euclidean')
        model.fit(X_train, y_train)
        
        # 绘制决策边界
        plot_decision_boundary(model, X_train, y_train, k,
                             f'{output_dir}/figures/decision_boundary_k{k}.png')
    
    print("\n✓ 实验 C1 完成！")
