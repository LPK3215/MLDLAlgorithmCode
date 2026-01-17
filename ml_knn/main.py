"""
KNN 算法案例项目 - 主入口
运行所有实验并输出结果
"""
import argparse
import os
import time
import sys
from datetime import datetime

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(__file__))

from src.data.data_loader import load_data
from src.data.preprocessing import scale_features
from src.experiments.exp_a1_k_sensitivity import run_experiment_a1
from src.experiments.exp_a2_distance_comparison import run_experiment_a2
from src.experiments.exp_b1_baseline_comparison import run_experiment_b1
from src.experiments.exp_c1_decision_boundary import run_experiment_c1
from src.experiments.exp_c2_nearest_neighbors import run_experiment_c2
from src.experiments.exp_a3_scaling_impact import run_experiment_a3


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='KNN 算法案例项目')
    
    parser.add_argument('--data_path', type=str, default=None,
                       help='用户数据路径（CSV/Excel）')
    parser.add_argument('--target_col', type=str, default='target',
                       help='目标列名')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'A1', 'A2', 'B1', 'C1', 'C2', 'A3'],
                       help='指定运行的实验')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--test_size', type=float, default=0.3,
                       help='测试集比例')
    parser.add_argument('--k_default', type=int, default=5,
                       help='默认 K 值')
    
    return parser.parse_args()


def initialize_output_dirs(output_dir='outputs'):
    """初始化输出目录"""
    os.makedirs(f'{output_dir}/metrics', exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    os.makedirs(f'{output_dir}/logs', exist_ok=True)


def log_message(message, log_file='outputs/logs/experiment_log.txt'):
    """记录日志"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 初始化输出目录
    output_dir = 'outputs'
    initialize_output_dirs(output_dir)
    
    # 清空日志文件
    log_file = f'{output_dir}/logs/experiment_log.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"KNN 算法案例项目 - 实验日志\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
    
    # 打印欢迎信息
    log_message("\n" + "="*60)
    log_message("KNN 算法案例项目")
    log_message("="*60)
    log_message(f"随机种子: {args.seed}")
    log_message(f"测试集比例: {args.test_size}")
    log_message(f"默认 K 值: {args.k_default}")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 加载数据
        log_message("\n" + "-"*60)
        log_message("1. 加载数据")
        log_message("-"*60)
        
        if args.data_path:
            log_message(f"数据源: 用户数据 ({args.data_path})")
            log_message(f"目标列: {args.target_col}")
            
            # 判断文件类型
            if args.data_path.endswith('.csv'):
                mode = 'csv'
            elif args.data_path.endswith('.xlsx') or args.data_path.endswith('.xls'):
                mode = 'excel'
            else:
                raise ValueError("不支持的文件格式，请使用 CSV 或 Excel 文件")
            
            X_train, X_test, y_train, y_test = load_data(
                mode=mode,
                data_path=args.data_path,
                target_col=args.target_col,
                test_size=args.test_size,
                random_state=args.seed
            )
        else:
            log_message("数据源: 合成数据 (synthetic)")
            X_train, X_test, y_train, y_test = load_data(
                mode='synthetic',
                test_size=args.test_size,
                random_state=args.seed,
                n_samples=500,
                n_features=10,
                n_classes=3
            )
        
        log_message(f"✓ 数据加载完成")
        log_message(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
        log_message(f"  测试集: {X_test.shape[0]} 样本")
        log_message(f"  类别数: {len(set(y_train))}")
        
        # 特征标准化（用于大部分实验）
        log_message("\n" + "-"*60)
        log_message("2. 特征标准化")
        log_message("-"*60)
        X_train_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_test, method='standard'
        )
        log_message("✓ 特征标准化完成 (StandardScaler)")
        
        # 运行实验
        log_message("\n" + "-"*60)
        log_message("3. 运行实验")
        log_message("-"*60)
        
        experiments = {
            'A1': ('K 值敏感性分析', lambda: run_experiment_a1(
                X_train_scaled, X_test_scaled, y_train, y_test, output_dir
            )),
            'A2': ('距离度量对比', lambda: run_experiment_a2(
                X_train_scaled, X_test_scaled, y_train, y_test, output_dir, args.k_default
            )),
            'B1': ('基线模型对照', lambda: run_experiment_b1(
                X_train_scaled, X_test_scaled, y_train, y_test, output_dir, args.k_default
            )),
            'C1': ('决策边界可视化', lambda: run_experiment_c1(output_dir)),
            'C2': ('最近邻样本可视化', lambda: run_experiment_c2(output_dir, args.k_default)),
            'A3': ('特征标准化影响', lambda: run_experiment_a3(
                X_train, X_test, y_train, y_test, output_dir, args.k_default
            ))
        }
        
        # 选择要运行的实验
        if args.experiment == 'all':
            selected_experiments = experiments
        else:
            selected_experiments = {args.experiment: experiments[args.experiment]}
        
        # 运行选定的实验
        for exp_id, (exp_name, exp_func) in selected_experiments.items():
            log_message(f"\n运行实验 {exp_id}: {exp_name}")
            exp_func()
        
        # 计算总运行时间
        total_time = time.time() - start_time
        
        # 打印总结
        log_message("\n" + "="*60)
        log_message("实验完成！")
        log_message("="*60)
        log_message(f"总运行时间: {total_time:.2f} 秒")
        log_message(f"\n输出结果位置:")
        log_message(f"  指标 (CSV): {output_dir}/metrics/")
        log_message(f"  图表 (PNG): {output_dir}/figures/")
        log_message(f"  日志 (TXT): {output_dir}/logs/experiment_log.txt")
        log_message("\n查看结果:")
        log_message(f"  cd {output_dir}/figures")
        log_message(f"  ls -lh")
        
    except Exception as e:
        log_message(f"\n❌ 错误: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
