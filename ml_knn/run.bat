@echo off
REM KNN 算法项目快速启动脚本
REM 自动激活虚拟环境并运行项目

echo ========================================
echo KNN 算法案例项目 - 快速启动
echo ========================================
echo.

REM 激活虚拟环境
echo [1/3] 激活虚拟环境 mldl_algo311...
call conda activate mldl_algo311
if errorlevel 1 (
    echo 错误：无法激活虚拟环境 mldl_algo311
    echo 请先创建虚拟环境：conda create -n mldl_algo311 python=3.11 -y
    pause
    exit /b 1
)
echo ✓ 虚拟环境已激活
echo.

REM 检查依赖
echo [2/3] 检查依赖...
python -c "import numpy, sklearn, matplotlib, pandas, seaborn" 2>nul
if errorlevel 1 (
    echo 警告：缺少依赖包，正在安装...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误：依赖安装失败
        pause
        exit /b 1
    )
)
echo ✓ 依赖检查完成
echo.

REM 运行项目
echo [3/3] 运行项目...
echo.
python main.py %*

echo.
echo ========================================
echo 运行完成！
echo ========================================
echo 查看结果：
echo   - 指标：outputs/metrics/
echo   - 图表：outputs/figures/
echo   - 日志：outputs/logs/experiment_log.txt
echo.
pause
