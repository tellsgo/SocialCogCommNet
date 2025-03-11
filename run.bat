@echo off
chcp 65001 >nul
echo 安装依赖...
pip install -r requirements.txt

echo 创建结果目录...
if not exist results mkdir results

echo 开始训练...
python main.py --mode train ^
    --n_episodes 10 ^
    --batch_size 32 ^
    --learning_rate 0.001 ^
    --hidden_dim 32 ^
    --comm_dim 8 ^
    --memory_dim 32 ^
    --max_steps 5 ^
    --print_interval 1 ^
    --eval_interval 5 ^
    --save_dir ./results ^
    --debug

echo 训练完成！

echo 查找最新的模型文件...
set "latest_model="
for /f "delims=" %%a in ('dir /b /s "results\*\model_*.pt" 2^>nul') do (
    if not defined latest_model (
        set "latest_model=%%a"
    )
)

if defined latest_model (
    echo 找到最新模型: %latest_model%
    
    echo 开始测试...
    python main.py --mode test --model_path "%latest_model%" --debug
) else (
    echo 未找到训练好的模型文件
    echo 正在查看结果目录内容:
    dir /s results
)

pause 