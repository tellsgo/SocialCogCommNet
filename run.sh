#!/bin/bash

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 创建结果目录
mkdir -p results

# 训练模型
echo "开始训练..."
python main.py --mode train \
    --n_episodes 1000 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --hidden_dim 64 \
    --comm_dim 16 \
    --memory_dim 64 \
    --max_steps 50 \
    --print_interval 10 \
    --eval_interval 100 \
    --save_dir ./results

# 等待训练完成
echo "训练完成！"

# 查找最新的模型文件
latest_model=$(find ./results -name "model_*.pt" | sort -V | tail -n 1)

if [ -n "$latest_model" ]; then
    echo "找到最新模型: $latest_model"
    
    # 测试模型
    echo "开始测试..."
    python main.py --mode test --model_path "$latest_model"
else
    echo "未找到训练好的模型文件"
fi 