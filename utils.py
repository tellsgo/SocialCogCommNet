import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from datetime import datetime

def visualize_training(rewards_history, rewards_history_no_comm, 
                     losses_history, losses_history_no_comm,
                     success_rates, success_rates_no_comm,
                     save_dir, episode=None):
    """可视化训练过程"""
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录时间
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_ep{episode}" if episode else f"_{timestamp}"
    
    # 绘制奖励图
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, 'b-', label='有通信')
    plt.plot(rewards_history_no_comm, 'r-', label='无通信')
    plt.xlabel('回合')
    plt.ylabel('平均奖励')
    plt.title('训练奖励')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/rewards{suffix}.png")
    plt.close()
    
    # 绘制损失图
    plt.figure(figsize=(12, 6))
    plt.plot(losses_history, 'b-', label='有通信')
    plt.plot(losses_history_no_comm, 'r-', label='无通信')
    plt.xlabel('更新步骤')
    plt.ylabel('损失')
    plt.title('训练损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/losses{suffix}.png")
    plt.close()
    
    # 绘制成功率图
    plt.figure(figsize=(12, 6))
    plt.plot(success_rates, 'b-', label='有通信')
    plt.plot(success_rates_no_comm, 'r-', label='无通信')
    plt.xlabel('回合')
    plt.ylabel('成功率')
    plt.title('任务成功率')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/success_rates{suffix}.png")
    plt.close()
    
    # 绘制通信提升图
    if len(rewards_history) == len(rewards_history_no_comm):
        plt.figure(figsize=(12, 6))
        comm_improvements = [r_c - r_nc for r_c, r_nc in zip(rewards_history, rewards_history_no_comm)]
        plt.plot(comm_improvements, 'g-')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('回合')
        plt.ylabel('奖励提升')
        plt.title('通信带来的奖励提升')
        plt.grid(True)
        plt.savefig(f"{save_dir}/comm_improvements{suffix}.png")
        plt.close()

def visualize_communication(model, env, device, save_dir, episode=None):
    """可视化智能体间的通信内容"""
    # 准备测试场景
    if hasattr(env.current_game, 'get_test_scenarios'):
        test_scenarios = env.current_game.get_test_scenarios()
    else:
        # 如果游戏没有定义测试场景，创建10个随机场景
        test_scenarios = [None] * 10
    
    # 收集每个场景的通信内容
    comm_data = []
    
    for i, scenario in enumerate(test_scenarios):
        # 重置环境
        obs = env.reset(scenario=scenario)
        
        # 分析通信
        messages = model.analyze_communication(obs, device)
        comm_data.append(messages)
    
    # 为每个智能体绘制通信热力图
    for agent_idx in range(env.n_agents):
        plt.figure(figsize=(10, 8))
        
        # 提取该智能体在所有场景下的通信消息
        agent_messages = [data[agent_idx] for data in comm_data]
        
        # 构建热力图数据
        comm_dim = agent_messages[0].shape[1]
        heat_data = np.array(agent_messages).reshape(len(agent_messages), comm_dim)
        
        # 绘制热力图
        ax = plt.gca()
        sns.heatmap(heat_data, cmap='viridis', ax=ax)
        
        # 设置标题和标签
        game_name = env.get_current_game_name()
        plt.title(f"{game_name} - 智能体 {agent_idx + 1} 通信内容 (回合 {episode})")
        plt.xlabel('通信维度')
        plt.ylabel('场景')
        
        # 保存图片
        suffix = f"_ep{episode}" if episode else ""
        plt.savefig(f"{save_dir}/{game_name}_agent{agent_idx+1}_comm{suffix}.png")
        plt.close()

def visualize_multi_game_performance(results, save_dir):
    """可视化模型在多个游戏上的表现"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取数据
    games = list(results.keys())
    rewards = [results[game]['reward'] for game in games]
    success_rates = [results[game]['success_rate'] for game in games]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制奖励图
    ax1.bar(games, rewards, color='blue', alpha=0.7)
    ax1.set_title('各游戏平均奖励')
    ax1.set_ylabel('平均奖励')
    ax1.set_xticklabels(games, rotation=45, ha='right')
    
    # 绘制成功率图
    ax2.bar(games, success_rates, color='green', alpha=0.7)
    ax2.set_title('各游戏成功率')
    ax2.set_ylabel('成功率')
    ax2.set_xticklabels(games, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/multi_game_performance.png")
    plt.close()

def visualize_curriculum_learning(game_statistics, save_dir):
    """可视化课程学习过程"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 提取数据
    games = list(game_statistics.keys())
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    # 绘制每个游戏的学习曲线
    for i, game in enumerate(games):
        rewards = game_statistics[game]['rewards']
        episodes = range(game_statistics[game]['episodes'])
        
        # 使用固定颜色循环
        color = colors[i % len(colors)]
        
        plt.plot(episodes, rewards, '-', color=color, label=game)
    
    plt.xlabel('游戏内回合')
    plt.ylabel('平均奖励')
    plt.title('课程学习过程中各游戏的学习曲线')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"{save_dir}/curriculum_learning.png")
    plt.close()

def analyze_communication_correlation(model, env, device, save_dir):
    """分析通信内容与游戏状态的相关性"""
    # 准备测试场景
    if hasattr(env.current_game, 'get_test_scenarios'):
        test_scenarios = env.current_game.get_test_scenarios()
    else:
        # 如果游戏没有定义测试场景，创建10个随机场景
        test_scenarios = [None] * 10
    
    # 收集数据
    states = []
    communications = []
    
    for scenario in test_scenarios:
        # 重置环境
        obs = env.reset(scenario=scenario)
        
        # 记录状态
        if hasattr(env.current_game, 'state'):
            states.append(env.current_game.state.copy())
        
        # 分析通信
        messages = model.analyze_communication(obs, device)
        flat_messages = []
        for agent_messages in messages:
            flat_messages.extend(agent_messages.flatten())
        
        communications.append(flat_messages)
    
    # 如果收集到状态数据，分析相关性
    if states and len(states[0]) > 0:
        # 转换为numpy数组
        states_array = np.array(states)
        comm_array = np.array(communications)
        
        # 计算相关性
        correlation_matrix = np.zeros((states_array.shape[1], comm_array.shape[1]))
        
        for i in range(states_array.shape[1]):
            for j in range(comm_array.shape[1]):
                if np.std(states_array[:, i]) > 0 and np.std(comm_array[:, j]) > 0:
                    correlation_matrix[i, j] = np.corrcoef(states_array[:, i], comm_array[:, j])[0, 1]
        
        # 可视化相关性
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
        plt.title('状态-通信相关性矩阵')
        plt.xlabel('通信维度')
        plt.ylabel('状态维度')
        
        game_name = env.get_current_game_name()
        plt.savefig(f"{save_dir}/{game_name}_state_comm_correlation.png")
        plt.close() 