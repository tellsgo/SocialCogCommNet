import torch
import json
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from environment import CooperativeEnvironment
from model import SocialCognitiveCommNet

def test(model_path, config):
    """
    测试模型性能
    
    参数:
        model_path: 模型路径
        config: 配置字典
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = CooperativeEnvironment(curriculum_learning=config.get('curriculum_learning', False))
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取当前游戏名称
    game_name = checkpoint.get('game', 'SimpleCoordinationGame')
    print(f"加载模型: {model_path}")
    print(f"游戏: {game_name}")
    
    # 手动切换到评估的游戏
    if game_name != env.get_current_game_name():
        # 尝试切换到目标游戏
        found = False
        while env.get_current_game_name() != game_name:
            success, _, _ = env.next_game(None, None)
            if not success:
                print(f"无法找到游戏: {game_name}，使用当前游戏: {env.get_current_game_name()}")
                break
    
    # 创建模型
    model = SocialCognitiveCommNet(
        input_dim=env.state_dim,
        hidden_dim=config['hidden_dim'],
        comm_dim=config['comm_dim'],
        memory_dim=config['memory_dim'],
        n_agents=env.n_agents,
        n_actions=env.n_actions
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 无通信对照组
    model_no_comm = SocialCognitiveCommNet(
        input_dim=env.state_dim,
        hidden_dim=config['hidden_dim'],
        comm_dim=config['comm_dim'],
        memory_dim=config['memory_dim'],
        n_agents=env.n_agents,
        n_actions=env.n_actions
    ).to(device)
    model_no_comm.load_state_dict(checkpoint['model_state_dict'])
    model_no_comm.eval()
    
    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(model_path), "test_results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"测试结果将保存到: {save_dir}")
    
    # 测试所有游戏
    if config.get('curriculum_learning', False):
        print("在所有游戏上评估模型")
        all_game_results = env.evaluate_all_games(model, device)
        all_game_results_no_comm = env.evaluate_all_games(model_no_comm, device, communication=False)
        
        # 保存结果
        with open(f"{save_dir}/all_games_evaluation.txt", "w") as f:
            f.write("模型在所有游戏上的评估结果:\n\n")
            for game_name, results in all_game_results.items():
                f.write(f"游戏: {game_name}\n")
                f.write(f"  有通信平均奖励: {results['reward']:.4f}\n")
                f.write(f"  有通信成功率: {results['success_rate']:.2f}\n")
                f.write(f"  无通信平均奖励: {all_game_results_no_comm[game_name]['reward']:.4f}\n")
                f.write(f"  无通信成功率: {all_game_results_no_comm[game_name]['success_rate']:.2f}\n")
                f.write(f"  通信提升: {results['reward'] - all_game_results_no_comm[game_name]['reward']:.4f}\n\n")
        
        # 可视化
        visualize_all_games_comparison(all_game_results, all_game_results_no_comm, save_dir)
    else:
        # 仅在当前游戏上评估
        print(f"在游戏 {env.get_current_game_name()} 上评估模型")
        
        # 获取评估场景
        test_scenarios = env.current_game.get_test_scenarios()
        
        # 测试有通信
        rewards, success_rate, communication_data = evaluate_scenarios(model, env, test_scenarios, device, communication=True)
        
        # 测试无通信
        rewards_no_comm, success_rate_no_comm, _ = evaluate_scenarios(model_no_comm, env, test_scenarios, device, communication=False)
        
        # 保存结果
        with open(f"{save_dir}/current_game_evaluation.txt", "w") as f:
            f.write(f"游戏: {env.get_current_game_name()}\n")
            f.write(f"有通信平均奖励: {rewards:.4f}\n")
            f.write(f"有通信成功率: {success_rate:.2f}\n")
            f.write(f"无通信平均奖励: {rewards_no_comm:.4f}\n")
            f.write(f"无通信成功率: {success_rate_no_comm:.2f}\n")
            f.write(f"通信提升: {rewards - rewards_no_comm:.4f}\n")
        
        # 可视化通信
        visualize_communication(communication_data, save_dir, env.get_current_game_name())
    
    print("测试完成!")
    return

def evaluate_scenarios(model, env, scenarios, device, communication=True):
    """评估模型在特定场景下的表现"""
    rewards_list = []
    success_count = 0
    comm_data = []
    
    for scenario in scenarios:
        # 设置场景
        obs = env.reset(scenario=scenario)
        
        # 初始化状态
        hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
        social_memories = [model.init_social_memory().to(device) for _ in range(env.n_agents)]
        
        # 收集通信数据
        if communication:
            # 将观察转为张量
            obs_tensor = []
            for i in range(env.n_agents):
                obs_tensor.append(torch.FloatTensor(obs[i]).unsqueeze(0).to(device))
            
            # 分析通信
            messages = model.analyze_communication(obs_tensor, device)
            comm_data.append(messages)
        
        # 选择动作
        actions = model.select_actions(
            [torch.FloatTensor(o).unsqueeze(0).to(device) for o in obs],
            hidden_states,
            social_memories,
            epsilon=0.0,
            communication=communication
        )
        
        # 执行动作
        _, rewards, _, info = env.step(actions)
        avg_reward = sum(rewards) / env.n_agents
        rewards_list.append(avg_reward)
        
        # 记录成功
        if "success" in info and info["success"]:
            success_count += 1
    
    # 计算平均奖励和成功率
    avg_reward = sum(rewards_list) / len(rewards_list) if rewards_list else 0
    success_rate = success_count / len(scenarios) if scenarios else 0
    
    return avg_reward, success_rate, comm_data

def visualize_communication(comm_data, save_dir, game_name):
    """可视化通信内容"""
    if not comm_data:
        return
    
    plt.figure(figsize=(12, 8))
    
    # 分析每个智能体的通信
    for agent_idx in range(len(comm_data[0])):
        # 提取该智能体在所有场景下的通信内容
        agent_messages = [data[agent_idx] for data in comm_data]
        
        # 计算消息维度
        comm_dim = agent_messages[0].shape[1]
        
        # 创建子图
        ax = plt.subplot(len(comm_data[0]), 1, agent_idx + 1)
        
        # 构建热力图数据
        heat_data = np.array(agent_messages).reshape(len(agent_messages), comm_dim)
        
        # 绘制热力图
        sns.heatmap(heat_data, cmap="viridis", ax=ax)
        
        # 设置标题和标签
        ax.set_title(f"智能体 {agent_idx + 1} 通信内容")
        ax.set_xlabel("通信维度")
        ax.set_ylabel("场景")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{game_name}_communication.png")
    plt.close()

def visualize_all_games_comparison(results_with_comm, results_no_comm, save_dir):
    """可视化所有游戏上的性能比较"""
    games = list(results_with_comm.keys())
    rewards_with_comm = [results_with_comm[game]['reward'] for game in games]
    success_with_comm = [results_with_comm[game]['success_rate'] for game in games]
    rewards_no_comm = [results_no_comm[game]['reward'] for game in games]
    success_no_comm = [results_no_comm[game]['success_rate'] for game in games]
    
    # 计算通信提升
    comm_improvement = [w - n for w, n in zip(rewards_with_comm, rewards_no_comm)]
    
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    
    # 奖励图表
    ax1.bar(range(len(games)), rewards_with_comm, width=0.4, label='有通信', color='blue', alpha=0.7)
    ax1.bar([i + 0.4 for i in range(len(games))], rewards_no_comm, width=0.4, label='无通信', color='red', alpha=0.7)
    ax1.set_title('各游戏平均奖励')
    ax1.set_ylabel('平均奖励')
    ax1.set_xticks([i + 0.2 for i in range(len(games))])
    ax1.set_xticklabels(games)
    ax1.legend()
    
    # 成功率图表
    ax2.bar(range(len(games)), success_with_comm, width=0.4, label='有通信', color='blue', alpha=0.7)
    ax2.bar([i + 0.4 for i in range(len(games))], success_no_comm, width=0.4, label='无通信', color='red', alpha=0.7)
    ax2.set_title('各游戏成功率')
    ax2.set_ylabel('成功率')
    ax2.set_xticks([i + 0.2 for i in range(len(games))])
    ax2.set_xticklabels(games)
    ax2.legend()
    
    # 通信提升图表
    ax3.bar(range(len(games)), comm_improvement, width=0.6, label='通信提升', color='green', alpha=0.7)
    ax3.set_title('通信带来的奖励提升')
    ax3.set_ylabel('奖励提升')
    ax3.set_xticks(range(len(games)))
    ax3.set_xticklabels(games)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_games_comparison.png")
    plt.close()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试社会认知通信网络')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--config_path', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--curriculum', action='store_true', default=False, help='是否启用课程学习')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # 添加命令行参数到配置中
    config['curriculum_learning'] = args.curriculum
    
    # 运行测试
    test(args.model_path, config) 