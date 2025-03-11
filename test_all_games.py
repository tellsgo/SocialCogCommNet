import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from games.simple_coordination import SimpleCoordinationGame
from games.asymmetric_info import AsymmetricInfoGame
from games.sequential_decision import SequentialDecisionGame
from games.partial_observable import PartialObservableGame
from environment import CooperativeEnvironment
from model import SocialCognitiveCommNet
from games.state_processor import StateProcessor
import json

def test_game(game_name, epochs=100, hidden_dim=128, comm_dim=64, memory_dim=64, 
             learning_rate=1e-4, batch_size=32, buffer_size=10000):
    """
    测试单个游戏的性能和通信演化
    
    参数:
        game_name: 游戏名称，如'SimpleCoordinationGame'
        epochs: 训练回合数
        hidden_dim: 隐藏层维度
        comm_dim: 通信维度
        memory_dim: 社会记忆维度
        learning_rate: 学习率
        batch_size: 批量大小
        buffer_size: 经验回放缓冲区大小
    """
    # 创建保存目录
    results_dir = f"./results/{game_name}_test"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建游戏
    if game_name == "SimpleCoordinationGame":
        game = SimpleCoordinationGame()
    elif game_name == "AsymmetricInfoGame":
        game = AsymmetricInfoGame()
    elif game_name == "SequentialDecisionGame":
        game = SequentialDecisionGame()
    elif game_name == "PartialObservableGame":
        game = PartialObservableGame()
    else:
        raise ValueError(f"未知的游戏名称: {game_name}")
    
    # 创建环境
    env = CooperativeEnvironment(game=game)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SocialCognitiveCommNet(
        input_dim=StateProcessor.UNIFIED_STATE_DIM,
        hidden_dim=hidden_dim,
        comm_dim=comm_dim,
        memory_dim=memory_dim,
        n_agents=env.n_agents,
        n_actions=env.n_actions
    ).to(device)
    
    # 比较模型 - 无通信
    model_no_comm = SocialCognitiveCommNet(
        input_dim=StateProcessor.UNIFIED_STATE_DIM,
        hidden_dim=hidden_dim,
        comm_dim=comm_dim,
        memory_dim=memory_dim,
        n_agents=env.n_agents,
        n_actions=env.n_actions
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_no_comm = torch.optim.Adam(model_no_comm.parameters(), lr=learning_rate)
    
    # 创建经验回放缓冲区
    from collections import deque
    buffer = deque(maxlen=buffer_size)
    buffer_no_comm = deque(maxlen=buffer_size)
    
    # 记录训练数据
    rewards_history = []
    rewards_history_no_comm = []
    success_rates = []
    success_rates_no_comm = []
    
    # 训练循环
    print(f"开始训练游戏: {game_name}")
    for epoch in range(epochs):
        # 重置环境
        obs = env.reset()
        
        # 初始化隐藏状态和社会记忆
        hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
        social_memories = [model.init_social_memory().to(device) for _ in range(env.n_agents)]
        
        hidden_states_no_comm = [model_no_comm.init_hidden().to(device) for _ in range(env.n_agents)]
        social_memories_no_comm = [model_no_comm.init_social_memory().to(device) for _ in range(env.n_agents)]
        
        # 转换观察为张量
        obs_tensor = []
        for i in range(env.n_agents):
            obs_tensor.append(torch.FloatTensor(obs[i]).unsqueeze(0).to(device))
        
        # 有通信模型选择动作
        epsilon = max(0.05, 1.0 - epoch / epochs)
        actions = model.select_actions(obs_tensor, hidden_states, social_memories, epsilon)
        
        # 执行动作
        next_obs, rewards, done, info = env.step(actions)
        
        # 记录结果
        success = info.get("success", False)
        rewards_history.append(sum(rewards) / env.n_agents)
        success_rates.append(1 if success else 0)
        
        # 无通信模型选择动作
        actions_no_comm = model_no_comm.select_actions(
            obs_tensor, hidden_states_no_comm, social_memories_no_comm, epsilon, communication=False
        )
        
        # 执行动作
        next_obs_no_comm, rewards_no_comm, done_no_comm, info_no_comm = env.step(actions_no_comm)
        
        # 记录结果
        success_no_comm = info_no_comm.get("success", False)
        rewards_history_no_comm.append(sum(rewards_no_comm) / env.n_agents)
        success_rates_no_comm.append(1 if success_no_comm else 0)
        
        # 简单训练（这里简化了训练过程）
        if epoch > 0 and epoch % 20 == 0:
            print(f"回合 {epoch}/{epochs}:")
            print(f"  有通信奖励: {np.mean(rewards_history[-20:]):.4f}, 成功率: {np.mean(success_rates[-20:]):.2f}")
            print(f"  无通信奖励: {np.mean(rewards_history_no_comm[-20:]):.4f}, 成功率: {np.mean(success_rates_no_comm[-20:]):.2f}")
    
    # 保存结果
    results = {
        "game_name": game_name,
        "epochs": epochs,
        "rewards_history": rewards_history,
        "rewards_history_no_comm": rewards_history_no_comm,
        "success_rates": success_rates,
        "success_rates_no_comm": success_rates_no_comm,
        "avg_reward": np.mean(rewards_history[-20:]),
        "avg_reward_no_comm": np.mean(rewards_history_no_comm[-20:]),
        "avg_success_rate": np.mean(success_rates[-20:]),
        "avg_success_rate_no_comm": np.mean(success_rates_no_comm[-20:])
    }
    
    # 保存训练结果
    with open(f"{results_dir}/results.json", "w") as f:
        json.dump({
            "rewards": rewards_history,
            "rewards_no_comm": rewards_history_no_comm,
            "success_rates": success_rates,
            "success_rates_no_comm": success_rates_no_comm
        }, f)
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(rewards_history, 'b-', label='有通信')
    plt.plot(rewards_history_no_comm, 'r-', label='无通信')
    plt.title(f'{game_name} - 平均奖励')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(success_rates, 'b-', label='有通信')
    plt.plot(success_rates_no_comm, 'r-', label='无通信')
    plt.title(f'{game_name} - 成功率')
    plt.xlabel('回合')
    plt.ylabel('成功率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_results.png")
    
    # 保存模型
    torch.save(model.state_dict(), f"{results_dir}/model.pt")
    
    # 分析通信
    from games.game_manager import GameManager
    game_manager = GameManager([game], save_dir=results_dir)
    game_manager.log_communication_evolution(model, epochs)
    
    print(f"游戏 {game_name} 测试完成，结果已保存到 {results_dir}")
    
    return results

def test_all_games(epochs=100):
    """测试所有游戏"""
    games = [
        "SimpleCoordinationGame",
        "AsymmetricInfoGame",
        "SequentialDecisionGame",
        "PartialObservableGame"
    ]
    
    all_results = {}
    
    for game_name in games:
        print(f"\n===== 测试游戏: {game_name} =====\n")
        results = test_game(game_name, epochs=epochs)
        all_results[game_name] = results
    
    # 比较所有游戏的结果
    compare_games_results(all_results)
    
    return all_results

def compare_games_results(all_results):
    """比较所有游戏的结果"""
    # 创建保存目录
    results_dir = "./results/all_games_comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    # 提取数据
    games = list(all_results.keys())
    rewards = [all_results[game]["avg_reward"] for game in games]
    rewards_no_comm = [all_results[game]["avg_reward_no_comm"] for game in games]
    success_rates = [all_results[game]["avg_success_rate"] for game in games]
    success_rates_no_comm = [all_results[game]["avg_success_rate_no_comm"] for game in games]
    
    # 计算通信提升
    comm_improvement = [rewards[i] - rewards_no_comm[i] for i in range(len(games))]
    success_improvement = [success_rates[i] - success_rates_no_comm[i] for i in range(len(games))]
    
    # 绘制结果
    plt.figure(figsize=(15, 10))
    
    # 奖励图
    plt.subplot(2, 2, 1)
    x = np.arange(len(games))
    width = 0.35
    plt.bar(x - width/2, rewards, width, label='有通信')
    plt.bar(x + width/2, rewards_no_comm, width, label='无通信')
    plt.title('各游戏平均奖励')
    plt.xticks(x, games, rotation=45)
    plt.ylabel('奖励')
    plt.legend()
    plt.grid(True)
    
    # 成功率图
    plt.subplot(2, 2, 2)
    plt.bar(x - width/2, success_rates, width, label='有通信')
    plt.bar(x + width/2, success_rates_no_comm, width, label='无通信')
    plt.title('各游戏成功率')
    plt.xticks(x, games, rotation=45)
    plt.ylabel('成功率')
    plt.legend()
    plt.grid(True)
    
    # 通信提升
    plt.subplot(2, 2, 3)
    plt.bar(x, comm_improvement, width, label='奖励提升')
    plt.title('通信带来的奖励提升')
    plt.xticks(x, games, rotation=45)
    plt.ylabel('奖励提升')
    plt.grid(True)
    
    # 成功率提升
    plt.subplot(2, 2, 4)
    plt.bar(x, success_improvement, width, label='成功率提升')
    plt.title('通信带来的成功率提升')
    plt.xticks(x, games, rotation=45)
    plt.ylabel('成功率提升')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/games_comparison.png")
    
    # 保存比较结果
    with open(f"{results_dir}/comparison_results.txt", "w") as f:
        f.write("游戏比较结果:\n\n")
        
        for i, game in enumerate(games):
            f.write(f"游戏: {game}\n")
            f.write(f"  有通信平均奖励: {rewards[i]:.4f}\n")
            f.write(f"  无通信平均奖励: {rewards_no_comm[i]:.4f}\n")
            f.write(f"  通信奖励提升: {comm_improvement[i]:.4f}\n")
            f.write(f"  有通信成功率: {success_rates[i]:.2f}\n")
            f.write(f"  无通信成功率: {success_rates_no_comm[i]:.2f}\n")
            f.write(f"  通信成功率提升: {success_improvement[i]:.2f}\n\n")
        
        # 计算加权平均提升
        avg_comm_improvement = np.mean(comm_improvement)
        avg_success_improvement = np.mean(success_improvement)
        
        f.write("\n总体通信效果:\n")
        f.write(f"  平均奖励提升: {avg_comm_improvement:.4f}\n")
        f.write(f"  平均成功率提升: {avg_success_improvement:.2f}\n")
    
    print(f"所有游戏比较结果已保存到 {results_dir}")

if __name__ == "__main__":
    # 测试所有游戏
    test_all_games(epochs=100) 