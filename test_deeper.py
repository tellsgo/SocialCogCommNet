import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from games.simple_coordination import SimpleCoordinationGame
from environment import CooperativeEnvironment
from model import SocialCognitiveCommNet
from games.state_processor import StateProcessor

def train_model(epochs=100, hidden_dim=128, comm_dim=64, memory_dim=64, learning_rate=0.001):
    """
    对简单协调游戏进行更深入的训练
    
    参数:
        epochs: 训练回合数
        hidden_dim: 隐藏层维度
        comm_dim: 通信维度
        memory_dim: 社会记忆维度
        learning_rate: 学习率
        
    返回:
        训练好的模型和游戏环境
    """
    print("="*60)
    print("深度训练 - 简单协调游戏 (SimpleCoordinationGame)")
    print("="*60)
    
    # 创建保存目录
    results_dir = "./results/deeper_training"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建游戏和环境
    game = SimpleCoordinationGame()
    env = CooperativeEnvironment(game=game)
    
    print(f"环境初始化成功")
    print(f"  游戏名称: {game.name}")
    print(f"  智能体数量: {env.n_agents}")
    print(f"  动作数量: {env.n_actions}")
    print(f"  状态维度: {StateProcessor.UNIFIED_STATE_DIM}")
    print(f"  通信维度: {comm_dim}")
    
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
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录训练数据
    rewards_history = []
    success_rates = []
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(epochs):
        epoch_rewards = []
        epoch_successes = []
        
        # 每个回合训练多个场景
        for _ in range(10):  # 每回合10个场景
            # 重置环境
            obs = env.reset()
            
            # 初始化隐藏状态和社会记忆
            hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
            social_memories = [model.init_social_memory().to(device) for _ in range(env.n_agents)]
            
            # 转换观察为张量
            obs_tensor = []
            for i in range(env.n_agents):
                obs_tensor.append(torch.FloatTensor(obs[i]).unsqueeze(0).to(device))
            
            # 选择动作
            epsilon = max(0.1, 1.0 - epoch / epochs)
            actions = model.select_actions(obs_tensor, hidden_states, social_memories, epsilon)
            
            # 执行动作
            next_obs, rewards, done, info = env.step(actions)
            
            # 记录结果
            success = info.get("success", False)
            epoch_rewards.append(sum(rewards) / env.n_agents)
            epoch_successes.append(1 if success else 0)
            
            # 进行简单的Q学习更新
            # 将下一个观察转换为张量
            next_obs_tensor = []
            for i in range(env.n_agents):
                next_obs_tensor.append(torch.FloatTensor(next_obs[i]).unsqueeze(0).to(device))
            
            # 获取当前的Q值和下一个状态的Q值
            q_values, state_values, _ = model.get_q_values(obs_tensor, hidden_states, social_memories)
            next_q_values, _, _ = model.get_q_values(next_obs_tensor, hidden_states, social_memories)
            
            # 为每个智能体进行简单的Q学习更新
            optimizer.zero_grad()
            loss = 0
            gamma = 0.99  # 折扣因子
            
            for i in range(env.n_agents):
                # 目标Q值 - 将奖励转换为标量
                reward_scalar = float(rewards[i])
                target_q = torch.tensor([reward_scalar], dtype=torch.float32, device=device)
                target_q += gamma * next_q_values[i].max(1)[0].detach()
                
                # 当前Q值 (对所选动作)
                current_q = q_values[i][0, actions[i]]
                
                # 添加到损失
                loss += torch.nn.functional.mse_loss(current_q, target_q)
            
            loss /= env.n_agents
            loss.backward()
            optimizer.step()
        
        # 记录本回合的平均表现
        rewards_history.append(np.mean(epoch_rewards))
        success_rates.append(np.mean(epoch_successes))
        
        # 报告训练进度
        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_reward = np.mean(rewards_history[-10:]) if epoch >= 10 else np.mean(rewards_history)
            avg_success = np.mean(success_rates[-10:]) if epoch >= 10 else np.mean(success_rates)
            print(f"回合 {epoch}/{epochs}:")
            print(f"  奖励: {avg_reward:.4f}, 成功率: {avg_success:.2f}")
    
    # 保存模型
    torch.save(model.state_dict(), f"{results_dir}/model.pt")
    print(f"\n模型已保存到 {results_dir}/model.pt")
    
    # 绘制学习曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('平均奖励')
    plt.xlabel('训练回合')
    plt.ylabel('奖励')
    
    plt.subplot(1, 2, 2)
    plt.plot(success_rates)
    plt.title('成功率')
    plt.xlabel('训练回合')
    plt.ylabel('成功率')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/learning_curve.png")
    
    print(f"学习曲线已保存到 {results_dir}/learning_curve.png")
    
    return model, env, device

def test_communication(model, env, device, n_scenarios=5):
    """测试智能体之间的通信"""
    game = env.current_game
    
    print("\n" + "="*60)
    print("通信测试")
    print("="*60)
    
    # 获取测试场景
    test_scenarios = game.get_test_scenarios()
    
    if not test_scenarios:
        print("没有可用的测试场景")
        return
    
    # 运行多次测试，收集统计数据
    success_count = 0
    reward_sum = 0
    
    for i, scenario in enumerate(test_scenarios[:n_scenarios]):
        print(f"\n场景 {i+1}:")
        
        # 重置游戏
        obs = game.reset(scenario)
        
        # 处理观察为统一维度
        processed_obs = StateProcessor.process_observations(game.name, obs)
        
        # 将观察转换为张量
        obs_tensor = []
        for j in range(game.n_agents):
            obs_tensor.append(torch.FloatTensor(processed_obs[j]).unsqueeze(0).to(device))
        
        # 初始化隐藏状态和社会记忆
        hidden_states = [model.init_hidden().to(device) for _ in range(game.n_agents)]
        social_memories = [model.init_social_memory().to(device) for _ in range(game.n_agents)]
        
        # 测试两种情况：有通信和无通信
        # 有通信
        print("【有通信】")
        comm_messages = model.get_communication(obs_tensor, hidden_states)
        
        # 打印通信内容
        for agent_idx, message in enumerate(comm_messages):
            print(f"智能体 {agent_idx+1} 发送消息:")
            message_np = message.cpu().detach().numpy()
            
            # 找出最活跃的通信维度
            active_dims = np.argsort(np.abs(message_np).flatten())[-5:]  # 取前5个最活跃维度
            for dim in active_dims:
                value = message_np.flatten()[dim]
                print(f"  维度 {dim}: {value:.4f}")
        
        # 选择动作（有通信）
        actions_with_comm = model.select_actions(obs_tensor, hidden_states, social_memories, epsilon=0.0, communication=True)
        
        # 执行动作
        next_obs, rewards_with_comm, done, info = game.step(actions_with_comm)
        print(f"\n选择动作: {actions_with_comm}")
        print(f"获得奖励: {rewards_with_comm}")
        print(f"成功: {info.get('success', False)}")
        
        if info.get("success", False):
            success_count += 1
        reward_sum += sum(rewards_with_comm) / game.n_agents
        
        # 重置游戏，测试无通信情况
        print("\n【无通信】")
        obs = game.reset(scenario)
        processed_obs = StateProcessor.process_observations(game.name, obs)
        
        # 将观察转换为张量
        obs_tensor = []
        for j in range(game.n_agents):
            obs_tensor.append(torch.FloatTensor(processed_obs[j]).unsqueeze(0).to(device))
        
        # 初始化隐藏状态和社会记忆
        hidden_states = [model.init_hidden().to(device) for _ in range(game.n_agents)]
        social_memories = [model.init_social_memory().to(device) for _ in range(game.n_agents)]
        
        # 选择动作（无通信）
        actions_no_comm = model.select_actions(obs_tensor, hidden_states, social_memories, epsilon=0.0, communication=False)
        
        # 执行动作
        next_obs, rewards_no_comm, done, info = game.step(actions_no_comm)
        print(f"选择动作: {actions_no_comm}")
        print(f"获得奖励: {rewards_no_comm}")
        print(f"成功: {info.get('success', False)}")
        
        # 比较有无通信的奖励差异
        print(f"\n通信提升: {sum(rewards_with_comm) / len(rewards_with_comm) - sum(rewards_no_comm) / len(rewards_no_comm):.4f}")
    
    print("\n" + "="*60)
    print(f"测试结果 ({n_scenarios}个场景):")
    print(f"  平均奖励: {reward_sum / n_scenarios:.4f}")
    print(f"  成功率: {success_count / n_scenarios:.2f}")
    print("="*60)

def analyze_communication_importance(model, env, device):
    """分析通信对于任务完成的重要性"""
    game = env.current_game
    
    print("\n" + "="*60)
    print("通信重要性分析")
    print("="*60)
    
    # 准备测试数据
    n_trials = 50
    rewards_with_comm = []
    success_with_comm = []
    rewards_without_comm = []
    success_without_comm = []
    
    for i in range(n_trials):
        # 重置环境
        obs = env.reset()
        
        # 设置场景编号（用于输出信息）
        scenario_id = i + 1
        
        # 测试有通信的情况
        hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
        social_memories = [model.init_social_memory().to(device) for _ in range(env.n_agents)]
        
        # 转换观察为张量
        obs_tensor = []
        for j in range(env.n_agents):
            obs_tensor.append(torch.FloatTensor(obs[j]).unsqueeze(0).to(device))
        
        # 选择动作（有通信）
        actions = model.select_actions(obs_tensor, hidden_states, social_memories, epsilon=0.0, communication=True)
        
        # 执行动作
        _, rewards, _, info = env.step(actions)
        
        # 记录结果
        rewards_with_comm.append(sum(rewards) / env.n_agents)
        success_with_comm.append(1 if info.get("success", False) else 0)
        
        # 重置环境，测试无通信情况
        obs = env.reset()
        
        # 转换观察为张量
        obs_tensor = []
        for j in range(env.n_agents):
            obs_tensor.append(torch.FloatTensor(obs[j]).unsqueeze(0).to(device))
        
        # 初始化隐藏状态和社会记忆
        hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
        social_memories = [model.init_social_memory().to(device) for _ in range(env.n_agents)]
        
        # 选择动作（无通信）
        actions = model.select_actions(obs_tensor, hidden_states, social_memories, epsilon=0.0, communication=False)
        
        # 执行动作
        _, rewards, _, info = env.step(actions)
        
        # 记录结果
        rewards_without_comm.append(sum(rewards) / env.n_agents)
        success_without_comm.append(1 if info.get("success", False) else 0)
    
    # 计算和显示统计数据
    avg_reward_with_comm = np.mean(rewards_with_comm)
    success_rate_with_comm = np.mean(success_with_comm)
    avg_reward_without_comm = np.mean(rewards_without_comm)
    success_rate_without_comm = np.mean(success_without_comm)
    
    print(f"有通信:")
    print(f"  平均奖励: {avg_reward_with_comm:.4f}")
    print(f"  成功率: {success_rate_with_comm:.2f}")
    
    print(f"\n无通信:")
    print(f"  平均奖励: {avg_reward_without_comm:.4f}")
    print(f"  成功率: {success_rate_without_comm:.2f}")
    
    print(f"\n通信提升:")
    print(f"  奖励提升: {avg_reward_with_comm - avg_reward_without_comm:.4f}")
    print(f"  成功率提升: {(success_rate_with_comm - success_rate_without_comm) * 100:.2f}%")
    
    # 创建可视化
    results_dir = "./results/deeper_training"
    
    labels = ['有通信', '无通信']
    rewards = [avg_reward_with_comm, avg_reward_without_comm]
    success_rates = [success_rate_with_comm, success_rate_without_comm]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # 奖励对比
    ax[0].bar(labels, rewards, color=['green', 'red'])
    ax[0].set_ylabel('平均奖励')
    ax[0].set_title('通信对奖励的影响')
    
    # 成功率对比
    ax[1].bar(labels, success_rates, color=['green', 'red'])
    ax[1].set_ylabel('成功率')
    ax[1].set_title('通信对成功率的影响')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/communication_importance.png")
    
    print(f"\n通信重要性分析已保存到 {results_dir}/communication_importance.png")

if __name__ == "__main__":
    # 训练模型
    model, env, device = train_model(epochs=100)
    
    # 测试通信
    test_communication(model, env, device)
    
    # 分析通信重要性
    analyze_communication_importance(model, env, device) 