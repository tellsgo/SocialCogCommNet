import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

from environment import CooperativeEnvironment
from model import SocialCognitiveCommNet

# 设置随机种子以便复现结果
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        存储一条经验
        
        参数:
            state: 当前状态 [agent_1_state, agent_2_state, ...]
            action: 动作 [agent_1_action, agent_2_action, ...]
            reward: 奖励 [agent_1_reward, agent_2_reward, ...]
            next_state: 下一个状态 [agent_1_next_state, agent_2_next_state, ...]
            done: 是否结束 (布尔值)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        从经验回放中随机采样
        
        参数:
            batch_size: 批量大小
            
        返回:
            batch: (states, actions, rewards, next_states, dones) 元组
        """
        batch = random.sample(self.memory, batch_size)
        # 解包经验成批次
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

# 训练函数
def train(config):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu")
    print(f"Using device: {device}")
    
    # 创建环境
    env = CooperativeEnvironment(curriculum_learning=config.get('curriculum_learning', False))
    
    # 创建模型
    model = SocialCognitiveCommNet(
        input_dim=env.state_dim,
        hidden_dim=config['hidden_dim'],
        comm_dim=config['comm_dim'],
        memory_dim=config['memory_dim'],
        n_agents=env.n_agents,
        n_actions=env.n_actions
    ).to(device)
    
    # 为无通信版本创建单独的模型 (用于比较)
    model_no_comm = SocialCognitiveCommNet(
        input_dim=env.state_dim,
        hidden_dim=config['hidden_dim'],
        comm_dim=config['comm_dim'],
        memory_dim=config['memory_dim'],
        n_agents=env.n_agents,
        n_actions=env.n_actions
    ).to(device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    optimizer_no_comm = optim.Adam(model_no_comm.parameters(), lr=config['learning_rate'])
    
    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer(config['buffer_capacity'])
    replay_buffer_no_comm = ReplayBuffer(config['buffer_capacity'])
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config['save_dir'], timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存到: {save_dir}")
    
    # 保存配置
    with open(f"{save_dir}/config.txt", "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # 统计数据
    rewards_history = []
    rewards_history_no_comm = []
    losses_history = []
    losses_history_no_comm = []
    success_rates = []
    success_rates_no_comm = []
    
    # 是否需要切换到下一个游戏
    next_game_triggered = False
    
    # 训练开始时间
    start_time = time.time()
    
    # 在每个游戏上的统计数据
    game_statistics = {}
    game_episodes = 0
    
    # 当前游戏名称
    current_game = env.get_current_game_name()
    game_statistics[current_game] = {
        "rewards": [],
        "success_rates": [],
        "episodes": 0
    }
    
    # 训练循环
    for episode in range(config['n_episodes']):
        # 重置环境
        obs = env.reset()
        
        # 初始化隐藏状态和社会记忆
        hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
        social_memories = [model.init_social_memory().to(device) for _ in range(env.n_agents)]
        
        hidden_states_no_comm = [model_no_comm.init_hidden().to(device) for _ in range(env.n_agents)]
        social_memories_no_comm = [model_no_comm.init_social_memory().to(device) for _ in range(env.n_agents)]
        
        # 训练统计
        episode_rewards = []
        episode_rewards_no_comm = []
        success_count = 0
        success_count_no_comm = 0
        steps = 0
        
        # 单回合训练
        done = False
        while not done and steps < config['max_steps']:
            steps += 1
            
            # 将观察转换为张量
            obs_tensor = []
            for i in range(env.n_agents):
                obs_tensor.append(torch.FloatTensor(obs[i]).unsqueeze(0).to(device))
            
            # 减小探索率
            epsilon = config['epsilon_end'] + (config['epsilon_start'] - config['epsilon_end']) * \
                        np.exp(-1. * episode * config['max_steps'] / config['epsilon_decay'])
            
            # 有通信模型选择动作
            actions = model.select_actions(
                obs_tensor, hidden_states, social_memories, epsilon
            )
            
            # 执行动作并获取奖励
            next_obs, rewards, done, info = env.step(actions)
            episode_rewards.append(sum(rewards) / env.n_agents)
            
            # 记录步骤是否成功（完成目标）
            if "success" in info and info["success"]:
                success_count += 1
            
            # 储存经验
            replay_buffer.push(obs, actions, rewards, next_obs, done)
            
            # 如果经验足够，进行学习
            if len(replay_buffer) >= config['batch_size']:
                loss = optimize_model(model, optimizer, replay_buffer, criterion, config, device)
                losses_history.append(loss)
            
            # 更新观察
            obs = next_obs
            
            # 无通信模型选择动作
            actions_no_comm = model_no_comm.select_actions(
                obs_tensor, hidden_states_no_comm, social_memories_no_comm, epsilon, communication=False
            )
            
            # 执行动作并获取奖励
            next_obs_no_comm, rewards_no_comm, done_no_comm, info_no_comm = env.step(actions_no_comm)
            episode_rewards_no_comm.append(sum(rewards_no_comm) / env.n_agents)
            
            # 记录步骤是否成功（完成目标）
            if "success" in info_no_comm and info_no_comm["success"]:
                success_count_no_comm += 1
            
            # 储存经验
            replay_buffer_no_comm.push(obs, actions_no_comm, rewards_no_comm, next_obs_no_comm, done_no_comm)
            
            # 如果经验足够，进行学习
            if len(replay_buffer_no_comm) >= config['batch_size']:
                loss_no_comm = optimize_model(model_no_comm, optimizer_no_comm, replay_buffer_no_comm, criterion, config, device, communication=False)
                losses_history_no_comm.append(loss_no_comm)
        
        # 计算平均奖励和损失
        avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        avg_reward_no_comm = sum(episode_rewards_no_comm) / len(episode_rewards_no_comm) if episode_rewards_no_comm else 0
        
        episode_loss = sum(losses_history[-steps:]) / steps if losses_history and steps > 0 else 0
        episode_loss_no_comm = sum(losses_history_no_comm[-steps:]) / steps if losses_history_no_comm and steps > 0 else 0
        
        # 计算成功率
        success_rate = success_count / steps if steps > 0 else 0
        success_rate_no_comm = success_count_no_comm / steps if steps > 0 else 0
        
        # 记录历史
        rewards_history.append(avg_reward)
        rewards_history_no_comm.append(avg_reward_no_comm)
        success_rates.append(success_rate)
        success_rates_no_comm.append(success_rate_no_comm)
        
        # 记录当前游戏的统计数据
        game_episodes += 1
        current_game = env.get_current_game_name()
        if current_game not in game_statistics:
            game_statistics[current_game] = {
                "rewards": [],
                "success_rates": [],
                "episodes": 0
            }
        game_statistics[current_game]["rewards"].append(avg_reward)
        game_statistics[current_game]["success_rates"].append(success_rate)
        game_statistics[current_game]["episodes"] += 1
        
        # 课程学习：检查是否需要转向下一个游戏
        if config.get('curriculum_learning', False) and not next_game_triggered:
            # 通过平均成功率检查是否已经学习得很好
            recent_success_rates = success_rates[-min(5, len(success_rates)):]
            avg_success_rate = sum(recent_success_rates) / len(recent_success_rates)
            
            # 如果最近5次回合的平均成功率超过60%并且至少已经玩了10个回合，则转向下一个游戏
            if avg_success_rate > 0.6 and game_episodes >= 10:
                # 保存当前游戏的模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode,
                    'game': current_game
                }, f"{save_dir}/model_{current_game}.pt")
                
                # 记录通信演化
                env.log_communication(model, episode)
                
                # 切换到下一个游戏
                success, new_model, new_optimizer = env.next_game(model, optimizer)
                if success:
                    # 更新模型和优化器
                    model = new_model
                    optimizer = new_optimizer
                    
                    # 为无通信模型创建完全相同的新模型
                    # 注意：这里我们不使用load_state_dict，而是重新创建模型
                    from games.state_processor import StateProcessor
                    model_no_comm = SocialCognitiveCommNet(
                        input_dim=StateProcessor.UNIFIED_STATE_DIM,  # 使用统一状态维度
                        hidden_dim=config['hidden_dim'],
                        comm_dim=config['comm_dim'],
                        memory_dim=config['memory_dim'],
                        n_agents=env.n_agents,
                        n_actions=env.n_actions
                    ).to(device)
                    
                    # 手动复制参数，避免尺寸不匹配问题
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name in model_no_comm.state_dict():
                                target_param = model_no_comm.state_dict()[name]
                                if param.shape == target_param.shape:
                                    model_no_comm.state_dict()[name].copy_(param)
                    
                    optimizer_no_comm = optim.Adam(model_no_comm.parameters(), lr=config['learning_rate'])
                    
                    # 清空回放缓冲区
                    replay_buffer = ReplayBuffer(config['buffer_capacity'])
                    replay_buffer_no_comm = ReplayBuffer(config['buffer_capacity'])
                    
                    # 重置游戏计数器
                    game_episodes = 0
                    next_game_triggered = False
                else:
                    # 如果没有下一个游戏，设置已触发标志避免重复检查
                    next_game_triggered = True
        
        # 打印训练信息
        if (episode + 1) % config['print_interval'] == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode {episode+1}/{config['n_episodes']} - 耗时: {elapsed_time:.2f}秒")
            print(f"游戏: {current_game}")
            print(f"  有通信平均奖励: {rewards_history[-1]:.4f}, 损失: {episode_loss:.4f}, 成功率: {success_rate:.2f}")
            print(f"  无通信平均奖励: {rewards_history_no_comm[-1]:.4f}, 损失: {episode_loss_no_comm:.4f}, 成功率: {success_rate_no_comm:.2f}")
            
            # 评估模型
            if (episode + 1) % config['eval_interval'] == 0:
                eval_reward, eval_success_rate = env.current_game.evaluate(model, device)
                eval_reward_no_comm, eval_success_rate_no_comm = env.current_game.evaluate(model_no_comm, device, communication=False)
                print(f"  有通信评估奖励: {eval_reward:.4f}, 成功率: {eval_success_rate:.2f}")
                print(f"  无通信评估奖励: {eval_reward_no_comm:.4f}, 成功率: {eval_success_rate_no_comm:.2f}")
                print(f"  通信提升: {eval_reward - eval_reward_no_comm:.4f}")
                
                # 保存模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode,
                    'game': current_game
                }, f"{save_dir}/model_{episode+1}.pt")
                
                # 当前游戏的通信分析
                env.log_communication(model, episode+1)
                
                # 可视化训练进展
                visualize_training(
                    rewards_history, rewards_history_no_comm,
                    losses_history, losses_history_no_comm,
                    success_rates, success_rates_no_comm,
                    save_dir, episode+1
                )
    
    # 训练结束后的完整可视化
    visualize_training(
        rewards_history, rewards_history_no_comm,
        losses_history, losses_history_no_comm,
        success_rates, success_rates_no_comm,
        save_dir, config['n_episodes']
    )
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': config['n_episodes'],
        'game': current_game
    }, f"{save_dir}/model_final.pt")
    
    # 在所有游戏上评估最终模型
    if config.get('curriculum_learning', False):
        all_games_results = env.evaluate_all_games(model, device)
        
        # 保存评估结果
        with open(f"{save_dir}/evaluation_results.txt", "w") as f:
            f.write("最终模型在所有游戏上的评估结果:\n\n")
            for game_name, results in all_games_results.items():
                f.write(f"游戏: {game_name}\n")
                f.write(f"  平均奖励: {results['reward']:.4f}\n")
                f.write(f"  成功率: {results['success_rate']:.2f}\n\n")
    
    print(f"训练完成! 总耗时: {time.time() - start_time:.2f}秒")
    
    return model, rewards_history

# 优化模型
def optimize_model(model, optimizer, replay_buffer, criterion, config, device, communication=True):
    """
    优化模型参数
    
    参数:
        model: 模型
        optimizer: 优化器
        replay_buffer: 经验回放缓冲区
        criterion: 损失函数
        config: 配置字典
        device: 计算设备
        communication: 是否使用通信
    
    返回:
        loss_value: 当前批次的损失值
    """
    # 如果经验不足，不进行学习
    if len(replay_buffer) < config['batch_size']:
        return 0.0
    
    # 从经验回放中采样
    batch = replay_buffer.sample(config['batch_size'])
    states, actions, rewards, next_states, dones = batch
    
    # 转换为张量
    state_tensors = []
    next_state_tensors = []
    for i in range(len(states[0])):  # 对每个智能体
        states_i = [s[i] for s in states]
        next_states_i = [s[i] for s in next_states]
        state_tensors.append(torch.FloatTensor(states_i).to(device))
        next_state_tensors.append(torch.FloatTensor(next_states_i).to(device))
    
    actions_tensor = torch.LongTensor(actions).to(device)
    rewards_tensor = torch.FloatTensor(rewards).to(device)
    dones_tensor = torch.FloatTensor(dones).to(device)
    
    # 初始化隐藏状态和社会记忆
    batch_size = len(states)
    hidden_states = [model.init_hidden(batch_size).to(device) for _ in range(len(states[0]))]
    social_memories = [model.init_social_memory(batch_size).to(device) for _ in range(len(states[0]))]
    
    # 计算当前Q值
    action_values, state_values, _ = model.get_q_values(
        state_tensors, hidden_states, social_memories, communication
    )
    
    q_values = []
    for i in range(len(action_values)):
        q_value = action_values[i].gather(1, actions_tensor[:, i].unsqueeze(1)).squeeze(1)
        q_values.append(q_value)
    
    # 计算目标Q值
    next_hidden_states = [model.init_hidden(batch_size).to(device) for _ in range(len(states[0]))]
    next_social_memories = [model.init_social_memory(batch_size).to(device) for _ in range(len(states[0]))]
    
    with torch.no_grad():
        next_action_values, next_state_values, _ = model.get_q_values(
            next_state_tensors, next_hidden_states, next_social_memories, communication
        )
        
        next_q_values = []
        for i in range(len(next_action_values)):
            next_q_value = next_action_values[i].max(1)[0]
            next_q_values.append(next_q_value)
        
        # 使用平均值进行多智能体Q学习
        avg_next_q = torch.stack(next_q_values).mean(0)
        target_q = rewards_tensor.mean(1) + (1 - dones_tensor) * config['gamma'] * avg_next_q
    
    # 计算损失 (使用所有智能体的平均损失)
    loss = 0
    for i in range(len(q_values)):
        loss += criterion(q_values[i], target_q)
    
    loss /= len(q_values)
    
    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    if config.get('grad_clip', 0) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
    
    optimizer.step()
    
    return loss.item()

# 评估函数
def evaluate(model, env, config, device, communication=True, n_episodes=10):
    model.eval()  # 设置为评估模式
    total_reward = 0
    total_success = 0
    
    with torch.no_grad():
        for _ in range(n_episodes):
            states = env.reset()
            hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
            social_memories = [model.init_social_memory().to(device) for _ in range(env.n_agents)]
            episode_reward = 0
            success_count = 0
            
            for _ in range(config['max_steps']):
                # 将状态转换为张量
                states_tensor = [torch.FloatTensor(state).unsqueeze(0).to(device) for state in states]
                
                # 选择行动 (无探索)
                actions, hidden_states, social_memories = model.select_actions(
                    states_tensor, hidden_states, social_memories, epsilon=0.0
                )
                
                # 执行行动
                states, rewards, done, _ = env.step(actions)
                
                # 记录成功次数
                if sum(rewards) > 0:  # 如果获得正奖励，视为成功
                    success_count += 1
                
                # 累积奖励
                episode_reward += sum(rewards) / env.n_agents
            
            total_reward += episode_reward / config['max_steps']
            total_success += success_count / config['max_steps']
    
    model.train()  # 恢复为训练模式
    return total_reward / n_episodes, total_success / n_episodes

# 分析通信内容
def analyze_communication(model, env, config, device, save_dir, episode):
    """分析智能体之间的通信内容"""
    model.eval()  # 设置为评估模式
    
    with torch.no_grad():
        states = env.reset()
        hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
        social_memories = [model.init_social_memory().to(device) for _ in range(env.n_agents)]
        
        # 将状态转换为张量
        states_tensor = [torch.FloatTensor(state).unsqueeze(0).to(device) for state in states]
        
        # 前向传播
        _, _, _, comm_messages = model.forward(
            states_tensor, hidden_states, social_memories, communication=True
        )
        
        # 创建通信分析目录
        comm_dir = f"{save_dir}/communication_ep{episode}"
        os.makedirs(comm_dir, exist_ok=True)
        
        # 分析通信内容
        for i in range(env.n_agents):
            message = comm_messages[i].cpu().numpy()
            
            # 分离两种模态
            action_intent = message[0, :config['comm_dim']//2]  # 离散部分
            env_info = message[0, config['comm_dim']//2:]  # 连续部分
            
            with open(f"{comm_dir}/agent{i}_message.txt", "w") as f:
                f.write(f"智能体 {i} 通信内容:\n")
                f.write(f"  行动意图 (离散): {action_intent}\n")
                f.write(f"  环境信息 (连续): {env_info}\n")
            
            # 可视化通信内容
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.bar(range(len(action_intent)), action_intent)
            plt.title(f'智能体 {i} 行动意图')
            plt.xlabel('通信位')
            plt.ylabel('值')
            plt.ylim(-0.1, 1.1)  # 适合二进制值
            
            plt.subplot(1, 2, 2)
            plt.bar(range(len(env_info)), env_info)
            plt.title(f'智能体 {i} 环境信息')
            plt.xlabel('通信位')
            plt.ylabel('值')
            plt.ylim(-1.1, 1.1)  # 适合Tanh范围
            
            plt.tight_layout()
            plt.savefig(f"{comm_dir}/agent{i}_message.png")
            plt.close()
        
        # 模拟多步交互分析通信变化
        plt.figure(figsize=(15, 10))
        all_messages = []
        
        for step in range(min(10, config['max_steps'])):
            # 前向传播
            actions, hidden_states, social_memories, comm_messages = model.forward(
                states_tensor, hidden_states, social_memories, communication=True
            )
            all_messages.append(comm_messages.cpu().numpy())
            
            # 选择行动
            actions = [torch.argmax(actions_logits[0]).item() for actions_logits in actions]
            
            # 执行行动
            next_states, rewards, done, _ = env.step(actions)
            states = next_states
            states_tensor = [torch.FloatTensor(state).unsqueeze(0).to(device) for state in states]
        
        # 可视化通信演化
        all_messages = np.array(all_messages)  # [step, agent, batch, comm_dim]
        
        for i in range(env.n_agents):
            plt.subplot(env.n_agents, 1, i+1)
            # 提取该智能体的所有消息
            agent_messages = all_messages[:, i, 0, :]  # [step, comm_dim]
            plt.imshow(agent_messages, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f"智能体 {i} 通信随时间变化")
            plt.xlabel('通信维度')
            plt.ylabel('时间步')
        
        plt.tight_layout()
        plt.savefig(f"{comm_dir}/communication_evolution.png")
        plt.close()
    
    model.train()  # 恢复为训练模式

# 可视化训练进展
def visualize_training(rewards, rewards_no_comm, losses, losses_no_comm, success_rates, success_rates_no_comm, save_dir, episode):
    """可视化训练进展"""
    # 创建可视化目录
    vis_dir = f"{save_dir}/visualization"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 绘制奖励曲线
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards, label='有通信', color='blue')
    plt.plot(rewards_no_comm, label='无通信', color='red', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('平均奖励')
    plt.title('训练奖励')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制平滑后的奖励曲线 (使用移动平均)
    window_size = min(20, len(rewards))
    if window_size > 0:
        smooth_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        smooth_rewards_no_comm = np.convolve(rewards_no_comm, np.ones(window_size)/window_size, mode='valid')
        
        plt.subplot(2, 2, 2)
        plt.plot(smooth_rewards, label='有通信 (平滑)', color='blue')
        plt.plot(smooth_rewards_no_comm, label='无通信 (平滑)', color='red', linestyle='--')
        plt.xlabel('Episode')
        plt.ylabel('平滑平均奖励')
        plt.title(f'平滑训练奖励 (窗口={window_size})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制损失曲线
    plt.subplot(2, 2, 3)
    plt.plot(losses, label='有通信', color='blue')
    plt.plot(losses_no_comm, label='无通信', color='red', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('损失')
    plt.title('训练损失')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制成功率曲线
    plt.subplot(2, 2, 4)
    plt.plot(success_rates, label='有通信', color='blue')
    plt.plot(success_rates_no_comm, label='无通信', color='red', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('成功率')
    plt.title('任务成功率')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/training_curves_ep{episode}.png")
    plt.close()
    
    # 保存训练数据
    np.savez(
        f"{vis_dir}/training_data_ep{episode}.npz",
        rewards=rewards,
        rewards_no_comm=rewards_no_comm,
        losses=losses,
        losses_no_comm=losses_no_comm,
        success_rates=success_rates,
        success_rates_no_comm=success_rates_no_comm
    )
    
    # 绘制通信收益
    if len(rewards) > 0:
        comm_gain = [r - r_nc for r, r_nc in zip(rewards, rewards_no_comm)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(comm_gain, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Episode')
        plt.ylabel('通信收益')
        plt.title('通信带来的收益 (有通信奖励 - 无通信奖励)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加移动平均线
        if len(comm_gain) >= window_size:
            smooth_gain = np.convolve(comm_gain, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smooth_gain, color='blue', label=f'移动平均 (窗口={window_size})')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/communication_gain_ep{episode}.png")
        plt.close() 