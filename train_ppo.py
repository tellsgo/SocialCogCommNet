import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns
from games.simple_coordination import SimpleCoordinationGame
from games.asymmetric_info import AsymmetricInfoGame
from games.sequential_decision import SequentialDecisionGame
from games.partial_observable import PartialObservableGame
from environment import CooperativeEnvironment
from model import SocialCognitiveCommNet
from games.state_processor import StateProcessor

# 用于PPO算法的记忆缓冲区
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.hidden_states = []
        self.social_memories = []
        self.batch_size = batch_size
    
    def store(self, state, action, log_prob, value, reward, done, hidden_state, social_memory):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.hidden_states.append(hidden_state)
        self.social_memories.append(social_memory)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.hidden_states = []
        self.social_memories = []
    
    def get_batches(self):
        batch_start = 0
        batch_size = self.batch_size
        n_samples = len(self.states)
        
        # 创建数据索引列表并打乱
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        batches = []
        while batch_start < n_samples:
            batch_indices = indices[batch_start:batch_start+batch_size]
            
            state_batch = [self.states[i] for i in batch_indices]
            action_batch = [self.actions[i] for i in batch_indices]
            log_prob_batch = [self.log_probs[i] for i in batch_indices]
            value_batch = [self.values[i] for i in batch_indices]
            reward_batch = [self.rewards[i] for i in batch_indices]
            done_batch = [self.dones[i] for i in batch_indices]
            hidden_state_batch = [self.hidden_states[i] for i in batch_indices]
            social_memory_batch = [self.social_memories[i] for i in batch_indices]
            
            batches.append((
                state_batch, action_batch, log_prob_batch, value_batch, 
                reward_batch, done_batch, hidden_state_batch, social_memory_batch
            ))
            
            batch_start += batch_size
        
        return batches

# PPO Agent
class PPOAgent:
    def __init__(self, model, optimizer, memory, gamma=0.99, gae_lambda=0.95, 
                 policy_clip=0.2, value_coef=0.5, entropy_coef=0.01, n_epochs=4, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.memory = memory
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.device = device
    
    def choose_actions(self, obs, hidden_states, social_memories, epsilon=0.0, communication=True):
        with torch.no_grad():
            # 前向传播获取动作和值
            action_probs, state_values, new_social_memories = self.model.forward(
                obs, hidden_states, social_memories, communication
            )
            
            # 选择动作
            actions = []
            log_probs = []
            for i in range(len(action_probs)):
                # 应用epsilon-贪婪策略
                if np.random.random() < epsilon:
                    action = np.random.randint(0, action_probs[i].size(1))
                    # 创建分类分布并计算对数概率
                    dist = Categorical(F.softmax(action_probs[i], dim=1))
                    log_prob = dist.log_prob(torch.tensor([action], device=self.device))
                else:
                    # 创建分类分布并采样动作
                    dist = Categorical(F.softmax(action_probs[i], dim=1))
                    action = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor([action], device=self.device))
                
                actions.append(action)
                log_probs.append(log_prob)
            
            return actions, log_probs, state_values, new_social_memories
    
    def learn(self):
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        entropy_loss_total = 0
        n_batches = 0
        
        # 遍历指定的轮数
        for _ in range(self.n_epochs):
            # 获取数据批次
            batches = self.memory.get_batches()
            n_batches += len(batches)
            
            # 处理每个批次
            for states, actions, old_log_probs, old_values, rewards, dones, hidden_states, social_memories in batches:
                n_agents = len(states[0])
                batch_size = len(states)
                
                # 计算每个智能体的优势
                advantages = []
                returns = []
                
                for agent_idx in range(n_agents):
                    # 提取当前智能体的值和奖励
                    values = torch.tensor([v[agent_idx].item() for v in old_values], device=self.device)
                    rewards_agent = torch.tensor([r for r in rewards], device=self.device)
                    dones_agent = torch.tensor([d for d in dones], device=self.device)
                    
                    # 计算GAE (Generalized Advantage Estimation)
                    gae = 0
                    adv_batch = []
                    returns_batch = []
                    # 逆序计算，从最后一个时间步开始
                    for t in reversed(range(batch_size)):
                        # 处理最后一个时间步或结束状态
                        if t == batch_size - 1 or dones_agent[t]:
                            delta = rewards_agent[t] - values[t]
                            gae = delta
                        else:
                            # 标准GAE公式：delta = r + gamma * V(s') - V(s)
                            delta = rewards_agent[t] + self.gamma * values[t+1] - values[t]
                            # GAE累积：GAE = delta + gamma * lambda * GAE'
                            gae = delta + self.gamma * self.gae_lambda * gae
                        
                        # 累积优势
                        adv_batch.insert(0, gae)
                        # 计算G值（折扣回报）
                        returns_batch.insert(0, gae + values[t])
                    
                    advantages.append(torch.tensor(adv_batch, device=self.device))
                    returns.append(torch.tensor(returns_batch, device=self.device))
                
                # 标准化优势
                for i in range(n_agents):
                    if len(advantages[i]) > 1:  # 确保有足够的数据来标准化
                        advantages[i] = (advantages[i] - advantages[i].mean()) / (advantages[i].std() + 1e-10)
                
                # 前向传播和计算损失
                policy_loss = 0
                value_loss = 0
                entropy_loss = 0
                
                for t in range(batch_size):
                    # 获取当前状态的新预测
                    obs_tensor = []
                    for i in range(n_agents):
                        obs_tensor.append(states[t][i])
                    
                    h_states = [hidden_states[t][i] for i in range(n_agents)]
                    s_memories = [social_memories[t][i] for i in range(n_agents)]
                    
                    action_probs, new_values, _ = self.model.forward(obs_tensor, h_states, s_memories)
                    
                    # 计算每个智能体的损失
                    for i in range(n_agents):
                        # 创建分布
                        dist = Categorical(F.softmax(action_probs[i], dim=1))
                        
                        # 获取新预测的对数概率
                        # 检查actions的类型和结构
                        if isinstance(actions[t], int):
                            # 如果actions[t]是整数，说明只有一个智能体的动作
                            action = torch.tensor([actions[t]], device=self.device)
                        else:
                            # 否则，假设actions[t]是一个列表或类似的可索引对象
                            action = torch.tensor([actions[t][i]], device=self.device)
                        new_log_prob = dist.log_prob(action)
                        
                        # 计算旧的和新的概率比率
                        old_log_prob = old_log_probs[t][i]
                        ratio = torch.exp(new_log_prob - old_log_prob)
                        
                        # PPO裁剪
                        adv = advantages[i][t]
                        surrogate1 = ratio * adv
                        surrogate2 = torch.clamp(ratio, 1-self.policy_clip, 1+self.policy_clip) * adv
                        
                        # 策略损失 (使用PPO的裁剪目标)
                        p_loss = -torch.min(surrogate1, surrogate2)
                        
                        # 值函数损失
                        # 确保维度匹配
                        target_value = returns[i][t].unsqueeze(0)
                        if new_values[i].dim() > 1 and target_value.dim() == 1:
                            target_value = target_value.unsqueeze(0)
                        elif new_values[i].dim() == 1 and target_value.dim() > 1:
                            new_values[i] = new_values[i].unsqueeze(0)
                        
                        # 确保形状完全匹配
                        if new_values[i].shape != target_value.shape:
                            target_value = target_value.view(new_values[i].shape)
                            
                        v_loss = F.mse_loss(new_values[i], target_value)
                        
                        # 熵损失 (鼓励探索)
                        e_loss = -dist.entropy()
                        
                        # 累加每个智能体的损失
                        policy_loss += p_loss
                        value_loss += v_loss
                        entropy_loss += e_loss
                
                # 平均每个批次中的所有智能体和时间步的损失
                policy_loss /= (batch_size * n_agents)
                value_loss /= (batch_size * n_agents)
                entropy_loss /= (batch_size * n_agents)
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 优化器步骤
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                # 累计总损失
                total_loss += loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_loss_total += entropy_loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / n_batches
        avg_policy_loss = policy_loss_total / n_batches
        avg_value_loss = value_loss_total / n_batches
        avg_entropy_loss = entropy_loss_total / n_batches
        
        # 清空记忆
        self.memory.clear()
        
        return avg_loss, avg_policy_loss, avg_value_loss, avg_entropy_loss

def train_model_ppo(game_name, epochs=200, hidden_dim=128, comm_dim=64, memory_dim=64, learning_rate=0.0003,
                   batch_size=32, n_epochs=4, gamma=0.99, gae_lambda=0.95, policy_clip=0.2, 
                   value_coef=0.5, entropy_coef=0.01, episodes_per_epoch=10):
    """
    使用PPO算法训练社会认知通信网络
    
    参数:
        game_name: 游戏名称，如'SimpleCoordinationGame'
        epochs: 训练轮数
        hidden_dim: 隐藏层维度
        comm_dim: 通信维度
        memory_dim: 社会记忆维度
        learning_rate: 学习率
        batch_size: 批量大小
        n_epochs: PPO更新的轮数
        gamma: 折扣因子
        gae_lambda: GAE lambda参数
        policy_clip: PPO裁剪参数
        value_coef: 值函数损失系数
        entropy_coef: 熵损失系数
        episodes_per_epoch: 每个训练轮次的回合数
        
    返回:
        训练好的模型和游戏环境
    """
    print("="*60)
    print(f"PPO训练 - {game_name}")
    print("="*60)
    
    # 创建保存目录
    results_dir = f"./results/ppo_{game_name}"
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建PPO记忆
    memory = PPOMemory(batch_size)
    
    # 创建PPO智能体
    agent = PPOAgent(
        model=model,
        optimizer=optimizer,
        memory=memory,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_clip=policy_clip,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        n_epochs=n_epochs,
        device=device
    )
    
    # 记录训练数据
    rewards_history = []
    success_rates = []
    loss_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_loss_history = []
    
    # 训练循环
    print("\n开始PPO训练...")
    for epoch in range(epochs):
        epoch_rewards = []
        epoch_successes = []
        
        # 每个回合训练多个场景
        for episode in range(episodes_per_epoch):
            episode_reward = 0
            episode_success = False
            
            # 重置环境
            obs = env.reset()
            done = False
            step = 0
            
            # 初始化隐藏状态和社会记忆
            hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
            social_memories = [model.init_social_memory().to(device) for _ in range(env.n_agents)]
            
            # 转换观察为张量
            obs_tensor = []
            for i in range(env.n_agents):
                obs_tensor.append(torch.FloatTensor(obs[i]).unsqueeze(0).to(device))
            
            # 游戏循环
            while not done and step < 5:  # 限制最大步数
                # 选择动作
                epsilon = max(0.05, 0.5 - epoch / epochs)  # 逐渐减小随机性
                actions, log_probs, state_values, new_social_memories = agent.choose_actions(
                    obs_tensor, hidden_states, social_memories, epsilon
                )
                
                # 执行动作
                next_obs, rewards, done, info = env.step(actions)
                
                # 转换下一个观察为张量
                next_obs_tensor = []
                for i in range(env.n_agents):
                    next_obs_tensor.append(torch.FloatTensor(next_obs[i]).unsqueeze(0).to(device))
                
                # 记录经验
                for i in range(env.n_agents):
                    agent.memory.store(
                        state=obs_tensor[i],
                        action=actions[i],
                        log_prob=log_probs[i],
                        value=state_values[i],
                        reward=float(rewards[i]),
                        done=done,
                        hidden_state=hidden_states[i],
                        social_memory=social_memories[i]
                    )
                
                # 更新状态
                obs = next_obs
                obs_tensor = next_obs_tensor
                hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
                social_memories = new_social_memories
                
                # 累积回报
                episode_reward += sum(rewards) / env.n_agents
                
                # 记录成功
                if info.get("success", False):
                    episode_success = True
                
                step += 1
            
            # 记录每个回合的结果
            epoch_rewards.append(episode_reward)
            epoch_successes.append(1 if episode_success else 0)
        
        # 学习
        if len(agent.memory.states) > 0:
            avg_loss, avg_policy_loss, avg_value_loss, avg_entropy_loss = agent.learn()
            loss_history.append(avg_loss)
            policy_loss_history.append(avg_policy_loss)
            value_loss_history.append(avg_value_loss)
            entropy_loss_history.append(avg_entropy_loss)
        
        # 记录本回合的平均表现
        rewards_history.append(np.mean(epoch_rewards))
        success_rates.append(np.mean(epoch_successes))
        
        # 报告训练进度
        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_reward = np.mean(rewards_history[-10:]) if epoch >= 10 else np.mean(rewards_history)
            avg_success = np.mean(success_rates[-10:]) if epoch >= 10 else np.mean(success_rates)
            print(f"回合 {epoch}/{epochs}:")
            print(f"  奖励: {avg_reward:.4f}, 成功率: {avg_success:.2f}")
            if len(loss_history) > 0:
                avg_loss = np.mean(loss_history[-10:]) if epoch >= 10 else np.mean(loss_history)
                print(f"  损失: {avg_loss:.6f}")
    
    # 保存模型
    torch.save(model.state_dict(), f"{results_dir}/model.pt")
    print(f"\n模型已保存到 {results_dir}/model.pt")
    
    # 绘制学习曲线
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history)
    plt.title('平均奖励')
    plt.xlabel('训练回合')
    plt.ylabel('奖励')
    
    plt.subplot(2, 2, 2)
    plt.plot(success_rates)
    plt.title('成功率')
    plt.xlabel('训练回合')
    plt.ylabel('成功率')
    
    if len(loss_history) > 0:
        plt.subplot(2, 2, 3)
        plt.plot(loss_history, label='总损失')
        plt.plot(policy_loss_history, label='策略损失')
        plt.plot(value_loss_history, label='值函数损失')
        plt.title('训练损失')
        plt.xlabel('训练回合')
        plt.ylabel('损失')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(entropy_loss_history)
        plt.title('熵损失')
        plt.xlabel('训练回合')
        plt.ylabel('熵')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/learning_curve.png")
    
    print(f"学习曲线已保存到 {results_dir}/learning_curve.png")
    
    return model, env, device

def visualize_communication(model, env, device, n_scenarios=5, results_dir=None):
    """
    可视化智能体之间的通信内容
    
    参数:
        model: 训练好的模型
        env: 环境
        device: 计算设备
        n_scenarios: 要测试的场景数量
        results_dir: 结果保存目录
    """
    game = env.current_game
    game_name = game.name
    
    if results_dir is None:
        results_dir = f"./results/ppo_{game_name}/communication"
    
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("通信可视化")
    print("="*60)
    
    # 获取测试场景
    test_scenarios = game.get_test_scenarios()
    
    if not test_scenarios:
        print("没有可用的测试场景")
        return
    
    # 为每个场景创建通信可视化
    for i, scenario in enumerate(test_scenarios[:n_scenarios]):
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
        
        # 获取通信消息
        with torch.no_grad():
            comm_messages = model.get_communication(obs_tensor, hidden_states)
        
        # 测试两种情况：有通信和无通信
        # 有通信
        actions_with_comm = model.select_actions(obs_tensor, hidden_states, social_memories, epsilon=0.0, communication=True)
        
        # 无通信
        actions_no_comm = model.select_actions(obs_tensor, hidden_states, social_memories, epsilon=0.0, communication=False)
        
        # 创建通信矩阵可视化
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"场景 {i+1} - 通信可视化", fontsize=16)
        
        # 为每个智能体创建热力图
        for agent_idx, message in enumerate(comm_messages):
            message_np = message.cpu().detach().numpy().reshape(-1)
            
            # 使用热力图显示整个通信向量
            plt.subplot(game.n_agents, 2, agent_idx*2 + 1)
            sns.heatmap(
                message_np.reshape(1, -1), 
                cmap='viridis', 
                annot=False, 
                cbar=True,
                xticklabels=5,  # 每5个维度显示一个标签
                yticklabels=False
            )
            plt.title(f"智能体 {agent_idx+1} 通信向量")
            plt.xlabel("通信维度")
            
            # 找出最活跃的通信维度
            active_dims = np.argsort(np.abs(message_np))[-10:]  # 取前10个最活跃维度
            active_values = message_np[active_dims]
            
            # 使用条形图显示最活跃的维度
            plt.subplot(game.n_agents, 2, agent_idx*2 + 2)
            plt.bar(
                np.arange(len(active_dims)), 
                active_values, 
                color=[plt.cm.viridis(abs(v) / max(abs(active_values))) for v in active_values]
            )
            plt.xticks(np.arange(len(active_dims)), active_dims)
            plt.title(f"智能体 {agent_idx+1} 最活跃的通信维度")
            plt.xlabel("维度索引")
            plt.ylabel("通信值")
        
        # 添加动作和奖励信息
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为底部的文本留出空间
        
        # 在图表底部添加文本信息
        plt.figtext(0.5, 0.01, 
                   f"有通信: 动作={actions_with_comm}, 无通信: 动作={actions_no_comm}", 
                   ha="center", fontsize=10, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        # 保存图表
        plt.savefig(f"{results_dir}/comm_viz_scenario_{i+1}.png")
        print(f"场景 {i+1} 的通信可视化已保存到 {results_dir}/comm_viz_scenario_{i+1}.png")
        plt.close()
    
    # 创建通信重要性分析
    analyze_communication_importance(model, env, device, results_dir)

def analyze_communication_importance(model, env, device, results_dir=None):
    """分析通信对于任务完成的重要性"""
    game = env.current_game
    game_name = game.name
    
    if results_dir is None:
        results_dir = f"./results/ppo_{game_name}"
    
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
    labels = ['有通信', '无通信']
    rewards = [avg_reward_with_comm, avg_reward_without_comm]
    success_rates = [success_rate_with_comm, success_rate_without_comm]
    
    plt.figure(figsize=(12, 10))
    
    # 奖励对比
    plt.subplot(2, 1, 1)
    bars = plt.bar(labels, rewards, color=['green', 'red'])
    plt.ylabel('平均奖励')
    plt.title('通信对奖励的影响')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    # 成功率对比
    plt.subplot(2, 1, 2)
    bars = plt.bar(labels, success_rates, color=['green', 'red'])
    plt.ylabel('成功率')
    plt.title('通信对成功率的影响')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/communication_importance.png")
    
    print(f"\n通信重要性分析已保存到 {results_dir}/communication_importance.png")

if __name__ == "__main__":
    # 训练简单协调游戏
    print("\n" + "="*60)
    print("开始训练简单协调游戏")
    print("="*60)
    model, env, device = train_model_ppo(
        game_name="SimpleCoordinationGame",
        epochs=200,
        hidden_dim=128,
        comm_dim=64,
        memory_dim=64,
        learning_rate=0.0003,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        episodes_per_epoch=10
    )
    
    # 可视化通信
    visualize_communication(model, env, device, n_scenarios=3) 