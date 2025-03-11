import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from .base_algorithm import BaseAlgorithm

class PPOMemory:
    """PPO记忆缓冲区"""
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
    
    def push(self, state, action, log_prob, value, reward, done, hidden_state=None, social_memory=None):
        """
        存储经验
        
        参数:
            state: 状态 (观察张量)
            action: 动作
            log_prob: 动作对数概率
            value: 状态值估计
            reward: 奖励
            done: 是否结束
            hidden_state: 隐藏状态 (可选)
            social_memory: 社会记忆 (可选)
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.hidden_states.append(hidden_state)
        self.social_memories.append(social_memory)
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.hidden_states = []
        self.social_memories = []
    
    def get_batches(self):
        """获取数据批次"""
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
    
    def __len__(self):
        """缓冲区大小"""
        return len(self.states)


class PPOAlgorithm(BaseAlgorithm):
    """PPO算法实现"""
    
    def _set_default_config(self):
        """设置默认配置"""
        super()._set_default_config()
        
        # PPO特定配置
        ppo_defaults = {
            'ppo_epochs': 4,        # PPO更新的轮数
            'gae_lambda': 0.95,     # GAE lambda参数
            'policy_clip': 0.2,     # PPO裁剪参数
            'value_coef': 0.5,      # 值函数损失系数
            'entropy_coef': 0.01,   # 熵损失系数
            'max_grad_norm': 0.5,   # 梯度范数上限
            'use_gae': True,        # 是否使用GAE
            'normalize_advantages': True,  # 是否标准化优势
            'clip_value': True,     # 是否裁剪值函数
            'use_standardized_rewards': False,  # 是否标准化奖励
        }
        
        # 更新配置
        for key, value in ppo_defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _init_components(self):
        """初始化PPO特定组件"""
        # 创建PPO记忆
        self.memory = PPOMemory(self.config['batch_size'])
        
        # 训练步数计数器
        self.train_steps = 0
        
        # 奖励标准化
        if self.config['use_standardized_rewards']:
            self.reward_mean = 0
            self.reward_std = 1
            self.reward_count = 0
    
    def select_actions(self, observations, hidden_states, social_memories):
        """
        选择动作
        
        参数:
            observations: 观察列表，每个元素是一个智能体的观察张量
            hidden_states: 隐藏状态列表
            social_memories: 社会记忆列表
            
        返回:
            actions: 动作列表，每个元素是一个智能体的动作
        """
        # 是否使用通信
        use_comm = self.config['use_communication']
        
        # 随机探索率
        epsilon = self.config['epsilon']
        
        with torch.no_grad():
            # 前向传播
            action_probs, state_values, new_social_memories = self.model.forward(
                observations, 
                hidden_states, 
                social_memories,
                communication=use_comm
            )
            
            n_agents = len(observations)
            actions = []
            log_probs = []
            
            # 为每个智能体选择动作
            for i in range(n_agents):
                # 创建分类分布
                dist = Categorical(F.softmax(action_probs[i], dim=1))
                
                # ε-贪婪探索
                if np.random.random() < epsilon:
                    # 随机动作
                    action = np.random.randint(0, self.model.n_actions)
                    action_tensor = torch.tensor([action], device=self.device)
                    log_prob = dist.log_prob(action_tensor)
                else:
                    # 采样动作
                    action_tensor = dist.sample()
                    action = action_tensor.item()
                    log_prob = dist.log_prob(action_tensor)
                
                actions.append(action)
                log_probs.append(log_prob.item())
        
        return actions, log_probs, state_values, new_social_memories
    
    def store_experience(self, state, action, reward, next_state, done, info=None):
        """
        存储经验到PPO记忆
        
        参数:
            state: 状态 (包含observations, hidden_states, social_memories, log_probs, values)
            action: 动作列表
            reward: 奖励列表，每个元素对应一个智能体的奖励
            next_state: 下一个状态
            done: 是否结束
            info: 额外信息
        """
        observations, hidden_states, social_memories, log_probs, values = state
        
        # 将张量转换为CPU上的numpy数组
        cpu_observations = [obs.cpu().detach() for obs in observations]
        cpu_hidden_states = [h.cpu().detach() for h in hidden_states] if hidden_states else None
        cpu_social_memories = [s.cpu().detach() for s in social_memories] if social_memories else None
        
        # 存储经验 (为每个智能体分别存储)
        n_agents = len(observations)
        for i in range(n_agents):
            # 如果使用标准化奖励，更新统计信息
            if self.config['use_standardized_rewards']:
                self.reward_count += 1
                delta = reward[i] - self.reward_mean
                self.reward_mean += delta / self.reward_count
                delta2 = reward[i] - self.reward_mean
                self.reward_std += delta * delta2
                
                # 计算标准差
                std = np.sqrt(self.reward_std / max(1, self.reward_count - 1))
                std = max(std, 1e-6)  # 避免除以零
                
                # 标准化奖励
                normalized_reward = (reward[i] - self.reward_mean) / std
            else:
                normalized_reward = reward[i]
            
            self.memory.push(
                cpu_observations[i], 
                action[i], 
                log_probs[i] if log_probs else None,
                values[i].item() if values else None,
                normalized_reward, 
                done,
                cpu_hidden_states[i] if cpu_hidden_states else None,
                cpu_social_memories[i] if cpu_social_memories else None
            )
    
    def update(self, batch=None):
        """
        更新模型参数
        
        参数:
            batch: 经验批次（如果为None，则使用内部记忆）
            
        返回:
            loss_info: 包含损失信息的字典
        """
        # 检查记忆是否有足够样本
        if len(self.memory) == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'total_loss': 0.0
            }
        
        # 配置参数
        device = self.device
        gamma = self.config['gamma']
        gae_lambda = self.config['gae_lambda']
        policy_clip = self.config['policy_clip']
        value_coef = self.config['value_coef']
        entropy_coef = self.config['entropy_coef']
        clip_value = self.config['clip_value']
        normalize_advantages = self.config['normalize_advantages']
        
        # 获取所有批次
        batches = self.memory.get_batches()
        
        # 统计损失
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_batches = len(batches)
        
        # 多次更新
        for _ in range(self.config['ppo_epochs']):
            for state_batch, action_batch, old_log_prob_batch, old_value_batch, reward_batch, done_batch, hidden_batch, memory_batch in batches:
                batch_size = len(state_batch)
                
                # 转换为张量
                states = torch.stack([torch.FloatTensor(s).to(device) for s in state_batch])
                actions = torch.tensor(action_batch, dtype=torch.long).to(device)
                old_log_probs = torch.tensor(old_log_prob_batch, dtype=torch.float32).to(device)
                old_values = torch.tensor(old_value_batch, dtype=torch.float32).to(device)
                rewards = torch.tensor(reward_batch, dtype=torch.float32).to(device)
                dones = torch.tensor(done_batch, dtype=torch.bool).to(device)
                
                # 计算优势估计和回报
                if self.config['use_gae']:
                    # 使用GAE计算优势估计
                    advantages = torch.zeros_like(rewards)
                    gae = 0
                    for t in reversed(range(batch_size)):
                        if t == batch_size - 1:
                            next_value = 0
                            next_non_terminal = 1.0 - dones[t].float()
                        else:
                            next_value = old_values[t+1]
                            next_non_terminal = 1.0 - dones[t].float()
                        
                        delta = rewards[t] + gamma * next_value * next_non_terminal - old_values[t]
                        gae = delta + gamma * gae_lambda * next_non_terminal * gae
                        advantages[t] = gae
                    
                    returns = advantages + old_values
                else:
                    # 常规方法
                    returns = torch.zeros_like(rewards)
                    for t in reversed(range(batch_size)):
                        if t == batch_size - 1:
                            returns[t] = rewards[t]
                        else:
                            returns[t] = rewards[t] + gamma * returns[t+1] * (1 - dones[t].float())
                    
                    advantages = returns - old_values
                
                # 标准化优势
                if normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                
                # 初始化隐藏状态和社会记忆
                if hidden_batch[0] is not None:
                    # 确保隐藏状态和社会记忆的维度正确
                    hidden_states = torch.FloatTensor(hidden_batch[0]).to(device)
                    social_memories = torch.FloatTensor(memory_batch[0]).to(device)
                    
                    # 如果是批量数据，需要确保维度正确
                    if hidden_states.dim() > 2:
                        hidden_states = hidden_states.squeeze(0)
                    if social_memories.dim() > 2:
                        social_memories = social_memories.squeeze(0)
                else:
                    # 初始化为正确的维度
                    hidden_states = self.model.init_hidden(1).to(device).squeeze(0)
                    social_memories = self.model.init_social_memory(1).to(device).squeeze(0)
                
                # 重新计算当前策略下的动作概率和值
                use_comm = self.config['use_communication']
                # 创建观察列表
                obs_list = []
                for i in range(self.model.n_agents):
                    # 如果只有一个智能体的数据，则重复使用
                    if i < len(states):
                        obs_list.append(states[i].unsqueeze(0) if states[i].dim() == 1 else states[i])
                    else:
                        obs_list.append(states[0].unsqueeze(0) if states[0].dim() == 1 else states[0])
                
                # 创建隐藏状态和社会记忆列表
                h_states_list = [hidden_states for _ in range(self.model.n_agents)]
                s_memories_list = [social_memories for _ in range(self.model.n_agents)]
                
                # 前向传播
                action_probs, current_values, _ = self.model.forward(
                    obs_list,
                    h_states_list,
                    s_memories_list,
                    communication=use_comm
                )
                
                # 创建分布
                dist = Categorical(F.softmax(action_probs[0], dim=1))
                
                # 计算当前动作的对数概率
                current_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # 计算比率和裁剪
                ratio = torch.exp(current_log_probs - old_log_probs)
                
                # PPO裁剪目标
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - policy_clip, 1.0 + policy_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 值函数裁剪
                if clip_value:
                    current_values_clipped = old_values + torch.clamp(
                        current_values[0] - old_values, -policy_clip, policy_clip
                    )
                    # 确保维度匹配
                    current_values_squeezed = current_values[0].view(-1)
                    current_values_clipped = current_values_clipped.view(-1)
                    returns = returns.view(-1)
                    
                    value_loss = 0.5 * torch.max(
                        F.mse_loss(current_values_squeezed, returns),
                        F.mse_loss(current_values_clipped, returns)
                    )
                else:
                    # 确保维度匹配
                    current_values_squeezed = current_values[0].view(-1)
                    returns = returns.view(-1)
                    value_loss = F.mse_loss(current_values_squeezed, returns)
                
                # 熵损失
                entropy_loss = -entropy
                
                # 总损失
                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if self.config['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['max_grad_norm']
                    )
                
                self.optimizer.step()
                
                # 累积统计信息
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # 计算平均损失
        avg_policy_loss = total_policy_loss / (n_batches * self.config['ppo_epochs'])
        avg_value_loss = total_value_loss / (n_batches * self.config['ppo_epochs'])
        avg_entropy_loss = total_entropy_loss / (n_batches * self.config['ppo_epochs'])
        avg_total_loss = avg_policy_loss + value_coef * avg_value_loss + entropy_coef * avg_entropy_loss
        
        # 更新训练步数
        self.train_steps += 1
        
        # 清空记忆
        self.memory.clear()
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_total_loss
        }
    
    def get_algorithm_state(self):
        """
        获取算法特定状态（用于保存）
        
        返回:
            state_dict: 状态字典
        """
        state_dict = {
            'train_steps': self.train_steps,
            'epsilon': self.config['epsilon']
        }
        
        if self.config['use_standardized_rewards']:
            state_dict.update({
                'reward_mean': self.reward_mean,
                'reward_std': self.reward_std,
                'reward_count': self.reward_count
            })
        
        return state_dict
    
    def set_algorithm_state(self, state_dict):
        """
        设置算法特定状态（用于加载）
        
        参数:
            state_dict: 状态字典
        """
        if 'train_steps' in state_dict:
            self.train_steps = state_dict['train_steps']
        
        if 'epsilon' in state_dict:
            self.config['epsilon'] = state_dict['epsilon']
        
        if self.config['use_standardized_rewards']:
            if 'reward_mean' in state_dict:
                self.reward_mean = state_dict['reward_mean']
            
            if 'reward_std' in state_dict:
                self.reward_std = state_dict['reward_std']
            
            if 'reward_count' in state_dict:
                self.reward_count = state_dict['reward_count'] 