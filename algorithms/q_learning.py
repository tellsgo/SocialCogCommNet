import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .base_algorithm import BaseAlgorithm

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, hidden_state=None, social_memory=None):
        """
        存储经验
        
        参数：
            state: 状态 (列表，每个元素是一个智能体的观察)
            action: 动作 (列表，每个元素是一个智能体的动作)
            reward: 奖励 (列表，每个元素是一个智能体的奖励)
            next_state: 下一个状态 (列表)
            done: 是否结束 (布尔值)
            hidden_state: 隐藏状态 (可选)
            social_memory: 社会记忆 (可选)
        """
        experience = (state, action, reward, next_state, done, hidden_state, social_memory)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """采样批次数据"""
        if len(self.buffer) < batch_size:
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(self.buffer, batch_size)
        
        # 解构样本
        state_batch = [e[0] for e in batch]
        action_batch = [e[1] for e in batch]
        reward_batch = [e[2] for e in batch]
        next_state_batch = [e[3] for e in batch]
        done_batch = [e[4] for e in batch]
        hidden_state_batch = [e[5] for e in batch]
        social_memory_batch = [e[6] for e in batch]
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, hidden_state_batch, social_memory_batch
    
    def __len__(self):
        return len(self.buffer)


class QLearningAlgorithm(BaseAlgorithm):
    """Q学习算法实现"""
    
    def _set_default_config(self):
        """设置默认配置"""
        super()._set_default_config()
        
        # Q学习特定配置
        q_defaults = {
            'buffer_capacity': 10000,  # 经验缓冲区容量
            'target_update': 10,       # 目标网络更新频率
            'tau': 0.005,              # 软更新系数
            'use_double_q': True,      # 是否使用Double Q-learning
            'use_per': False,          # 是否使用优先经验回放
            'use_dueling': False,      # 是否使用Dueling结构
        }
        
        # 更新配置
        for key, value in q_defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _init_components(self):
        """初始化Q学习特定组件"""
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(self.config['buffer_capacity'])
        
        # 创建目标网络（如果使用）
        if self.config.get('target_update', 0) > 0:
            self.target_model = type(self.model)(
                input_dim=self.model.input_dim,
                hidden_dim=self.model.hidden_dim,
                comm_dim=self.model.comm_dim,
                memory_dim=self.model.memory_dim,
                n_agents=self.model.n_agents,
                n_actions=self.model.n_actions
            ).to(self.device)
            
            # 初始化目标网络参数
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()  # 目标网络仅用于评估
        else:
            self.target_model = None
        
        # 训练步数计数器
        self.train_steps = 0
    
    def select_actions(self, observations, hidden_states, social_memories):
        """
        选择动作（使用ε-贪婪策略）
        
        参数:
            observations: 观察列表，每个元素是一个智能体的观察张量
            hidden_states: 隐藏状态列表
            social_memories: 社会记忆列表
            
        返回:
            actions: 动作列表
        """
        epsilon = self.config['epsilon']
        n_agents = len(observations)
        actions = []
        
        # 是否使用通信
        use_comm = self.config['use_communication']
        
        with torch.no_grad():
            # 获取Q值
            q_values, _, new_social_memories = self.model.forward(
                observations, 
                hidden_states, 
                social_memories,
                communication=use_comm
            )
            
            # 对每个智能体使用ε-贪婪策略选择动作
            for i in range(n_agents):
                if random.random() < epsilon:
                    # 随机探索
                    action = random.randint(0, self.model.n_actions - 1)
                else:
                    # 贪婪动作
                    action = q_values[i].max(1)[1].item()
                
                actions.append(action)
        
        return actions
    
    def store_experience(self, state, action, reward, next_state, done, info=None):
        """
        存储经验到回放缓冲区
        
        参数:
            state: 状态 (包含observations, hidden_states, social_memories)
            action: 动作列表
            reward: 奖励列表，每个元素对应一个智能体的奖励
            next_state: 下一个状态
            done: 是否结束
            info: 额外信息
        """
        observations, hidden_states, social_memories = state
        next_observations, next_hidden_states, next_social_memories = next_state
        
        # 将张量转换为CPU上的numpy数组
        cpu_observations = [obs.cpu() for obs in observations]
        cpu_next_observations = [obs.cpu() for obs in next_observations]
        cpu_hidden_states = [h.cpu() for h in hidden_states] if hidden_states else None
        cpu_social_memories = [s.cpu() for s in social_memories] if social_memories else None
        
        # 存储经验
        self.replay_buffer.push(
            cpu_observations, 
            action, 
            reward, 
            cpu_next_observations, 
            done,
            cpu_hidden_states,
            cpu_social_memories
        )
    
    def update(self, batch=None):
        """
        更新模型参数
        
        参数:
            batch: 经验批次（如果为None，则从缓冲区采样）
            
        返回:
            loss_info: 包含损失信息的字典
        """
        # 检查回放缓冲区是否有足够样本
        if len(self.replay_buffer) < self.config['batch_size']:
            return {'q_loss': 0.0}
        
        # 从回放缓冲区采样批次
        if batch is None:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, hidden_batch, memory_batch = \
                self.replay_buffer.sample(self.config['batch_size'])
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, hidden_batch, memory_batch = batch
        
        # 转换为张量
        device = self.device
        gamma = self.config['gamma']
        n_agents = self.model.n_agents
        use_comm = self.config['use_communication']
        
        # 计算Q值
        q_loss = 0.0
        
        for agent_idx in range(n_agents):
            # 处理当前智能体的数据
            obs_batch = torch.stack([torch.FloatTensor(s[agent_idx]).to(device) for s in state_batch])
            next_obs_batch = torch.stack([torch.FloatTensor(s[agent_idx]).to(device) for s in next_state_batch])
            
            # 初始化隐藏状态和社会记忆
            if hidden_batch[0] is not None:
                hidden_states = torch.stack([h[agent_idx].to(device) for h in hidden_batch])
                social_memories = torch.stack([m[agent_idx].to(device) for m in memory_batch])
            else:
                hidden_states = self.model.init_hidden(self.config['batch_size']).to(device)
                social_memories = self.model.init_social_memory(self.config['batch_size']).to(device)
            
            # 将其他智能体的观察打包为列表
            obs_list = [obs_batch]
            next_obs_list = [next_obs_batch]
            
            for idx in range(n_agents):
                if idx != agent_idx:
                    obs_list.append(torch.stack([torch.FloatTensor(s[idx]).to(device) for s in state_batch]))
                    next_obs_list.append(torch.stack([torch.FloatTensor(s[idx]).to(device) for s in next_state_batch]))
            
            # 当前Q值
            current_q_values, _, _ = self.model.get_q_values(
                obs_list,
                [hidden_states for _ in range(n_agents)],
                [social_memories for _ in range(n_agents)],
                communication=use_comm
            )
            
            # 对当前动作的Q值
            actions = torch.LongTensor([a[agent_idx] for a in action_batch]).unsqueeze(1).to(device)
            current_q = current_q_values[agent_idx].gather(1, actions)
            
            # 目标Q值
            if self.target_model is not None:
                # 使用目标网络
                with torch.no_grad():
                    if self.config['use_double_q']:
                        # Double Q-learning
                        next_q_values, _, _ = self.model.get_q_values(
                            next_obs_list, 
                            [hidden_states for _ in range(n_agents)],
                            [social_memories for _ in range(n_agents)],
                            communication=use_comm
                        )
                        
                        # 选择动作使用在线网络，但值使用目标网络
                        next_actions = next_q_values[agent_idx].max(1)[1].unsqueeze(1)
                        
                        target_q_values, _, _ = self.target_model.get_q_values(
                            next_obs_list, 
                            [hidden_states for _ in range(n_agents)],
                            [social_memories for _ in range(n_agents)],
                            communication=use_comm
                        )
                        
                        next_q = target_q_values[agent_idx].gather(1, next_actions)
                    else:
                        # 标准Q-learning
                        next_q_values, _, _ = self.target_model.get_q_values(
                            next_obs_list, 
                            [hidden_states for _ in range(n_agents)],
                            [social_memories for _ in range(n_agents)],
                            communication=use_comm
                        )
                        
                        next_q = next_q_values[agent_idx].max(1)[0].unsqueeze(1)
            else:
                # 不使用目标网络
                with torch.no_grad():
                    next_q_values, _, _ = self.model.get_q_values(
                        next_obs_list, 
                        [hidden_states for _ in range(n_agents)],
                        [social_memories for _ in range(n_agents)],
                        communication=use_comm
                    )
                    
                    next_q = next_q_values[agent_idx].max(1)[0].unsqueeze(1)
            
            # 计算目标值
            rewards = torch.FloatTensor([r[agent_idx] for r in reward_batch]).unsqueeze(1).to(device)
            dones = torch.FloatTensor([float(d) for d in done_batch]).unsqueeze(1).to(device)
            
            # 贝尔曼方程
            target_q = rewards + gamma * next_q * (1 - dones)
            
            # 计算损失
            agent_loss = F.mse_loss(current_q, target_q)
            q_loss += agent_loss
        
        # 平均每个智能体的损失
        q_loss /= n_agents
        
        # 反向传播
        self.optimizer.zero_grad()
        q_loss.backward()
        
        # 梯度裁剪
        if self.config['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
        
        # 更新参数
        self.optimizer.step()
        
        # 更新目标网络（如果使用）
        self.train_steps += 1
        if self.target_model is not None:
            if self.config.get('tau', 0) > 0:
                # 软更新
                for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - self.config['tau']) + 
                        param.data * self.config['tau']
                    )
            elif self.train_steps % self.config['target_update'] == 0:
                # 硬更新
                self.target_model.load_state_dict(self.model.state_dict())
        
        return {'q_loss': q_loss.item()}
    
    def get_algorithm_state(self):
        """
        获取算法特定状态（用于保存）
        
        返回:
            state_dict: 状态字典
        """
        return {
            'train_steps': self.train_steps,
            'epsilon': self.config['epsilon']
        }
    
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