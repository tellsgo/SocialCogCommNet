import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class TheoryOfMindModule(nn.Module):
    """
    理论心智模块 - 尝试预测其他智能体的心理状态和行动
    """
    def __init__(self, input_dim, hidden_dim, n_agents, n_actions):
        super(TheoryOfMindModule, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        print(f"TheoryOfMindModule初始化: input_dim={input_dim}, hidden_dim={hidden_dim}")
        
        # 预测其他智能体的行动倾向
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions * (n_agents - 1))
        )
        
        # 预测其他智能体的内部状态
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * (n_agents - 1))
        )
    
    def forward(self, x, social_memory):
        """
        预测其他智能体的心理状态和行动
        
        参数:
            x: 当前智能体的隐藏状态 [batch_size, hidden_dim]
            social_memory: 社会记忆 [batch_size, hidden_dim]
            
        返回:
            predicted_actions: 预测的其他智能体行动概率
            predicted_states: 预测的其他智能体内部状态
        """
        print(f"TheoryOfMindModule forward: x.shape={x.shape}, social_memory.shape={social_memory.shape}")
        
        # 结合隐藏状态和社会记忆
        combined = torch.cat([x, social_memory], dim=1)
        print(f"Combined shape: {combined.shape}")
        
        # 预测其他智能体的行动倾向
        action_logits = self.action_predictor(combined)
        print(f"Action logits shape: {action_logits.shape}")
        predicted_actions = action_logits.view(-1, self.n_agents - 1, self.n_actions)
        print(f"Predicted actions shape: {predicted_actions.shape}")
        predicted_actions = F.softmax(predicted_actions, dim=2)
        
        # 预测其他智能体的内部状态
        predicted_states = self.state_predictor(combined)
        print(f"Raw predicted states shape: {predicted_states.shape}")
        predicted_states = predicted_states.view(-1, self.n_agents - 1, self.hidden_dim)
        print(f"Final predicted states shape: {predicted_states.shape}")
        
        return predicted_actions, predicted_states


class CommunicationStrategyModule(nn.Module):
    """
    通信策略模块 - 基于社会认知生成通信内容
    """
    def __init__(self, input_dim, hidden_dim, comm_dim, n_agents):
        super(CommunicationStrategyModule, self).__init__()
        self.n_agents = n_agents
        self.comm_dim = comm_dim
        
        # 多模态通信生成器
        # 模态1: 行动意图通信 (离散)
        self.action_intent_comm = nn.Sequential(
            nn.Linear(input_dim + hidden_dim * (n_agents - 1) + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, comm_dim // 2)
        )
        
        # 模态2: 环境信息通信 (连续)
        self.env_info_comm = nn.Sequential(
            nn.Linear(input_dim + hidden_dim * (n_agents - 1) + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, comm_dim // 2),
            nn.Tanh()  # 使用Tanh确保输出在[-1,1]范围内
        )
    
    def forward(self, x, predicted_states, social_memory):
        """
        生成通信内容
        
        参数:
            x: 当前智能体的隐藏状态 [batch_size, hidden_dim]
            predicted_states: 预测的其他智能体内部状态 [batch_size, n_agents-1, hidden_dim]
            social_memory: 社会记忆 [batch_size, memory_dim]
            
        返回:
            message: 生成的通信内容 [batch_size, comm_dim]
        """
        # 将预测状态展平
        flat_states = predicted_states.reshape(predicted_states.size(0), -1)
        
        # 结合输入、预测状态和社会记忆
        combined = torch.cat([x, flat_states, social_memory], dim=1)
        
        # 生成两种模态的通信内容
        action_intent = self.action_intent_comm(combined)
        env_info = self.env_info_comm(combined)
        
        # 合并两种模态
        message = torch.cat([action_intent, env_info], dim=1)
        
        return message


class SocialRelationshipTracker(nn.Module):
    """
    社会关系追踪器 - 记录与其他智能体的互动历史
    """
    def __init__(self, hidden_dim, comm_dim, memory_dim, n_agents):
        super(SocialRelationshipTracker, self).__init__()
        self.n_agents = n_agents
        
        # 更新社会记忆
        self.memory_updater = nn.GRUCell(
            input_size=(n_agents - 1) * (comm_dim + hidden_dim),
            hidden_size=memory_dim
        )
    
    def forward(self, social_memory, sent_messages, received_messages, others_states):
        """
        更新社会记忆
        
        参数:
            social_memory: 当前社会记忆 [batch_size, memory_dim]
            sent_messages: 发送的消息 [batch_size, comm_dim]
            received_messages: 接收的消息 [batch_size, (n_agents-1)*comm_dim]
            others_states: 其他智能体的状态 [batch_size, (n_agents-1)*hidden_dim]
            
        返回:
            new_memory: 更新后的社会记忆 [batch_size, memory_dim]
        """
        # 结合接收的消息和其他智能体状态
        combined = torch.cat([received_messages, others_states], dim=1)
        
        # 更新社会记忆
        new_memory = self.memory_updater(combined, social_memory)
        
        return new_memory


class SocialCognitiveCommNet(nn.Module):
    """
    社会认知通信网络 - 结合社会认知与通信学习
    """
    def __init__(self, input_dim, hidden_dim, comm_dim, memory_dim, n_agents, n_actions):
        super(SocialCognitiveCommNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.comm_dim = comm_dim
        self.memory_dim = memory_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        # 编码网络：将观察编码为隐藏状态
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 通信：产生通信消息
        self.comm_gen = nn.Sequential(
            nn.Linear(hidden_dim, comm_dim),
            nn.Tanh()  # 限制通信值的范围
        )
        
        # 通信处理：处理接收到的通信
        self.comm_processor = nn.Sequential(
            nn.Linear(comm_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 社会记忆：追踪其他智能体的行为模式
        self.memory_update = nn.GRUCell(hidden_dim, memory_dim)
        
        # 决策网络：根据隐藏状态和社会记忆做出决策
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # 价值网络：状态价值估计
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, hidden_states, social_memories, communication=True):
        # 检查obs列表是否为空
        if not obs or len(obs) == 0:
            raise ValueError("观察列表不能为空")
        
        # 确保obs列表长度足够
        if len(obs) < self.n_agents:
            # 如果观察列表长度不足，复制第一个观察填充
            obs = obs + [obs[0]] * (self.n_agents - len(obs))
            
        # 确保hidden_states列表长度足够
        if len(hidden_states) < self.n_agents:
            # 如果隐藏状态列表长度不足，复制第一个隐藏状态填充
            hidden_states = hidden_states + [hidden_states[0]] * (self.n_agents - len(hidden_states))
        
        # 确保social_memories列表长度足够
        if len(social_memories) < self.n_agents:
            # 如果社交记忆列表长度不足，复制第一个社交记忆填充
            social_memories = social_memories + [social_memories[0]] * (self.n_agents - len(social_memories))
            
        batch_size = obs[0].size(0)
        
        # 编码观察
        encoded_states = []
        for i in range(self.n_agents):
            encoded = self.encoder(obs[i])
            encoded_states.append(encoded)
        
        # 通信阶段
        if communication:
            # 生成通信消息
            messages = [self.comm_gen(state) for state in encoded_states]
            
            # 处理接收到的消息
            processed_states = []
            for i in range(self.n_agents):
                # 排除自己的消息
                others_messages = []
                for j in range(self.n_agents):
                    if i != j:
                        others_messages.append(messages[j])
                
                if others_messages:
                    # 聚合其他智能体的消息
                    avg_message = torch.stack(others_messages).mean(dim=0)
                    processed_message = self.comm_processor(avg_message)
                    # 组合原始状态和处理后的消息
                    combined_state = encoded_states[i] + processed_message
                else:
                    combined_state = encoded_states[i]
                
                processed_states.append(combined_state)
        else:
            # 没有通信
            processed_states = encoded_states
        
        # 更新社会记忆
        new_social_memories = []
        for i in range(self.n_agents):
            new_memory = self.memory_update(processed_states[i], social_memories[i])
            new_social_memories.append(new_memory)
        
        # 动作和价值估计
        action_values = []
        state_values = []
        for i in range(self.n_agents):
            # 确保张量维度正确
            proc_state = processed_states[i]
            new_memory = new_social_memories[i]
            
            # 如果张量是一维的，添加批次维度
            if proc_state.dim() == 1:
                proc_state = proc_state.unsqueeze(0)
            if new_memory.dim() == 1:
                new_memory = new_memory.unsqueeze(0)
                
            # 连接处理后的状态和新的社交记忆
            combined = torch.cat([proc_state, new_memory], dim=1)
            action_value = self.action_head(combined)
            state_value = self.value_head(combined)
            action_values.append(action_value)
            state_values.append(state_value)
        
        return action_values, state_values, new_social_memories
    
    def init_hidden(self, batch_size=1):
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_dim)
    
    def init_social_memory(self, batch_size=1):
        """初始化社会记忆"""
        return torch.zeros(batch_size, self.memory_dim)
    
    def select_actions(self, obs, hidden_states, social_memories, epsilon=0.0, communication=True):
        """选择动作（用于测试或部署）"""
        self.eval()  # 评估模式
        
        with torch.no_grad():
            action_values, _, new_social_memories = self.forward(obs, hidden_states, social_memories, communication)
            
            # ε-贪婪策略
            actions = []
            for i in range(self.n_agents):
                if random.random() < epsilon:
                    # 随机动作
                    action = random.randint(0, self.n_actions - 1)
                else:
                    # 最佳动作
                    action = action_values[i].max(1)[1].item()
                actions.append(action)
        
        self.train()  # 恢复训练模式
        return actions
    
    def get_q_values(self, obs, hidden_states, social_memories, communication=True):
        """获取Q值（用于训练）"""
        action_values, state_values, new_social_memories = self.forward(obs, hidden_states, social_memories, communication)
        return action_values, state_values, new_social_memories
    
    def get_communication(self, obs, hidden_states):
        """获取通信内容（用于分析）"""
        batch_size = obs[0].size(0)
        
        # 编码观察
        encoded_states = []
        for i in range(self.n_agents):
            encoded = self.encoder(obs[i])
            encoded_states.append(encoded)
        
        # 生成通信消息
        messages = [self.comm_gen(state) for state in encoded_states]
        return messages
    
    def analyze_communication(self, observations, device=None):
        """分析通信内容"""
        if device is None:
            device = next(self.parameters()).device
        
        # 将观察转换为张量
        obs_tensor = []
        for i in range(self.n_agents):
            if isinstance(observations[i], np.ndarray):
                obs_tensor.append(torch.FloatTensor(observations[i]).unsqueeze(0).to(device))
            else:
                obs_tensor.append(observations[i].unsqueeze(0).to(device))
        
        # 初始化隐藏状态
        hidden_states = [self.init_hidden().to(device) for _ in range(self.n_agents)]
        
        # 获取通信消息
        with torch.no_grad():
            messages = self.get_communication(obs_tensor, hidden_states)
        
        # 分析消息
        message_data = []
        for i in range(self.n_agents):
            message_data.append(messages[i].detach().cpu().numpy())
        
        return message_data
        
    def transfer_learning(self, new_state_dim=None, new_action_dim=None):
        """
        为课程学习实现知识迁移，在状态维度或动作维度变化时保留尽可能多的已学习知识
        
        参数:
            new_state_dim: 新游戏的状态维度 (实际上不再使用，我们统一使用8维状态)
            new_action_dim: 新游戏的动作维度
            
        返回:
            更新后的模型
        """
        device = next(self.parameters()).device
        
        # 使用统一的状态维度 - 我们不再需要更改输入维度
        from games.state_processor import StateProcessor
        fixed_state_dim = StateProcessor.UNIFIED_STATE_DIM
        
        # 仅当模型的输入维度与统一维度不一致时才更新
        if self.input_dim != fixed_state_dim:
            print(f"迁移学习: 状态维度从 {self.input_dim} 变为 {fixed_state_dim} (统一维度)")
            
            # 保存旧模型参数
            old_encoder_state = self.encoder.state_dict()
            
            # 更新输入维度
            self.input_dim = fixed_state_dim
            
            # 重新创建编码器
            self.encoder = nn.Sequential(
                nn.Linear(fixed_state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()
            )
            
            # 尝试恢复第二层权重（如果维度相同）
            try:
                second_layer = [m for m in self.encoder.modules() if isinstance(m, nn.Linear)][1]
                second_layer_old = [m for m in old_encoder_state.keys() if '2.weight' in m or '2.bias' in m]
                
                if second_layer_old:
                    for key in second_layer_old:
                        if key.endswith('weight'):
                            second_layer.weight.data.copy_(old_encoder_state[key])
                        elif key.endswith('bias'):
                            second_layer.bias.data.copy_(old_encoder_state[key])
                
                print("  成功迁移编码器第二层权重")
            except Exception as e:
                print(f"  警告: 无法迁移编码器权重 - {e}")
        
        # 处理动作维度变化
        if new_action_dim is not None and new_action_dim != self.n_actions:
            print(f"迁移学习: 动作维度从 {self.n_actions} 变为 {new_action_dim}")
            
            # 保存决策网络的隐藏层权重
            if isinstance(self.action_head, nn.Sequential) and len(self.action_head) >= 2:
                old_hidden_layer = None
                for i, layer in enumerate(self.action_head):
                    if i == 0 and isinstance(layer, nn.Linear):
                        old_hidden_layer = layer.state_dict()
                        break
            
            # 更新动作维度
            old_action_dim = self.n_actions
            self.n_actions = new_action_dim
            
            # 重新创建决策网络
            self.action_head = nn.Sequential(
                nn.Linear(self.hidden_dim + self.memory_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, new_action_dim)
            )
            
            # 恢复隐藏层权重
            if old_hidden_layer is not None:
                self.action_head[0].load_state_dict(old_hidden_layer)
                print("  成功迁移决策网络隐藏层权重")
        
        # 通信模块和社会记忆模块的参数保持不变
        # 因为它们的维度不受状态和动作维度变化的影响
        
        # 将模型移回原设备
        self.to(device)
        
        return self 