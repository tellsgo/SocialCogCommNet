import numpy as np
import torch
from abc import ABC, abstractmethod

class BaseGame(ABC):
    """
    所有游戏的基类，定义了游戏必须实现的接口。
    """
    def __init__(self):
        self.name = "BaseGame"
        self.state_dim = 0
        self.n_agents = 0
        self.n_actions = 0
    
    @abstractmethod
    def reset(self):
        """
        重置游戏状态
        
        返回:
            observation: 初始观察
        """
        pass
    
    @abstractmethod
    def step(self, actions):
        """
        执行动作并转移到下一个状态
        
        参数:
            actions: 所有智能体的动作列表
            
        返回:
            observations: 新的观察
            rewards: 所有智能体的奖励
            done: 游戏是否结束
            info: 额外信息
        """
        pass
    
    @abstractmethod
    def get_test_scenarios(self):
        """
        获取用于测试的场景列表
        
        返回:
            scenarios: 测试场景列表
        """
        pass
    
    def evaluate(self, model, device, n_episodes=10, communication=True):
        """
        评估模型性能
        
        参数:
            model: 要评估的模型
            device: 计算设备
            n_episodes: 评估的回合数
            communication: 是否启用通信
            
        返回:
            avg_reward: 平均奖励
            success_rate: 成功率
        """
        try:
            from games.state_processor import StateProcessor
        except ImportError:
            # 如果在导入过程中发生错误，定义一个简单的代理函数
            class SimpleProcessor:
                @staticmethod
                def process_observations(game_name, observations):
                    return observations
            StateProcessor = SimpleProcessor
        
        total_rewards = []
        success_count = 0
        
        for _ in range(n_episodes):
            # 重置环境
            obs = self.reset()
            done = False
            episode_reward = 0
            
            # 使用状态处理器处理观察
            obs = StateProcessor.process_observations(self.name, obs)
            
            # 初始化隐藏状态和社会记忆
            hidden_states = [model.init_hidden().to(device) for _ in range(self.n_agents)]
            social_memories = [model.init_social_memory().to(device) for _ in range(self.n_agents)]
            
            while not done:
                # 将观察转换为张量
                obs_tensor = [torch.FloatTensor(o).unsqueeze(0).to(device) for o in obs]
                
                # 选择动作
                actions = model.select_actions(
                    obs_tensor, hidden_states, social_memories, epsilon=0.0, communication=communication
                )
                
                # 执行动作
                obs, rewards, done, info = self.step(actions)
                
                # 使用状态处理器处理观察
                obs = StateProcessor.process_observations(self.name, obs)
                
                episode_reward += sum(rewards) / self.n_agents
                
                # 记录成功
                if "success" in info and info["success"]:
                    success_count += 1
                    break
            
            total_rewards.append(episode_reward)
        
        # 计算平均奖励和成功率
        avg_reward = sum(total_rewards) / n_episodes
        success_rate = success_count / n_episodes
        
        return avg_reward, success_rate 