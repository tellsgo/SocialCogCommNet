import torch
import os
import json
from .base_algorithm import BaseAlgorithm
from .q_learning import QLearningAlgorithm
from .ppo import PPOAlgorithm

class AlgorithmManager:
    """
    算法管理器，负责管理多种强化学习算法，便于比较和修改
    """
    def __init__(self, config=None):
        """
        初始化算法管理器
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        self.algorithms = {}
        self._register_algorithms()
    
    def _register_algorithms(self):
        """注册所有可用的算法"""
        self.algorithms = {
            "q_learning": QLearningAlgorithm,
            "ppo": PPOAlgorithm,
            # 未来可以添加更多算法
            # "a2c": A2CAlgorithm,
            # "maddpg": MADDPGAlgorithm,
        }
    
    def get_algorithm(self, name, model, optimizer, device, config=None):
        """
        获取指定名称的算法实例
        
        参数:
            name: 算法名称
            model: 模型实例
            optimizer: 优化器实例
            device: 计算设备
            config: 算法特定配置
            
        返回:
            algorithm: 算法实例
        """
        if name not in self.algorithms:
            raise ValueError(f"未知的算法: {name}，可用算法: {list(self.algorithms.keys())}")
        
        # 合并全局配置和算法特定配置
        merged_config = self.config.copy()
        if config:
            merged_config.update(config)
        
        # 创建算法实例
        return self.algorithms[name](model, optimizer, device, merged_config)
    
    def list_algorithms(self):
        """
        列出所有可用的算法
        
        返回:
            algorithm_list: 算法名称列表
        """
        return list(self.algorithms.keys())
    
    def create_experiment(self, algo_name, model, optimizer, device, config=None, save_dir="./results"):
        """
        创建实验
        
        参数:
            algo_name: 算法名称
            model: 模型实例
            optimizer: 优化器实例
            device: 计算设备
            config: 算法特定配置
            save_dir: 保存目录
            
        返回:
            experiment: 实验对象
        """
        # 获取算法实例
        algorithm = self.get_algorithm(algo_name, model, optimizer, device, config)
        
        # 创建实验
        return Experiment(algorithm, save_dir)
    
    def save_config(self, path):
        """
        保存全局配置
        
        参数:
            path: 保存路径
        """
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def load_config(self, path):
        """
        加载全局配置
        
        参数:
            path: 加载路径
        """
        with open(path, 'r') as f:
            self.config = json.load(f)


class Experiment:
    """实验类，用于训练和评估算法"""
    def __init__(self, algorithm, save_dir="./results"):
        """
        初始化实验
        
        参数:
            algorithm: 算法实例
            save_dir: 保存目录
        """
        self.algorithm = algorithm
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练记录
        self.rewards_history = []
        self.success_rates = []
        self.loss_history = []
        self.episode_count = 0
    
    def train_episode(self, env, n_episodes=1, max_steps=20, update_frequency=1):
        """
        训练指定回合数
        
        参数:
            env: 环境实例
            n_episodes: 回合数
            max_steps: 每回合最大步数
            update_frequency: 更新模型的频率
            
        返回:
            results: 训练结果字典
        """
        total_reward = 0
        success_count = 0
        
        for _ in range(n_episodes):
            # 重置环境
            obs = env.reset()
            
            # 初始化隐藏状态和社会记忆
            device = self.algorithm.device
            n_agents = env.n_agents
            hidden_states = [self.algorithm.model.init_hidden().to(device) for _ in range(n_agents)]
            social_memories = [self.algorithm.model.init_social_memory().to(device) for _ in range(n_agents)]
            
            # 转换观察为张量
            obs_tensor = []
            for i in range(n_agents):
                obs_tensor.append(torch.FloatTensor(obs[i]).unsqueeze(0).to(device))
            
            # 回合奖励
            episode_reward = 0
            episode_success = False
            
            # 回合步数
            for step in range(max_steps):
                # 根据算法类型选择不同的动作选择方法
                if isinstance(self.algorithm, PPOAlgorithm):
                    # PPO需要额外获取对数概率和值
                    actions, log_probs, values, new_social_memories = self.algorithm.select_actions(
                        obs_tensor, hidden_states, social_memories
                    )
                else:
                    # 其他算法（如Q-learning）
                    actions = self.algorithm.select_actions(
                        obs_tensor, hidden_states, social_memories
                    )
                    log_probs = None
                    values = None
                    new_social_memories = None
                
                # 执行动作
                next_obs, rewards, done, info = env.step(actions)
                
                # 转换下一个观察为张量
                next_obs_tensor = []
                for i in range(n_agents):
                    next_obs_tensor.append(torch.FloatTensor(next_obs[i]).unsqueeze(0).to(device))
                
                # 更新社会记忆
                if new_social_memories is not None:
                    social_memories = new_social_memories
                
                # 存储经验
                if isinstance(self.algorithm, PPOAlgorithm):
                    # PPO特定的存储方式
                    state = (obs_tensor, hidden_states, social_memories, log_probs, values)
                    next_state = (next_obs_tensor, hidden_states, social_memories, None, None)
                else:
                    # 其他算法
                    state = (obs_tensor, hidden_states, social_memories)
                    next_state = (next_obs_tensor, hidden_states, social_memories)
                
                self.algorithm.store_experience(state, actions, rewards, next_state, done, info)
                
                # 更新状态
                obs = next_obs
                obs_tensor = next_obs_tensor
                
                # 累积奖励
                episode_reward += sum(rewards) / n_agents
                
                # 检查是否成功
                if info.get("success", False):
                    episode_success = True
                
                # 检查是否结束
                if done:
                    break
            
            # 记录回合结果
            total_reward += episode_reward
            if episode_success:
                success_count += 1
            
            # 回合计数
            self.episode_count += 1
            
            # 更新模型
            if self.episode_count % update_frequency == 0:
                loss_info = self.algorithm.update()
                self.loss_history.append(loss_info)
            
            # 衰减探索率
            self.algorithm.decay_exploration()
        
        # 计算平均奖励和成功率
        avg_reward = total_reward / n_episodes
        success_rate = success_count / n_episodes
        
        # 记录
        self.rewards_history.append(avg_reward)
        self.success_rates.append(success_rate)
        
        return {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "episode_count": self.episode_count,
            "exploration_rate": self.algorithm.get_exploration_rate()
        }
    
    def evaluate(self, env, n_episodes=10, max_steps=20, use_communication=True):
        """
        评估算法
        
        参数:
            env: 环境实例
            n_episodes: 评估回合数
            max_steps: 每回合最大步数
            use_communication: 是否使用通信
            
        返回:
            results: 评估结果字典
        """
        # 设置通信状态
        old_comm_state = self.algorithm.config['use_communication']
        self.algorithm.set_communication_enabled(use_communication)
        
        # 记录奖励和成功率
        rewards = []
        success_count = 0
        
        # 评估模式
        self.algorithm.model.eval()
        
        with torch.no_grad():
            for _ in range(n_episodes):
                # 重置环境
                obs = env.reset()
                
                # 初始化隐藏状态和社会记忆
                device = self.algorithm.device
                n_agents = env.n_agents
                hidden_states = [self.algorithm.model.init_hidden().to(device) for _ in range(n_agents)]
                social_memories = [self.algorithm.model.init_social_memory().to(device) for _ in range(n_agents)]
                
                # 转换观察为张量
                obs_tensor = []
                for i in range(n_agents):
                    obs_tensor.append(torch.FloatTensor(obs[i]).unsqueeze(0).to(device))
                
                # 回合奖励
                episode_reward = 0
                
                # 回合步数
                for step in range(max_steps):
                    # 选择动作 (评估时不需要记录log_prob和value)
                    if isinstance(self.algorithm, PPOAlgorithm):
                        actions, _, _, new_social_memories = self.algorithm.select_actions(
                            obs_tensor, hidden_states, social_memories
                        )
                    else:
                        actions = self.algorithm.select_actions(
                            obs_tensor, hidden_states, social_memories
                        )
                        new_social_memories = None
                    
                    # 执行动作 (epsilon=0, 纯贪婪)
                    next_obs, rewards_step, done, info = env.step(actions)
                    
                    # 转换下一个观察为张量
                    next_obs_tensor = []
                    for i in range(n_agents):
                        next_obs_tensor.append(torch.FloatTensor(next_obs[i]).unsqueeze(0).to(device))
                    
                    # 更新社会记忆
                    if new_social_memories is not None:
                        social_memories = new_social_memories
                    
                    # 更新状态
                    obs = next_obs
                    obs_tensor = next_obs_tensor
                    
                    # 累积奖励
                    episode_reward += sum(rewards_step) / n_agents
                    
                    # 检查是否结束
                    if done:
                        break
                
                # 记录回合结果
                rewards.append(episode_reward)
                if info.get("success", False):
                    success_count += 1
        
        # 恢复通信状态
        self.algorithm.set_communication_enabled(old_comm_state)
        
        # 恢复训练模式
        self.algorithm.model.train()
        
        # 计算平均奖励和成功率
        avg_reward = sum(rewards) / n_episodes
        success_rate = success_count / n_episodes
        
        return {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "use_communication": use_communication
        }
    
    def compare_communication(self, env, n_episodes=20, max_steps=20):
        """
        比较有无通信的表现差异
        
        参数:
            env: 环境实例
            n_episodes: 评估回合数
            max_steps: 每回合最大步数
            
        返回:
            results: 比较结果字典
        """
        # 评估有通信的表现
        with_comm_results = self.evaluate(env, n_episodes, max_steps, use_communication=True)
        
        # 评估无通信的表现
        without_comm_results = self.evaluate(env, n_episodes, max_steps, use_communication=False)
        
        # 计算差异
        reward_diff = with_comm_results["avg_reward"] - without_comm_results["avg_reward"]
        success_diff = with_comm_results["success_rate"] - without_comm_results["success_rate"]
        
        return {
            "with_comm": with_comm_results,
            "without_comm": without_comm_results,
            "reward_diff": reward_diff,
            "success_diff": success_diff
        }
    
    def save(self, name=None):
        """
        保存模型和训练状态
        
        参数:
            name: 保存文件名，如果为None，则使用episode_count
        """
        if name is None:
            name = f"model_{self.episode_count}"
        
        model_path = os.path.join(self.save_dir, f"{name}.pt")
        self.algorithm.save(model_path)
        
        # 保存训练历史
        history_path = os.path.join(self.save_dir, f"{name}_history.json")
        history = {
            "rewards": self.rewards_history,
            "success_rates": self.success_rates,
            "episode_count": self.episode_count
        }
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        return model_path, history_path
    
    def load(self, name=None):
        """
        加载模型和训练状态
        
        参数:
            name: 加载文件名，如果为None，则使用最新的checkpoint
        """
        if name is None:
            # 查找最新的checkpoint
            checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith('.pt')]
            if not checkpoints:
                raise FileNotFoundError(f"在 {self.save_dir} 中没有找到checkpoint")
            
            # 按修改时间排序
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x)), reverse=True)
            name = checkpoints[0].replace('.pt', '')
        
        model_path = os.path.join(self.save_dir, f"{name}.pt")
        self.algorithm.load(model_path)
        
        # 加载训练历史
        history_path = os.path.join(self.save_dir, f"{name}_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                
                self.rewards_history = history.get("rewards", [])
                self.success_rates = history.get("success_rates", [])
                self.episode_count = history.get("episode_count", 0)
        
        return model_path 