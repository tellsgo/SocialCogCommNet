import torch
import numpy as np
from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """
    算法基类，定义通用接口
    所有具体算法应该继承这个类并实现必要的方法
    """
    def __init__(self, model, optimizer, device, config=None):
        """
        初始化算法
        
        参数:
            model: 模型实例
            optimizer: 优化器实例
            device: 计算设备
            config: 配置参数
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config or {}
        
        # 设置默认配置
        self._set_default_config()
        
        # 初始化算法特定组件
        self._init_components()
    
    def _set_default_config(self):
        """
        设置默认配置参数
        子类可以覆盖此方法以设置特定的默认值
        """
        defaults = {
            'gamma': 0.99,            # 折扣因子
            'learning_rate': 1e-4,    # 学习率
            'grad_clip': 1.0,         # 梯度裁剪
            'batch_size': 64,         # 批量大小
            'use_communication': True, # 是否使用通信
            'epsilon': 0.1,           # 探索率
            'epsilon_decay': 0.995,   # 探索率衰减
            'min_epsilon': 0.01,      # 最小探索率
        }
        
        # 更新配置，保留用户提供的值
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    @abstractmethod
    def _init_components(self):
        """
        初始化算法特定组件
        必须由子类实现
        """
        pass
    
    @abstractmethod
    def select_actions(self, observations, hidden_states, social_memories):
        """
        选择动作
        
        参数:
            observations: 智能体的观察，列表，每个元素是一个智能体的观察张量
            hidden_states: 隐藏状态，列表
            social_memories: 社会记忆，列表
            
        返回:
            actions: 动作列表
        """
        pass
    
    @abstractmethod
    def update(self, batch):
        """
        更新模型参数
        
        参数:
            batch: 经验批次
            
        返回:
            loss_info: 损失信息字典
        """
        pass
    
    @abstractmethod
    def store_experience(self, state, action, reward, next_state, done, info=None):
        """
        存储经验
        
        参数:
            state: 状态
            action: 动作
            reward: 奖励（列表，每个元素对应一个智能体的奖励）
            next_state: 下一个状态
            done: 是否结束
            info: 额外信息
        """
        pass
    
    def decay_exploration(self):
        """
        衰减探索率
        """
        if 'epsilon' in self.config:
            self.config['epsilon'] = max(
                self.config['min_epsilon'],
                self.config['epsilon'] * self.config['epsilon_decay']
            )
    
    def get_exploration_rate(self):
        """
        获取当前探索率
        
        返回:
            epsilon: 探索率
        """
        return self.config.get('epsilon', 0.0)
    
    def save(self, path):
        """
        保存模型和算法状态
        
        参数:
            path: 保存路径
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config,
            'algorithm_state': self.get_algorithm_state()
        }
        torch.save(save_dict, path)
    
    def load(self, path):
        """
        加载模型和算法状态
        
        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config.update(checkpoint.get('config', {}))
        self.set_algorithm_state(checkpoint.get('algorithm_state', {}))
    
    @abstractmethod
    def get_algorithm_state(self):
        """
        获取算法特定状态（用于保存）
        
        返回:
            state_dict: 状态字典
        """
        pass
    
    @abstractmethod
    def set_algorithm_state(self, state_dict):
        """
        设置算法特定状态（用于加载）
        
        参数:
            state_dict: 状态字典
        """
        pass
    
    def set_communication_enabled(self, enabled):
        """
        设置是否启用通信
        
        参数:
            enabled: 布尔值，是否启用通信
        """
        self.config['use_communication'] = enabled 