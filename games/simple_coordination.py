import numpy as np
from .base_game import BaseGame

class SimpleCoordinationGame(BaseGame):
    """
    简单协调游戏：两个智能体需要选择互补的行动以获得最大奖励。
    
    游戏规则:
    1. 两个智能体各自有3种可能的行动 (0, 1, 2)
    2. 如果两个智能体选择互补的行动 (和为2)，获得+1奖励
    3. 如果两个智能体选择相同的行动，获得-0.5奖励
    4. 其他情况获得0奖励
    """
    def __init__(self):
        """初始化简单协调游戏"""
        super().__init__()
        self.name = "SimpleCoordinationGame"
        self.state_dim = 4  # 环境状态维度
        self.n_agents = 2
        self.n_actions = 3
        self.reset()
        
    def reset(self, scenario=None):
        """
        重置游戏状态
        
        参数:
            scenario: 可选的场景参数，用于测试特定场景
            
        返回:
            observations: 初始观察
        """
        if scenario is not None:
            # 使用指定场景
            self.state = scenario.copy()
        else:
            # 随机初始化
            self.state = np.zeros(self.state_dim)
            self.state[:2] = np.random.rand(2)  # 随机初始化环境状态的前两个维度
        
        # 每个智能体观察到完整的环境状态
        observations = [self.state.copy() for _ in range(self.n_agents)]
        return observations
    
    def step(self, actions):
        """
        执行动作并返回新的观察、奖励和是否结束
        """
        # 更新环境状态
        self.state[2:] = actions
        
        # 计算奖励
        if actions[0] + actions[1] == 2:  # 互补行动
            rewards = [1.0, 1.0]
            success = True
        elif actions[0] == actions[1]:  # 相同行动
            rewards = [-0.5, -0.5]
            success = False
        else:  # 其他情况
            rewards = [0.0, 0.0]
            success = False
        
        # 游戏每次step后就结束
        done = True
        
        # 准备返回值
        observations = [self.state.copy() for _ in range(self.n_agents)]
        info = {"success": success}
        
        return observations, rewards, done, info
    
    def get_test_scenarios(self):
        """
        获取用于测试的场景
        
        返回:
            scenarios: 测试场景列表
        """
        # 创建一系列测试场景
        scenarios = []
        
        # 场景1: 环境状态为[0.1, 0.2, 0, 0]
        scenario1 = np.zeros(self.state_dim)
        scenario1[0] = 0.1
        scenario1[1] = 0.2
        scenarios.append(scenario1)
        
        # 场景2: 环境状态为[0.5, 0.5, 0, 0]
        scenario2 = np.zeros(self.state_dim)
        scenario2[0] = 0.5
        scenario2[1] = 0.5
        scenarios.append(scenario2)
        
        # 场景3: 环境状态为[0.9, 0.9, 0, 0]
        scenario3 = np.zeros(self.state_dim)
        scenario3[0] = 0.9
        scenario3[1] = 0.9
        scenarios.append(scenario3)
        
        return scenarios 