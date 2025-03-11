import numpy as np
from .base_game import BaseGame

class PartialObservableGame(BaseGame):
    """
    部分可观察游戏：环境有很多特征，但每个智能体只能观察到一部分特征。
    这是最复杂的一个游戏，需要智能体通过高效的通信来共享信息。
    
    游戏规则:
    1. 环境有一个6维的状态
    2. 每个智能体只能观察其中的3个维度
    3. 智能体需要基于环境状态做出最佳的协调动作
    4. 全局最优解需要共享信息和协调
    """
    def __init__(self):
        """初始化部分可观察游戏"""
        super().__init__()
        self.name = "PartialObservableGame"
        self.state_dim = 6  # 环境状态维度
        self.n_agents = 2
        self.n_actions = 3
        
        # 定义观察掩码，指定每个智能体可以观察到哪些维度
        self.observation_masks = [
            np.array([1, 1, 1, 0, 0, 0]),  # 智能体1的掩码
            np.array([0, 0, 0, 1, 1, 1])   # 智能体2的掩码
        ]
        
        # 定义决策规则：根据环境状态的各种模式确定最佳动作
        # 这些规则既考虑单个智能体可观察的特征，也考虑需要通信才能获取的信息
        self.action_rules = [
            # 规则1：如果特征0和特征3的和大于1，智能体1应该选择0，智能体2应该选择2
            # 这条规则需要通信，因为两个智能体无法独立观察到所有特征
            lambda s: (0, 2) if s[0] + s[3] > 1 else None,
            
            # 规则2：如果特征1和特征4的和小于0.5，两个智能体都应该选择1
            lambda s: (1, 1) if s[1] + s[4] < 0.5 else None,
            
            # 规则3：如果特征2和特征5的差的绝对值大于0.7，智能体1应该选择2，智能体2应该选择0
            lambda s: (2, 0) if abs(s[2] - s[5]) > 0.7 else None,
            
            # 规则4：默认规则，如果前面的规则都不适用
            lambda s: (np.random.randint(3), np.random.randint(3))
        ]
        
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
            # 随机生成环境状态
            self.state = np.random.rand(self.state_dim)
        
        # 为每个智能体生成部分观察
        observations = self.get_masked_observations(self.state)
        return observations
    
    def step(self, actions):
        """
        执行动作并返回新的观察、奖励和是否结束
        
        参数:
            actions: 智能体的动作，形如 [agent1_action, agent2_action]
            
        返回:
            observations: 下一步观察
            rewards: 每个智能体的奖励
            done: 游戏是否结束
            info: 额外信息
        """
        # 获取最优动作
        optimal_actions = self.get_optimal_actions(self.state)
        success = False
        
        # 计算奖励
        if actions[0] == optimal_actions[0] and actions[1] == optimal_actions[1]:
            rewards = [1.0, 1.0]
            success = True
        else:
            rewards = [-0.5, -0.5]
        
        # 生成部分观察
        observations = self.get_masked_observations(self.state)
        
        # 游戏只有一步
        done = True
        
        return observations, rewards, done, {"success": success}
    
    def get_optimal_actions(self, state):
        """根据环境状态确定最佳动作组合"""
        for rule in self.action_rules:
            result = rule(state)
            if result is not None:
                return result
        
        # 默认情况不应该到达这里，因为最后一个规则总是返回值
        return (0, 0)
    
    def get_test_scenarios(self):
        """
        获取用于测试的场景
        
        返回:
            scenarios: 测试场景列表
        """
        scenarios = []
        
        # 规则1测试：前三维大于阈值，后三维小于阈值
        scenario1 = np.zeros(self.state_dim)
        scenario1[:3] = 0.8  # 前三维 > 0.5
        scenario1[3:] = 0.2  # 后三维 < 0.5
        scenarios.append(scenario1)
        
        # 规则2测试：前三维小于阈值，后三维大于阈值
        scenario2 = np.zeros(self.state_dim)
        scenario2[:3] = 0.2  # 前三维 < 0.5
        scenario2[3:] = 0.8  # 后三维 > 0.5
        scenarios.append(scenario2)
        
        # 规则3测试：前三维和后三维所有维度都 > 0.7
        scenario3 = np.zeros(self.state_dim)
        scenario3[:] = 0.8  # 所有维度 > 0.7
        scenarios.append(scenario3)
        
        # 规则4测试：前三维和后三维所有维度都 < 0.3
        scenario4 = np.zeros(self.state_dim)
        scenario4[:] = 0.2  # 所有维度 < 0.3
        scenarios.append(scenario4)
        
        # 规则5测试：所有维度之和 > 4
        scenario5 = np.zeros(self.state_dim)
        scenario5[:] = 0.7  # 所有维度之和 = 4.2
        scenarios.append(scenario5)
        
        return scenarios
    
    def get_masked_observations(self, state):
        """为一个状态生成掩码后的观察"""
        observations = []
        for mask in self.observation_masks:
            obs = state * mask
            observations.append(obs)
        return observations 