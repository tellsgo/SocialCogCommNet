import numpy as np
from .base_game import BaseGame

class AsymmetricInfoGame(BaseGame):
    """
    信息不对称游戏：两个智能体观察到不同的信息，需要通过通信来协调行动。
    
    游戏规则:
    1. 两个智能体各自有3种可能的行动 (0, 1, 2)
    2. 每个智能体只能观察到环境的一部分信息
    3. 如果两个智能体选择与环境匹配的正确行动组合，获得+1奖励
    4. 如果选择错误的行动组合，获得-0.5奖励
    """
    def __init__(self):
        """初始化非对称信息游戏"""
        super().__init__()
        self.name = "AsymmetricInfoGame"
        self.state_dim = 3  # 环境状态维度（环境状态 + 两个智能体观察掩码）
        self.n_agents = 2
        self.n_actions = 3
        
        # 定义正确的行动组合
        self.correct_actions = {
            0: [0, 2],  # 当环境状态为0时，第一个智能体应选0，第二个智能体应选2
            1: [1, 1],  # 当环境状态为1时，两个智能体都应选1
            2: [2, 0]   # 当环境状态为2时，第一个智能体应选2，第二个智能体应选0
        }
        
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
            # 随机选择环境状态 (0, 1 或 2)
            self.state = np.zeros(self.state_dim)
            self.state[0] = np.random.randint(0, 3)  # 环境状态为整数0,1,2
        
        # 为每个智能体生成观察，根据掩码隐藏部分信息
        observations = []
        
        # 智能体1观察掩码为[1, 0, 0]，只能观察到第一个维度
        obs1 = np.zeros(self.state_dim)
        obs1[0] = float(self.state[0])  # 确保是标量值
        
        # 智能体2观察掩码为[0, 1, 1]，只能观察到第二、三个维度
        obs2 = np.zeros(self.state_dim)
        obs2[1:] = self.state[1:]  # 看不到环境状态，但可以看到其他信息
        
        observations.append(obs1)
        observations.append(obs2)
        
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
        # 更新状态
        self.state[1:] = actions  # 记录两个智能体的行动
        
        # 检查是否选择了正确的行动组合
        correct_actions = self.correct_actions[int(self.state[0])]
        success = False
        
        if actions[0] == correct_actions[0] and actions[1] == correct_actions[1]:
            rewards = [1.0, 1.0]
            success = True
        else:
            rewards = [-0.5, -0.5]
        
        # 准备每个智能体的观察
        mask1 = np.array([1, 0, 0])  # 智能体1的掩码
        mask2 = np.array([0, 1, 1])  # 智能体2的掩码
        
        obs1 = self.state * mask1
        obs2 = self.state * mask2
        
        observations = [obs1, obs2]
        done = True  # 每个回合只有一步
        
        return observations, rewards, done, {"success": success}
    
    def get_test_scenarios(self):
        """
        获取用于测试的场景
        
        返回:
            scenarios: 测试场景列表
        """
        # 创建不同环境状态的测试场景
        scenarios = []
        
        # 环境状态为0的场景
        scenario0 = np.zeros(self.state_dim)
        scenario0[0] = 0  # 环境状态为0
        scenarios.append(scenario0)
        
        # 环境状态为1的场景
        scenario1 = np.zeros(self.state_dim)
        scenario1[0] = 1  # 环境状态为1
        scenarios.append(scenario1)
        
        # 环境状态为2的场景
        scenario2 = np.zeros(self.state_dim)
        scenario2[0] = 2  # 环境状态为2
        scenarios.append(scenario2)
        
        return scenarios 