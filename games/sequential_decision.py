import numpy as np
from .base_game import BaseGame

class SequentialDecisionGame(BaseGame):
    """
    序列决策游戏：两个智能体需要在多个连续的步骤中做出决策。
    
    游戏规则:
    1. 游戏持续多个步骤 (默认3步)
    2. 每个智能体在每一步选择动作 (0, 1, 2)
    3. 环境状态会随着智能体的动作而变化
    4. 每步的奖励取决于两个智能体的动作组合和当前环境状态
    5. 智能体需要考虑长期收益，而不仅仅是即时奖励
    """
    def __init__(self, max_steps=3):
        """初始化序列决策游戏"""
        super().__init__()
        self.name = "SequentialDecisionGame"
        self.max_steps = max_steps
        self.current_step = 0
        self.state_dim = 5  # 环境状态(1) + 当前步骤(1) + 双方之前的行动(2*1=2) + 单步奖励(1)
        self.n_agents = 2
        self.n_actions = 3
        
        # 定义不同环境状态下的最佳动作组合
        self.optimal_actions = {
            0: {  # 环境状态0
                0: [0, 2],  # 步骤0的最佳动作
                1: [1, 1],  # 步骤1的最佳动作
                2: [2, 0]   # 步骤2的最佳动作
            },
            1: {  # 环境状态1
                0: [2, 0],
                1: [0, 2],
                2: [1, 1]
            },
            2: {  # 环境状态2
                0: [1, 1],
                1: [2, 0],
                2: [0, 2]
            }
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
        # 重置步骤计数器
        self.current_step = 0
        
        if scenario is not None:
            # 使用指定场景
            self.state = scenario.copy()
        else:
            # 随机选择初始环境状态 (0, 1 或 2)
            self.state = np.zeros(self.state_dim)
            self.state[0] = np.random.randint(0, 3)  # 环境状态
            self.state[1] = self.current_step  # 当前步骤
            # 其余部分为0，表示没有之前的动作
        
        # 每个智能体观察到完整的状态
        observations = [self.state.copy() for _ in range(self.n_agents)]
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
        # 记录动作
        self.state[2:4] = actions
        
        # 检查是否选择了最佳动作组合
        optimal_actions = self.optimal_actions[int(self.state[0])][self.current_step]
        success = False
        
        if actions[0] == optimal_actions[0] and actions[1] == optimal_actions[1]:
            rewards = [1.0, 1.0]
            success = True
        else:
            rewards = [-0.5, -0.5]
        
        # 记录单步奖励
        self.state[4] = rewards[0]  # 假设两个智能体获得相同的奖励
        
        # 更新环境状态
        if self.current_step < self.max_steps - 1:
            # 根据智能体的动作更新环境状态
            action_sum = np.sum(actions)
            if action_sum >= 3:
                self.state[0] = (self.state[0] + 1) % 3
            elif action_sum <= 1:
                self.state[0] = (self.state[0] - 1) % 3
            # 如果和为2，环境状态不变
        
        # 检查游戏是否结束
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # 更新当前步骤
        self.state[1] = self.current_step
        
        # 每个智能体观察到完整的状态
        observations = [self.state.copy() for _ in range(self.n_agents)]
        
        return observations, rewards, done, {"success": success}
    
    def get_test_scenarios(self):
        """
        获取用于测试的场景
        
        返回:
            scenarios: 测试场景列表
        """
        scenarios = []
        
        # 为每个可能的初始环境状态创建场景
        for env_state in range(3):
            # 创建初始状态 [环境状态, 当前步骤, 上一步动作1, 上一步动作2, 奖励]
            scenario = np.zeros(self.state_dim)
            scenario[0] = env_state  # 环境状态
            scenario[1] = 0  # 当前步骤
            # 其他位置（上一步动作和奖励）都是0
            
            scenarios.append(scenario)
        
        return scenarios 