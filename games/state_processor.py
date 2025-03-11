import numpy as np

class StateProcessor:
    """
    状态预处理器 - 为课程学习提供统一的状态表示
    
    将不同游戏的状态映射到统一的状态表示空间，保证不同游戏之间的状态表示一致性，
    从而支持更有效的知识迁移。
    """
    
    # 统一状态维度 - 所有游戏的状态都会被处理成这个维度
    UNIFIED_STATE_DIM = 8
    
    @staticmethod
    def process_state(game_name, raw_state):
        """
        将原始状态转换为统一维度的状态表示
        
        参数:
            game_name: 游戏名称
            raw_state: 原始状态
            
        返回:
            统一维度的状态表示
        """
        # 初始化统一状态表示为全零向量
        unified_state = np.zeros(StateProcessor.UNIFIED_STATE_DIM)
        
        if game_name == "SimpleCoordinationGame":
            # SimpleCoordinationGame: 环境状态(4维)
            # 统一表示: [环境状态(4维), 0, 0, 0, 0]
            unified_state[:4] = raw_state
            
        elif game_name == "AsymmetricInfoGame":
            # AsymmetricInfoGame: 环境状态(3维)
            # 统一表示: [环境状态(3维), 0, 0, 0, 0, 0]
            unified_state[:3] = raw_state
            
        elif game_name == "SequentialDecisionGame":
            # SequentialDecisionGame: 环境状态(1) + 当前步骤(1) + 上一步动作(2) + 单步奖励(1)
            # 统一表示: [环境状态(1), 0, 0, 0, 当前步骤(1), 上一步动作(2), 单步奖励(1), 0]
            unified_state[0] = raw_state[0]  # 环境状态
            unified_state[4] = raw_state[1]  # 当前步骤
            unified_state[5:7] = raw_state[2:4]  # 上一步动作
            unified_state[7] = raw_state[4]  # 单步奖励
            
        elif game_name == "PartialObservableGame":
            # PartialObservableGame: 环境状态(6维)
            # 统一表示: [环境状态(6维), 0, 0]
            unified_state[:6] = raw_state
            
        else:
            # 未知游戏，直接复制原始状态并截断或填充
            copy_len = min(len(raw_state), StateProcessor.UNIFIED_STATE_DIM)
            unified_state[:copy_len] = raw_state[:copy_len]
        
        return unified_state
    
    @staticmethod
    def process_observations(game_name, observations):
        """
        处理所有智能体的观察
        
        参数:
            game_name: 游戏名称
            observations: 原始观察列表 [agent1_obs, agent2_obs, ...]
            
        返回:
            处理后的观察列表
        """
        processed_observations = []
        for obs in observations:
            processed_observations.append(StateProcessor.process_state(game_name, obs))
        return processed_observations 