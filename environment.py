import torch
import numpy as np
from games.base_game import BaseGame
from games.simple_coordination import SimpleCoordinationGame
from games.game_manager import GameManager
from games.state_processor import StateProcessor
import os

class CooperativeEnvironment:
    """协作环境，用于训练和评估多智能体协作"""
    
    def __init__(self, game=None, curriculum_learning=False):
        """
        初始化协作环境
        
        参数:
            game: 指定的游戏，默认为SimpleCoordinationGame
            curriculum_learning: 是否启用课程学习
        """
        self.curriculum_learning = curriculum_learning
        
        # 游戏初始化
        if game is not None:
            self.current_game = game
        else:
            self.current_game = SimpleCoordinationGame()
        
        # 获取游戏参数 - 始终使用统一的状态维度
        self.state_dim = StateProcessor.UNIFIED_STATE_DIM
        self.n_agents = self.current_game.n_agents
        self.n_actions = self.current_game.n_actions
        
        # 初始化游戏管理器（用于课程学习）
        if self.curriculum_learning:
            from games.asymmetric_info import AsymmetricInfoGame
            from games.sequential_decision import SequentialDecisionGame
            from games.partial_observable import PartialObservableGame
            
            # 创建游戏列表，按难度排序
            games = [
                SimpleCoordinationGame(),
                AsymmetricInfoGame(),
                SequentialDecisionGame(),
                PartialObservableGame()
            ]
            
            # 创建游戏管理器
            self.game_manager = GameManager(games, save_dir="./results")
            print("游戏管理器已初始化，包含", len(games), "个游戏")
            for i, game in enumerate(games):
                print(f"  游戏 {i+1}: {game.name}")
            
            # 设置当前游戏为第一个游戏
            self.current_game = self.game_manager.get_current_game()
        
        print(f"环境初始化成功：使用游戏 '{self.current_game.name}'")
        print(f"  状态维度: {self.state_dim} (统一状态维度)")
        print(f"  动作数量: {self.n_actions}")
    
    def reset(self, scenario=None):
        """
        重置环境
        
        参数:
            scenario: 可选的场景参数
            
        返回:
            初始观察
        """
        # 重置游戏
        if scenario is not None:
            observations = self.current_game.reset(scenario)
        else:
            observations = self.current_game.reset()
        
        # 处理观察为统一维度
        game_name = self.get_current_game_name()
        return StateProcessor.process_observations(game_name, observations)
    
    def step(self, actions):
        """
        执行一步环境交互
        
        参数:
            actions: 智能体的动作
            
        返回:
            observations: 下一个观察
            rewards: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 执行游戏步骤
        observations, rewards, done, info = self.current_game.step(actions)
        
        # 处理观察为统一维度
        game_name = self.get_current_game_name()
        processed_observations = StateProcessor.process_observations(game_name, observations)
        
        return processed_observations, rewards, done, info
    
    def next_game(self, model=None, optimizer=None):
        """
        切换到下一个难度的游戏
        
        参数:
            model: 当前模型，用于知识迁移
            optimizer: 当前优化器，用于重新初始化
            
        返回:
            success: 是否成功切换到下一个游戏
            model: 更新后的模型（如果提供）
            optimizer: 更新后的优化器（如果提供）
        """
        if not self.curriculum_learning:
            print("警告: 非课程学习模式下调用了next_game")
            return False, model, optimizer
        
        # 获取当前游戏名称
        current_game_name = self.get_current_game_name()
        
        # 记录当前游戏的通信演化
        if model is not None:
            self.log_communication(model, f"before_switch_{current_game_name}")
        
        # 切换到下一个游戏
        success, next_game = self.game_manager.next_game()
        
        if not success:
            print("已经是最后一个游戏，无法切换")
            return False, model, optimizer
        
        # 更新当前游戏
        next_game_name = self.get_current_game_name()
        self.current_game = next_game
        print(f"转移到下一个游戏: '{next_game_name}'")
        
        # 更新环境参数
        self.state_dim = self.current_game.state_dim
        self.n_agents = self.current_game.n_agents
        self.n_actions = self.current_game.n_actions
        print(f"切换到新游戏: '{next_game_name}'")
        print(f"  状态维度: {self.state_dim}")
        print(f"  智能体数量: {self.n_agents}")
        print(f"  动作数量: {self.n_actions}")
        
        # 如果提供了模型，进行知识迁移
        if model is not None:
            # 使用模型的知识迁移方法
            model = model.transfer_learning(new_state_dim=self.state_dim, 
                                            new_action_dim=self.n_actions)
            
            # 如果提供了优化器，重新初始化
            if optimizer is not None:
                # 使用相同的学习率重新创建优化器
                import torch.optim as optim
                lr = optimizer.param_groups[0]['lr']
                optimizer = optim.Adam(model.parameters(), lr=lr)
                print(f"  优化器已重新初始化，学习率: {lr}")
        
        return True, model, optimizer
    
    def get_current_game_name(self):
        """获取当前游戏名称"""
        if self.curriculum_learning:
            # 从GameManager获取游戏名称
            return self.game_manager.get_current_game_name()
        else:
            # 直接从当前游戏获取名称
            return self.current_game.name
    
    def evaluate_all_games(self, model, device):
        """
        在所有游戏上评估模型
        
        参数:
            model: 要评估的模型
            device: 计算设备
            
        返回:
            结果字典: {游戏名称: {reward: 平均奖励, success_rate: 成功率}}
        """
        if not self.curriculum_learning:
            print("警告: 非课程学习模式下调用了evaluate_all_games")
            return {self.current_game.name: {"reward": 0, "success_rate": 0}}
        
        results = {}
        
        # 保存当前游戏
        current_game = self.current_game
        current_game_idx = self.game_manager.current_game_idx
        
        # 在每个游戏上评估
        for i, game in enumerate(self.game_manager.games):
            # 切换到当前游戏
            self.game_manager.current_game_idx = i
            self.current_game = game
            
            # 评估
            print(f"评估游戏: {game.name}")
            reward, success_rate = game.evaluate(model, device)
            
            # 记录结果
            results[game.name] = {
                "reward": reward,
                "success_rate": success_rate
            }
        
        # 恢复当前游戏
        self.game_manager.current_game_idx = current_game_idx
        self.current_game = current_game
        
        return results
    
    def log_communication(self, model, episode):
        """
        记录智能体之间的通信内容
        
        参数:
            model: 要分析的模型
            episode: 当前回合数
        """
        if self.curriculum_learning:
            # 使用游戏管理器的通信日志功能
            self.game_manager.log_communication_evolution(model, episode)
        else:
            # 为单个游戏创建通信日志
            log_dir = os.path.join("./results", "communication_logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # 获取测试场景
            if hasattr(self.current_game, 'get_test_scenarios'):
                test_scenarios = self.current_game.get_test_scenarios()
                if test_scenarios and len(test_scenarios) > 0:
                    # 使用第一个测试场景
                    scenario = test_scenarios[0]
                    
                    # 重置游戏并获取观察
                    obs = self.current_game.reset(scenario)
                    
                    # 使用状态预处理器
                    processed_obs = StateProcessor.process_observations(self.current_game.name, obs)
                    
                    # 分析通信
                    try:
                        device = next(model.parameters()).device
                        messages = model.analyze_communication(processed_obs, device)
                        
                        # 保存通信内容
                        filename = os.path.join(log_dir, f"{self.current_game.name}_ep{episode}.txt")
                        with open(filename, "w") as f:
                            f.write(f"游戏: {self.current_game.name}\n")
                            f.write(f"回合: {episode}\n\n")
                            
                            for i, message in enumerate(messages):
                                f.write(f"智能体 {i+1} 通信内容:\n")
                                f.write(str(message.cpu().numpy()))
                                f.write("\n\n")
                        
                        print(f"通信演化记录已保存到 {filename}")
                    except Exception as e:
                        print(f"记录通信演化时出错: {e}") 