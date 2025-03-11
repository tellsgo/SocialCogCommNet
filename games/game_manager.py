import numpy as np
import os
import torch
from .base_game import BaseGame
from .simple_coordination import SimpleCoordinationGame
from .asymmetric_info import AsymmetricInfoGame
from .sequential_decision import SequentialDecisionGame
from .partial_observable import PartialObservableGame
import matplotlib.pyplot as plt
import seaborn as sns

class GameManager:
    """
    游戏管理器，负责管理多种游戏，支持课程学习
    """
    def __init__(self, games, save_dir="./results"):
        """
        初始化游戏管理器
        
        参数:
            games: 游戏对象列表，按难度递增排序
            save_dir: 模型和结果的保存目录
        """
        # 存储游戏实例
        self.games = games
        self.save_dir = save_dir
        
        # 创建游戏名称到游戏实例的映射
        self.games_dict = {}
        self.game_order = []
        
        for game in games:
            game_name = game.name
            self.games_dict[game_name] = game
            self.game_order.append(game_name)
        
        # 当前游戏索引
        self.current_game_idx = 0
    
    def get_current_game(self):
        """获取当前游戏"""
        return self.games[self.current_game_idx]
    
    def get_current_game_name(self):
        """获取当前游戏名称"""
        return self.game_order[self.current_game_idx]
    
    def next_game(self):
        """切换到下一个游戏"""
        if self.current_game_idx < len(self.games) - 1:
            self.current_game_idx += 1
            return True, self.get_current_game()
        else:
            return False, self.get_current_game()
    
    def reset_curriculum(self):
        """重置课程，从最简单的游戏开始"""
        self.current_game_idx = 0
        return self.get_current_game()
    
    def get_game_by_name(self, name):
        """通过名称获取游戏"""
        if name in self.games_dict:
            return self.games_dict[name]
        return None
    
    def get_all_games(self):
        """获取所有游戏"""
        return self.games_dict
    
    def evaluate_all_games(self, model, device, communication=True):
        """在所有游戏上评估模型"""
        results = {}
        
        for game_name, game in self.games_dict.items():
            # 获取测试场景
            test_scenarios = game.get_test_scenarios()
            
            rewards_list = []
            success_count = 0
            
            for scenario in test_scenarios:
                # 重置游戏
                obs = game.reset(scenario)
                
                # 初始化状态
                hidden_states = [model.init_hidden().to(device) for _ in range(game.n_agents)]
                social_memories = [model.init_social_memory().to(device) for _ in range(game.n_agents)]
                
                # 转换观察为张量
                obs_tensor = []
                for i in range(game.n_agents):
                    obs_tensor.append(torch.FloatTensor(obs[i]).unsqueeze(0).to(device))
                
                # 选择动作
                actions = model.select_actions(
                    obs_tensor,
                    hidden_states,
                    social_memories,
                    epsilon=0.0,
                    communication=communication
                )
                
                # 执行动作
                _, rewards, _, info = game.step(actions)
                avg_reward = sum(rewards) / game.n_agents
                rewards_list.append(avg_reward)
                
                # 记录成功
                if "success" in info and info["success"]:
                    success_count += 1
            
            # 计算平均奖励和成功率
            avg_reward = sum(rewards_list) / len(rewards_list) if rewards_list else 0
            success_rate = success_count / len(test_scenarios) if test_scenarios else 0
            
            # 保存结果
            results[game_name] = {
                "reward": avg_reward,
                "success_rate": success_rate
            }
        
        return results
    
    def analyze_communication_evolution(self, models, save_dir):
        """分析通信的演化规律"""
        if not models:
            return
        
        # 创建保存通信分析的目录
        os.makedirs(os.path.join(save_dir, "communication_evolution"), exist_ok=True)
        
        # 对每个游戏进行分析
        for game_name, game in self.games_dict.items():
            # 获取测试场景
            test_scenarios = game.get_test_scenarios()
            
            # 跳过没有测试场景的游戏
            if not test_scenarios:
                continue
            
            # 选择一个代表性场景来分析通信
            scenario = test_scenarios[0]
            
            # 准备收集通信数据
            comm_data = []
            
            # 对每个模型获取通信内容
            for i, model in enumerate(models):
                model.eval()
                
                with torch.no_grad():
                    # 重置游戏
                    obs = game.reset(scenario)
                    
                    # 分析通信
                    comm_messages = model.analyze_communication(obs)
                    
                    # 保存数据
                    comm_data.append({
                        "episode": i * 100,  # 假设每100回合保存一次模型
                        "messages": comm_messages
                    })
            
            # 可视化通信演化
            self._visualize_comm_evolution(comm_data, game_name, save_dir)
    
    def _visualize_comm_evolution(self, comm_data, game_name, save_dir):
        """可视化通信内容的演化"""
        if not comm_data:
            return
        
        # 获取智能体数量和通信维度
        n_agents = len(comm_data[0]["messages"])
        comm_dim = comm_data[0]["messages"][0].shape[1]
        
        # 为每个智能体创建一个图
        for agent_idx in range(n_agents):
            plt.figure(figsize=(12, 8))
            
            # 提取该智能体在所有时间点的通信内容
            agent_messages = []
            episodes = []
            
            for data in comm_data:
                agent_messages.append(data["messages"][agent_idx][0])
                episodes.append(data["episode"])
            
            # 构建热力图数据
            heat_data = np.array(agent_messages)
            
            # 创建热力图
            ax = plt.gca()
            sns.heatmap(heat_data, cmap="viridis", ax=ax, 
                       yticklabels=episodes)
            
            # 设置标题和标签
            plt.title(f"{game_name} - 智能体 {agent_idx + 1} 通信内容演化")
            plt.xlabel("通信维度")
            plt.ylabel("训练回合")
            
            # 保存图片
            filename = os.path.join(save_dir, "communication_evolution", 
                                   f"{game_name}_agent{agent_idx+1}_comm_evolution.png")
            plt.savefig(filename)
            plt.close()
    
    def transfer_model(self, model, optimizer, next_game=None):
        """
        将模型从当前游戏转移到下一个游戏
        
        参数:
            model: 当前训练好的模型
            optimizer: 优化器
            next_game: 指定的下一个游戏索引，如果为None则使用当前游戏索引+1
            
        返回:
            迁移后的模型
        """
        # 保存当前游戏的模型
        current_game = self.get_current_game()
        save_path = os.path.join(self.save_dir, f"model_{current_game.name}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'game': current_game.name
        }, save_path)
        print(f"当前游戏 '{current_game.name}' 的模型已保存到 {save_path}")
        
        # 移动到下一个游戏
        if next_game is not None:
            self.current_game_idx = next_game
        else:
            if not self.next_game():
                print("已经是最后一个游戏，无法继续")
                return model, optimizer
        
        next_game = self.get_current_game()
        print(f"转移到下一个游戏: '{next_game.name}'")
        
        # 如果下一个游戏需要修改模型结构，这里可以添加调整逻辑
        # 例如，如果输入维度或输出维度变化了
        
        return model, optimizer
    
    def log_communication_evolution(self, model, episode):
        """
        记录通信内容的演化，分析智能体间的通信模式
        
        参数:
            model: 当前模型
            episode: 当前回合数
        """
        # 创建通信日志目录
        log_dir = os.path.join(self.save_dir, "communication_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 获取当前游戏
        current_game = self.get_current_game()
        game_name = current_game.name
        
        # 获取测试场景
        if hasattr(current_game, 'get_test_scenarios'):
            test_scenarios = current_game.get_test_scenarios()
            if test_scenarios and len(test_scenarios) > 0:
                # 使用状态处理器处理测试场景
                from games.state_processor import StateProcessor
                
                # 创建分析结果文件
                filename = os.path.join(log_dir, f"{game_name}_ep{episode}.txt")
                with open(filename, "w") as f:
                    f.write(f"游戏: {game_name}\n")
                    f.write(f"回合: {episode}\n\n")
                    f.write("通信分析:\n\n")
                    
                    # 分析每个测试场景下的通信
                    for i, scenario in enumerate(test_scenarios[:3]):  # 只分析前3个场景，避免太多输出
                        f.write(f"场景 {i+1}:\n")
                        
                        # 重置游戏
                        try:
                            obs = current_game.reset(scenario)
                            
                            # 处理观察为统一维度
                            processed_obs = StateProcessor.process_observations(game_name, obs)
                            
                            # 分析通信
                            try:
                                device = next(model.parameters()).device
                                
                                # 将观察转换为张量
                                obs_tensor = []
                                for i in range(current_game.n_agents):
                                    obs_tensor.append(torch.FloatTensor(processed_obs[i]).unsqueeze(0).to(device))
                                
                                # 获取通信消息
                                comm_messages = model.get_communication(obs_tensor)
                                
                                # 记录通信内容
                                for agent_idx, message in enumerate(comm_messages):
                                    f.write(f"智能体 {agent_idx+1} 发送消息:\n")
                                    message_np = message.cpu().detach().numpy()
                                    
                                    # 找出最活跃的通信维度
                                    active_dims = np.argsort(np.abs(message_np).flatten())[-3:]  # 取前3个最活跃维度
                                    
                                    for dim in active_dims:
                                        value = message_np.flatten()[dim]
                                        f.write(f"  维度 {dim}: {value:.4f}\n")
                                    
                                    f.write("\n")
                                
                                # 执行动作
                                actions = model.select_actions(
                                    obs_tensor, 
                                    [model.init_hidden().to(device) for _ in range(current_game.n_agents)],
                                    [model.init_social_memory().to(device) for _ in range(current_game.n_agents)],
                                    epsilon=0.0
                                )
                                
                                # 记录动作和奖励
                                next_obs, rewards, done, info = current_game.step(actions)
                                f.write(f"选择动作: {actions}\n")
                                f.write(f"获得奖励: {rewards}\n")
                                f.write(f"成功: {info.get('success', False)}\n\n")
                                
                            except Exception as e:
                                f.write(f"分析通信时出错: {e}\n\n")
                        except Exception as e:
                            f.write(f"重置游戏时出错: {e}\n\n")
                    
                    # 记录通信模式的总结分析
                    f.write("\n通信模式总结:\n")
                    f.write("================\n")
                    
                    if game_name == "SimpleCoordinationGame":
                        f.write("简单协调游戏中，通信的主要作用是协调双方的动作选择。\n")
                        f.write("由于智能体能完全观察环境，通信主要用于表达自己的意图。\n")
                    
                    elif game_name == "AsymmetricInfoGame":
                        f.write("非对称信息游戏中，通信的主要作用是共享各自的私有信息。\n")
                        f.write("智能体1需要传递环境状态信息，智能体2需要根据这些信息选择正确的行动。\n")
                    
                    elif game_name == "SequentialDecisionGame":
                        f.write("序列决策游戏中，通信的主要作用是协调长期策略。\n")
                        f.write("智能体需要考虑当前步骤和历史信息，共同制定多步骤的策略。\n")
                    
                    elif game_name == "PartialObservableGame":
                        f.write("部分可观察游戏中，通信的主要作用是共享各自观察到的环境部分。\n")
                        f.write("每个智能体只能观察到环境的部分状态，需要通过通信合成完整的环境信息。\n")
                
                print(f"通信演化分析已保存到 {filename}")
                return
        
        # 如果没有测试场景，则简单记录
        filename = os.path.join(log_dir, f"{game_name}_ep{episode}.txt")
        with open(filename, "w") as f:
            f.write(f"游戏: {game_name}\n")
            f.write(f"回合: {episode}\n\n")
            f.write("通信演化记录\n")
            f.write("由于缺少测试场景，无法分析通信内容\n")
        
        print(f"通信演化记录已保存到 {filename}")
    
    def evaluate_transfer(self, model, device):
        """评估模型在所有游戏上的表现"""
        results = {}
        current_idx = self.current_game_idx
        
        for i, game in enumerate(self.games):
            self.current_game_idx = i
            print(f"在游戏 '{game.name}' 上评估...")
            
            # 评估模型在当前游戏上的表现
            reward, success_rate = game.evaluate(model, device)
            results[game.name] = {
                "reward": reward,
                "success_rate": success_rate
            }
            
            print(f"  奖励: {reward:.4f}, 成功率: {success_rate:.2f}")
        
        # 恢复当前游戏索引
        self.current_game_idx = current_idx
        
        return results 