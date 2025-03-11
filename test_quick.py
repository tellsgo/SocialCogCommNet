import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from games.simple_coordination import SimpleCoordinationGame
from games.asymmetric_info import AsymmetricInfoGame
from games.sequential_decision import SequentialDecisionGame
from games.partial_observable import PartialObservableGame
from environment import CooperativeEnvironment
from model import SocialCognitiveCommNet
from games.state_processor import StateProcessor
import json

def test_game(game_name, epochs=50, hidden_dim=64, comm_dim=32, memory_dim=32):
    """
    测试单个游戏的性能和通信演化
    
    参数:
        game_name: 游戏名称，如'SimpleCoordinationGame'
        epochs: 训练回合数
    """
    # 创建保存目录
    results_dir = f"./results/{game_name}_test"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建游戏
    if game_name == "SimpleCoordinationGame":
        game = SimpleCoordinationGame()
    elif game_name == "AsymmetricInfoGame":
        game = AsymmetricInfoGame()
    elif game_name == "SequentialDecisionGame":
        game = SequentialDecisionGame()
    elif game_name == "PartialObservableGame":
        game = PartialObservableGame()
    else:
        raise ValueError(f"未知的游戏名称: {game_name}")
    
    # 创建环境
    env = CooperativeEnvironment(game=game)
    print(f"环境初始化成功：使用游戏 '{game_name}'")
    print(f"  状态维度: {StateProcessor.UNIFIED_STATE_DIM} (统一状态维度)")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SocialCognitiveCommNet(
        input_dim=StateProcessor.UNIFIED_STATE_DIM,
        hidden_dim=hidden_dim,
        comm_dim=comm_dim,
        memory_dim=memory_dim,
        n_agents=env.n_agents,
        n_actions=env.n_actions
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 记录训练数据
    rewards_history = []
    success_rates = []
    
    # 训练循环
    print(f"开始训练游戏: {game_name}")
    for epoch in range(epochs):
        # 重置环境
        obs = env.reset()
        
        # 初始化隐藏状态和社会记忆
        hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
        social_memories = [model.init_social_memory().to(device) for _ in range(env.n_agents)]
        
        # 转换观察为张量
        obs_tensor = []
        for i in range(env.n_agents):
            obs_tensor.append(torch.FloatTensor(obs[i]).unsqueeze(0).to(device))
        
        # 选择动作
        epsilon = max(0.1, 1.0 - epoch / epochs)
        actions = model.select_actions(obs_tensor, hidden_states, social_memories, epsilon)
        
        # 执行动作
        next_obs, rewards, done, info = env.step(actions)
        
        # 记录结果
        success = info.get("success", False)
        rewards_history.append(sum(rewards) / env.n_agents)
        success_rates.append(1 if success else 0)
        
        # 简单训练（这里简化了训练过程）
        if epoch > 0 and epoch % 10 == 0:
            print(f"回合 {epoch}/{epochs}:")
            print(f"  奖励: {np.mean(rewards_history[-10:]):.4f}, 成功率: {np.mean(success_rates[-10:]):.2f}")
    
    # 保存模型
    torch.save(model.state_dict(), f"{results_dir}/model.pt")
    
    # 分析通信
    log_communication_evolution(model, game, epochs, results_dir)
    
    print(f"游戏 {game_name} 测试完成，结果已保存到 {results_dir}")
    
    # 返回通信日志文件路径
    return f"{results_dir}/communication_logs/{game_name}_ep{epochs}.txt"

def log_communication_evolution(model, game, episode, save_dir):
    """
    记录通信内容的演化，分析智能体间的通信模式
    
    参数:
        model: 当前模型
        game: 当前游戏
        episode: 当前回合数
        save_dir: 保存目录
    """
    # 创建通信日志目录
    log_dir = os.path.join(save_dir, "communication_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    game_name = game.name
    
    # 获取测试场景
    if hasattr(game, 'get_test_scenarios'):
        test_scenarios = game.get_test_scenarios()
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
                for i, scenario in enumerate(test_scenarios[:3]):  # 只分析前3个场景
                    f.write(f"场景 {i+1}:\n")
                    
                    # 重置游戏
                    try:
                        obs = game.reset(scenario)
                        
                        # 处理观察为统一维度
                        processed_obs = StateProcessor.process_observations(game_name, obs)
                        
                        # 分析通信
                        try:
                            device = next(model.parameters()).device
                            
                            # 将观察转换为张量
                            obs_tensor = []
                            for j in range(game.n_agents):
                                obs_tensor.append(torch.FloatTensor(processed_obs[j]).unsqueeze(0).to(device))
                            
                            # 初始化隐藏状态
                            hidden_states = [model.init_hidden().to(device) for _ in range(game.n_agents)]
                            
                            # 获取通信消息
                            comm_messages = model.get_communication(obs_tensor, hidden_states)
                            
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
                            social_memories = [model.init_social_memory().to(device) for _ in range(game.n_agents)]
                            actions = model.select_actions(
                                obs_tensor, 
                                hidden_states,
                                social_memories,
                                epsilon=0.0
                            )
                            
                            # 记录动作和奖励
                            next_obs, rewards, done, info = game.step(actions)
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

def display_communication_analysis(log_file):
    """显示通信分析结果"""
    try:
        with open(log_file, "r") as f:
            content = f.read()
            print("\n" + "="*50)
            print(f"通信分析结果 ({log_file}):")
            print(content)
    except Exception as e:
        print(f"读取通信日志文件时出错: {e}")

if __name__ == "__main__":
    # 测试SimpleCoordinationGame
    log_file = test_game("SimpleCoordinationGame", epochs=50)
    display_communication_analysis(log_file)
    
    # 测试AsymmetricInfoGame
    log_file = test_game("AsymmetricInfoGame", epochs=50)
    display_communication_analysis(log_file) 