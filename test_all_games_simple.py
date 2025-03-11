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

def test_game(game_name, epochs=30, hidden_dim=64, comm_dim=32, memory_dim=32):
    """
    测试单个游戏的性能和通信模式
    
    参数:
        game_name: 游戏名称，如'SimpleCoordinationGame'
        epochs: 训练回合数
        
    返回:
        (模型, 游戏实例)
    """
    print("\n" + "="*50)
    print(f"测试游戏: {game_name}")
    print("="*50)
    
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
    
    print(f"环境初始化成功")
    print(f"  游戏名称: {game.name}")
    print(f"  智能体数量: {env.n_agents}")
    print(f"  动作数量: {env.n_actions}")
    print(f"  状态维度: {StateProcessor.UNIFIED_STATE_DIM}")
    
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
    print("\n开始训练...")
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
        
        # 报告训练进度
        if epoch > 0 and epoch % 5 == 0:
            avg_reward = np.mean(rewards_history[-5:])
            avg_success = np.mean(success_rates[-5:])
            print(f"回合 {epoch}/{epochs}:")
            print(f"  奖励: {avg_reward:.4f}, 成功率: {avg_success:.2f}")
    
    # 保存模型
    torch.save(model.state_dict(), f"{results_dir}/model.pt")
    print(f"\n模型已保存到 {results_dir}/model.pt")
    
    # 测试游戏性能
    print("\n测试游戏性能...")
    success_count = 0
    reward_sum = 0
    test_count = 10
    
    for _ in range(test_count):
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
        actions = model.select_actions(obs_tensor, hidden_states, social_memories, epsilon=0.0)
        
        # 执行动作
        _, rewards, _, info = env.step(actions)
        reward_sum += sum(rewards) / env.n_agents
        
        if info.get("success", False):
            success_count += 1
    
    print(f"测试结果:")
    print(f"  平均奖励: {reward_sum / test_count:.4f}")
    print(f"  成功率: {success_count / test_count:.2f}")
    
    # 测试通信
    print("\n测试通信...")
    test_communication(model, game, device)
    
    return model, game

def test_communication(model, game, device):
    """
    测试智能体之间的通信
    
    参数:
        model: 模型
        game: 游戏实例
        device: 设备
    """
    # 获取测试场景
    test_scenarios = game.get_test_scenarios()
    
    if not test_scenarios:
        print("没有可用的测试场景")
        return
    
    # 测试每个场景
    for i, scenario in enumerate(test_scenarios[:2]):  # 仅测试前两个场景
        print(f"\n场景 {i+1}:")
        
        # 重置游戏
        obs = game.reset(scenario)
        
        # 处理观察为统一维度
        processed_obs = StateProcessor.process_observations(game.name, obs)
        
        # 将观察转换为张量
        obs_tensor = []
        for j in range(game.n_agents):
            obs_tensor.append(torch.FloatTensor(processed_obs[j]).unsqueeze(0).to(device))
        
        # 初始化隐藏状态和社会记忆
        hidden_states = [model.init_hidden().to(device) for _ in range(game.n_agents)]
        social_memories = [model.init_social_memory().to(device) for _ in range(game.n_agents)]
        
        # 获取通信消息
        comm_messages = model.get_communication(obs_tensor, hidden_states)
        
        # 打印通信内容
        for agent_idx, message in enumerate(comm_messages):
            print(f"智能体 {agent_idx+1} 发送消息:")
            message_np = message.cpu().detach().numpy()
            
            # 找出最活跃的通信维度
            active_dims = np.argsort(np.abs(message_np).flatten())[-3:]  # 取前3个最活跃维度
            for dim in active_dims:
                value = message_np.flatten()[dim]
                print(f"  维度 {dim}: {value:.4f}")
        
        # 选择动作
        actions = model.select_actions(obs_tensor, hidden_states, social_memories, epsilon=0.0)
        
        # 执行动作
        next_obs, rewards, done, info = game.step(actions)
        print(f"\n选择动作: {actions}")
        print(f"获得奖励: {rewards}")
        print(f"成功: {info.get('success', False)}")

def analyze_communication_patterns(model, game, device):
    """
    分析通信模式
    
    参数:
        model: 训练好的模型
        game: 游戏实例
        device: 设备
    """
    print("\n通信模式分析:")
    print("-" * 30)
    
    game_name = game.name
    
    if game_name == "SimpleCoordinationGame":
        print("简单协调游戏中，通信的主要作用是协调双方的动作选择。")
        print("由于智能体能完全观察环境，通信主要用于表达自己的意图。")
    
    elif game_name == "AsymmetricInfoGame":
        print("非对称信息游戏中，通信的主要作用是共享各自的私有信息。")
        print("智能体1需要传递环境状态信息，智能体2需要根据这些信息选择正确的行动。")
    
    elif game_name == "SequentialDecisionGame":
        print("序列决策游戏中，通信的主要作用是协调长期策略。")
        print("智能体需要考虑当前步骤和历史信息，共同制定多步骤的策略。")
    
    elif game_name == "PartialObservableGame":
        print("部分可观察游戏中，通信的主要作用是共享各自观察到的环境部分。")
        print("每个智能体只能观察到环境的部分状态，需要通过通信合成完整的环境信息。")

def test_all_games(epochs=30):
    """
    测试所有游戏
    
    参数:
        epochs: 每个游戏的训练回合数
    """
    print("\n" + "="*50)
    print("开始测试所有游戏")
    print("="*50)
    
    # 定义所有游戏
    games = [
        "SimpleCoordinationGame",
        "AsymmetricInfoGame",
        "SequentialDecisionGame",
        "PartialObservableGame"
    ]
    
    # 循环测试每个游戏
    for game_name in games:
        model, game = test_game(game_name, epochs=epochs)
        analyze_communication_patterns(model, game, 
                                      torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("\n" + "-"*50)
    
    print("\n所有游戏测试完成")

if __name__ == "__main__":
    # 测试所有游戏
    test_all_games(epochs=30) 