import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from environment import CooperativeEnvironment
from model import SocialCognitiveCommNet
from games.simple_coordination import SimpleCoordinationGame
from games.asymmetric_info import AsymmetricInfoGame
from games.sequential_decision import SequentialDecisionGame
from games.partial_observable import PartialObservableGame
from algorithms import AlgorithmManager

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练社会认知通信网络')
    
    # 训练参数
    parser.add_argument('--algorithm', type=str, default='q_learning', 
                        choices=['q_learning', 'ppo'], help='训练算法')
    parser.add_argument('--game', type=str, default='SimpleCoordinationGame',
                        choices=['SimpleCoordinationGame', 'AsymmetricInfoGame', 
                                'SequentialDecisionGame', 'PartialObservableGame'],
                        help='游戏名称')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--episodes_per_epoch', type=int, default=5, help='每轮训练的回合数')
    parser.add_argument('--max_steps', type=int, default=20, help='每回合最大步数')
    parser.add_argument('--eval_interval', type=int, default=20, help='评估间隔')
    parser.add_argument('--save_interval', type=int, default=50, help='保存间隔')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--comm_dim', type=int, default=64, help='通信维度')
    parser.add_argument('--memory_dim', type=int, default=64, help='社会记忆维度')
    
    # 算法参数
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--buffer_capacity', type=int, default=10000, help='经验缓冲区容量')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='初始探索率')
    parser.add_argument('--epsilon_end', type=float, default=0.05, help='最终探索率')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='探索率衰减')
    
    # PPO特定参数
    parser.add_argument('--ppo_epochs', type=int, default=4, help='PPO更新轮数')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda参数')
    parser.add_argument('--policy_clip', type=float, default=0.2, help='PPO裁剪参数')
    parser.add_argument('--value_coef', type=float, default=0.5, help='值函数损失系数')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='熵损失系数')
    
    # 其他参数
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./results', help='保存目录')
    
    return parser.parse_args()

def create_game(game_name):
    """创建游戏实例"""
    if game_name == 'SimpleCoordinationGame':
        return SimpleCoordinationGame()
    elif game_name == 'AsymmetricInfoGame':
        return AsymmetricInfoGame()
    elif game_name == 'SequentialDecisionGame':
        return SequentialDecisionGame()
    elif game_name == 'PartialObservableGame':
        return PartialObservableGame()
    else:
        raise ValueError(f"未知的游戏: {game_name}")

def create_algorithm_config(args):
    """创建算法配置"""
    config = {
        'gamma': args.gamma,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epsilon': args.epsilon_start,
        'epsilon_decay': args.epsilon_decay,
        'min_epsilon': args.epsilon_end,
        'use_communication': True,
    }
    
    # 添加Q学习特定配置
    if args.algorithm == 'q_learning':
        config.update({
            'buffer_capacity': args.buffer_capacity,
            'target_update': 10,
            'tau': 0.005,
            'use_double_q': True,
        })
    
    # 添加PPO特定配置
    elif args.algorithm == 'ppo':
        config.update({
            'ppo_epochs': args.ppo_epochs,
            'gae_lambda': args.gae_lambda,
            'policy_clip': args.policy_clip,
            'value_coef': args.value_coef,
            'entropy_coef': args.entropy_coef,
            'use_gae': True,
            'normalize_advantages': True,
            'clip_value': True,
        })
    
    return config

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存目录
    save_dir = os.path.join(args.save_dir, f"{args.game}_{args.algorithm}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建游戏和环境
    game = create_game(args.game)
    env = CooperativeEnvironment(game=game)
    
    print(f"环境初始化成功")
    print(f"  游戏名称: {game.name}")
    print(f"  智能体数量: {env.n_agents}")
    print(f"  动作数量: {env.n_actions}")
    
    # 创建模型
    from games.state_processor import StateProcessor
    model = SocialCognitiveCommNet(
        input_dim=StateProcessor.UNIFIED_STATE_DIM,
        hidden_dim=args.hidden_dim,
        comm_dim=args.comm_dim,
        memory_dim=args.memory_dim,
        n_agents=env.n_agents,
        n_actions=env.n_actions
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 创建算法配置
    algorithm_config = create_algorithm_config(args)
    
    # 创建算法管理器和实验
    algo_manager = AlgorithmManager()
    experiment = algo_manager.create_experiment(
        args.algorithm, 
        model, 
        optimizer, 
        device, 
        algorithm_config,
        save_dir
    )
    
    # 训练循环
    print(f"\n开始训练 {args.game} 使用 {args.algorithm} 算法")
    print(f"训练参数: 轮数={args.epochs}, 每轮回合={args.episodes_per_epoch}, 最大步数={args.max_steps}")
    
    # 记录训练过程
    epochs = []
    rewards = []
    success_rates = []
    epsilon_history = []
    
    for epoch in range(args.epochs):
        # 训练
        result = experiment.train_episode(
            env, 
            n_episodes=args.episodes_per_epoch, 
            max_steps=args.max_steps
        )
        
        # 记录
        epochs.append(epoch)
        rewards.append(result["avg_reward"])
        success_rates.append(result["success_rate"])
        epsilon_history.append(result["exploration_rate"])
        
        # 输出训练信息
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"轮次 {epoch}/{args.epochs}:")
            print(f"  奖励: {result['avg_reward']:.4f}, 成功率: {result['success_rate']:.2f}, " +
                 f"探索率: {result['exploration_rate']:.4f}")
        
        # 定期评估
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 评估有通信和无通信的表现
            compare_result = experiment.compare_communication(env, n_episodes=10, max_steps=args.max_steps)
            
            print(f"\n评估结果 (轮次 {epoch}):")
            print(f"  有通信 - 奖励: {compare_result['with_comm']['avg_reward']:.4f}, " +
                 f"成功率: {compare_result['with_comm']['success_rate']:.2f}")
            print(f"  无通信 - 奖励: {compare_result['without_comm']['avg_reward']:.4f}, " +
                 f"成功率: {compare_result['without_comm']['success_rate']:.2f}")
            print(f"  通信提升 - 奖励: {compare_result['reward_diff']:.4f}, " + 
                 f"成功率: {compare_result['success_diff'] * 100:.2f}%")
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            model_path, _ = experiment.save(f"model_epoch_{epoch}")
            print(f"\n模型已保存到 {model_path}")
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 10))
    
    # 奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, rewards)
    plt.title('平均奖励')
    plt.xlabel('训练轮次')
    plt.ylabel('奖励')
    
    # 成功率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, success_rates)
    plt.title('成功率')
    plt.xlabel('训练轮次')
    plt.ylabel('成功率')
    
    # 探索率曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, epsilon_history)
    plt.title('探索率')
    plt.xlabel('训练轮次')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    
    print(f"\n训练完成，学习曲线已保存到 {os.path.join(save_dir, 'learning_curves.png')}")
    
    # 最终评估
    print("\n进行最终评估...")
    final_result = experiment.compare_communication(env, n_episodes=50, max_steps=args.max_steps)
    
    print(f"\n最终评估结果:")
    print(f"  有通信 - 奖励: {final_result['with_comm']['avg_reward']:.4f}, " +
         f"成功率: {final_result['with_comm']['success_rate']:.2f}")
    print(f"  无通信 - 奖励: {final_result['without_comm']['avg_reward']:.4f}, " +
         f"成功率: {final_result['without_comm']['success_rate']:.2f}")
    print(f"  通信提升 - 奖励: {final_result['reward_diff']:.4f}, " + 
         f"成功率: {final_result['success_diff'] * 100:.2f}%")

if __name__ == "__main__":
    main() 