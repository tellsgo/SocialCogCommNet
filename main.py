import argparse
import torch
import numpy as np
import os
import sys
import traceback
import json
import random
from train import train, evaluate, set_seed
from model import SocialCognitiveCommNet
from environment import CooperativeEnvironment

# 设置日志文件
log_file = open('debug_log.txt', 'w')

def log(message):
    """将消息同时输出到控制台和日志文件"""
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

def parse_args():
    parser = argparse.ArgumentParser(description='社会认知通信网络训练')
    
    # 环境参数
    parser.add_argument('--max_steps', type=int, default=50, help='每个episode的最大步数')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--comm_dim', type=int, default=16, help='通信维度')
    parser.add_argument('--memory_dim', type=int, default=64, help='社会记忆维度')
    
    # 训练参数
    parser.add_argument('--n_episodes', type=int, default=1000, help='训练的episode数量')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--buffer_capacity', type=int, default=10000, help='经验回放缓冲区容量')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='初始探索率')
    parser.add_argument('--epsilon_end', type=float, default=0.05, help='最终探索率')
    parser.add_argument('--epsilon_decay', type=float, default=200, help='探索率衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./results', help='保存结果的目录')
    parser.add_argument('--print_interval', type=int, default=10, help='打印间隔')
    parser.add_argument('--eval_interval', type=int, default=100, help='评估间隔')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式')
    parser.add_argument('--model_path', type=str, default=None, help='测试模式下加载的模型路径')
    parser.add_argument('--use_cuda', action='store_true', help='是否使用CUDA加速')
    parser.add_argument('--visualize_only', action='store_true', help='只进行可视化，不训练')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--config_path', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--curriculum', action='store_true', default=False, help='是否启用课程学习')
    
    return parser.parse_args()

def main():
    try:
        log("程序开始执行...")
        
        # 解析参数
        args = parse_args()
        log("参数解析完成")
        
        log(f"Python版本: {sys.version}")
        log(f"PyTorch版本: {torch.__version__}")
        log(f"NumPy版本: {np.__version__}")
        
        # 设置随机种子
        set_seed(args.seed)
        log(f"随机种子设置为: {args.seed}")
        
        # 检查CUDA可用性
        if args.use_cuda and not torch.cuda.is_available():
            log("警告: CUDA不可用，将使用CPU")
            args.use_cuda = False
            
        log(f"使用设备: {'CUDA' if args.use_cuda and torch.cuda.is_available() else 'CPU'}")
        
        # 加载配置
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        
        # 添加命令行参数到配置中
        config['curriculum_learning'] = args.curriculum
        
        # 确保保存目录存在
        os.makedirs(args.save_dir, exist_ok=True)
        log(f"保存目录: {os.path.abspath(args.save_dir)}")
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
        
        # 根据模式运行
        if args.mode == 'train':
            log("开始训练社会认知通信网络...")
            log(f"使用设备: {device}")
            
            log("配置参数:")
            for k, v in config.items():
                log(f"  {k}: {v}")
            
            # 测试环境
            log("测试环境...")
            try:
                env = CooperativeEnvironment()
                log(f"环境创建成功: {env.n_agents} 智能体, {env.n_actions} 动作")
                
                obs = env.reset()
                log(f"环境重置成功，观察形状: {np.array(obs).shape}")
                
                # 测试模型
                log("测试模型...")
                model = SocialCognitiveCommNet(
                    input_dim=env.state_dim,
                    hidden_dim=config['hidden_dim'],
                    comm_dim=config['comm_dim'],
                    memory_dim=config['memory_dim'],
                    n_agents=env.n_agents,
                    n_actions=env.n_actions
                ).to(device)
                log(f"模型创建成功")
                
                # 初始化隐藏状态和社会记忆
                hidden_states = [model.init_hidden().to(device) for _ in range(env.n_agents)]
                social_memories = [model.init_social_memory().to(device) for _ in range(env.n_agents)]
                log(f"隐藏状态和社会记忆初始化成功")
                
                # 将观察转换为张量
                obs_tensor = [torch.FloatTensor(o).unsqueeze(0).to(device) for o in obs]
                log(f"观察转换为张量成功")
                
                # 前向传播
                actions_logits, new_hidden_states, new_social_memories, comm_messages = model.forward(
                    obs_tensor, hidden_states, social_memories, communication=True
                )
                log(f"前向传播成功")
                
                # 选择动作
                actions = [torch.argmax(a[0]).item() for a in actions_logits]
                log(f"选择动作成功: {actions}")
                
                # 执行动作
                next_obs, rewards, done, info = env.step(actions)
                log(f"执行动作成功: 奖励={rewards}")
                
                log("模型测试成功，开始训练...")
                
                # 开始训练
                model, rewards_history = train(config)
                log("训练完成！")
            except Exception as e:
                log(f"训练过程中出错: {e}")
                traceback.print_exc(file=log_file)
                
        elif args.visualize_only:
            log("进行可视化分析...")
            
            # 加载训练数据并可视化
            if not args.model_path:
                log("错误：可视化模式需要指定模型路径")
                return
                
            # 加载训练数据
            try:
                npz_file = args.model_path.replace('.pt', '_data.npz')
                if os.path.exists(npz_file):
                    data = np.load(npz_file)
                    
                    # 提取数据
                    rewards = data['rewards']
                    rewards_no_comm = data['rewards_no_comm']
                    if 'losses' in data:
                        losses = data['losses']
                        losses_no_comm = data['losses_no_comm']
                    else:
                        losses = []
                        losses_no_comm = []
                    
                    if 'success_rates' in data:
                        success_rates = data['success_rates']
                        success_rates_no_comm = data['success_rates_no_comm']
                    else:
                        success_rates = []
                        success_rates_no_comm = []
                    
                    # 进行可视化
                    from train import visualize_training
                    visualize_training(
                        rewards, rewards_no_comm,
                        losses, losses_no_comm,
                        success_rates, success_rates_no_comm,
                        os.path.dirname(args.model_path), len(rewards)
                    )
                    log("可视化完成！")
                else:
                    log(f"错误：找不到训练数据文件 {npz_file}")
            except Exception as e:
                log(f"可视化出错: {e}")
                traceback.print_exc(file=log_file)
                
        else:
            log("开始测试社会认知通信网络...")
            # 创建环境
            env = CooperativeEnvironment()
            
            # 创建模型
            model = SocialCognitiveCommNet(
                input_dim=env.state_dim,
                hidden_dim=config['hidden_dim'],
                comm_dim=config['comm_dim'],
                memory_dim=config['memory_dim'],
                n_agents=env.n_agents,
                n_actions=env.n_actions
            ).to(device)
            
            # 加载模型
            if args.model_path:
                try:
                    checkpoint = torch.load(args.model_path, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        log(f"模型已从 {args.model_path} 加载 (episode {checkpoint['episode']})")
                    else:
                        model.load_state_dict(checkpoint)
                        log(f"模型已从 {args.model_path} 加载")
                except Exception as e:
                    log(f"加载模型出错: {e}")
                    traceback.print_exc(file=log_file)
                    return
            else:
                log("错误：测试模式需要指定模型路径")
                return
            
            # 评估模型
            try:
                reward_with_comm, success_rate_comm = evaluate(model, env, config, device, communication=True, n_episodes=100)
                reward_without_comm, success_rate_no_comm = evaluate(model, env, config, device, communication=False, n_episodes=100)
                
                log(f"有通信评估奖励: {reward_with_comm:.4f}, 成功率: {success_rate_comm:.2f}")
                log(f"无通信评估奖励: {reward_without_comm:.4f}, 成功率: {success_rate_no_comm:.2f}")
                log(f"通信带来的提升: {reward_with_comm - reward_without_comm:.4f}")
                
                # 分析通信内容
                from train import analyze_communication
                analyze_communication(model, env, config, device, args.save_dir, 'test')
            except Exception as e:
                log(f"测试过程中出错: {e}")
                traceback.print_exc(file=log_file)
    
    except Exception as e:
        log(f"程序运行出错: {e}")
        traceback.print_exc(file=log_file)
    
    finally:
        # 关闭日志文件
        log("程序执行结束")
        log_file.close()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='社会认知通信网络')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式：训练或测试')
    parser.add_argument('--model_path', type=str, default=None, help='测试模式下加载模型的路径')
    parser.add_argument('--config_path', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--curriculum', action='store_true', default=False, help='是否启用课程学习')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # 设置随机种子，确保结果可重现
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 添加命令行参数到配置中
    config['curriculum_learning'] = args.curriculum
    
    # 运行模式选择
    if args.mode == 'train':
        # 训练模式
        print("开始训练社会认知通信网络...")
        try:
            model, rewards_history = train(config)
            print("训练完成!")
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            traceback.print_exc()
    else:
        # 测试模式
        print("开始测试社会认知通信网络...")
        try:
            if args.model_path is None:
                raise ValueError("测试模式需要指定模型路径 (--model_path)")
            
            test(args.model_path, config)
            print("测试完成!")
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            traceback.print_exc() 