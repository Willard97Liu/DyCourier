from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from copy import deepcopy
import random
import math
import os
from glob import glob
from methods.ML.model import NN
from pathlib import Path
import sys
import csv
import matplotlib.pyplot as plt


path = Path(os.path.dirname(__file__))
sys.path.insert(1, str(path.parent.absolute()))

result_path = path.parent.parent  # 表示当前脚本的上上一层目录
results_dir = result_path / "results"
results_dir.mkdir(parents=True, exist_ok=True)

log_path = results_dir / "training_log.csv"


def select_action(state, policy_net):
    logits = policy_net(state)  # shape: [1, n_actions]
    return logits



def rollout_one(env, ACTIONS, select_action, policy_net, gamma=0.99, device='cpu'):
    
    env.reset()
    
    t = 5
    
    visible_orders = [o for o in env.active_orders if o.order_time <= t]
        
    
    state = env.state_manager.compute_state(
                t, env.courier_scheduler, visible_orders
            )
    
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    episode_reward = 0 
    
    trajectory = []
    
    all_new_couriers = []

    for t in env.decision_epochs:
        
        # 先选择动作
        
        logits = select_action(state, policy_net).squeeze(1)
        probs = torch.softmax(logits, dim=-1)
        # print(probs)
        action_dist = torch.distributions.Categorical(probs)
        action_id = action_dist.sample()
        log_prob = action_dist.log_prob(action_id)
        trajectory.append(log_prob.squeeze()) 

        
        action = ACTIONS[action_id]
        
        # 再用动作进行交互
        
        current_lost = env.step(t, action, visible_orders)
        
        # 要将每次的action 进行记录保留，以便后面计算总共的骑手的惩罚
        
        a1, a1_5 = action
       
        new_couriers = [1]*a1 + [1.5]*a1_5
        all_new_couriers.extend(new_couriers)
        
        t_next = t + env.config.decision_interval
        
        next_state = env.state_manager.compute_state(
                t, env.courier_scheduler, visible_orders
            )
        
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        
        visible_orders = [o for o in env.active_orders if o.order_time <= t_next]
    
    courier_penalty = sum(env.config.K_c[c] for c in all_new_couriers)
    
    lost_penalty = env.config.K_lost * current_lost
    
    # 最终奖励
    episode_reward = courier_penalty + lost_penalty  # 通常使用负值作为惩罚
    
    return trajectory, episode_reward

def compute_loss(all_trajs):
    """
    all_trajs: List of (log_probs_list, total_reward)
    """
    rewards = [r for _, r in all_trajs]
    
    baseline = np.mean(rewards)

    total_loss = 0.0
    for traj, reward in all_trajs:
        logprob_sum = torch.stack(traj).sum()
        advantage = reward - baseline
        total_loss += -logprob_sum * advantage

    
    return total_loss / len(all_trajs), baseline 


def load_latest_model(policy_net, model_path_prefix):
    model_files = sorted(
        glob(str(results_dir / f"{model_path_prefix}_ep*.pt")),
        key=lambda x: int(x.split("_ep")[-1].split(".pt")[0]),
        reverse=True,
    )
    if model_files:
        latest_file = model_files[0]
        print(f"Loading model: {latest_file}")
        policy_net.load_state_dict(torch.load(latest_file))
        return int(latest_file.split("_ep")[-1].split(".pt")[0])
    return 0

def train_reinforce(
    env_fn,
    hidden_layers,
    epoch = 10000,
    num_samples = 16,
    n_env = 8,
    
    
    GAMMA = 0.99,
    eval_interval = 100,
    save = True,
    model_path = 'model_DQN',
):
    device = torch.device(
        # "cuda" if torch.cuda.is_available() else
        # "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    len_state = 7
    ACTIONS = [
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (1, 1),
    (0, 2)]
    
    
    n_actions = len(ACTIONS)
    policy_net = NN(len_state, deepcopy(hidden_layers), n_actions)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

    
    if save:
        start_epoch = load_latest_model(policy_net, model_path)
    else:
        start_epoch = 0
    
    
    np.random.seed(42)
    
    rewards = []  # epoch -> avg_reward
    losses = []
    
    
    for epoch in range(start_epoch, epoch): 
        all_trajs = []
        for sample_id in range(num_samples):  # 每个 epoch 多个订单样本
            env = env_fn(seed=np.random.randint(1e6))  # 保证订单多样性
            results = Parallel(n_jobs=n_env)(
                delayed(rollout_one)(
                copy.deepcopy(env),
                ACTIONS,
                select_action,
                policy_net,
                gamma=GAMMA,
                device=device,
            )
                for _ in range(n_env)
            )
            for traj, reward in results:
                all_trajs.append((traj, reward))
        loss, baseline  = compute_loss(all_trajs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        rewards.append(baseline)
        losses.append(loss.item())
        
        # 在更新参数之后保存模型
        if save and (epoch + 1) % eval_interval == 0:
            save_path = results_dir / f"{model_path}_ep{epoch + 1}.pt"
            torch.save(policy_net.state_dict(), save_path)
            print(f"[Save] Model saved to {save_path}")
            
            x = list(range(1, len(rewards) + 1))  # 显式指定 epoch 编号从 1 到当前

            plt.figure()
            plt.plot(x, rewards, label="Average Reward")
            plt.plot(x, losses, label="Loss")
            plt.title("Training Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(results_dir / f"train_curve_ep{epoch + 1}.png")
            plt.close()



        print(f"[epoch {epoch}], loss: {loss:.4f}, average_reward: {baseline:.2f}")








