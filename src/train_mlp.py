import torch
import numpy as np
import os
import csv
import argparse
import time
import datetime
from env import MultiEchelonSupplyChainEnv
from bdh import MLP_GPU
from ppo import PPOAgent, RolloutBuffer, get_history
from willems_loader import get_willems_config

class VectorSingleAgentEnv:
    def __init__(self, make_env_fn, num_envs):
        self.envs = [make_env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
    def reset(self, seed=42):
        obs_list = [self.envs[i].reset(seed=seed+i)[0] for i in range(self.num_envs)]
        return np.stack(obs_list)
        
    def step(self, actions):
        obs_list, reward_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i in range(self.num_envs):
            obs, reward, terminated, truncated, info = self.envs[i].step(actions[i])
            if terminated or truncated:
                reset_obs, _ = self.envs[i].reset()
                obs_list.append(reset_obs)
            else:
                obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(terminated)
            trunc_list.append(truncated)
            info_list.append(info)
            
        return (
            np.stack(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(term_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            info_list
        )

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_iterations", type=int, default=5000)
    args = parser.parse_args()

    config = get_willems_config(1)
    
    # Enable centralized mode
    env_fn = lambda: MultiEchelonSupplyChainEnv(config, mode="centralized")
    envs = VectorSingleAgentEnv(env_fn, num_envs=64)
    
    device = get_device()
    print("Using device:", device)
    
    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]
    
    model = MLP_GPU(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=128).to(device)
    ppo_agent = PPOAgent(model, lr=1e-4)
    
    save_dir = "SynapSCIM_mlp_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    log_csv_path = os.path.join(save_dir, "training_log.csv")
    
    # Write header if new log
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "mean_reward", "actor_loss", "critic_loss", "entropy", "fill_rate"])
            
    num_envs = envs.num_envs
    env_buffers = [RolloutBuffer() for _ in range(num_envs)]
    main_buffer = RolloutBuffer()
    
    obs_batch = envs.reset(seed=42)
    episode_obs = [[obs_batch[i]] for i in range(num_envs)]
    current_episode_rewards = np.zeros(num_envs)
    episode_rewards = []
    
    T_context = 10
    steps_per_rollout = 31
    
    start_time = time.time()
    
    for iteration in range(1, args.total_iterations + 1):
        for buffer in env_buffers:
            buffer.clear()
            
        rollout_total_demand = 0.0
        rollout_unfilled_demand = 0.0
        
        for step in range(steps_per_rollout):
            hist_obs_list = []
            for i in range(num_envs):
                t = len(episode_obs[i]) - 1
                hist_obs_list.append(get_history(episode_obs[i], t, T_context))
            hist_obs_batch = np.array(hist_obs_list, dtype=np.float32)
            
            hist_obs_batch_t = torch.tensor(hist_obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_mu, action_std, state_value = model(hist_obs_batch_t)
                
            dist = torch.distributions.Normal(action_mu, action_std)
            act_t = dist.sample()
            log_probs = dist.log_prob(act_t).sum(dim=-1).cpu().numpy()
            
            actions_unclipped = act_t.cpu().numpy()
            actions_clipped = np.clip(actions_unclipped, 0.0, 1.0)
            values = state_value.cpu().numpy().squeeze(-1)
            
            next_obs, rewards, terminateds, truncateds, infos = envs.step(actions_clipped)
            
            for i in range(num_envs):
                current_episode_rewards[i] += rewards[i]
                
                demands = infos[i]["demands"]
                backorders = envs.envs[i].ret_backorders
                rollout_total_demand += np.sum(demands)
                rollout_unfilled_demand += np.sum(np.minimum(demands, backorders))
                
                env_buffers[i].hist_states.append(hist_obs_batch[i])
                env_buffers[i].actions.append(actions_unclipped[i])
                env_buffers[i].log_probs.append(log_probs[i])
                env_buffers[i].rewards.append(rewards[i])
                env_buffers[i].dones.append(terminateds[i])
                env_buffers[i].values.append(values[i])
                
                if terminateds[i] or truncateds[i]:
                    episode_rewards.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0.0
                    episode_obs[i] = [next_obs[i]]
                else:
                    episode_obs[i].append(next_obs[i])
                    
        # Compute GAE
        for i in range(num_envs):
            last_t = len(episode_obs[i]) - 1
            last_hist = get_history(episode_obs[i], last_t, T_context)
            last_hist_t = torch.as_tensor(last_hist, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, final_value = model(last_hist_t)
            env_buffers[i].compute_gae(final_value.item())
            
        main_buffer.clear()
        for i in range(num_envs):
            main_buffer.hist_states.extend(env_buffers[i].hist_states)
            main_buffer.actions.extend(env_buffers[i].actions)
            main_buffer.log_probs.extend(env_buffers[i].log_probs)
            main_buffer.rewards.extend(env_buffers[i].rewards)
            main_buffer.dones.extend(env_buffers[i].dones)
            main_buffer.values.extend(env_buffers[i].values)
            main_buffer.advantages.extend(env_buffers[i].advantages)
            main_buffer.value_targets.extend(env_buffers[i].value_targets)
            
        update_info = ppo_agent.update(main_buffer, batch_size=128, epochs=4)
        
        fill_rate = max(0.0, 100.0 * (1.0 - (rollout_unfilled_demand / (rollout_total_demand + 1e-8))))
        mean_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) > 0 else 0.0
        
        if iteration % 100 == 0 or iteration == 1:
            print(f"Iteration {iteration:04d}/{args.total_iterations} | "
                  f"Mean Reward: {mean_reward:8.2f} | "
                  f"Fill Rate: {fill_rate:6.2f}%")
                  
        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, mean_reward, update_info["actor_loss"], update_info["critic_loss"], update_info["entropy"], fill_rate])
            
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "mlp_ppo_model_final.pt"))
    print("Training finished! Final model saved.")

if __name__ == "__main__":
    main()
