import torch
import numpy as np
import os
import csv
import argparse
import matplotlib.pyplot as plt
from env import MultiEchelonSupplyChainEnv
from bdh import BDH_GPU
from ppo import MultiAgentPPOAgent, RolloutBuffer, get_history
from willems_loader import get_willems_config

# Helper to pad observations
def pad_obs(obs, target_dim):
    if len(obs) < target_dim:
        return np.pad(obs, (0, target_dim - len(obs)), mode='constant')
    return obs[:target_dim]

def train_mappo(network_id=1, total_iterations=1000, rollout_steps=2000, T_context=5, save_path_wh="bdh_mappo_wh.pt", save_path_ret="bdh_mappo_ret.pt"):
    print(f"Initializing Decentralized MAPPO training on Willems Network {network_id}...")
    
    # 1. Load config and initialize multi-agent environment
    config = get_willems_config(network_id)
    env = MultiEchelonSupplyChainEnv(config, mode="multi_agent")
    
    wh_obs_dim = env.observation_spaces["warehouse"].shape[0]
    max_ret_obs_dim = max(env.observation_spaces[f"retailer_{i}"].shape[0] for i in range(env.num_retailers))
    
    print(f"Warehouse obs dim: {wh_obs_dim}")
    print(f"Retailer max obs dim: {max_ret_obs_dim}")
    
    # 2. Instantiate networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_wh = BDH_GPU(obs_dim=wh_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
    model_ret = BDH_GPU(obs_dim=max_ret_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
    
    mappo_agent = MultiAgentPPOAgent(model_wh, model_ret, lr=1e-4)
    buffer_wh = RolloutBuffer()
    buffer_ret = RolloutBuffer()
    
    # 3. Setup metrics logging
    os.makedirs("reports/decentralized_mappo", exist_ok=True)
    log_csv_path = "reports/decentralized_mappo/training_log.csv"
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "joint_reward", "wh_actor_loss", "wh_critic_loss", "ret_actor_loss", "ret_critic_loss", "fill_rate"])
        
    metrics_history = {
        "iteration": [],
        "joint_reward": [],
        "wh_actor_loss": [],
        "wh_critic_loss": [],
        "ret_actor_loss": [],
        "ret_critic_loss": [],
        "fill_rate": []
    }
    
    # 4. Rollout parameter initializations
    obs_dict, _ = env.reset(seed=42)
    hidden_wh = model_wh.init_recurrent_states(1, device)
    hidden_rets = [model_ret.init_recurrent_states(1, device) for _ in range(env.num_retailers)]
    
    wh_history = [obs_dict["warehouse"]]
    ret_histories = [[pad_obs(obs_dict[f"retailer_{i}"], max_ret_obs_dim)] for i in range(env.num_retailers)]
    
    current_step = 0
    
    print("Starting rollout loop...")
    for iteration in range(1, total_iterations + 1):
        buffer_wh.clear()
        buffer_ret.clear()
        
        rollout_total_demand = 0.0
        rollout_unfilled_demand = 0.0
        
        # Collect rollout steps
        for _ in range(rollout_steps):
            # Warehouse forward recurrent pass
            obs_wh_t = torch.tensor(obs_dict["warehouse"], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                mu_wh, std_wh, val_wh, next_hidden_wh = model_wh.forward_recurrent(obs_wh_t, current_step, hidden_wh)
                
            dist_wh = torch.distributions.Normal(mu_wh, std_wh)
            act_wh_t = dist_wh.sample()
            log_prob_wh = dist_wh.log_prob(act_wh_t).sum(dim=-1).item()
            action_wh_unclipped = act_wh_t.cpu().numpy()[0]
            action_wh_clipped = np.clip(action_wh_unclipped, 0.0, 1.0)
            
            # Retailers forward recurrent pass
            action_dict = {"warehouse": action_wh_clipped}
            retailer_vals = []
            retailer_log_probs = []
            retailer_actions_unclipped = []
            next_hidden_rets = []
            
            for i in range(env.num_retailers):
                padded = pad_obs(obs_dict[f"retailer_{i}"], max_ret_obs_dim)
                obs_ret_t = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    mu_ret, std_ret, val_ret, next_hidden_ret = model_ret.forward_recurrent(obs_ret_t, current_step, hidden_rets[i])
                    
                dist_ret = torch.distributions.Normal(mu_ret, std_ret)
                act_ret_t = dist_ret.sample()
                log_prob_ret = dist_ret.log_prob(act_ret_t).sum(dim=-1).item()
                action_ret_unclipped = act_ret_t.cpu().numpy()[0]
                action_ret_clipped = np.clip(action_ret_unclipped, 0.0, 1.0)
                
                action_dict[f"retailer_{i}"] = action_ret_clipped
                retailer_vals.append(val_ret.item())
                retailer_log_probs.append(log_prob_ret)
                retailer_actions_unclipped.append(action_ret_unclipped)
                next_hidden_rets.append(next_hidden_ret)
                
            # Step Multi-Agent Environment
            next_obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict = env.step(action_dict)
            
            # Track demands and backorders for Fill Rate calculation
            info = infos_dict["warehouse"]
            demands = info["demands"]
            backorders = env.ret_backorders
            rollout_total_demand += np.sum(demands)
            rollout_unfilled_demand += np.sum(np.minimum(demands, backorders))
            
            # Store warehouse transition
            hist_wh = get_history(wh_history, len(wh_history) - 1, T_context)
            buffer_wh.hist_states.append(hist_wh)
            buffer_wh.actions.append(action_wh_unclipped)
            buffer_wh.log_probs.append(log_prob_wh)
            buffer_wh.rewards.append(rewards_dict["warehouse"])
            buffer_wh.dones.append(terminations_dict["warehouse"])
            buffer_wh.values.append(val_wh.item())
            
            # Store retailer transitions (Parameter-Sharing)
            for i in range(env.num_retailers):
                hist_ret = get_history(ret_histories[i], len(ret_histories[i]) - 1, T_context)
                buffer_ret.hist_states.append(hist_ret)
                buffer_ret.actions.append([retailer_actions_unclipped[i]])
                buffer_ret.log_probs.append(retailer_log_probs[i])
                buffer_ret.rewards.append(rewards_dict[f"retailer_{i}"])
                buffer_ret.dones.append(terminations_dict[f"retailer_{i}"])
                buffer_ret.values.append(retailer_vals[i])
                
            # Append next step observations to history lists
            wh_history.append(next_obs_dict["warehouse"])
            for i in range(env.num_retailers):
                ret_histories[i].append(pad_obs(next_obs_dict[f"retailer_{i}"], max_ret_obs_dim))
                
            obs_dict = next_obs_dict
            hidden_wh = next_hidden_wh
            hidden_rets = next_hidden_rets
            current_step += 1
            
            if terminations_dict["warehouse"] or truncations_dict["warehouse"]:
                obs_dict, _ = env.reset()
                hidden_wh = model_wh.init_recurrent_states(1, device)
                hidden_rets = [model_ret.init_recurrent_states(1, device) for _ in range(env.num_retailers)]
                wh_history = [obs_dict["warehouse"]]
                ret_histories = [[pad_obs(obs_dict[f"retailer_{i}"], max_ret_obs_dim)] for i in range(env.num_retailers)]
                current_step = 0
                
        # Compute GAE for Warehouse
        obs_wh_last = torch.tensor(obs_dict["warehouse"], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, final_val_wh, _ = model_wh.forward_recurrent(obs_wh_last, current_step, hidden_wh)
        buffer_wh.compute_gae(final_val_wh.item())
        
        # Compute GAE for Retailers
        final_val_rets = []
        for i in range(env.num_retailers):
            padded_last = pad_obs(obs_dict[f"retailer_{i}"], max_ret_obs_dim)
            obs_ret_last = torch.tensor(padded_last, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, _, final_val_ret, _ = model_ret.forward_recurrent(obs_ret_last, current_step, hidden_rets[i])
            final_val_rets.append(final_val_ret.item())
        buffer_ret.compute_gae(float(np.mean(final_val_rets)))
        
        # PPO Update (parallel backprop)
        update_info = mappo_agent.update(buffer_wh, buffer_ret, batch_size=128, epochs=3)
        
        # Compute rollout Fill Rate
        fill_rate = max(0.0, 100.0 * (1.0 - (rollout_unfilled_demand / (rollout_total_demand + 1e-8))))
        mean_reward = np.mean(buffer_wh.rewards)
        
        # Logging progress
        print(f"Iteration {iteration:04d}/{total_iterations} | "
              f"Joint Reward: {mean_reward:8.4f} | "
              f"Fill Rate: {fill_rate:6.2f}% | "
              f"WH Actor Loss: {update_info['wh_actor_loss']:6.4f} | "
              f"Ret Actor Loss: {update_info['ret_actor_loss']:6.4f}")
              
        # Save metrics history
        metrics_history["iteration"].append(iteration)
        metrics_history["joint_reward"].append(mean_reward)
        metrics_history["wh_actor_loss"].append(update_info["wh_actor_loss"])
        metrics_history["wh_critic_loss"].append(update_info["wh_critic_loss"])
        metrics_history["ret_actor_loss"].append(update_info["ret_actor_loss"])
        metrics_history["ret_critic_loss"].append(update_info["ret_critic_loss"])
        metrics_history["fill_rate"].append(fill_rate)
        
        # Append to CSV log
        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, mean_reward, 
                update_info["wh_actor_loss"], update_info["wh_critic_loss"], 
                update_info["ret_actor_loss"], update_info["ret_critic_loss"], 
                fill_rate
            ])
            
        # Checkpoints saving
        if iteration in [10000, 15000, 20000]:
            checkpoint_wh = f"bdh_mappo_wh_{iteration}.pt"
            checkpoint_ret = f"bdh_mappo_ret_{iteration}.pt"
            torch.save(model_wh.state_dict(), checkpoint_wh)
            torch.save(model_ret.state_dict(), checkpoint_ret)
            print(f"MAPPO checkpoints saved at iteration {iteration}")
            
    # Save final model
    torch.save(model_wh.state_dict(), save_path_wh)
    torch.save(model_ret.state_dict(), save_path_ret)
    print(f"Final MAPPO models saved to {save_path_wh} and {save_path_ret}")
    
    # Generate and save training progress plot
    print("Generating training progress plot...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    
    # Plot Mean Reward
    axes[0, 0].plot(metrics_history["iteration"], metrics_history["joint_reward"], color="#1f77b4", linewidth=2)
    axes[0, 0].set_title("Joint Episode Mean Reward", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Iteration", fontsize=10)
    axes[0, 0].set_ylabel("Reward (negative cost)", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    
    # Plot Fill Rate
    axes[0, 1].plot(metrics_history["iteration"], metrics_history["fill_rate"], color="#2ca02c", linewidth=2)
    axes[0, 1].set_title("Type II Service Level (Fill Rate %)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Iteration", fontsize=10)
    axes[0, 1].set_ylabel("Fill Rate (%)", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    
    # Plot WH Losses
    axes[1, 0].plot(metrics_history["iteration"], metrics_history["wh_actor_loss"], label="Actor Loss", color="#ff7f0e", linewidth=1.5)
    axes[1, 0].plot(metrics_history["iteration"], metrics_history["wh_critic_loss"], label="Critic Loss (x0.1)", color="#d62728", linewidth=1.5, alpha=0.7)
    axes[1, 0].set_title("Warehouse Agent Losses", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Iteration", fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    
    # Plot Ret Losses
    axes[1, 1].plot(metrics_history["iteration"], metrics_history["ret_actor_loss"], label="Actor Loss", color="#1f77b4", linewidth=1.5)
    axes[1, 1].plot(metrics_history["iteration"], metrics_history["ret_critic_loss"], label="Critic Loss (x0.1)", color="#9467bd", linewidth=1.5, alpha=0.7)
    axes[1, 1].set_title("Retailer Agent Losses (Shared)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Iteration", fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    
    # Empty panels for grid formatting
    axes[2, 0].axis("off")
    axes[2, 1].axis("off")
    
    plt.tight_layout()
    plot_path = "reports/decentralized_mappo/training_progress.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training progress plot saved to {plot_path}")
    
    return model_wh, model_ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train decentralized MAPPO cooperative supply chain controller.")
    parser.add_argument("--network_id", type=int, default=1, help="Willems network ID.")
    parser.add_argument("--total_iterations", type=int, default=1000, help="Number of training iterations.")
    parser.add_argument("--rollout_steps", type=int, default=2000, help="Steps collected per iteration.")
    parser.add_argument("--save_path_wh", type=str, default="bdh_mappo_wh.pt", help="Filepath to save warehouse model.")
    parser.add_argument("--save_path_ret", type=str, default="bdh_mappo_ret.pt", help="Filepath to save retailer model.")
    args = parser.parse_args()
    
    train_mappo(
        network_id=args.network_id,
        total_iterations=args.total_iterations,
        rollout_steps=args.rollout_steps,
        save_path_wh=args.save_path_wh,
        save_path_ret=args.save_path_ret
    )
