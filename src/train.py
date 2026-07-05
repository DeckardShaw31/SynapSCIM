import torch
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from env import MultiEchelonSupplyChainEnv
from bdh import BDH_GPU
from ppo import PPOAgent, RolloutBuffer, get_history
from willems_loader import get_willems_config

def train_synapscim(network_id=1, total_iterations=1000, rollout_steps=4000, T_context=10, save_path="bdh_ppo_model_3000.pt"):
    print(f"Initializing SynapSCIM training on Willems Network {network_id}...")
    
    # 1. Load config and initialize environment
    config = get_willems_config(network_id)
    env = MultiEchelonSupplyChainEnv(config)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"Observation space dimension: {obs_dim}")
    print(f"Action space dimension: {act_dim}")
    
    # 2. Instantiate policy model (custom scaled-down BDH-GPU)
    # Using small parameters for efficient RL training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = BDH_GPU(
        obs_dim=obs_dim,
        act_dim=act_dim,
        D=32,       # Embedding dimension
        H=2,        # Attention heads
        N=256,      # Total neurons
        L=2,        # Layers
        dropout=0.05
    ).to(device)
    
    # 3. Instantiate PPO agent and Rollout buffer
    ppo_agent = PPOAgent(model, lr=1e-4)
    buffer = RolloutBuffer()
    
    # 4. Rollout execution parameters
    episode_obs = []
    obs, _ = env.reset(seed=42)
    episode_obs.append(obs)
    
    episode_rewards = []
    current_episode_reward = 0.0
    
    # Setup metrics logging
    os.makedirs("reports/centralized_ppo", exist_ok=True)
    log_csv_path = "reports/centralized_ppo/training_log.csv"
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "mean_reward", "actor_loss", "critic_loss", "entropy"])
        
    metrics_history = {
        "iteration": [],
        "mean_reward": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": []
    }
    
    print("Starting rollout loop...")
    for iteration in range(1, total_iterations + 1):
        buffer.clear()
        
        # Collect rollout steps
        for _ in range(rollout_steps):
            # Construct history segment
            t = len(episode_obs) - 1
            hist_obs = get_history(episode_obs, t, T_context)
            
            # Convert to PyTorch tensor and run forward pass
            hist_obs_t = torch.tensor(hist_obs, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dim: [1, T_context, obs_dim]
            
            with torch.no_grad():
                action_mu, action_std, state_value = model(hist_obs_t)
                
            # Sample action
            dist = torch.distributions.Normal(action_mu, action_std)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t).sum(dim=-1)
            
            action_unclipped = action_t.cpu().numpy()[0]
            action_clipped = np.clip(action_unclipped, 0.0, 1.0)
            log_prob = log_prob_t.item()
            val = state_value.item()
            
            # Step the environment
            next_obs, reward, terminated, truncated, info = env.step(action_clipped)
            current_episode_reward += reward
            
            # Store transition in buffer (storing unclipped actions)
            buffer.hist_states.append(hist_obs)
            buffer.actions.append(action_unclipped)
            buffer.log_probs.append(log_prob)
            buffer.rewards.append(reward)
            buffer.dones.append(terminated)
            buffer.values.append(val)
            
            # Update episode observations list
            episode_obs.append(next_obs)
            
            if terminated or truncated:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                
                # Reset environment and episode obs history
                obs, _ = env.reset()
                episode_obs = [obs]
                
        # Get value for the final step to compute GAE bootstrapping
        last_t = len(episode_obs) - 1
        last_hist = get_history(episode_obs, last_t, T_context)
        last_hist_t = torch.tensor(last_hist, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, final_value = model(last_hist_t)
            
        # Compute Generalized Advantage Estimation
        buffer.compute_gae(final_value.item())
        
        # Run PPO update epochs
        update_info = ppo_agent.update(buffer, batch_size=128, epochs=4)
        
        # Logging progress
        mean_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) > 0 else 0.0
        print(f"Iteration {iteration:03d}/{total_iterations} | "
              f"Mean Reward (last 20 ep): {mean_reward:8.2f} | "
              f"Actor Loss: {update_info['actor_loss']:6.4f} | "
              f"Critic Loss: {update_info['critic_loss']:6.2f} | "
              f"Entropy: {update_info['entropy']:5.3f}")
              
        # Save to metrics history dict
        metrics_history["iteration"].append(iteration)
        metrics_history["mean_reward"].append(mean_reward)
        metrics_history["actor_loss"].append(update_info["actor_loss"])
        metrics_history["critic_loss"].append(update_info["critic_loss"])
        metrics_history["entropy"].append(update_info["entropy"])
        
        # Append to CSV log file
        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, mean_reward, update_info["actor_loss"], update_info["critic_loss"], update_info["entropy"]])
        
    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to {save_path}")
    
    # Generate and save training progress plot
    print("Generating training progress plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot Mean Reward
    axes[0, 0].plot(metrics_history["iteration"], metrics_history["mean_reward"], color="#1f77b4", linewidth=2)
    axes[0, 0].set_title("Episode Mean Reward", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Iteration", fontsize=10)
    axes[0, 0].set_ylabel("Reward (negative cost)", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    
    # Plot Critic Loss
    axes[0, 1].plot(metrics_history["iteration"], metrics_history["critic_loss"], color="#ff7f0e", linewidth=2)
    axes[0, 1].set_title("Critic Value Loss (MSE)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Iteration", fontsize=10)
    axes[0, 1].set_ylabel("Loss Value", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    
    # Plot Actor Loss
    axes[1, 0].plot(metrics_history["iteration"], metrics_history["actor_loss"], color="#2ca02c", linewidth=2)
    axes[1, 0].set_title("Actor Policy Loss", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Iteration", fontsize=10)
    axes[1, 0].set_ylabel("Loss Value", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    
    # Plot Entropy
    axes[1, 1].plot(metrics_history["iteration"], metrics_history["entropy"], color="#d62728", linewidth=2)
    axes[1, 1].set_title("Policy Entropy", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Iteration", fontsize=10)
    axes[1, 1].set_ylabel("Entropy Value", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plot_path = "reports/centralized_ppo/training_progress.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training progress plot saved to {plot_path}")
    
    return model

if __name__ == "__main__":
    train_synapscim(network_id=1, total_iterations=3000)
