import torch
import numpy as np
import os
import csv
import argparse
import matplotlib.pyplot as plt
from env import MultiEchelonSupplyChainEnv
from bdh import BDH_GPU
from ppo import PPOAgent, RolloutBuffer, get_history
from willems_loader import get_willems_config

def get_device():
    """Detects the fastest available hardware accelerator (TPU, GPU, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        return device
    try:
        import torch_xla
        device = torch_xla.device()
        print("TPU accelerator detected via PyTorch XLA.")
        return device
    except (ImportError, AttributeError):
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print("TPU accelerator detected via PyTorch XLA.")
            return device
        except ImportError:
            pass
    return torch.device("cpu")

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
    device = get_device()
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
    
    # PyTorch 2.0+ Model Compilation for speedups (skip if on CPU or TPU to avoid Inductor exceptions)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            print("Compiling model for speed optimization...")
            model = torch.compile(model)
        except Exception as e:
            print(f"Skipping model compilation: {e}")
            
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
        writer.writerow(["iteration", "mean_reward", "actor_loss", "critic_loss", "entropy", "fill_rate"])
        
    metrics_history = {
        "iteration": [],
        "mean_reward": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "fill_rate": []
    }
    
    print("Starting rollout loop...")
    for iteration in range(1, total_iterations + 1):
        buffer.clear()
        
        # Track total demand and unfilled demand for Type II Service Level (Fill Rate) during rollouts
        rollout_total_demand = 0.0
        rollout_unfilled_demand = 0.0
        
        # Collect rollout steps
        for _ in range(rollout_steps):
            # Construct history segment
            t = len(episode_obs) - 1
            hist_obs = get_history(episode_obs, t, T_context)
            
            # Convert to PyTorch tensor and run forward pass (using as_tensor for performance)
            hist_obs_t = torch.as_tensor(hist_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
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
            
            # Track demands and backorders for Fill Rate calculation
            demands = info["demands"]
            backorders = env.ret_backorders
            rollout_total_demand += np.sum(demands)
            rollout_unfilled_demand += np.sum(np.minimum(demands, backorders))
            
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
        last_hist_t = torch.as_tensor(last_hist, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, final_value = model(last_hist_t)
            
        # Compute Generalized Advantage Estimation
        buffer.compute_gae(final_value.item())
        
        # Run PPO update epochs
        update_info = ppo_agent.update(buffer, batch_size=128, epochs=4)
        
        # Compute rollouts Fill Rate (Type II Service Level)
        fill_rate = max(0.0, 100.0 * (1.0 - (rollout_unfilled_demand / (rollout_total_demand + 1e-8))))
        
        # Logging progress
        mean_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) > 0 else 0.0
        print(f"Iteration {iteration:04d}/{total_iterations} | "
              f"Mean Reward (last 20 ep): {mean_reward:8.2f} | "
              f"Fill Rate: {fill_rate:6.2f}% | "
              f"Actor Loss: {update_info['actor_loss']:6.4f} | "
              f"Critic Loss: {update_info['critic_loss']:6.2f} | "
              f"Entropy: {update_info['entropy']:5.3f}")
              
        # Save to metrics history dict
        metrics_history["iteration"].append(iteration)
        metrics_history["mean_reward"].append(mean_reward)
        metrics_history["actor_loss"].append(update_info["actor_loss"])
        metrics_history["critic_loss"].append(update_info["critic_loss"])
        metrics_history["entropy"].append(update_info["entropy"])
        metrics_history["fill_rate"].append(fill_rate)
        
        # Append to CSV log file
        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, mean_reward, update_info["actor_loss"], update_info["critic_loss"], update_info["entropy"], fill_rate])
            
        # Save model checkpoints at specific milestones
        if iteration in [10000, 15000, 20000]:
            # Unwrap compiled model state dict if compiled
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            checkpoint_path = f"bdh_ppo_model_{iteration}.pt"
            torch.save(state, checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")
        
    # Save the final trained model
    state_final = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_final, save_path)
    print(f"Model successfully saved to {save_path}")
    
    # Generate and save training progress plot
    print("Generating training progress plot...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    
    # Plot Mean Reward
    axes[0, 0].plot(metrics_history["iteration"], metrics_history["mean_reward"], color="#1f77b4", linewidth=2)
    axes[0, 0].set_title("Episode Mean Reward", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Iteration", fontsize=10)
    axes[0, 0].set_ylabel("Reward (negative cost)", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    
    # Plot Fill Rate (Service Level)
    axes[0, 1].plot(metrics_history["iteration"], metrics_history["fill_rate"], color="#2ca02c", linewidth=2)
    axes[0, 1].set_title("Type II Service Level (Fill Rate %)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Iteration", fontsize=10)
    axes[0, 1].set_ylabel("Fill Rate (%)", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    
    # Plot Critic Loss
    axes[1, 0].plot(metrics_history["iteration"], metrics_history["critic_loss"], color="#ff7f0e", linewidth=2)
    axes[1, 0].set_title("Critic Value Loss (MSE)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Iteration", fontsize=10)
    axes[1, 0].set_ylabel("Loss Value", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    
    # Plot Actor Loss
    axes[1, 1].plot(metrics_history["iteration"], metrics_history["actor_loss"], color="#d62728", linewidth=2)
    axes[1, 1].set_title("Actor Policy Loss", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Iteration", fontsize=10)
    axes[1, 1].set_ylabel("Loss Value", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    
    # Plot Entropy
    axes[2, 0].plot(metrics_history["iteration"], metrics_history["entropy"], color="#9467bd", linewidth=2)
    axes[2, 0].set_title("Policy Entropy", fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel("Iteration", fontsize=10)
    axes[2, 0].set_ylabel("Entropy Value", fontsize=10)
    axes[2, 0].grid(True, linestyle="--", alpha=0.6)
    
    # Disable the 6th axis panel
    axes[2, 1].axis("off")
    
    plt.tight_layout()
    plot_path = "reports/centralized_ppo/training_progress.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training progress plot saved to {plot_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train centralized BDH-PPO supply chain controller.")
    parser.add_argument("--network_id", type=int, default=1, help="Willems network ID.")
    parser.add_argument("--total_iterations", type=int, default=3000, help="Number of training iterations.")
    parser.add_argument("--rollout_steps", type=int, default=4000, help="Steps collected per iteration.")
    parser.add_argument("--save_path", type=str, default="bdh_ppo_model_3000.pt", help="Filepath to save final model weights.")
    args = parser.parse_args()
    
    train_synapscim(
        network_id=args.network_id,
        total_iterations=args.total_iterations,
        rollout_steps=args.rollout_steps,
        save_path=args.save_path
    )
