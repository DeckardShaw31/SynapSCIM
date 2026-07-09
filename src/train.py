import torch
import numpy as np
import os
import csv
import argparse
import time
import datetime
import matplotlib.pyplot as plt
from env import MultiEchelonSupplyChainEnv
from bdh import BDH_GPU
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

def get_device(device_arg="auto"):
    """Detects the fastest available hardware accelerator or uses the user-specified one."""
    if device_arg != "auto":
        print(f"Explicitly selecting user-specified device: {device_arg}")
        return torch.device(device_arg)
        
    # 1. Check Intel GPU (XPU)
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            device = torch.device("xpu")
            print("Intel GPU (XPU) detected via Intel Extension for PyTorch.")
            return device
    except ImportError:
        pass
        
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print("Intel GPU (XPU) detected.")
        return device

    # 2. Check NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        return device
        
    # 3. Check TPU (PyTorch XLA)
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

def train_synapscim(network_id=1, total_iterations=20000, rollout_steps=4000, T_context=10, save_path="bdh_ppo_model_20000.pt", device_arg="auto", resume=False, num_envs=64):
    print(f"Initializing SynapSCIM training on Willems Network {network_id}...")
    
    # 1. Load config and initialize environment
    config = get_willems_config(network_id)
    print(f"Using {num_envs} vectorized environments for parallel rollout collection.")
    envs = VectorSingleAgentEnv(lambda: MultiEchelonSupplyChainEnv(config), num_envs)
    
    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]
    
    print(f"Observation space dimension: {obs_dim}")
    print(f"Action space dimension: {act_dim}")
    
    # 2. Instantiate policy model (custom scaled-down BDH-GPU)
    device = get_device(device_arg)
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
            
    # 3. Instantiate PPO agent and buffers
    ppo_agent = PPOAgent(model, lr=1e-4)
    main_buffer = RolloutBuffer()
    env_buffers = [RolloutBuffer() for _ in range(num_envs)]
    
    # Setup metrics logging relative to save_path folder
    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)
    log_csv_path = os.path.join(save_dir, "training_log.csv")
    
    # Check for resume options
    start_iteration = 1
    if resume and os.path.exists(save_path):
        print(f"Resuming training from checkpoint: {save_path}")
        try:
            model.load_state_dict(torch.load(save_path, map_location=device))
            if os.path.exists(log_csv_path):
                with open(log_csv_path, "r", encoding="utf-8") as f:
                    reader = list(csv.reader(f))
                    if len(reader) > 1:
                        # Find the last logged iteration
                        start_iteration = int(reader[-1][0]) + 1
                        print(f"Resuming from iteration {start_iteration}")
        except Exception as e:
            print(f"Could not load checkpoint or log files: {e}. Starting from iteration 1.")
            start_iteration = 1
            
    # If not resuming or files do not exist, write CSV header
    if start_iteration == 1:
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
    
    # Load past logs into metrics history if resuming
    if start_iteration > 1 and os.path.exists(log_csv_path):
        try:
            with open(log_csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader) # skip header
                for row in reader:
                    if len(row) >= 6:
                        metrics_history["iteration"].append(int(row[0]))
                        metrics_history["mean_reward"].append(float(row[1]))
                        metrics_history["actor_loss"].append(float(row[2]))
                        metrics_history["critic_loss"].append(float(row[3]))
                        metrics_history["entropy"].append(float(row[4]))
                        metrics_history["fill_rate"].append(float(row[5]))
        except Exception as e:
            print(f"Error loading past CSV logs: {e}")
            
    # 4. Rollout execution parameters
    obs = envs.reset(seed=42)
    episode_obs = [[obs[i]] for i in range(num_envs)]
    current_episode_rewards = [0.0] * num_envs
    episode_rewards = []
    
    steps_per_env = rollout_steps // num_envs
    
    print("Starting rollout loop...")
    start_time = time.time()
    
    for iteration in range(start_iteration, total_iterations + 1):
        for b in env_buffers:
            b.clear()
            
        # Track total demand and unfilled demand for Type II Service Level (Fill Rate) during rollouts
        rollout_total_demand = 0.0
        rollout_unfilled_demand = 0.0
        
        # Collect rollout steps batched
        for step in range(steps_per_env):
            # Construct history segment for all envs
            hist_obs_list = []
            for i in range(num_envs):
                t = len(episode_obs[i]) - 1
                hist_obs_list.append(get_history(episode_obs[i], t, T_context))
                
            hist_obs_batch = np.stack(hist_obs_list)
            
            # Convert to PyTorch tensor and run forward pass
            hist_obs_t = torch.as_tensor(hist_obs_batch, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                action_mu, action_std, state_value = model(hist_obs_t)
                
            # Sample action
            dist = torch.distributions.Normal(action_mu, action_std)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t).sum(dim=-1)
            
            actions_unclipped = action_t.cpu().numpy()
            actions_clipped = np.clip(actions_unclipped, 0.0, 1.0)
            log_probs = log_prob_t.cpu().numpy()
            values = state_value.cpu().numpy().squeeze(-1)
            
            # Step the environments in parallel
            next_obs, rewards, terminateds, truncateds, infos = envs.step(actions_clipped)
            
            for i in range(num_envs):
                current_episode_rewards[i] += rewards[i]
                
                # Track demands and backorders for Fill Rate calculation
                demands = infos[i]["demands"]
                backorders = envs.envs[i].ret_backorders
                rollout_total_demand += np.sum(demands)
                rollout_unfilled_demand += np.sum(np.minimum(demands, backorders))
                
                # Store transition in respective env buffer
                env_buffers[i].hist_states.append(hist_obs_batch[i])
                env_buffers[i].actions.append(actions_unclipped[i])
                env_buffers[i].log_probs.append(log_probs[i])
                env_buffers[i].rewards.append(rewards[i])
                env_buffers[i].dones.append(terminateds[i])
                env_buffers[i].values.append(values[i])
                
                # Update episode observations list
                if terminateds[i] or truncateds[i]:
                    episode_rewards.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0.0
                    episode_obs[i] = [next_obs[i]]
                else:
                    episode_obs[i].append(next_obs[i])
                    
        # Compute GAE for each environment's trajectory
        for i in range(num_envs):
            last_t = len(episode_obs[i]) - 1
            last_hist = get_history(episode_obs[i], last_t, T_context)
            last_hist_t = torch.as_tensor(last_hist, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, final_value = model(last_hist_t)
            env_buffers[i].compute_gae(final_value.item())
            
        # Merge all environments' buffers into one main buffer
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
            
        # Run PPO update epochs (utilizing the GPU with batch sizes of 128)
        update_info = ppo_agent.update(main_buffer, batch_size=128, epochs=4)
        
        # Compute rollouts Fill Rate (Type II Service Level)
        fill_rate = max(0.0, 100.0 * (1.0 - (rollout_unfilled_demand / (rollout_total_demand + 1e-8))))
        
        # Logging progress
        mean_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) > 0 else 0.0
        
        # Compute ETA
        elapsed_time = time.time() - start_time
        avg_time_per_iter = elapsed_time / (iteration - start_iteration + 1)
        remaining_iters = total_iterations - iteration
        eta_seconds = int(avg_time_per_iter * remaining_iters)
        eta_str = str(datetime.timedelta(seconds=eta_seconds))
        
        print(f"Iteration {iteration:04d}/{total_iterations} | "
              f"Mean Reward (last 20 ep): {mean_reward:8.2f} | "
              f"Fill Rate: {fill_rate:6.2f}% | "
              f"Actor Loss: {update_info['actor_loss']:6.4f} | "
              f"Critic Loss: {update_info['critic_loss']:6.2f} | "
              f"Entropy: {update_info['entropy']:5.3f} | "
              f"ETA: {eta_str}")
              
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
        if iteration in [10000, 15000, 20000] or (iteration % 1000 == 0):
            # Unwrap compiled model state dict if compiled
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            checkpoint_path = os.path.join(save_dir, f"bdh_ppo_model_{iteration}.pt")
            torch.save(state, checkpoint_path)
            # Also save to default save_path so user can easily --resume from it
            torch.save(state, save_path)
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
    plot_path = os.path.join(save_dir, "training_progress.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training progress plot saved to {plot_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train centralized BDH-PPO supply chain controller.")
    parser.add_argument("--network_id", type=int, default=1, help="Willems network ID.")
    parser.add_argument("--total_iterations", type=int, default=20000, help="Number of training iterations.")
    parser.add_argument("--rollout_steps", type=int, default=4000, help="Steps collected per iteration.")
    parser.add_argument("--save_path", type=str, default="bdh_ppo_model_20000.pt", help="Filepath to save final model weights.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "xpu", "xla"], help="Specify hardware device.")
    parser.add_argument("--resume", action="store_true", help="Resume training from save_path checkpoint if it exists.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments for vectorized rollouts.")
    args = parser.parse_args()
    
    train_synapscim(
        network_id=args.network_id,
        total_iterations=args.total_iterations,
        rollout_steps=args.rollout_steps,
        save_path=args.save_path,
        device_arg=args.device,
        resume=args.resume,
        num_envs=args.num_envs
    )
