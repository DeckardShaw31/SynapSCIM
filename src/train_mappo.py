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
from ppo import MultiAgentPPOAgent, RolloutBuffer, get_history
from willems_loader import get_willems_config

# Helper to pad observations
def pad_obs(obs, target_dim):
    if len(obs) < target_dim:
        return np.pad(obs, (0, target_dim - len(obs)), mode='constant')
    return obs[:target_dim]

class VectorMultiAgentEnv:
    def __init__(self, make_env_fn, num_envs):
        self.envs = [make_env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.num_retailers = self.envs[0].num_retailers
        self.observation_spaces = self.envs[0].observation_spaces
        
    def reset(self, seed=42):
        obs_dicts = [self.envs[i].reset(seed=seed+i)[0] for i in range(self.num_envs)]
        batched_obs = {}
        for key in obs_dicts[0].keys():
            batched_obs[key] = np.stack([obs_dicts[i][key] for i in range(self.num_envs)])
        return batched_obs
        
    def step(self, actions_dict):
        obs_list, reward_list, term_list, trunc_list, info_list = [], [], [], [], []
        
        for i in range(self.num_envs):
            env_actions = {}
            for key in actions_dict.keys():
                env_actions[key] = actions_dict[key][i]
                
            obs, rewards, terminated, truncated, info = self.envs[i].step(env_actions)
            
            if terminated or truncated:
                reset_obs, _ = self.envs[i].reset()
                obs_list.append(reset_obs)
            else:
                obs_list.append(obs)
                
            reward_list.append(rewards)
            term_list.append(terminated)
            trunc_list.append(truncated)
            info_list.append(info)
            
        batched_obs = {}
        for key in obs_list[0].keys():
            batched_obs[key] = np.stack([obs_list[i][key] for i in range(self.num_envs)])
            
        batched_rewards = {}
        for key in reward_list[0].keys():
            batched_rewards[key] = np.array([reward_list[i][key] for i in range(self.num_envs)], dtype=np.float32)
            
        batched_terms = {}
        for key in term_list[0].keys():
            batched_terms[key] = np.array([term_list[i][key] for i in range(self.num_envs)], dtype=bool)
            
        batched_truncs = {}
        for key in trunc_list[0].keys():
            batched_truncs[key] = np.array([trunc_list[i][key] for i in range(self.num_envs)], dtype=bool)
            
        return batched_obs, batched_rewards, batched_terms, batched_truncs, info_list

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

def train_mappo(network_id=1, total_iterations=20000, rollout_steps=2000, T_context=5, save_path_wh="bdh_mappo_wh_20000.pt", save_path_ret="bdh_mappo_ret_20000.pt", device_arg="auto", resume=False, num_envs=64):
    print(f"Initializing Decentralized MAPPO training on Willems Network {network_id}...")
    
    # 1. Load config and initialize multi-agent environment
    config = get_willems_config(network_id)
    print(f"Using {num_envs} vectorized environments for parallel rollout collection.")
    envs = VectorMultiAgentEnv(lambda: MultiEchelonSupplyChainEnv(config, mode="multi_agent"), num_envs)
    
    wh_obs_dim = envs.observation_spaces["warehouse"].shape[0]
    max_ret_obs_dim = max(envs.observation_spaces[f"retailer_{i}"].shape[0] for i in range(envs.num_retailers))
    
    print(f"Warehouse obs dim: {wh_obs_dim}")
    print(f"Retailer max obs dim: {max_ret_obs_dim}")
    
    # 2. Instantiate networks on the fastest available device
    device = get_device(device_arg)
    print(f"Using device: {device}")
    model_wh = BDH_GPU(obs_dim=wh_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
    model_ret = BDH_GPU(obs_dim=max_ret_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
    
    # PyTorch 2.0+ Model Compilation for speedups (skip if on CPU or TPU to avoid Inductor exceptions)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            print("Compiling models for speed optimization...")
            model_wh = torch.compile(model_wh)
            model_ret = torch.compile(model_ret)
        except Exception as e:
            print(f"Skipping model compilation: {e}")
            
    mappo_agent = MultiAgentPPOAgent(model_wh, model_ret, lr=1e-4)
    main_buffer_wh = RolloutBuffer()
    main_buffer_ret = RolloutBuffer()
    env_buffers_wh = [RolloutBuffer() for _ in range(num_envs)]
    env_buffers_ret = [RolloutBuffer() for _ in range(num_envs)]
    
    # 3. Setup metrics logging relative to save_path_wh folder
    save_dir = os.path.dirname(os.path.abspath(save_path_wh))
    os.makedirs(save_dir, exist_ok=True)
    log_csv_path = os.path.join(save_dir, "training_log.csv")
    
    start_iteration = 1
    if resume and os.path.exists(save_path_wh) and os.path.exists(save_path_ret):
        print(f"Resuming training from checkpoints: {save_path_wh} & {save_path_ret}")
        try:
            state_dict_wh = torch.load(save_path_wh, map_location=device)
            state_dict_ret = torch.load(save_path_ret, map_location=device)
            
            # Automatically strip _orig_mod. prefix from compiled checkpoints
            from collections import OrderedDict
            new_state_wh = OrderedDict()
            for k, v in state_dict_wh.items():
                if k.startswith("_orig_mod."):
                    new_state_wh[k[10:]] = v
                else:
                    new_state_wh[k] = v
                    
            new_state_ret = OrderedDict()
            for k, v in state_dict_ret.items():
                if k.startswith("_orig_mod."):
                    new_state_ret[k[10:]] = v
                else:
                    new_state_ret[k] = v
                    
            model_wh.load_state_dict(new_state_wh)
            model_ret.load_state_dict(new_state_ret)
            if os.path.exists(log_csv_path):
                with open(log_csv_path, "r", encoding="utf-8") as f:
                    reader = list(csv.reader(f))
                    if len(reader) > 1:
                        start_iteration = int(reader[-1][0]) + 1
                        print(f"Resuming from iteration {start_iteration}")
            else:
                # Fallback: scan save_dir for bdh_mappo_wh_X.pt milestone files
                import glob
                import re
                milestones = glob.glob(os.path.join(save_dir, "bdh_mappo_wh_*.pt"))
                iters = []
                for m in milestones:
                    match = re.search(r"bdh_mappo_wh_(\d+)\.pt", os.path.basename(m))
                    if match:
                        iters.append(int(match.group(1)))
                if len(iters) > 0:
                    start_iteration = max(iters) + 1
                    print(f"Log CSV not found, but found milestone checkpoints. Resuming from iteration {start_iteration}")
        except Exception as e:
            print(f"Could not load checkpoint or log files: {e}. Starting from iteration 1.")
            start_iteration = 1
            
    # Write CSV header if starting fresh
    if start_iteration == 1:
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
    
    # Load past logs if resuming
    if start_iteration > 1 and os.path.exists(log_csv_path):
        try:
            with open(log_csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 7:
                        metrics_history["iteration"].append(int(row[0]))
                        metrics_history["joint_reward"].append(float(row[1]))
                        metrics_history["wh_actor_loss"].append(float(row[2]))
                        metrics_history["wh_critic_loss"].append(float(row[3]))
                        metrics_history["ret_actor_loss"].append(float(row[4]))
                        metrics_history["ret_critic_loss"].append(float(row[5]))
                        metrics_history["fill_rate"].append(float(row[6]))
        except Exception as e:
            print(f"Error loading past CSV logs: {e}")
    
    # 4. Rollout parameter initializations
    obs_dict = envs.reset(seed=42)
    hidden_wh = model_wh.init_recurrent_states(num_envs, device)
    hidden_rets = model_ret.init_recurrent_states(num_envs * envs.num_retailers, device)
    
    wh_histories = [[obs_dict["warehouse"][i]] for i in range(num_envs)]
    ret_histories = [[[pad_obs(obs_dict[f"retailer_{j}"][i], max_ret_obs_dim)] for i in range(num_envs)] for j in range(envs.num_retailers)]
    
    current_step = 0
    steps_per_env = rollout_steps // num_envs
    
    print("Starting rollout loop...")
    start_time = time.time()
    
    for iteration in range(start_iteration, total_iterations + 1):
        for b in env_buffers_wh:
            b.clear()
        for b in env_buffers_ret:
            b.clear()
        
        rollout_total_demand = 0.0
        rollout_unfilled_demand = 0.0
        
        # Collect rollout steps
        for step in range(steps_per_env):
            # Warehouse forward recurrent pass
            obs_wh_batch = obs_dict["warehouse"]
            obs_wh_t = torch.as_tensor(obs_wh_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                mu_wh, std_wh, val_wh, next_hidden_wh = model_wh.forward_recurrent(obs_wh_t, current_step, hidden_wh)
                
            dist_wh = torch.distributions.Normal(mu_wh, std_wh)
            act_wh_t = dist_wh.sample()
            log_prob_wh_batch = dist_wh.log_prob(act_wh_t).sum(dim=-1).cpu().numpy()
            action_wh_unclipped = act_wh_t.cpu().numpy()
            action_wh_clipped = np.clip(action_wh_unclipped, 0.0, 1.0)
            values_wh = val_wh.cpu().numpy().squeeze(-1)
            
            # Retailers BATCH forward pass
            ret_obs_list = []
            for j in range(envs.num_retailers):
                for i in range(num_envs):
                    padded = pad_obs(obs_dict[f"retailer_{j}"][i], max_ret_obs_dim)
                    ret_obs_list.append(padded)
            
            obs_ret_batch = torch.as_tensor(np.array(ret_obs_list), dtype=torch.float32, device=device)
            with torch.no_grad():
                mu_ret, std_ret, val_ret, next_hidden_rets = model_ret.forward_recurrent(obs_ret_batch, current_step, hidden_rets)
                
            dist_ret = torch.distributions.Normal(mu_ret, std_ret)
            act_ret_batch = dist_ret.sample()
            log_prob_ret_batch = dist_ret.log_prob(act_ret_batch).sum(dim=-1).cpu().numpy()
            
            act_ret_np = act_ret_batch.cpu().numpy()
            val_ret_np = val_ret.cpu().numpy().squeeze(-1)
            
            # Map back to build actions dict
            actions_dict = {"warehouse": action_wh_clipped}
            for j in range(envs.num_retailers):
                actions_dict[f"retailer_{j}"] = []
                for i in range(num_envs):
                    idx = j * num_envs + i
                    action_ret_unclipped = act_ret_np[idx]
                    action_ret_clipped = np.clip(action_ret_unclipped, 0.0, 1.0)
                    actions_dict[f"retailer_{j}"].append(action_ret_clipped)
                actions_dict[f"retailer_{j}"] = np.stack(actions_dict[f"retailer_{j}"])
                
            # Step Multi-Agent Environment
            next_obs_dict, rewards_dict, term_dict, trunc_dict, infos_dict = envs.step(actions_dict)
            
            # Demands tracking for Fill Rate calculation
            for i in range(num_envs):
                info = infos_dict[i]["warehouse"]
                demands = info["demands"]
                backorders = envs.envs[i].ret_backorders
                rollout_total_demand += np.sum(demands)
                rollout_unfilled_demand += np.sum(np.minimum(demands, backorders))
                
                # Store Warehouse transition
                hist_wh = get_history(wh_histories[i], len(wh_histories[i]) - 1, T_context)
                env_buffers_wh[i].hist_states.append(hist_wh)
                env_buffers_wh[i].actions.append(action_wh_unclipped[i])
                env_buffers_wh[i].log_probs.append(log_prob_wh_batch[i])
                env_buffers_wh[i].rewards.append(rewards_dict["warehouse"][i])
                env_buffers_wh[i].dones.append(term_dict["warehouse"][i])
                env_buffers_wh[i].values.append(values_wh[i])
                
                # Store Retailer transitions
                for j in range(envs.num_retailers):
                    idx = j * num_envs + i
                    hist_ret = get_history(ret_histories[j][i], len(ret_histories[j][i]) - 1, T_context)
                    env_buffers_ret[i].hist_states.append(hist_ret)
                    env_buffers_ret[i].actions.append([act_ret_np[idx]])
                    env_buffers_ret[i].log_probs.append(log_prob_ret_batch[idx])
                    env_buffers_ret[i].rewards.append(rewards_dict[f"retailer_{j}"][i])
                    env_buffers_ret[i].dones.append(term_dict[f"retailer_{j}"][i])
                    env_buffers_ret[i].values.append(val_ret_np[idx])
                    
                # In-place reset of recurrent hidden states and history paths if environment resets
                if term_dict["warehouse"][i] or trunc_dict["warehouse"][i]:
                    wh_histories[i] = [next_obs_dict["warehouse"][i]]
                    for j in range(envs.num_retailers):
                        ret_histories[j][i] = [pad_obs(next_obs_dict[f"retailer_{j}"][i], max_ret_obs_dim)]
                        
                    # Reset hidden states to zeros in-place
                    for l in range(model_wh.L):
                        hidden_wh[l][i].zero_()
                    for l in range(model_ret.L):
                        for j in range(envs.num_retailers):
                            idx_rst = j * num_envs + i
                            hidden_rets[l][idx_rst].zero_()
                else:
                    wh_histories[i].append(next_obs_dict["warehouse"][i])
                    for j in range(envs.num_retailers):
                        ret_histories[j][i].append(pad_obs(next_obs_dict[f"retailer_{j}"][i], max_ret_obs_dim))
                        
            obs_dict = next_obs_dict
            hidden_wh = next_hidden_wh
            hidden_rets = next_hidden_rets
            current_step += 1
            
            # If all envs synchronized reset
            if all(term_dict["warehouse"]) or all(trunc_dict["warehouse"]):
                current_step = 0
                
        # Compute GAE for Warehouse in each env
        for i in range(num_envs):
            obs_wh_last = torch.as_tensor(obs_dict["warehouse"][i], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, final_val_wh, _ = model_wh.forward_recurrent(obs_wh_last, current_step, [hidden_wh[l][i:i+1] for l in range(model_wh.L)])
            env_buffers_wh[i].compute_gae(final_val_wh.item())
            
        # Compute GAE for Retailers in each env
        for i in range(num_envs):
            ret_obs_last_list = []
            for j in range(envs.num_retailers):
                ret_obs_last_list.append(pad_obs(obs_dict[f"retailer_{j}"][i], max_ret_obs_dim))
            obs_ret_last_batch = torch.as_tensor(np.array(ret_obs_last_list), dtype=torch.float32, device=device)
            env_hidden_rets = [
                torch.stack([hidden_rets[l][j * num_envs + i] for j in range(envs.num_retailers)])
                for l in range(model_ret.L)
            ]
            with torch.no_grad():
                _, _, final_val_rets_t, _ = model_ret.forward_recurrent(obs_ret_last_batch, current_step, env_hidden_rets)
            final_val_rets = final_val_rets_t.squeeze(-1).cpu().numpy()
            env_buffers_ret[i].compute_gae(float(np.mean(final_val_rets)))
            
        # Merge all trajectories
        main_buffer_wh.clear()
        main_buffer_ret.clear()
        for i in range(num_envs):
            main_buffer_wh.hist_states.extend(env_buffers_wh[i].hist_states)
            main_buffer_wh.actions.extend(env_buffers_wh[i].actions)
            main_buffer_wh.log_probs.extend(env_buffers_wh[i].log_probs)
            main_buffer_wh.rewards.extend(env_buffers_wh[i].rewards)
            main_buffer_wh.dones.extend(env_buffers_wh[i].dones)
            main_buffer_wh.values.extend(env_buffers_wh[i].values)
            main_buffer_wh.advantages.extend(env_buffers_wh[i].advantages)
            main_buffer_wh.value_targets.extend(env_buffers_wh[i].value_targets)
            
            main_buffer_ret.hist_states.extend(env_buffers_ret[i].hist_states)
            main_buffer_ret.actions.extend(env_buffers_ret[i].actions)
            main_buffer_ret.log_probs.extend(env_buffers_ret[i].log_probs)
            main_buffer_ret.rewards.extend(env_buffers_ret[i].rewards)
            main_buffer_ret.dones.extend(env_buffers_ret[i].dones)
            main_buffer_ret.values.extend(env_buffers_ret[i].values)
            main_buffer_ret.advantages.extend(env_buffers_ret[i].advantages)
            main_buffer_ret.value_targets.extend(env_buffers_ret[i].value_targets)
            
        # PPO Update (parallel backprop)
        update_info = mappo_agent.update(main_buffer_wh, main_buffer_ret, batch_size=128, epochs=3)
        
        # Compute rollout Fill Rate
        fill_rate = max(0.0, 100.0 * (1.0 - (rollout_unfilled_demand / (rollout_total_demand + 1e-8))))
        mean_reward = np.mean(main_buffer_wh.rewards)
        
        # Compute ETA
        elapsed_time = time.time() - start_time
        avg_time_per_iter = elapsed_time / (iteration - start_iteration + 1)
        remaining_iters = total_iterations - iteration
        eta_seconds = int(avg_time_per_iter * remaining_iters)
        eta_str = str(datetime.timedelta(seconds=eta_seconds))
        
        # Logging progress
        print(f"Iteration {iteration:04d}/{total_iterations} | "
              f"Joint Reward: {mean_reward:8.4f} | "
              f"Fill Rate: {fill_rate:6.2f}% | "
              f"WH Actor Loss: {update_info['wh_actor_loss']:6.4f} | "
              f"Ret Actor Loss: {update_info['ret_actor_loss']:6.4f} | "
              f"ETA: {eta_str}")
              
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
        if iteration in [10000, 15000, 20000] or (iteration % 1000 == 0):
            if hasattr(model_wh, "_orig_mod"):
                wh_state = model_wh._orig_mod.state_dict()
            elif hasattr(model_wh, "module"):
                wh_state = model_wh.module.state_dict()
            else:
                wh_state = model_wh.state_dict()
                
            if hasattr(model_ret, "_orig_mod"):
                ret_state = model_ret._orig_mod.state_dict()
            elif hasattr(model_ret, "module"):
                ret_state = model_ret.module.state_dict()
            else:
                ret_state = model_ret.state_dict()
                
            checkpoint_wh = os.path.join(save_dir, f"bdh_mappo_wh_{iteration}.pt")
            checkpoint_ret = os.path.join(save_dir, f"bdh_mappo_ret_{iteration}.pt")
            torch.save(wh_state, checkpoint_wh)
            torch.save(ret_state, checkpoint_ret)
            # Also save to default paths for easy resume
            torch.save(wh_state, save_path_wh)
            torch.save(ret_state, save_path_ret)
            print(f"MAPPO checkpoints saved at iteration {iteration}")
            
    # Save final model
    if hasattr(model_wh, "_orig_mod"):
        wh_state_final = model_wh._orig_mod.state_dict()
    elif hasattr(model_wh, "module"):
        wh_state_final = model_wh.module.state_dict()
    else:
        wh_state_final = model_wh.state_dict()
        
    if hasattr(model_ret, "_orig_mod"):
        ret_state_final = model_ret._orig_mod.state_dict()
    elif hasattr(model_ret, "module"):
        ret_state_final = model_ret.module.state_dict()
    else:
        ret_state_final = model_ret.state_dict()
        
    torch.save(wh_state_final, save_path_wh)
    torch.save(ret_state_final, save_path_ret)
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
    plot_path = os.path.join(save_dir, "training_progress.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training progress plot saved to {plot_path}")
    
    return model_wh, model_ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train decentralized MAPPO cooperative supply chain controller.")
    parser.add_argument("--network_id", type=int, default=1, help="Willems network ID.")
    parser.add_argument("--total_iterations", type=int, default=20000, help="Number of training iterations.")
    parser.add_argument("--rollout_steps", type=int, default=2000, help="Steps collected per iteration.")
    parser.add_argument("--save_path_wh", type=str, default="bdh_mappo_wh_20000.pt", help="Filepath to save warehouse model.")
    parser.add_argument("--save_path_ret", type=str, default="bdh_mappo_ret_20000.pt", help="Filepath to save retailer model.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "xpu", "xla"], help="Specify hardware device.")
    parser.add_argument("--resume", action="store_true", help="Resume training from save paths checkpoints if they exist.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments for vectorized rollouts.")
    args = parser.parse_args()
    
    train_mappo(
        network_id=args.network_id,
        total_iterations=args.total_iterations,
        rollout_steps=args.rollout_steps,
        save_path_wh=args.save_path_wh,
        save_path_ret=args.save_path_ret,
        device_arg=args.device,
        resume=args.resume,
        num_envs=args.num_envs
    )
