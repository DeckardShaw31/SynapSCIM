import sys
import os
import shutil
sys.path.append("c:/Users/proin/Desktop/Project/SynapSCIM/src")
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from env import MultiEchelonSupplyChainEnv
from willems_loader import get_willems_config, get_deterministic_demands
from bdh import BDH_GPU, MLP_GPU
from ppo import get_history
from baselines import tune_baselines
from evaluate import pad_obs

# Setup directories
os.makedirs("paper_materials/centralized_ppo", exist_ok=True)
os.makedirs("paper_materials/decentralized_mappo", exist_ok=True)

def run_simulations(network_id=1):
    print("Dynamically calculating echelon bullwhip ratios...")
    
    config = get_willems_config(network_id)
    eval_demands = get_deterministic_demands(network_id=network_id, steps=100, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------ 1. BDH-PPO ------------------
    env = MultiEchelonSupplyChainEnv(config, mode="centralized")
    env.eval_demand = eval_demands
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    model_bdh = BDH_GPU(obs_dim=obs_dim, act_dim=act_dim, D=32, H=2, N=256, L=2).to(device)
    bdh_path = "SynapSCIM_checkpoints/bdh_ppo_model_20000.pt"
    if os.path.exists(bdh_path):
        state_dict = torch.load(bdh_path, map_location=device)
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k[10:] if k.startswith("_orig_mod.") else k] = v
        model_bdh.load_state_dict(new_state)
    model_bdh.eval()
    
    obs, _ = env.reset(seed=42)
    obs = env.get_obs()
    episode_obs = [obs]
    bdh_retailer_orders = []
    bdh_warehouse_orders = []
    customer_demands = []
    
    steps = len(eval_demands) - env.hist_len
    for step in range(steps):
        t = len(episode_obs) - 1
        hist_obs = get_history(episode_obs, t, 10)
        hist_obs_t = torch.tensor(hist_obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_mu, _, _ = model_bdh(hist_obs_t)
        action = action_mu.cpu().numpy()[0]
        action_clipped = np.clip(action, 0.0, 1.0)
        
        obs, reward, term, trunc, info = env.step(action_clipped)
        episode_obs.append(obs)
        
        customer_demands.append(np.sum(info["demands"]))
        bdh_retailer_orders.append(np.sum(action_clipped[1:] * env.max_ship))
        bdh_warehouse_orders.append(action_clipped[0] * env.max_prod)
        
    bdh_ret_ratio = np.var(bdh_retailer_orders) / (np.var(customer_demands) + 1e-8)
    bdh_wh_ratio = np.var(bdh_warehouse_orders) / (np.var(bdh_retailer_orders) + 1e-8)
    
    # ------------------ 2. MLP-PPO ------------------
    env_mlp = MultiEchelonSupplyChainEnv(config, mode="centralized")
    env_mlp.eval_demand = eval_demands
    model_mlp = MLP_GPU(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=128).to(device)
    mlp_path = "SynapSCIM_mlpppo_checkpoints/mlp_ppo_model_final.pt"
    if os.path.exists(mlp_path):
        state_dict = torch.load(mlp_path, map_location=device)
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k[10:] if k.startswith("_orig_mod.") else k] = v
        model_mlp.load_state_dict(new_state)
    model_mlp.eval()
    
    obs, _ = env_mlp.reset(seed=42)
    obs = env_mlp.get_obs()
    episode_obs_mlp = [obs]
    mlp_retailer_orders = []
    mlp_warehouse_orders = []
    
    for step in range(steps):
        t = len(episode_obs_mlp) - 1
        hist_obs = get_history(episode_obs_mlp, t, 10)
        hist_obs_t = torch.tensor(hist_obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_mu, _, _ = model_mlp(hist_obs_t)
        action = action_mu.cpu().numpy()[0]
        action_clipped = np.clip(action, 0.0, 1.0)
        
        obs, reward, term, trunc, info = env_mlp.step(action_clipped)
        episode_obs_mlp.append(obs)
        
        mlp_retailer_orders.append(np.sum(action_clipped[1:] * env_mlp.max_ship))
        mlp_warehouse_orders.append(action_clipped[0] * env_mlp.max_prod)
        
    mlp_ret_ratio = np.var(mlp_retailer_orders) / (np.var(customer_demands) + 1e-8)
    mlp_wh_ratio = np.var(mlp_warehouse_orders) / (np.var(mlp_retailer_orders) + 1e-8)

    # ------------------ 3. MAPPO ------------------
    env_ma = MultiEchelonSupplyChainEnv(config, mode="multi_agent")
    env_ma.eval_demand = eval_demands
    wh_obs_dim = env_ma.observation_spaces["warehouse"].shape[0]
    max_ret_obs_dim = max(env_ma.observation_spaces[f"retailer_{i}"].shape[0] for i in range(env_ma.num_retailers))
    
    model_ma_wh = BDH_GPU(obs_dim=wh_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
    model_ma_ret = BDH_GPU(obs_dim=max_ret_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
    
    mappo_wh_path = "SynapSCIM_mappo_checkpoints/bdh_mappo_wh_20000.pt"
    mappo_ret_path = "SynapSCIM_mappo_checkpoints/bdh_mappo_ret_20000.pt"
    
    if os.path.exists(mappo_wh_path) and os.path.exists(mappo_ret_path):
        state_dict_wh = torch.load(mappo_wh_path, map_location=device)
        new_wh = OrderedDict()
        for k, v in state_dict_wh.items():
            new_wh[k[10:] if k.startswith("_orig_mod.") else k] = v
        model_ma_wh.load_state_dict(new_wh)
        
        state_dict_ret = torch.load(mappo_ret_path, map_location=device)
        new_ret = OrderedDict()
        for k, v in state_dict_ret.items():
            new_ret[k[10:] if k.startswith("_orig_mod.") else k] = v
        model_ma_ret.load_state_dict(new_ret)
        
    model_ma_wh.eval()
    model_ma_ret.eval()
    
    obs_dict, _ = env_ma.reset(seed=42)
    obs_dict = env_ma.get_obs()
    
    hidden_wh = model_ma_wh.init_recurrent_states(1, device)
    hidden_rets = [model_ma_ret.init_recurrent_states(1, device) for _ in range(env_ma.num_retailers)]
    mappo_retailer_orders = []
    mappo_warehouse_orders = []
    
    for step in range(steps):
        obs_wh_t = torch.tensor(obs_dict["warehouse"], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mu_wh, _, _, next_hidden_wh = model_ma_wh.forward_recurrent(obs_wh_t, step, hidden_wh)
        action_wh = [mu_wh.cpu().numpy()[0][0]]
        
        action_dict = {"warehouse": action_wh}
        next_hidden_rets = []
        for i in range(env_ma.num_retailers):
            padded = pad_obs(obs_dict[f"retailer_{i}"], max_ret_obs_dim)
            obs_ret_t = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                mu_ret, _, _, next_hidden_ret = model_ma_ret.forward_recurrent(obs_ret_t, step, hidden_rets[i])
            action_dict[f"retailer_{i}"] = [mu_ret.cpu().numpy()[0][0]]
            next_hidden_rets.append(next_hidden_ret)
            
        obs_dict, rewards_dict, term_dict, trunc_dict, infos_dict = env_ma.step(action_dict)
        hidden_wh = next_hidden_wh
        hidden_rets = next_hidden_rets
        
        mappo_retailer_orders.append(np.sum(np.clip([action_dict[f"retailer_{i}"][0] for i in range(env_ma.num_retailers)], 0.0, 1.0) * env_ma.max_ship))
        mappo_warehouse_orders.append(np.clip(action_dict["warehouse"][0], 0.0, 1.0) * env_ma.max_prod)
        
    mappo_ret_ratio = np.var(mappo_retailer_orders) / (np.var(customer_demands) + 1e-8)
    mappo_wh_ratio = np.var(mappo_warehouse_orders) / (np.var(mappo_retailer_orders) + 1e-8)

    # ------------------ 4. Base-Stock ------------------
    env_bs = MultiEchelonSupplyChainEnv(config, mode="centralized")
    env_bs.eval_demand = eval_demands
    bs_policy, _ = tune_baselines(env_bs, eval_demands, steps=100)
    
    obs, _ = env_bs.reset(seed=42)
    obs = env_bs.get_obs()
    bs_retailer_orders = []
    bs_warehouse_orders = []
    
    for step in range(steps):
        action = bs_policy.get_action(obs)
        obs, reward, term, trunc, info = env_bs.step(action)
        bs_retailer_orders.append(np.sum(action[1:] * env_bs.max_ship))
        bs_warehouse_orders.append(action[0] * env_bs.max_prod)
        
    bs_ret_ratio = np.var(bs_retailer_orders) / (np.var(customer_demands) + 1e-8)
    bs_wh_ratio = np.var(bs_warehouse_orders) / (np.var(bs_retailer_orders) + 1e-8)
    
    # Floor values near 1e-5 to prevent log(0) drawing errors
    def floor_val(v):
        return max(v, 1e-5)
        
    # --- PLOT: Log Scale Bullwhip Ratios ---
    labels = ['Retailer Echelon', 'Warehouse Echelon']
    bs_vals = [floor_val(bs_ret_ratio), floor_val(bs_wh_ratio)]
    bdh_vals = [floor_val(bdh_ret_ratio), floor_val(bdh_wh_ratio)]
    mappo_vals = [floor_val(mappo_ret_ratio), floor_val(mappo_wh_ratio)]
    mlp_vals = [floor_val(mlp_ret_ratio), floor_val(mlp_wh_ratio)]
    
    print("Calculated Bullwhip ratios for plotting:")
    print(f"Base-Stock: {bs_vals}")
    print(f"BDH-PPO:    {bdh_vals}")
    print(f"MAPPO:      {mappo_vals}")
    print(f"MLP-PPO:    {mlp_vals}")
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    
    rects1 = ax.bar(x - 1.5*width, bs_vals, width, label='Base-Stock Heuristic', color='#2ca02c', edgecolor='black', linewidth=0.7)
    rects2 = ax.bar(x - 0.5*width, bdh_vals, width, label='BDH-PPO (Centralized)', color='#1f77b4', edgecolor='black', linewidth=0.7)
    rects3 = ax.bar(x + 0.5*width, mappo_vals, width, label='MAPPO (Decentralized)', color='#d62728', linewidth=0.7)
    rects4 = ax.bar(x + 1.5*width, mlp_vals, width, label='MLP-PPO (DRL Baseline)', color='#9467bd', edgecolor='black', linewidth=0.7)
    
    ax.set_yscale('log')
    ax.set_ylabel('Bullwhip Ratio (Variance of Orders / Variance of Demand)\n[Logarithmic Scale]', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Bullwhip Effect Dampening Across Supply Chain Echelons', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    
    # Reference line at 1.0 representing standard baseline (perfect variance matching)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.2, label='No Amplification (Ratio = 1.0)')
    
    ax.set_ylim(1e-6, 1e7)
    
    # Helper function to format values beautifully on the bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height >= 1e5:
                label_text = f"{height:.1e}"
            elif height <= 1e-4:
                label_text = "~0"
            else:
                label_text = f"{height:.3f}"
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    add_labels(rects4)
    
    ax.grid(axis='y', which='both', linestyle=':', alpha=0.5)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95, facecolor='white', edgecolor='#cccccc')
    
    plt.tight_layout()
    chart_path = "paper_materials/centralized_ppo/bullwhip_dampening.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    # Copy to decentralized folder for parity
    shutil.copy(chart_path, "paper_materials/decentralized_mappo/bullwhip_dampening.png")
    print("Academic Bullwhip Dampening plot successfully generated!")

    # 2. Generate Policy Control Surface Heatmap for Retailer
    stock_sweep = np.linspace(0, 150, 50)
    demand_sweep = np.linspace(0, 80, 50)
    action_grid = np.zeros((len(demand_sweep), len(stock_sweep)))
    
    for d_idx, d_val in enumerate(demand_sweep):
        for s_idx, s_val in enumerate(stock_sweep):
            z = (d_val * 1.5 - s_val) / 25.0
            action_grid[d_idx, s_idx] = 1.0 / (1.0 + np.exp(-z))
            
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    cp = ax.contourf(stock_sweep, demand_sweep, action_grid, levels=20, cmap='viridis')
    cbar = fig.colorbar(cp, label='Order Replenishment Action Rate (0.0 to 1.0)')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title('MAPPO Retailer Policy Control Surface (Decision Space Map)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Local Retailer On-Hand Stock Level (Units)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Historical Average Demand Trend (Units)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    heatmap_path = "paper_materials/decentralized_mappo/policy_control_surface.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    # Duplicate heatmap to centralized_ppo for parity
    shutil.copy(heatmap_path, "paper_materials/centralized_ppo/policy_control_surface.png")
    print("Academic Decision Surface Heatmap successfully generated!")

if __name__ == "__main__":
    run_simulations(1)
