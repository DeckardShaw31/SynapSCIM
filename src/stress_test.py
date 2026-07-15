import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from env import MultiEchelonSupplyChainEnv
from bdh import BDH_GPU, MLP_GPU
from ppo import get_history
from willems_loader import get_willems_config, get_deterministic_demands
from baselines import tune_baselines
from generate_report import pad_obs

def run_stress_test_simulation(env, policy_type, policy_model, eval_demands_shock, T_context=10):
    env.eval_demand = eval_demands_shock
    obs_raw, _ = env.reset(seed=42)
    
    # Track trajectories
    wh_stock_history = []
    ret_stock_history = []
    backorder_history = []
    cost_history = []
    
    # Store original capacity limits
    original_max_prod = env.max_prod
    original_max_ship = env.max_ship.copy()
    
    steps = len(eval_demands_shock) - env.hist_len
    
    # Centralized PPO execution list
    episode_obs = [obs_raw]
    
    # MAPPO execution parameters
    if policy_type == "mappo":
        model_wh, model_ret = policy_model
        wh_obs_dim = env.observation_spaces["warehouse"].shape[0]
        max_ret_obs_dim = max(env.observation_spaces[f"retailer_{i}"].shape[0] for i in range(env.num_retailers))
        hidden_wh = model_wh.init_recurrent_states(1, next(model_wh.parameters()).device)
        hidden_rets = [model_ret.init_recurrent_states(1, next(model_ret.parameters()).device) for _ in range(env.num_retailers)]
        obs_dict = obs_raw
        
    obs = obs_raw
    total_cost = 0.0
    
    for step in range(steps):
        # 1. Trigger severe logistics capacity outage from Day 30 to Day 50 (10% capacity)
        is_disrupted = (30 <= step < 50)
        if is_disrupted:
            env.max_prod = original_max_prod * 0.1
            env.max_ship = [s * 0.1 for s in original_max_ship]
        else:
            env.max_prod = original_max_prod
            env.max_ship = original_max_ship
            
        # 2. Get action depending on policy type
        if policy_type in ["bdh_ppo", "mlp_ppo"]:
            t = len(episode_obs) - 1
            hist_obs = get_history(episode_obs, t, T_context)
            device = next(policy_model.parameters()).device
            hist_obs_t = torch.tensor(hist_obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mu, _, _ = policy_model(hist_obs_t)
            action = action_mu.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            episode_obs.append(obs)
            step_cost = info["total_cost"]
            
        elif policy_type == "mappo":
            device = next(model_wh.parameters()).device
            obs_wh_t = torch.tensor(obs_dict["warehouse"], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                mu_wh, _, _, next_hidden_wh = model_wh.forward_recurrent(obs_wh_t, step, hidden_wh)
            action_wh = [mu_wh.cpu().numpy()[0][0]]
            
            action_dict = {"warehouse": action_wh}
            next_hidden_rets = []
            for i in range(env.num_retailers):
                padded = pad_obs(obs_dict[f"retailer_{i}"], max_ret_obs_dim)
                obs_ret_t = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    mu_ret, _, _, next_hidden_ret = model_ret.forward_recurrent(obs_ret_t, step, hidden_rets[i])
                action_dict[f"retailer_{i}"] = [mu_ret.cpu().numpy()[0][0]]
                next_hidden_rets.append(next_hidden_ret)
                
            obs_dict, rewards_dict, term_dict, trunc_dict, infos_dict = env.step(action_dict)
            hidden_wh = next_hidden_wh
            hidden_rets = next_hidden_rets
            step_cost = infos_dict["warehouse"]["total_cost"]
            obs = obs_dict
            
        else:  # Baselines
            action = policy_model.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step_cost = info["total_cost"]
            
        # Record trajectories
        wh_stock_history.append(env.wh_stock)
        ret_stock_history.append(env.ret_stocks.copy())
        backorder_history.append(env.ret_backorders.copy())
        total_cost += step_cost
        cost_history.append(total_cost)
        
    # Restore capacities
    env.max_prod = original_max_prod
    env.max_ship = original_max_ship
    env.eval_demand = None
    
    return {
        "wh_stock": wh_stock_history,
        "ret_stock": ret_stock_history,
        "backorders": backorder_history,
        "cumulative_cost": cost_history
    }

def run_stress_test(network_id=1, model_path_ppo=None, model_path_mappo_wh="bdh_mappo_wh.pt", model_path_mappo_ret="bdh_mappo_ret.pt"):
    print(f"\n--- Starting Disruption Stress Test on Willems Network {network_id} ---")
    
    config = get_willems_config(network_id)
    eval_demands = get_deterministic_demands(network_id=network_id, steps=100, seed=42)
    
    # Create the Disruption + Demand Surge scenario (Days 30 to 50)
    eval_demands_shock = [d.copy() for d in eval_demands]
    for step in range(30 + config["hist_len"], 50 + config["hist_len"]):
        eval_demands_shock[step] = [v * 2.5 for v in eval_demands_shock[step]]
        
    # 1. Load Centralized BDH-PPO model
    env_ce = MultiEchelonSupplyChainEnv(config, mode="centralized")
    obs_dim = env_ce.observation_space.shape[0]
    act_dim = env_ce.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bdh_model_ppo = BDH_GPU(obs_dim=obs_dim, act_dim=act_dim, D=32, H=2, N=256, L=2).to(device)
    if model_path_ppo is None:
        for candidate in [
            "SynapSCIM_checkpoints/bdh_ppo_model_20000.pt",
            "SynapSCIM_checkpoints/bdh_ppo_model.pt",
            "bdh_ppo_model_3000.pt",
            "bdh_ppo_model_1000.pt",
            "bdh_ppo_model.pt"
        ]:
            if os.path.exists(candidate):
                model_path_ppo = candidate
                break
        if model_path_ppo is None:
            model_path_ppo = "bdh_ppo_model.pt"
            
    if os.path.exists(model_path_ppo):
        print(f"Loading Centralized PPO weights from {model_path_ppo}...")
        state_dict = torch.load(model_path_ppo, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        bdh_model_ppo.load_state_dict(new_state_dict)
    else:
        print("[Warning] Centralized PPO weights not found. Using randomly initialized model.")
    bdh_model_ppo.eval()
    
    # 2. Load Cooperative MAPPO model
    env_ma = MultiEchelonSupplyChainEnv(config, mode="multi_agent")
    wh_obs_dim = env_ma.observation_spaces["warehouse"].shape[0]
    max_ret_obs_dim = max(env_ma.observation_spaces[f"retailer_{i}"].shape[0] for i in range(env_ma.num_retailers))
    
    model_ma_wh = BDH_GPU(obs_dim=wh_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
    model_ma_ret = BDH_GPU(obs_dim=max_ret_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
    
    # Search candidates for MAPPO if defaults are specified
    if model_path_mappo_wh == "bdh_mappo_wh.pt" and model_path_mappo_ret == "bdh_mappo_ret.pt":
        for cand_wh, cand_ret in [
            (f"SynapSCIM_mappo_checkpoints/bdh_mappo_wh_{network_id}_20000.pt", f"SynapSCIM_mappo_checkpoints/bdh_mappo_ret_{network_id}_20000.pt"),
            ("SynapSCIM_mappo_checkpoints/bdh_mappo_wh_20000.pt", "SynapSCIM_mappo_checkpoints/bdh_mappo_ret_20000.pt"),
            ("SynapSCIM_mappo_checkpoints/bdh_mappo_wh.pt", "SynapSCIM_mappo_checkpoints/bdh_mappo_ret.pt"),
            ("bdh_mappo_wh.pt", "bdh_mappo_ret.pt")
        ]:
            if os.path.exists(cand_wh) and os.path.exists(cand_ret):
                model_path_mappo_wh = cand_wh
                model_path_mappo_ret = cand_ret
                break
                
    if os.path.exists(model_path_mappo_wh) and os.path.exists(model_path_mappo_ret):
        print(f"Loading MAPPO weights from {model_path_mappo_wh} and {model_path_mappo_ret}...")
        # Auto prefix strip MAPPO
        state_dict_wh = torch.load(model_path_mappo_wh, map_location=device)
        from collections import OrderedDict
        new_wh = OrderedDict()
        for k, v in state_dict_wh.items():
            if k.startswith("_orig_mod."):
                new_wh[k[10:]] = v
            else:
                new_wh[k] = v
        model_ma_wh.load_state_dict(new_wh)
        
        state_dict_ret = torch.load(model_path_mappo_ret, map_location=device)
        new_ret = OrderedDict()
        for k, v in state_dict_ret.items():
            if k.startswith("_orig_mod."):
                new_ret[k[10:]] = v
            else:
                new_ret[k] = v
        model_ma_ret.load_state_dict(new_ret)
    else:
        print("[Warning] MAPPO weights not found. Using randomly initialized model.")
    model_ma_wh.eval()
    model_ma_ret.eval()
    
    # 2.5 Load MLP-PPO baseline model
    bdh_model_mlp = MLP_GPU(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=128).to(device)
    mlp_path = "SynapSCIM_mlpppo_checkpoints/mlp_ppo_model_final.pt"
    if os.path.exists(mlp_path):
        print(f"Loading MLP-PPO weights from {mlp_path}...")
        state_dict_mlp = torch.load(mlp_path, map_location=device)
        from collections import OrderedDict
        new_mlp = OrderedDict()
        for k, v in state_dict_mlp.items():
            if k.startswith("_orig_mod."):
                new_mlp[k[10:]] = v
            else:
                new_mlp[k] = v
        bdh_model_mlp.load_state_dict(new_mlp)
    else:
        print("[Warning] MLP-PPO weights not found. Using randomly initialized model.")
    bdh_model_mlp.eval()
    
    # 3. Instantiate tuned Base-Stock baseline
    bs_policy, _ = tune_baselines(env_ce, eval_demands_shock, steps=100)
    
    # 4. Run simulations under shock
    print("Simulating policies under disruption shock (Days 30-50)...")
    res_ppo = run_stress_test_simulation(env_ce, "bdh_ppo", bdh_model_ppo, eval_demands_shock)
    res_mappo = run_stress_test_simulation(env_ma, "mappo", (model_ma_wh, model_ma_ret), eval_demands_shock)
    res_mlp = run_stress_test_simulation(env_ce, "mlp_ppo", bdh_model_mlp, eval_demands_shock)
    res_bs = run_stress_test_simulation(env_ce, "base_stock", bs_policy, eval_demands_shock)
    
    # 5. Plot comparisons
    os.makedirs("reports/centralized_ppo", exist_ok=True)
    chart_path = "reports/centralized_ppo/disruption_stress_test.png"
    print("Generating disruption stress test plot...")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Subplot 1: Cumulative Cost comparison
    axes[0].plot(res_ppo["cumulative_cost"], label="BDH-PPO (Centralized Coordinated)", color="#1f77b4", linewidth=2.5)
    axes[0].plot(res_mappo["cumulative_cost"], label="MAPPO (Decentralized Information-Gap)", color="#d62728", linewidth=2, linestyle="--")
    axes[0].plot(res_mlp["cumulative_cost"], label="MLP-PPO (Centralized DRL Baseline)", color="#9467bd", linewidth=2, linestyle="-.")
    axes[0].plot(res_bs["cumulative_cost"], label="Base-Stock (Standard Industry Heuristic)", color="#2ca02c", linewidth=2)
    axes[0].axvspan(30, 50, color='red', alpha=0.15, label='Logistics Disruption Outage & Demand Surge')
    axes[0].set_title("Operational Cumulative Cost Reaction under Extreme Disruption Shock", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Total Cost ($)", fontsize=10)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Subplot 2: Total On-Hand Retailer Inventory comparison
    ppo_sum_stock = [np.sum(s) for s in res_ppo["ret_stock"]]
    mappo_sum_stock = [np.sum(s) for s in res_mappo["ret_stock"]]
    mlp_sum_stock = [np.sum(s) for s in res_mlp["ret_stock"]]
    bs_sum_stock = [np.sum(s) for s in res_bs["ret_stock"]]
    
    axes[1].plot(ppo_sum_stock, label="BDH-PPO", color="#1f77b4", linewidth=2.5)
    axes[1].plot(mappo_sum_stock, label="MAPPO", color="#d62728", linewidth=2, linestyle="--")
    axes[1].plot(mlp_sum_stock, label="MLP-PPO (Baseline)", color="#9467bd", linewidth=2, linestyle="-.")
    axes[1].plot(bs_sum_stock, label="Base-Stock", color="#2ca02c", linewidth=2)
    axes[1].axvspan(30, 50, color='red', alpha=0.15)
    axes[1].set_title("Total Retailer On-Hand Inventory Levels", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Stock Units", fontsize=10)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    # Subplot 3: Total Retailer Backorders (Shortages) comparison
    ppo_sum_bo = [np.sum(b) for b in res_ppo["backorders"]]
    mappo_sum_bo = [np.sum(b) for b in res_mappo["backorders"]]
    mlp_sum_bo = [np.sum(b) for b in res_mlp["backorders"]]
    bs_sum_bo = [np.sum(b) for b in res_bs["backorders"]]
    
    axes[2].plot(ppo_sum_bo, label="BDH-PPO", color="#1f77b4", linewidth=2.5)
    axes[2].plot(mappo_sum_bo, label="MAPPO", color="#d62728", linewidth=2, linestyle="--")
    axes[2].plot(mlp_sum_bo, label="MLP-PPO (Baseline)", color="#9467bd", linewidth=2, linestyle="-.")
    axes[2].plot(bs_sum_bo, label="Base-Stock", color="#2ca02c", linewidth=2)
    axes[2].axvspan(30, 50, color='red', alpha=0.15)
    axes[2].set_title("Total Retailer Customer Backorders (Shortages)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Backorder Units", fontsize=10)
    axes[2].set_xlabel("Time Steps (Days)", fontsize=10)
    axes[2].legend(loc="upper left")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Disruption stress test plot saved successfully to {chart_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Disruption stress testing.")
    parser.add_argument("--network_id", type=int, default=1, help="Willems network ID.")
    parser.add_argument("--model_path_ppo", type=str, default=None, help="Path to centralized PPO weights.")
    parser.add_argument("--model_path_mappo_wh", type=str, default="bdh_mappo_wh.pt", help="Path to MAPPO warehouse weights.")
    parser.add_argument("--model_path_mappo_ret", type=str, default="bdh_mappo_ret.pt", help="Path to MAPPO retailer weights.")
    args = parser.parse_args()
    
    run_stress_test(
        network_id=args.network_id,
        model_path_ppo=args.model_path_ppo,
        model_path_mappo_wh=args.model_path_mappo_wh,
        model_path_mappo_ret=args.model_path_mappo_ret
    )
