import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from env import MultiEchelonSupplyChainEnv
from bdh import BDH_GPU, MLP_GPU, GNN_PPO_Model
from ppo import get_history
from willems_loader import get_willems_config, get_deterministic_demands
from baselines import tune_baselines

# Utility helper to pad observations
def pad_obs(obs, target_dim):
    if len(obs) < target_dim:
        return np.pad(obs, (0, target_dim - len(obs)), mode='constant')
    return obs[:target_dim]

# Recurrent evaluation loop for Decentralized MAPPO
def run_mappo_evaluation(env, model_wh, model_ret, deterministic_demands, T_context=10):
    env.eval_demand = deterministic_demands
    obs_dict, _ = env.reset(seed=42)
    
    device = next(model_wh.parameters()).device
    max_ret_obs_dim = max(env.observation_spaces[f"retailer_{i}"].shape[0] for i in range(env.num_retailers))
    
    hidden_wh = model_wh.init_recurrent_states(1, device)
    hidden_rets = [model_ret.init_recurrent_states(1, device) for _ in range(env.num_retailers)]
    
    total_cost = 0.0
    holding_cost = 0.0
    backorder_cost = 0.0
    shipping_cost = 0.0
    production_cost = 0.0
    
    total_demand = 0.0
    total_unfilled = 0.0
    
    cost_trajectory = []
    wh_stock_trajectory = []
    ret_stock_trajectory = []
    backorder_trajectory = []
    
    steps = len(deterministic_demands) - env.hist_len
    
    for step in range(steps):
        # 1. Warehouse forward recurrent pass
        obs_wh_t = torch.as_tensor(obs_dict["warehouse"], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mu_wh, _, _, next_hidden_wh = model_wh.forward_recurrent(obs_wh_t, step, hidden_wh)
        action_wh = [mu_wh.cpu().numpy()[0][0]]
        
        # 2. Retailers forward recurrent pass
        action_dict = {"warehouse": action_wh}
        next_hidden_rets = []
        for i in range(env.num_retailers):
            padded = pad_obs(obs_dict[f"retailer_{i}"], max_ret_obs_dim)
            obs_ret_t = torch.as_tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                mu_ret, _, _, next_hidden_ret = model_ret.forward_recurrent(obs_ret_t, step, hidden_rets[i])
            action_dict[f"retailer_{i}"] = [mu_ret.cpu().numpy()[0][0]]
            next_hidden_rets.append(next_hidden_ret)
            
        # Step env
        obs_dict, rewards_dict, term_dict, trunc_dict, infos_dict = env.step(action_dict)
        hidden_wh = next_hidden_wh
        hidden_rets = next_hidden_rets
        
        # Accumulate metrics
        info = infos_dict["warehouse"]
        total_cost += info["total_cost"]
        holding_cost += info["holding_cost"]
        backorder_cost += info["backorder_cost"]
        shipping_cost += info["shipping_cost"]
        production_cost += info["production_cost"]
        
        demands = info["demands"]
        backorders = env.ret_backorders
        
        total_demand += np.sum(demands)
        total_unfilled += np.sum(np.minimum(demands, backorders))
        
        # Track step trajectories
        cost_trajectory.append(total_cost)
        wh_stock_trajectory.append(env.wh_stock)
        ret_stock_trajectory.append(env.ret_stocks.copy())
        backorder_trajectory.append(env.ret_backorders.copy())
        
        if term_dict["warehouse"] or trunc_dict["warehouse"]:
            break
            
    env.eval_demand = None
    service_level = max(0.0, 100.0 * (1.0 - (total_unfilled / (total_demand + 1e-8))))
    
    return {
        "total_cost": total_cost,
        "holding_cost": holding_cost,
        "backorder_cost": backorder_cost,
        "shipping_cost": shipping_cost,
        "production_cost": production_cost,
        "service_level": service_level,
        "cost_trajectory": cost_trajectory,
        "wh_stock_trajectory": wh_stock_trajectory,
        "ret_stock_trajectory": ret_stock_trajectory,
        "backorder_trajectory": backorder_trajectory
    }

def run_evaluation(env, policy_type, policy_model, deterministic_demands, T_context=10):
    """
    Run evaluation simulation on deterministic demands and return accumulated costs.
    """
    env.eval_demand = deterministic_demands
    obs, _ = env.reset(seed=42)
    
    episode_obs = [obs]
    
    total_cost = 0.0
    holding_cost = 0.0
    backorder_cost = 0.0
    shipping_cost = 0.0
    production_cost = 0.0
    
    total_demand = 0.0
    total_unfilled = 0.0
    
    cost_trajectory = []
    wh_stock_trajectory = []
    ret_stock_trajectory = []
    backorder_trajectory = []
    
    steps = len(deterministic_demands) - env.hist_len
    
    for step in range(steps):
        t = len(episode_obs) - 1
        hist_obs = get_history(episode_obs, t, T_context)
        
        # Decide action based on policy type
        if policy_type in ["bdh_ppo", "mlp_ppo", "gnn_ppo"]:
            device = next(policy_model.parameters()).device
            hist_obs_t = torch.tensor(hist_obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mu, _, _ = policy_model(hist_obs_t)
            action = action_mu.cpu().numpy()[0]
        else:
            # Traditional heuristics
            action = policy_model.get_action(obs)
            
        # Step env
        obs, reward, terminated, truncated, info = env.step(action)
        episode_obs.append(obs)
        
        # Accumulate metrics
        total_cost += info["total_cost"]
        holding_cost += info["holding_cost"]
        backorder_cost += info["backorder_cost"]
        shipping_cost += info["shipping_cost"]
        production_cost += info["production_cost"]
        
        demands = info["demands"]
        backorders = env.ret_backorders
        
        total_demand += np.sum(demands)
        total_unfilled += np.sum(np.minimum(demands, backorders))
        
        # Track step trajectories
        cost_trajectory.append(total_cost)
        wh_stock_trajectory.append(env.wh_stock)
        ret_stock_trajectory.append(env.ret_stocks.copy())
        backorder_trajectory.append(env.ret_backorders.copy())
        
        if terminated or truncated:
            break
            
    # Reset env demands
    env.eval_demand = None
    service_level = max(0.0, 100.0 * (1.0 - (total_unfilled / (total_demand + 1e-8))))
    
    return {
        "total_cost": total_cost,
        "holding_cost": holding_cost,
        "backorder_cost": backorder_cost,
        "shipping_cost": shipping_cost,
        "production_cost": production_cost,
        "service_level": service_level,
        "cost_trajectory": cost_trajectory,
        "wh_stock_trajectory": wh_stock_trajectory,
        "ret_stock_trajectory": ret_stock_trajectory,
        "backorder_trajectory": backorder_trajectory
    }

def evaluate_all(network_id=1, model_path=None, T_context=10):
    print(f"\n--- Starting Evaluation on Willems Network {network_id} ---")
    
    # 1. Load env and deterministic demands
    config = get_willems_config(network_id)
    env = MultiEchelonSupplyChainEnv(config)
    
    eval_demands = get_deterministic_demands(network_id=network_id, steps=100, seed=42)
    
    # 2. Instantiate and load BDH-PPO model
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bdh_model = BDH_GPU(
        obs_dim=obs_dim,
        act_dim=act_dim,
        D=32, H=2, N=256, L=2
    ).to(device)
    
    # Auto-detect best available trained weights if model_path is not specified
    if model_path is None:
        for candidate in [
            "SynapSCIM_checkpoints/bdh_ppo_model_20000.pt",
            "SynapSCIM_checkpoints/bdh_ppo_model.pt",
            "bdh_ppo_model_3000.pt",
            "bdh_ppo_model_1000.pt",
            "bdh_ppo_model.pt"
        ]:
            if os.path.exists(candidate):
                model_path = candidate
                break
        if model_path is None:
            model_path = "bdh_ppo_model.pt"
            
    if os.path.exists(model_path):
        print(f"Loading trained PPO weights from {model_path} on device {device}...")
        state_dict = torch.load(model_path, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        bdh_model.load_state_dict(new_state_dict)
    else:
        print(f"[Warning] Trained weights not found at {model_path}. Using randomly initialized model.")
        
    bdh_model.eval()
    
    # 3. Load Cooperative MAPPO models
    env_ma = MultiEchelonSupplyChainEnv(config, mode="multi_agent")
    wh_obs_dim = env_ma.observation_spaces["warehouse"].shape[0]
    max_ret_obs_dim = max(env_ma.observation_spaces[f"retailer_{i}"].shape[0] for i in range(env_ma.num_retailers))
    
    model_ma_wh = BDH_GPU(obs_dim=wh_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
    model_ma_ret = BDH_GPU(obs_dim=max_ret_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
    
    # Look for MAPPO checkpoints
    mappo_wh_path = None
    mappo_ret_path = None
    for cand_wh, cand_ret in [
        (f"SynapSCIM_mappo_checkpoints/bdh_mappo_wh_{network_id}_20000.pt", f"SynapSCIM_mappo_checkpoints/bdh_mappo_ret_{network_id}_20000.pt"),
        ("SynapSCIM_mappo_checkpoints/bdh_mappo_wh_20000.pt", "SynapSCIM_mappo_checkpoints/bdh_mappo_ret_20000.pt"),
        ("SynapSCIM_mappo_checkpoints/bdh_mappo_wh.pt", "SynapSCIM_mappo_checkpoints/bdh_mappo_ret.pt"),
        ("bdh_mappo_wh.pt", "bdh_mappo_ret.pt")
    ]:
        if os.path.exists(cand_wh) and os.path.exists(cand_ret):
            mappo_wh_path = cand_wh
            mappo_ret_path = cand_ret
            break
            
    if mappo_wh_path is not None and mappo_ret_path is not None:
        print(f"Loading MAPPO weights from {mappo_wh_path} and {mappo_ret_path}...")
        
        state_dict_wh = torch.load(mappo_wh_path, map_location=device)
        new_wh = OrderedDict()
        for k, v in state_dict_wh.items():
            if k.startswith("_orig_mod."):
                new_wh[k[10:]] = v
            else:
                new_wh[k] = v
        model_ma_wh.load_state_dict(new_wh)
        
        state_dict_ret = torch.load(mappo_ret_path, map_location=device)
        new_ret = OrderedDict()
        for k, v in state_dict_ret.items():
            if k.startswith("_orig_mod."):
                new_ret[k[10:]] = v
            else:
                new_ret[k] = v
        model_ma_ret.load_state_dict(new_ret)
    else:
        print("[Warning] Trained MAPPO weights not found. Using randomly initialized models.")
        
    model_ma_wh.eval()
    model_ma_ret.eval()
    
    # 4. Instantiate and load MLP-PPO baseline agent
    mlp_model = MLP_GPU(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=128).to(device)
    mlp_path = "SynapSCIM_mlpppo_checkpoints/mlp_ppo_model_final.pt"
    if os.path.exists(mlp_path):
        print(f"Loading trained MLP-PPO weights from {mlp_path}...")
        state_dict_mlp = torch.load(mlp_path, map_location=device)
        new_mlp = OrderedDict()
        for k, v in state_dict_mlp.items():
            if k.startswith("_orig_mod."):
                new_mlp[k[10:]] = v
            else:
                new_mlp[k] = v
        mlp_model.load_state_dict(new_mlp)
    else:
        print(f"[Warning] MLP weights not found at {mlp_path}. Using randomly initialized MLP model.")
    mlp_model.eval()

    # 4.2 Instantiate and load GNN-PPO baseline agent
    gnn_model = GNN_PPO_Model(obs_dim=obs_dim, act_dim=act_dim, num_nodes=act_dim, hidden_dim=64).to(device)
    gnn_path = "SynapSCIM_gnn_checkpoints/gnn_ppo_model_final.pt"
    if os.path.exists(gnn_path):
        print(f"Loading trained GNN-PPO weights from {gnn_path}...")
        state_dict_gnn = torch.load(gnn_path, map_location=device)
        new_gnn = OrderedDict()
        for k, v in state_dict_gnn.items():
            if k.startswith("_orig_mod."):
                new_gnn[k[10:]] = v
            else:
                new_gnn[k] = v
        gnn_model.load_state_dict(new_gnn)
    else:
        print(f"[Warning] GNN weights not found at {gnn_path}. Using randomly initialized GNN model.")
    gnn_model.eval()

    # 4.5 Instantiate and tune traditional heuristics
    print("Tuning traditional Base-Stock and (s, Q) policies...")
    bs_policy, sq_policy = tune_baselines(env, eval_demands, steps=100)
    
    # 5. Run simulations
    print("Running simulations...")
    bdh_results = run_evaluation(env, "bdh_ppo", bdh_model, eval_demands, T_context)
    mappo_results = run_mappo_evaluation(env_ma, model_ma_wh, model_ma_ret, eval_demands, T_context)
    mlp_results = run_evaluation(env, "mlp_ppo", mlp_model, eval_demands, T_context)
    gnn_results = run_evaluation(env, "gnn_ppo", gnn_model, eval_demands, T_context)
    bs_results = run_evaluation(env, "base_stock", bs_policy, eval_demands, T_context)
    sq_results = run_evaluation(env, "sq", sq_policy, eval_demands, T_context)
    
    # 6. Output comparison results to stdout
    print("\n=============================================================")
    print(f"               BENCHMARK RESULTS (Network {network_id})")
    print("=============================================================")
    print(f"{'Policy':<25} | {'Total Cost':<12} | {'Holding':<10} | {'Backorder':<10} | {'Service Level':<14}")
    print("-" * 80)
    print(f"{'BDH-PPO (Centralized)':<25} | {bdh_results['total_cost']:12.2f} | {bdh_results['holding_cost']:10.2f} | {bdh_results['backorder_cost']:10.2f} | {bdh_results['service_level']:12.2f}%")
    print(f"{'MAPPO (Decentralized)':<25} | {mappo_results['total_cost']:12.2f} | {mappo_results['holding_cost']:10.2f} | {mappo_results['backorder_cost']:10.2f} | {mappo_results['service_level']:12.2f}%")
    print(f"{'MLP-PPO (DRL Baseline)':<25} | {mlp_results['total_cost']:12.2f} | {mlp_results['holding_cost']:10.2f} | {mlp_results['backorder_cost']:10.2f} | {mlp_results['service_level']:12.2f}%")
    print(f"{'GNN-PPO (GNN Baseline)':<25} | {gnn_results['total_cost']:12.2f} | {gnn_results['holding_cost']:10.2f} | {gnn_results['backorder_cost']:10.2f} | {gnn_results['service_level']:12.2f}%")
    print(f"{'Base-Stock':<25} | {bs_results['total_cost']:12.2f} | {bs_results['holding_cost']:10.2f} | {bs_results['backorder_cost']:10.2f} | {bs_results['service_level']:12.2f}%")
    print(f"{'s, Q Policy':<25} | {sq_results['total_cost']:12.2f} | {sq_results['holding_cost']:10.2f} | {sq_results['backorder_cost']:10.2f} | {sq_results['service_level']:12.2f}%")
    print("=============================================================\n")
    
    # 7. Save results to report files
    os.makedirs("reports/centralized_ppo", exist_ok=True)
    
    # Save text report
    report_txt_path = "reports/centralized_ppo/benchmark_results.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=============================================================\n")
        f.write(f"               BENCHMARK RESULTS (Network {network_id})\n")
        f.write("=============================================================\n")
        f.write(f"{'Policy':<25} | {'Total Cost':<12} | {'Holding':<10} | {'Backorder':<10} | {'Service Level (Fill Rate)':<14}\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'BDH-PPO (Centralized)':<25} | {bdh_results['total_cost']:12.2f} | {bdh_results['holding_cost']:10.2f} | {bdh_results['backorder_cost']:10.2f} | {bdh_results['service_level']:12.2f}%\n")
        f.write(f"{'MAPPO (Decentralized)':<25} | {mappo_results['total_cost']:12.2f} | {mappo_results['holding_cost']:10.2f} | {mappo_results['backorder_cost']:10.2f} | {mappo_results['service_level']:12.2f}%\n")
        f.write(f"{'MLP-PPO (DRL Baseline)':<25} | {mlp_results['total_cost']:12.2f} | {mlp_results['holding_cost']:10.2f} | {mlp_results['backorder_cost']:10.2f} | {mlp_results['service_level']:12.2f}%\n")
        f.write(f"{'GNN-PPO (GNN Baseline)':<25} | {gnn_results['total_cost']:12.2f} | {gnn_results['holding_cost']:10.2f} | {gnn_results['backorder_cost']:10.2f} | {gnn_results['service_level']:12.2f}%\n")
        f.write(f"{'Base-Stock':<25} | {bs_results['total_cost']:12.2f} | {bs_results['holding_cost']:10.2f} | {bs_results['backorder_cost']:10.2f} | {bs_results['service_level']:12.2f}%\n")
        f.write(f"{'s, Q Policy':<25} | {sq_results['total_cost']:12.2f} | {sq_results['holding_cost']:10.2f} | {sq_results['backorder_cost']:10.2f} | {sq_results['service_level']:12.2f}%\n")
        f.write("=============================================================\n")
    print(f"Benchmark results table saved to {report_txt_path}")
    
    # Generate multi-panel comparison chart
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Subplot 1: Cumulative Cost
    axes[0].plot(bdh_results["cost_trajectory"], label="BDH-PPO (Centralized)", color="#1f77b4", linewidth=2)
    axes[0].plot(mappo_results["cost_trajectory"], label="MAPPO (Decentralized)", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].plot(mlp_results["cost_trajectory"], label="MLP-PPO (DRL Baseline)", color="#9467bd", linewidth=2, linestyle="-.")
    axes[0].plot(gnn_results["cost_trajectory"], label="GNN-PPO (GNN Baseline)", color="#8c564b", linewidth=2, linestyle=":")
    axes[0].plot(bs_results["cost_trajectory"], label="Base-Stock Heuristic", color="#2ca02c", linewidth=2)
    axes[0].plot(sq_results["cost_trajectory"], label="s, Q Policy Heuristic", color="#d62728", linewidth=2)
    axes[0].set_title("Cumulative Operational Cost Comparison", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Total Cost ($)", fontsize=10)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Subplot 2: Total Retailer Stock levels over time
    bdh_sum_stock = [np.sum(s) for s in bdh_results["ret_stock_trajectory"]]
    mappo_sum_stock = [np.sum(s) for s in mappo_results["ret_stock_trajectory"]]
    mlp_sum_stock = [np.sum(s) for s in mlp_results["ret_stock_trajectory"]]
    gnn_sum_stock = [np.sum(s) for s in gnn_results["ret_stock_trajectory"]]
    bs_sum_stock = [np.sum(s) for s in bs_results["ret_stock_trajectory"]]
    sq_sum_stock = [np.sum(s) for s in sq_results["ret_stock_trajectory"]]
    
    axes[1].plot(bdh_sum_stock, label="BDH-PPO (Centralized)", color="#1f77b4", linewidth=2)
    axes[1].plot(mappo_sum_stock, label="MAPPO (Decentralized)", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[1].plot(mlp_sum_stock, label="MLP-PPO (DRL Baseline)", color="#9467bd", linewidth=2, linestyle="-.")
    axes[1].plot(gnn_sum_stock, label="GNN-PPO (GNN Baseline)", color="#8c564b", linewidth=2, linestyle=":")
    axes[1].plot(bs_sum_stock, label="Base-Stock Heuristic", color="#2ca02c", linewidth=2)
    axes[1].plot(sq_sum_stock, label="s, Q Policy Heuristic", color="#d62728", linewidth=2)
    axes[1].set_title("Total Retailer On-Hand Stock Levels", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Stock Units", fontsize=10)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    # Subplot 3: Total Retailer Backorders over time
    bdh_sum_backorder = [np.sum(b) for b in bdh_results["backorder_trajectory"]]
    mappo_sum_backorder = [np.sum(b) for b in mappo_results["backorder_trajectory"]]
    mlp_sum_backorder = [np.sum(b) for b in mlp_results["backorder_trajectory"]]
    gnn_sum_backorder = [np.sum(b) for b in gnn_results["backorder_trajectory"]]
    bs_sum_backorder = [np.sum(b) for b in bs_results["backorder_trajectory"]]
    sq_sum_backorder = [np.sum(b) for b in sq_results["backorder_trajectory"]]
    
    axes[2].plot(bdh_sum_backorder, label="BDH-PPO (Centralized)", color="#1f77b4", linewidth=2)
    axes[2].plot(mappo_sum_backorder, label="MAPPO (Decentralized)", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[2].plot(mlp_sum_backorder, label="MLP-PPO (DRL Baseline)", color="#9467bd", linewidth=2, linestyle="-.")
    axes[2].plot(gnn_sum_backorder, label="GNN-PPO (GNN Baseline)", color="#8c564b", linewidth=2, linestyle=":")
    axes[2].plot(bs_sum_backorder, label="Base-Stock Heuristic", color="#2ca02c", linewidth=2)
    axes[2].plot(sq_sum_backorder, label="s, Q Policy Heuristic", color="#d62728", linewidth=2)
    axes[2].set_title("Total Retailer Backorders (Shortages)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Backorder Units", fontsize=10)
    axes[2].set_xlabel("Simulation Time Steps", fontsize=10)
    axes[2].legend(loc="upper left")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    chart_path = "reports/centralized_ppo/evaluation_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Evaluation comparison chart saved to {chart_path}")
    
    return {
        "bdh": bdh_results,
        "mappo": mappo_results,
        "mlp": mlp_results,
        "gnn": gnn_results,
        "base_stock": bs_results,
        "sq": sq_results
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate supply chain controllers.")
    parser.add_argument("--network_id", type=int, default=1, help="Willems network ID.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained BDH-PPO model weights.")
    args = parser.parse_args()
    
    evaluate_all(network_id=args.network_id, model_path=args.model_path)
