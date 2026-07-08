import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from env import MultiEchelonSupplyChainEnv
from bdh import BDH_GPU
from ppo import get_history
from willems_loader import get_willems_config, get_deterministic_demands
from baselines import tune_baselines

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
        if policy_type == "bdh_ppo":
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
        # Type II Service Level (Fill Rate) calculation: unsatisfied new demand is min(demands, backorders)
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
    
    # Calculate Service Level (Fill Rate: percentage of demand satisfied immediately)
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
        for candidate in ["bdh_ppo_model_3000.pt", "bdh_ppo_model_1000.pt", "bdh_ppo_model.pt"]:
            if os.path.exists(candidate):
                model_path = candidate
                break
        if model_path is None:
            model_path = "bdh_ppo_model.pt"
            
    if os.path.exists(model_path):
        print(f"Loading trained PPO weights from {model_path} on device {device}...")
        bdh_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[Warning] Trained weights not found at {model_path}. Using randomly initialized model.")
        
    bdh_model.eval()
    
    # 3. Instantiate and tune traditional heuristics
    print("Tuning traditional Base-Stock and (s, Q) policies...")
    bs_policy, sq_policy = tune_baselines(env, eval_demands, steps=100)
    
    # 4. Run simulations
    print("Running simulations...")
    bdh_results = run_evaluation(env, "bdh_ppo", bdh_model, eval_demands, T_context)
    bs_results = run_evaluation(env, "base_stock", bs_policy, eval_demands, T_context)
    sq_results = run_evaluation(env, "sq", sq_policy, eval_demands, T_context)
    
    # 5. Output comparison results to stdout
    print("\n=============================================================")
    print(f"               BENCHMARK RESULTS (Network {network_id})")
    print("=============================================================")
    print(f"{'Policy':<20} | {'Total Cost':<12} | {'Holding':<10} | {'Backorder':<10} | {'Service Level':<14}")
    print("-" * 75)
    print(f"{'BDH-PPO (Ours)':<20} | {bdh_results['total_cost']:12.2f} | {bdh_results['holding_cost']:10.2f} | {bdh_results['backorder_cost']:10.2f} | {bdh_results['service_level']:12.2f}%")
    print(f"{'Base-Stock':<20} | {bs_results['total_cost']:12.2f} | {bs_results['holding_cost']:10.2f} | {bs_results['backorder_cost']:10.2f} | {bs_results['service_level']:12.2f}%")
    print(f"{'s, Q Policy':<20} | {sq_results['total_cost']:12.2f} | {sq_results['holding_cost']:10.2f} | {sq_results['backorder_cost']:10.2f} | {sq_results['service_level']:12.2f}%")
    print("=============================================================\n")
    
    # 6. Save results to report files
    os.makedirs("reports/centralized_ppo", exist_ok=True)
    
    # Save text report
    report_txt_path = "reports/centralized_ppo/benchmark_results.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=============================================================\n")
        f.write(f"               BENCHMARK RESULTS (Network {network_id})\n")
        f.write("=============================================================\n")
        f.write(f"{'Policy':<20} | {'Total Cost':<12} | {'Holding':<10} | {'Backorder':<10} | {'Service Level (Fill Rate)':<14}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'BDH-PPO (Ours)':<20} | {bdh_results['total_cost']:12.2f} | {bdh_results['holding_cost']:10.2f} | {bdh_results['backorder_cost']:10.2f} | {bdh_results['service_level']:12.2f}%\n")
        f.write(f"{'Base-Stock':<20} | {bs_results['total_cost']:12.2f} | {bs_results['holding_cost']:10.2f} | {bs_results['backorder_cost']:10.2f} | {bs_results['service_level']:12.2f}%\n")
        f.write(f"{'s, Q Policy':<20} | {sq_results['total_cost']:12.2f} | {sq_results['holding_cost']:10.2f} | {sq_results['backorder_cost']:10.2f} | {sq_results['service_level']:12.2f}%\n")
        f.write("=============================================================\n")
    print(f"Benchmark results table saved to {report_txt_path}")
    
    # Generate multi-panel comparison chart
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Subplot 1: Cumulative Cost
    axes[0].plot(bdh_results["cost_trajectory"], label="BDH-PPO (Ours)", color="#1f77b4", linewidth=2)
    axes[0].plot(bs_results["cost_trajectory"], label="Base-Stock Heuristic", color="#2ca02c", linewidth=2)
    axes[0].plot(sq_results["cost_trajectory"], label="s, Q Policy Heuristic", color="#d62728", linewidth=2)
    axes[0].set_title("Cumulative Operational Cost Comparison", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Total Cost ($)", fontsize=10)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Subplot 2: Total Retailer Stock levels over time
    bdh_sum_stock = [np.sum(s) for s in bdh_results["ret_stock_trajectory"]]
    bs_sum_stock = [np.sum(s) for s in bs_results["ret_stock_trajectory"]]
    sq_sum_stock = [np.sum(s) for s in sq_results["ret_stock_trajectory"]]
    
    axes[1].plot(bdh_sum_stock, label="BDH-PPO (Ours)", color="#1f77b4", linewidth=2)
    axes[1].plot(bs_sum_stock, label="Base-Stock Heuristic", color="#2ca02c", linewidth=2)
    axes[1].plot(sq_sum_stock, label="s, Q Policy Heuristic", color="#d62728", linewidth=2)
    axes[1].set_title("Total Retailer On-Hand Stock Levels", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Stock Units", fontsize=10)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    # Subplot 3: Total Retailer Backorders over time
    bdh_sum_backorder = [np.sum(b) for b in bdh_results["backorder_trajectory"]]
    bs_sum_backorder = [np.sum(b) for b in bs_results["backorder_trajectory"]]
    sq_sum_backorder = [np.sum(b) for b in sq_results["backorder_trajectory"]]
    
    axes[2].plot(bdh_sum_backorder, label="BDH-PPO (Ours)", color="#1f77b4", linewidth=2)
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
