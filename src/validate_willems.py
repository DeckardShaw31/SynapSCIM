import os
import sys
import torch
import numpy as np
from env import MultiEchelonSupplyChainEnv
from bdh import BDH_GPU
from willems_loader import get_willems_config, get_deterministic_demands
from baselines import tune_baselines
from evaluate import run_evaluation

def validate_networks(network_ids=[1], model_path=None):
    print("\n=============================================================")
    print("           REAL-WORLD WILLEMS DATASET VALIDATION")
    print("=============================================================")
    
    os.makedirs("reports/centralized_ppo", exist_ok=True)
    report_path = "reports/centralized_ppo/willems_dataset_validation.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=============================================================\n")
        f.write("        WILLEMS (2008) DATASET BENCHMARK VALIDATION REPORT\n")
        f.write("=============================================================\n\n")
        
        for net_id in network_ids:
            f.write(f"--- Willems Network Topology ID: {net_id} ---\n")
            print(f"Validating Willems Network {net_id}...")
            
            # Load real-world Willems configuration
            config = get_willems_config(net_id)
            env = MultiEchelonSupplyChainEnv(config, mode="centralized")
            eval_demands = get_deterministic_demands(network_id=net_id, steps=100, seed=42)
            
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            
            # Load Centralized BDH-PPO model for this network
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = BDH_GPU(obs_dim=obs_dim, act_dim=act_dim, D=32, H=2, N=256, L=2).to(device)
            
            # Look for checkpoint specific to this network
            resolved_path = None
            if model_path is None:
                for candidate in [
                    f"SynapSCIM_checkpoints/bdh_ppo_model_{net_id}_20000.pt",
                    "SynapSCIM_checkpoints/bdh_ppo_model_20000.pt",
                    "SynapSCIM_checkpoints/bdh_ppo_model.pt",
                    f"bdh_ppo_model_{net_id}_20000.pt",
                    f"bdh_ppo_model_3000.pt",
                    f"bdh_ppo_model_1000.pt",
                    "bdh_ppo_model.pt"
                ]:
                    if os.path.exists(candidate):
                        resolved_path = candidate
                        break
            else:
                resolved_path = model_path
            
            is_random = True
            if resolved_path is not None:
                print(f"Loading model weights from {resolved_path}...")
                # Automatically strip _orig_mod. prefix from compiled checkpoints
                state_dict = torch.load(resolved_path, map_location=device)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("_orig_mod."):
                        new_state_dict[k[10:]] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
                is_random = False
            else:
                print(f"[Warning] No trained weights found for Network {net_id}. Using randomly initialized model.")
            
            model.eval()
            
            # Tune baselines
            print("Tuning traditional Base-Stock and (s, Q) policies...")
            bs_policy, sq_policy = tune_baselines(env, eval_demands, steps=100)
            
            # Run evaluations
            res_bdh = run_evaluation(env, "bdh_ppo", model, eval_demands)
            res_bs = run_evaluation(env, "base_stock", bs_policy, eval_demands)
            res_sq = run_evaluation(env, "sq", sq_policy, eval_demands)
            
            # Write to report file
            status_str = "Trained" if not is_random else "Random (Untrained)"
            f.write(f"Model Status: {status_str} | Weights Path: {resolved_path}\n")
            f.write(f"Parameters: Retailers: {env.num_retailers} | Lead Times (Ret): {env.lead_times} | Lead Time (Prod): {env.lead_time_prod}\n")
            f.write(f"{'Policy':<22} | {'Total Cost':<12} | {'Holding':<10} | {'Backorder':<10} | {'Fill Rate (SL)':<14}\n")
            f.write("-" * 75 + "\n")
            f.write(f"{'BDH-PPO (Ours)':<22} | {res_bdh['total_cost']:12.2f} | {res_bdh['holding_cost']:10.2f} | {res_bdh['backorder_cost']:10.2f} | {res_bdh['service_level']:12.2f}%\n")
            f.write(f"{'Base-Stock Baseline':<22} | {res_bs['total_cost']:12.2f} | {res_bs['holding_cost']:10.2f} | {res_bs['backorder_cost']:10.2f} | {res_bs['service_level']:12.2f}%\n")
            f.write(f"{'s, Q Policy Baseline':<22} | {res_sq['total_cost']:12.2f} | {res_sq['holding_cost']:10.2f} | {res_sq['backorder_cost']:10.2f} | {res_sq['service_level']:12.2f}%\n")
            f.write("=============================================================\n\n")
            
            print(f"Network {net_id} validation completed.")
            
    print(f"Real-world Willems dataset validation report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate Centralized PPO on Willems topologies.")
    parser.add_argument("--networks", type=str, default="1", help="Comma-separated Willems Network IDs (e.g. 1,14,30)")
    parser.add_argument("--model_path", type=str, default=None, help="Explicit path to model weights file.")
    args = parser.parse_args()
    
    net_list = [int(x.strip()) for x in args.networks.split(",")]
    validate_networks(net_list, model_path=args.model_path)
