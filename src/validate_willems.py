import os
import sys
import torch
import numpy as np
from collections import OrderedDict
from env import MultiEchelonSupplyChainEnv
from bdh import BDH_GPU
from willems_loader import get_willems_config, get_deterministic_demands
from baselines import tune_baselines
from evaluate import run_evaluation

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
        "service_level": service_level
    }

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
                state_dict = torch.load(resolved_path, map_location=device)
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("_orig_mod."):
                        new_state_dict[k[10:]] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
                is_random = False
            else:
                print(f"[Warning] No trained PPO weights found for Network {net_id}. Using randomly initialized model.")
            
            model.eval()
            
            # Load Cooperative MAPPO models
            env_ma = MultiEchelonSupplyChainEnv(config, mode="multi_agent")
            wh_obs_dim = env_ma.observation_spaces["warehouse"].shape[0]
            max_ret_obs_dim = max(env_ma.observation_spaces[f"retailer_{i}"].shape[0] for i in range(env_ma.num_retailers))
            
            model_ma_wh = BDH_GPU(obs_dim=wh_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
            model_ma_ret = BDH_GPU(obs_dim=max_ret_obs_dim, act_dim=1, D=32, H=2, N=256, L=2).to(device)
            
            # Look for MAPPO checkpoints
            mappo_wh_path = None
            mappo_ret_path = None
            for cand_wh, cand_ret in [
                (f"SynapSCIM_mappo_checkpoints/bdh_mappo_wh_{net_id}_20000.pt", f"SynapSCIM_mappo_checkpoints/bdh_mappo_ret_{net_id}_20000.pt"),
                ("SynapSCIM_mappo_checkpoints/bdh_mappo_wh_20000.pt", "SynapSCIM_mappo_checkpoints/bdh_mappo_ret_20000.pt"),
                ("SynapSCIM_mappo_checkpoints/bdh_mappo_wh.pt", "SynapSCIM_mappo_checkpoints/bdh_mappo_ret.pt"),
                ("bdh_mappo_wh.pt", "bdh_mappo_ret.pt")
            ]:
                if os.path.exists(cand_wh) and os.path.exists(cand_ret):
                    mappo_wh_path = cand_wh
                    mappo_ret_path = cand_ret
                    break
            
            is_mappo_random = True
            if mappo_wh_path is not None and mappo_ret_path is not None:
                print(f"Loading MAPPO weights from {mappo_wh_path} & {mappo_ret_path}...")
                
                # Load wh
                state_dict_wh = torch.load(mappo_wh_path, map_location=device)
                new_wh = OrderedDict()
                for k, v in state_dict_wh.items():
                    if k.startswith("_orig_mod."):
                        new_wh[k[10:]] = v
                    else:
                        new_wh[k] = v
                model_ma_wh.load_state_dict(new_wh)
                
                # Load ret
                state_dict_ret = torch.load(mappo_ret_path, map_location=device)
                new_ret = OrderedDict()
                for k, v in state_dict_ret.items():
                    if k.startswith("_orig_mod."):
                        new_ret[k[10:]] = v
                    else:
                        new_ret[k] = v
                model_ma_ret.load_state_dict(new_ret)
                
                is_mappo_random = False
            else:
                print(f"[Warning] No trained MAPPO weights found for Network {net_id}. Using randomly initialized models.")
            
            model_ma_wh.eval()
            model_ma_ret.eval()
            
            # Tune baselines
            print("Tuning traditional Base-Stock and (s, Q) policies...")
            bs_policy, sq_policy = tune_baselines(env, eval_demands, steps=100)
            
            # Run evaluations
            res_bdh = run_evaluation(env, "bdh_ppo", model, eval_demands)
            res_mappo = run_mappo_evaluation(env_ma, model_ma_wh, model_ma_ret, eval_demands)
            res_bs = run_evaluation(env, "base_stock", bs_policy, eval_demands)
            res_sq = run_evaluation(env, "sq", sq_policy, eval_demands)
            
            # Write to report file
            status_str = "Trained" if not is_random else "Random (Untrained)"
            status_ma_str = "Trained" if not is_mappo_random else "Random (Untrained)"
            f.write(f"Centralized PPO Status: {status_str} | Weights Path: {resolved_path}\n")
            f.write(f"Decentralized MAPPO Status: {status_ma_str} | WH Path: {mappo_wh_path} | RET Path: {mappo_ret_path}\n")
            f.write(f"Parameters: Retailers: {env.num_retailers} | Lead Times (Ret): {env.lead_times} | Lead Time (Prod): {env.lead_time_prod}\n")
            f.write(f"{'Policy':<22} | {'Total Cost':<12} | {'Holding':<10} | {'Backorder':<10} | {'Fill Rate (SL)':<14}\n")
            f.write("-" * 75 + "\n")
            f.write(f"{'BDH-PPO (Centralized)':<22} | {res_bdh['total_cost']:12.2f} | {res_bdh['holding_cost']:10.2f} | {res_bdh['backorder_cost']:10.2f} | {res_bdh['service_level']:12.2f}%\n")
            f.write(f"{'MAPPO (Decentralized)':<22} | {res_mappo['total_cost']:12.2f} | {res_mappo['holding_cost']:10.2f} | {res_mappo['backorder_cost']:10.2f} | {res_mappo['service_level']:12.2f}%\n")
            f.write(f"{'Base-Stock Baseline':<22} | {res_bs['total_cost']:12.2f} | {res_bs['holding_cost']:10.2f} | {res_bs['backorder_cost']:10.2f} | {res_bs['service_level']:12.2f}%\n")
            f.write(f"{'s, Q Policy Baseline':<22} | {res_sq['total_cost']:12.2f} | {res_sq['holding_cost']:10.2f} | {res_sq['backorder_cost']:10.2f} | {res_sq['service_level']:12.2f}%\n")
            f.write("=============================================================\n\n")
            
            print(f"Network {net_id} validation completed.")
            
    print(f"Real-world Willems dataset validation report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate Centralized and Decentralized PPO on Willems topologies.")
    parser.add_argument("--networks", type=str, default="1", help="Comma-separated Willems Network IDs (e.g. 1,14,30)")
    parser.add_argument("--model_path", type=str, default=None, help="Explicit path to model weights file.")
    args = parser.parse_args()
    
    net_list = [int(x.strip()) for x in args.networks.split(",")]
    validate_networks(net_list, model_path=args.model_path)
