import numpy as np
import pandas as pd
import os

def get_willems_config(network_id=1, max_retailers_limit=5):
    """
    Loads configuration parameters for the multi-echelon supply chain.
    It first attempts to dynamically parse the downloaded Willems (2008) Excel file ('data/msom-willems.xls').
    If the file is not present or an error occurs, it falls back to hardcoded representative configurations.
    
    Args:
        network_id (int): Network ID from 1 to 38.
        max_retailers_limit (int): Limit the action space size for RL by selecting the top-k retailers by demand.
    """
    # Resolve path relative to the willems_loader.py file location (project root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    excel_path = os.path.join(project_dir, "data", "msom-willems.xls")
    
    if os.path.exists(excel_path):
        try:
            # Format sheet prefix
            prefix = f"{network_id:02d}"
            
            # Read sheets
            df_sd = pd.read_excel(excel_path, sheet_name=f"{prefix}_SD")
            df_ll = pd.read_excel(excel_path, sheet_name=f"{prefix}_LL")
            
            # 1. Identify Retail stages
            retail_stages = df_sd[df_sd['stageClassification'] == 'Retail']
            if len(retail_stages) == 0:
                retail_stages = df_sd[df_sd['relDepth'] == 0]
                
            # Sort retailers by average demand descending and slice to limit action space size
            if 'avgDemand' in retail_stages.columns:
                retail_stages = retail_stages.sort_values(by='avgDemand', ascending=False)
            
            # Apply limit if necessary to keep RL continuous control stable
            if len(retail_stages) > max_retailers_limit:
                # Print a notice about the limit
                print(f"[Willems Loader] Network {network_id} has {len(retail_stages)} retailers. "
                      f"Limiting to top {max_retailers_limit} by demand to maintain stable RL action space.")
                retail_stages = retail_stages.iloc[:max_retailers_limit]
                
            num_retailers = len(retail_stages)
            retailer_names = list(retail_stages['Stage Name'])
            
            # 2. Identify Manufacturing stages
            manuf_stages = df_sd[df_sd['stageClassification'] == 'Manuf']
            if len(manuf_stages) == 0:
                manuf_stages = df_sd[df_sd['relDepth'] == 1]
                
            # 3. Holding Costs (scaled down to standard inventory range)
            wh_raw_cost = np.mean(manuf_stages['stageCost']) if len(manuf_stages) > 0 else 10.0
            cost_holding_wh = float(np.round(wh_raw_cost / 100.0 + 0.1, 2))
            
            cost_holding_ret = []
            for name in retailer_names:
                row = retail_stages[retail_stages['Stage Name'] == name].iloc[0]
                raw_cost = row['stageCost']
                if pd.isna(raw_cost) or raw_cost == 0:
                    # Retailers usually have higher holding costs than warehouses
                    cost_holding_ret.append(float(np.round(cost_holding_wh * 2.0, 2)))
                else:
                    cost_holding_ret.append(float(np.round(raw_cost / 100.0, 2)))
                    
            # 4. Lead Times
            lead_times = []
            for ret_name in retailer_names:
                sources = df_ll[df_ll['destinationStage'] == ret_name]['sourceStage'].tolist()
                source_times = []
                for src in sources:
                    src_row = df_sd[df_sd['Stage Name'] == src]
                    if len(src_row) > 0:
                        source_times.append(src_row.iloc[0]['stageTime'])
                
                # Mean lead time of sources, scale down if extremely large
                lt = int(np.round(np.mean(source_times))) if len(source_times) > 0 else 2
                # Clamp lead times between 1 and 5 for computational efficiency in queues
                lead_times.append(int(np.clip(lt // 2, 1, 5)))
                
            lead_time_prod = int(np.round(np.mean(manuf_stages['stageTime']))) if len(manuf_stages) > 0 else 2
            lead_time_prod = int(np.clip(lead_time_prod // 2, 1, 3))
            
            # 5. Demand profiles
            demand_mean = []
            demand_noise = []
            for name in retailer_names:
                row = retail_stages[retail_stages['Stage Name'] == name].iloc[0]
                mean_val = row.get('avgDemand', 25.0)
                noise_val = row.get('stdDevDemand', 3.0)
                
                if pd.isna(mean_val) or mean_val == 0:
                    mean_val = 25.0
                if pd.isna(noise_val) or noise_val == 0:
                    noise_val = mean_val * 0.12
                    
                # Scale demand down to standard ranges (e.g. between 10 and 50)
                scaled_mean = float(np.clip(mean_val / 2.0, 10.0, 50.0))
                scaled_noise = float(np.clip(noise_val / 2.0, 1.0, 8.0))
                
                demand_mean.append(float(np.round(scaled_mean, 2)))
                demand_noise.append(float(np.round(scaled_noise, 2)))
                
            # Default helper parameters
            return {
                "num_retailers": num_retailers,
                "lead_times": lead_times,
                "lead_time_prod": lead_time_prod,
                "cap_warehouse": float(num_retailers * 200.0),
                "cap_retailers": [150.0] * num_retailers,
                "max_prod": float(num_retailers * 40.0),
                "max_ship": [50.0] * num_retailers,
                "cost_holding_wh": cost_holding_wh,
                "cost_holding_ret": cost_holding_ret,
                "cost_backorder": [float(np.round(c * 5.0, 2)) for c in cost_holding_ret],
                "cost_shipping": [float(np.round(c * 1.5, 2)) for c in cost_holding_ret],
                "cost_production": float(np.round(cost_holding_wh * 2.0, 2)),
                "demand_amp": [float(np.round(m * 0.6, 2)) for m in demand_mean],
                "demand_period": [30.0] * num_retailers,
                "demand_mean": demand_mean,
                "demand_noise": demand_noise,
                "hist_len": 5,
                "max_steps": 100
            }
        except Exception as e:
            print(f"[Warning] Failed to parse Excel sheet for Network {network_id} due to: {e}. "
                  "Falling back to hardcoded configurations.")
                  
    # FALLBACK: Hardcoded representative configurations
    if network_id == 1:
        return {
            "num_retailers": 3,
            "lead_times": [2, 3, 4],
            "lead_time_prod": 1,
            "cap_warehouse": 500.0,
            "cap_retailers": [150.0, 150.0, 150.0],
            "max_prod": 100.0,
            "max_ship": [50.0, 50.0, 50.0],
            "cost_holding_wh": 0.5,
            "cost_holding_ret": [1.0, 1.2, 1.5],
            "cost_backorder": [5.0, 6.0, 7.5],
            "cost_shipping": [1.5, 2.0, 2.5],
            "cost_production": 2.0,
            "demand_amp": [15.0, 12.0, 10.0],
            "demand_period": [30.0, 30.0, 30.0],
            "demand_mean": [25.0, 20.0, 15.0],
            "demand_noise": [3.0, 2.5, 2.0],
            "hist_len": 5,
            "max_steps": 100
        }
    elif network_id == 14 or network_id == 30:
        # Hardcoded equivalent for Network 14 (Medium)
        M = 5
        return {
            "num_retailers": M,
            "lead_times": [2, 3, 2, 4, 3],
            "lead_time_prod": 2,
            "cap_warehouse": 800.0,
            "cap_retailers": [200.0] * M,
            "max_prod": 150.0,
            "max_ship": [40.0] * M,
            "cost_holding_wh": 0.4,
            "cost_holding_ret": [0.9, 1.1, 1.0, 1.3, 1.2],
            "cost_backorder": [4.5, 5.5, 5.0, 6.5, 6.0],
            "cost_shipping": [1.2, 1.5, 1.3, 1.8, 1.6],
            "cost_production": 1.8,
            "demand_amp": [15.0] * M,
            "demand_period": [30.0] * M,
            "demand_mean": [20.0, 22.0, 18.0, 25.0, 15.0],
            "demand_noise": [2.5] * M,
            "hist_len": 5,
            "max_steps": 100
        }
    else:
        # Default fallback for any other network id
        return get_willems_config(1)

def get_deterministic_demands(network_id=1, steps=100, seed=42):
    """
    Generates a deterministic sequence of customer demands for reproducible validation and testing.
    """
    config = get_willems_config(network_id)
    num_retailers = config["num_retailers"]
    amps = config["demand_amp"]
    periods = config["demand_period"]
    means = config["demand_mean"]
    noises = config["demand_noise"]
    
    # Set seed
    rng = np.random.default_rng(seed)
    
    demands = []
    for t in range(steps + config["hist_len"]):
        step_demands = []
        for i in range(num_retailers):
            val = amps[i] * np.sin(2.0 * np.pi * t / periods[i]) + means[i]
            val += rng.normal(0.0, noises[i])
            step_demands.append(max(0.0, val))
        demands.append(step_demands)
        
    return demands
