import numpy as np

class BaseStockPolicy:
    """
    A tuned Base-Stock (Order-Up-To) policy.
    Maintains the inventory position at each node at a target base-stock level S.
    """
    def __init__(self, env, safety_factors=None):
        self.env = env
        self.num_retailers = env.num_retailers
        self.lead_times = env.lead_times
        self.lead_time_prod = env.lead_time_prod
        
        # Calculate demand properties
        self.mean_demands = np.array(env.demand_mean)
        self.noise_demands = np.array(env.demand_noise)
        
        # Set safety factors (default tuned values)
        if safety_factors is None:
            self.k_retailers = [1.5] * self.num_retailers
            self.k_wh = 1.0
        else:
            self.k_retailers = safety_factors.get("k_retailers", [1.5] * self.num_retailers)
            self.k_wh = safety_factors.get("k_wh", 1.0)
            
        self.update_targets()

    def update_targets(self):
        # Target S for retailers: Mean demand over lead time + review period (1) + safety stock
        self.S_retailers = []
        for i in range(self.num_retailers):
            lt = self.lead_times[i]
            # Lead time + 1 review period
            coverage_period = lt + 1
            mean_demand_cov = self.mean_demands[i] * coverage_period
            std_demand_cov = self.noise_demands[i] * np.sqrt(coverage_period)
            
            S = mean_demand_cov + self.k_retailers[i] * std_demand_cov
            self.S_retailers.append(min(self.env.cap_retailers[i], S))
            
        # Target S for central warehouse
        # Warehouse covers factory lead time + average retailer lead time
        coverage_period_wh = self.lead_time_prod + 1
        mean_demand_wh = np.sum(self.mean_demands) * coverage_period_wh
        std_demand_wh = np.sqrt(np.sum(self.noise_demands**2)) * np.sqrt(coverage_period_wh)
        
        # S_wh represents system-wide inventory target
        self.S_wh = mean_demand_wh + self.k_wh * std_demand_wh + sum(self.S_retailers)
        self.S_wh = min(self.env.cap_warehouse + sum(self.env.cap_retailers), self.S_wh)

    def get_action(self, obs):
        # Parse observation vector
        # Warehouse stock: obs[0]
        wh_stock = obs[0]
        # Retailer stocks: obs[1 : 1+M]
        ret_stocks = obs[1 : 1 + self.num_retailers]
        # Retailer backorders: obs[1+M : 1+2M]
        ret_backorders = obs[1 + self.num_retailers : 1 + 2 * self.num_retailers]
        
        # Production pipeline queue
        start_idx = 1 + 2 * self.num_retailers
        end_idx = start_idx + self.lead_time_prod
        prod_pipeline = obs[start_idx:end_idx]
        
        # Shipment pipelines
        ship_pipelines = []
        curr_idx = end_idx
        for lt in self.lead_times:
            ship_pipelines.append(obs[curr_idx : curr_idx + lt])
            curr_idx += lt
            
        # 1. Retailer inventory positions and orders
        orders = []
        for i in range(self.num_retailers):
            # IP = on-hand - backorder + in-transit
            ip = ret_stocks[i] - ret_backorders[i] + np.sum(ship_pipelines[i])
            order = max(0.0, self.S_retailers[i] - ip)
            orders.append(order)
            
        # 2. Warehouse inventory position and production order
        # System-wide IP = warehouse stock + production in transit + sum of retailer IPs
        system_ip = wh_stock + np.sum(prod_pipeline) + np.sum([
            ret_stocks[i] - ret_backorders[i] + np.sum(ship_pipelines[i])
            for i in range(self.num_retailers)
        ])
        prod_order = max(0.0, self.S_wh - system_ip)
        
        # Map actions back to [0, 1] range
        act_prod_rate = np.clip(prod_order / self.env.max_prod, 0.0, 1.0)
        act_ship_rates = np.clip(np.array(orders) / np.array(self.env.max_ship), 0.0, 1.0)
        
        return np.concatenate([[act_prod_rate], act_ship_rates])

class sQPolicy:
    """
    A tuned (s, Q) policy.
    Places a fixed order Q when the inventory position falls below reorder point s.
    """
    def __init__(self, env, safety_factors=None):
        self.env = env
        self.num_retailers = env.num_retailers
        self.lead_times = env.lead_times
        self.lead_time_prod = env.lead_time_prod
        
        self.mean_demands = np.array(env.demand_mean)
        self.noise_demands = np.array(env.demand_noise)
        
        if safety_factors is None:
            self.k_retailers = [1.0] * self.num_retailers
            self.k_wh = 0.5
        else:
            self.k_retailers = safety_factors.get("k_retailers", [1.0] * self.num_retailers)
            self.k_wh = safety_factors.get("k_wh", 0.5)
            
        self.update_targets()

    def update_targets(self):
        # Target reorder points s and sizes Q
        self.s_retailers = []
        self.Q_retailers = []
        
        for i in range(self.num_retailers):
            lt = self.lead_times[i]
            # Reorder point s: Mean demand over lead time + safety stock
            s = self.mean_demands[i] * lt + self.k_retailers[i] * self.noise_demands[i] * np.sqrt(lt) if lt > 0 else 0.0
            # Order size Q: standard Economic Order Quantity (EOQ) heuristic approximation
            # We scale Q based on holding and shipping cost
            holding_cost = self.env.cost_holding_ret[i]
            setup_cost = self.env.cost_shipping[i] * 10.0  # Assumed setup cost overhead
            Q = np.sqrt(2.0 * self.mean_demands[i] * setup_cost / holding_cost)
            
            self.s_retailers.append(min(self.env.cap_retailers[i], s))
            self.Q_retailers.append(np.clip(Q, 10.0, self.env.max_ship[i]))
            
        # Warehouse reorder points s and sizes Q
        s_wh = np.sum(self.mean_demands) * self.lead_time_prod + self.k_wh * np.sqrt(np.sum(self.noise_demands**2)) * np.sqrt(self.lead_time_prod)
        setup_cost_wh = self.env.cost_production * 20.0
        Q_wh = np.sqrt(2.0 * np.sum(self.mean_demands) * setup_cost_wh / self.env.cost_holding_wh)
        
        self.s_wh = min(self.env.cap_warehouse, s_wh + sum(self.s_retailers))
        self.Q_wh = np.clip(Q_wh, 20.0, self.env.max_prod)

    def get_action(self, obs):
        wh_stock = obs[0]
        ret_stocks = obs[1 : 1 + self.num_retailers]
        ret_backorders = obs[1 + self.num_retailers : 1 + 2 * self.num_retailers]
        
        start_idx = 1 + 2 * self.num_retailers
        end_idx = start_idx + self.lead_time_prod
        prod_pipeline = obs[start_idx:end_idx]
        
        ship_pipelines = []
        curr_idx = end_idx
        for lt in self.lead_times:
            ship_pipelines.append(obs[curr_idx : curr_idx + lt])
            curr_idx += lt
            
        # 1. Retailer (s, Q) logic
        orders = []
        for i in range(self.num_retailers):
            ip = ret_stocks[i] - ret_backorders[i] + np.sum(ship_pipelines[i])
            if ip < self.s_retailers[i]:
                # Order Q
                orders.append(self.Q_retailers[i])
            else:
                orders.append(0.0)
                
        # 2. Warehouse (s, Q) logic
        system_ip = wh_stock + np.sum(prod_pipeline) + np.sum([
            ret_stocks[i] - ret_backorders[i] + np.sum(ship_pipelines[i])
            for i in range(self.num_retailers)
        ])
        
        if system_ip < self.s_wh:
            prod_order = self.Q_wh
        else:
            prod_order = 0.0
            
        # Map actions back to [0, 1] range
        act_prod_rate = np.clip(prod_order / self.env.max_prod, 0.0, 1.0)
        act_ship_rates = np.clip(np.array(orders) / np.array(self.env.max_ship), 0.0, 1.0)
        
        return np.concatenate([[act_prod_rate], act_ship_rates])

def tune_baselines(env, deterministic_demands, steps=100):
    """
    Runs a fast grid search to find the optimal safety factors for Base-Stock and (s, Q) policies
    on the provided demand sequence, minimizing total cost.
    """
    best_bs_cost = np.inf
    best_bs_factors = None
    best_sq_cost = np.inf
    best_sq_factors = None
    
    # Range of safety factors to search
    k_range = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    # 1. Tune Base Stock Policy
    for k_r in [1.0, 1.5, 2.0]:
        for k_w in [0.5, 1.0, 1.5]:
            factors = {"k_retailers": [k_r] * env.num_retailers, "k_wh": k_w}
            policy = BaseStockPolicy(env, factors)
            
            # Run simulation
            env.eval_demand = deterministic_demands
            obs, _ = env.reset(seed=42)
            total_cost = 0.0
            
            for _ in range(steps):
                action = policy.get_action(obs)
                obs, reward, term, trunc, info = env.step(action)
                total_cost += info["total_cost"]
                if term or trunc:
                    break
                    
            if total_cost < best_bs_cost:
                best_bs_cost = total_cost
                best_bs_factors = factors
                
    # 2. Tune (s, Q) Policy
    for k_r in [0.5, 1.0, 1.5]:
        for k_w in [0.0, 0.5, 1.0]:
            factors = {"k_retailers": [k_r] * env.num_retailers, "k_wh": k_w}
            policy = sQPolicy(env, factors)
            
            # Run simulation
            env.eval_demand = deterministic_demands
            obs, _ = env.reset(seed=42)
            total_cost = 0.0
            
            for _ in range(steps):
                action = policy.get_action(obs)
                obs, reward, term, trunc, info = env.step(action)
                total_cost += info["total_cost"]
                if term or trunc:
                    break
                    
            if total_cost < best_sq_cost:
                best_sq_cost = total_cost
                best_sq_factors = factors
                
    # Return tuned policies
    env.eval_demand = None
    tuned_bs = BaseStockPolicy(env, best_bs_factors)
    tuned_sq = sQPolicy(env, best_sq_factors)
    
    return tuned_bs, tuned_sq
