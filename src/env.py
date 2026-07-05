import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiEchelonSupplyChainEnv(gym.Env):
    """
    A custom Gymnasium environment representing a two-echelon supply chain.
    Supports both:
    1. Centralized Mode: Single agent controlling the entire network (production + M shipping actions).
    2. Multi-Agent Mode: M+1 independent cooperative agents (1 warehouse agent, M retailer agents)
       with localized partially observable states (POMDP) and a shared global reward to minimize system cost.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None, mode="centralized"):
        super().__init__()
        
        # Default configuration
        if config is None:
            config = {}
            
        self.mode = mode
        self.num_retailers = config.get("num_retailers", 3)
        self.lead_times = config.get("lead_times", [2, 3, 4])  # Lead times for shipments to retailers
        self.lead_time_prod = config.get("lead_time_prod", 1)  # Production lead time
        
        # Validate configuration
        assert len(self.lead_times) == self.num_retailers, "Length of lead times must match num_retailers."
        
        # Capacity limits
        self.cap_warehouse = config.get("cap_warehouse", 500.0)
        self.cap_retailers = config.get("cap_retailers", [150.0] * self.num_retailers)
        self.max_prod = config.get("max_prod", 100.0)
        self.max_ship = config.get("max_ship", [50.0] * self.num_retailers)
        
        # Cost parameters
        self.cost_holding_wh = config.get("cost_holding_wh", 0.5)
        self.cost_holding_ret = config.get("cost_holding_ret", [1.0] * self.num_retailers)
        self.cost_backorder = config.get("cost_backorder", [5.0] * self.num_retailers)
        self.cost_shipping = config.get("cost_shipping", [1.5] * self.num_retailers)
        self.cost_production = config.get("cost_production", 2.0)
        
        # Historical demand tracking length
        self.hist_len = config.get("hist_len", 5)
        
        # Demand profile for training (sinusoidal + stochastic noise)
        self.demand_amp = config.get("demand_amp", [15.0] * self.num_retailers)
        self.demand_period = config.get("demand_period", [30.0] * self.num_retailers)
        self.demand_mean = config.get("demand_mean", [25.0] * self.num_retailers)
        self.demand_noise = config.get("demand_noise", [3.0] * self.num_retailers)
        
        # Fixed demand trajectories for evaluation/testing (if provided)
        self.eval_demand = config.get("eval_demand", None)
        
        # Episode length
        self.reward_scale = config.get("reward_scale", 1000.0)
        self.max_steps = config.get("max_steps", 100)
        self.current_step = 0
        
        # Define agents list for multi-agent mode
        self.possible_agents = ["warehouse"] + [f"retailer_{i}" for i in range(self.num_retailers)]
        
        # Action space:
        if self.mode == "centralized":
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1 + self.num_retailers,),
                dtype=np.float32
            )
        else:
            # Multi-agent action spaces (dict)
            self.action_spaces = {
                agent: spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                for agent in self.possible_agents
            }
            self.action_space = self.action_spaces["warehouse"] # Gym standard fallback
            
        # State space size calculation for Centralized Mode:
        self.state_dim = (
            1 + 
            self.num_retailers * 2 + 
            self.lead_time_prod + 
            sum(self.lead_times) + 
            self.num_retailers * self.hist_len
        )
        
        if self.mode == "centralized":
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(self.state_dim,),
                dtype=np.float32
            )
        else:
            # Multi-agent observation spaces (dict) - POMDP
            self.observation_spaces = {}
            # Warehouse observations: on-hand, production pipeline, retailer backorders, demand history
            wh_obs_dim = 1 + self.lead_time_prod + self.num_retailers + self.num_retailers * self.hist_len
            self.observation_spaces["warehouse"] = spaces.Box(
                low=0.0, high=np.inf, shape=(wh_obs_dim,), dtype=np.float32
            )
            # Retailer observations: local stock, local backorder, local shipment pipeline, local demand history
            for i in range(self.num_retailers):
                ret_obs_dim = 1 + 1 + self.lead_times[i] + self.hist_len
                self.observation_spaces[f"retailer_{i}"] = spaces.Box(
                    low=0.0, high=np.inf, shape=(ret_obs_dim,), dtype=np.float32
                )
            self.observation_space = self.observation_spaces["warehouse"] # Gym standard fallback
            
        self.reset_env_state()

    def reset_env_state(self):
        # On-hand inventories (initialized to half of capacities)
        self.wh_stock = self.cap_warehouse / 2.0
        self.ret_stocks = np.array(self.cap_retailers, dtype=np.float32) / 2.0
        self.ret_backorders = np.zeros(self.num_retailers, dtype=np.float32)
        
        # Pipeline queues (FIFO)
        # Production queue
        self.prod_pipeline = [0.0] * self.lead_time_prod
        
        # In-transit queues: list of queues, one per retailer
        self.ship_pipelines = []
        for lt in self.lead_times:
            self.ship_pipelines.append([0.0] * lt)
            
        # Demand history initialization
        self.demand_history = np.zeros((self.num_retailers, self.hist_len), dtype=np.float32)
        
        self.current_step = 0

    def get_demand(self):
        """
        Generate demand at the current step for all retailers.
        """
        if self.eval_demand is not None:
            step = min(self.current_step, len(self.eval_demand) - 1)
            return np.array(self.eval_demand[step], dtype=np.float32)
            
        demands = []
        for i in range(self.num_retailers):
            amp = self.demand_amp[i]
            period = self.demand_period[i]
            mean_val = self.demand_mean[i]
            noise_std = self.demand_noise[i]
            
            # Sinusoidal component
            val = amp * np.sin(2.0 * np.pi * self.current_step / period) + mean_val
            # Add stochastic normal noise
            val += np.random.normal(0.0, noise_std)
            demands.append(max(0.0, val))
            
        return np.array(demands, dtype=np.float32)

    def get_centralized_obs(self):
        obs_parts = [
            np.array([self.wh_stock], dtype=np.float32),
            self.ret_stocks,
            self.ret_backorders,
            np.array(self.prod_pipeline, dtype=np.float32)
        ]
        for pipe in self.ship_pipelines:
            obs_parts.append(np.array(pipe, dtype=np.float32))
        obs_parts.append(self.demand_history.flatten())
        return np.concatenate(obs_parts)

    def get_multi_agent_obs(self):
        obs_dict = {}
        # 1. Warehouse observations
        wh_obs = np.concatenate([
            np.array([self.wh_stock], dtype=np.float32),
            np.array(self.prod_pipeline, dtype=np.float32),
            self.ret_backorders,
            self.demand_history.flatten()
        ])
        obs_dict["warehouse"] = wh_obs
        
        # 2. Retailer observations (POMDP local state)
        for i in range(self.num_retailers):
            ret_obs = np.concatenate([
                np.array([self.ret_stocks[i]], dtype=np.float32),
                np.array([self.ret_backorders[i]], dtype=np.float32),
                np.array(self.ship_pipelines[i], dtype=np.float32),
                self.demand_history[i]
            ])
            obs_dict[f"retailer_{i}"] = ret_obs
            
        return obs_dict

    def get_obs(self):
        if self.mode == "centralized":
            return self.get_centralized_obs()
        else:
            return self.get_multi_agent_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_env_state()
        
        # Populate initial demand history
        for _ in range(self.hist_len):
            d = self.get_demand()
            self.demand_history = np.roll(self.demand_history, -1, axis=1)
            self.demand_history[:, -1] = d
            self.current_step += 1
            
        self.current_step = 0
        return self.get_obs(), {}

    def step(self, action):
        # Action parsing based on execution mode
        if self.mode == "centralized":
            action = np.clip(action, 0.0, 1.0)
            act_prod_rate = action[0]
            act_ship_rates = action[1:]
        else:
            # Multi-agent action dictionary mapping
            act_prod_rate = np.clip(action.get("warehouse", [0.0])[0], 0.0, 1.0)
            act_ship_rates = np.array([
                np.clip(action.get(f"retailer_{i}", [0.0])[0], 0.0, 1.0)
                for i in range(self.num_retailers)
            ], dtype=np.float32)
        
        # Scale actions to physical limits
        proposed_prod = act_prod_rate * self.max_prod
        proposed_ships = act_ship_rates * np.array(self.max_ship, dtype=np.float32)
        
        # --- 1. Production Arrival at Warehouse ---
        arrived_prod = self.prod_pipeline[0] if self.lead_time_prod > 0 else proposed_prod
        if self.lead_time_prod > 0:
            self.prod_pipeline = self.prod_pipeline[1:] + [proposed_prod]
            
        available_stock = self.wh_stock + arrived_prod
        
        # --- 2. Shipping Allocations with Rationing ---
        total_proposed_ship = np.sum(proposed_ships)
        actual_ships = np.zeros(self.num_retailers, dtype=np.float32)
        
        if total_proposed_ship > 0:
            if total_proposed_ship <= available_stock:
                actual_ships = proposed_ships
            else:
                # Proportional stock rationing under warehouse stock deficit
                scale_factor = available_stock / total_proposed_ship
                actual_ships = proposed_ships * scale_factor
                
        # Update warehouse stock
        self.wh_stock = min(self.cap_warehouse, available_stock - np.sum(actual_ships))
        
        # --- 3. Shipments Transit and Retailer Arrivals ---
        arrived_shipments = np.zeros(self.num_retailers, dtype=np.float32)
        for i in range(self.num_retailers):
            lt = self.lead_times[i]
            if lt > 0:
                arrived_shipments[i] = self.ship_pipelines[i][0]
                self.ship_pipelines[i] = self.ship_pipelines[i][1:] + [actual_ships[i]]
            else:
                arrived_shipments[i] = actual_ships[i]
                
        # --- 4. Demand Realization and Retailer States ---
        current_demands = self.get_demand()
        self.demand_history = np.roll(self.demand_history, -1, axis=1)
        self.demand_history[:, -1] = current_demands
        
        for i in range(self.num_retailers):
            net_inventory = self.ret_stocks[i] - self.ret_backorders[i] + arrived_shipments[i] - current_demands[i]
            if net_inventory >= 0:
                self.ret_stocks[i] = min(self.cap_retailers[i], net_inventory)
                self.ret_backorders[i] = 0.0
            else:
                self.ret_stocks[i] = 0.0
                self.ret_backorders[i] = -net_inventory
                
        # --- 5. Shared Cost and Reward Calculation ---
        holding_wh = self.wh_stock * self.cost_holding_wh
        holding_ret = np.sum(self.ret_stocks * np.array(self.cost_holding_ret, dtype=np.float32))
        total_holding_cost = holding_wh + holding_ret
        total_backorder_cost = np.sum(self.ret_backorders * np.array(self.cost_backorder, dtype=np.float32))
        total_shipping_cost = np.sum(actual_ships * np.array(self.cost_shipping, dtype=np.float32))
        total_prod_cost = proposed_prod * self.cost_production
        
        total_cost = total_holding_cost + total_backorder_cost + total_shipping_cost + total_prod_cost
        reward = -total_cost / self.reward_scale
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "holding_cost": total_holding_cost,
            "backorder_cost": total_backorder_cost,
            "shipping_cost": total_shipping_cost,
            "production_cost": total_prod_cost,
            "total_cost": total_cost,
            "demands": current_demands,
            "actual_shipments": actual_ships,
            "actual_production": proposed_prod
        }
        
        if self.mode == "centralized":
            return self.get_obs(), reward, terminated, truncated, info
        else:
            # Multi-agent output format
            obs_dict = self.get_obs()
            rewards = {agent: reward for agent in self.possible_agents}
            terminations = {agent: terminated for agent in self.possible_agents}
            truncations = {agent: truncated for agent in self.possible_agents}
            infos = {agent: info for agent in self.possible_agents}
            return obs_dict, rewards, terminations, truncations, infos

    def render(self):
        print(f"Step: {self.current_step}")
        print(f"Warehouse Stock: {self.wh_stock:.2f}")
        print(f"Retailer Stocks: {self.ret_stocks}")
        print(f"Retailer Backorders: {self.ret_backorders}")
        print(f"In-Transit Pipelines: {[list(q) for q in self.ship_pipelines]}")
