import unittest
import numpy as np
import sys
import os

# Add src folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from env import MultiEchelonSupplyChainEnv

class TestMultiEchelonSupplyChainEnv(unittest.TestCase):
    def setUp(self):
        self.env = MultiEchelonSupplyChainEnv()

    def test_initial_state(self):
        obs, info = self.env.reset(seed=42)
        
        # Verify shape of state observation vector
        self.assertEqual(obs.shape[0], self.env.state_dim)
        
        # Verify that inventories are positive and correct
        self.assertEqual(self.env.wh_stock, self.env.cap_warehouse / 2.0)
        np.testing.assert_array_equal(self.env.ret_stocks, np.array(self.env.cap_retailers) / 2.0)
        np.testing.assert_array_equal(self.env.ret_backorders, np.zeros(self.env.num_retailers))

    def test_step_function(self):
        self.env.reset(seed=42)
        
        # Sample step with action [0.5, 0.5, 0.5, 0.5]
        action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        obs, reward, term, trunc, info = self.env.step(action)
        
        # Validate output shape
        self.assertEqual(obs.shape[0], self.env.state_dim)
        
        # Validate that costs and info are present
        self.assertIn("total_cost", info)
        self.assertIn("holding_cost", info)
        self.assertIn("backorder_cost", info)
        self.assertIn("shipping_cost", info)
        self.assertIn("production_cost", info)
        self.assertLessEqual(reward, 0.0)  # Rewards are negative costs

    def test_multi_agent_mode(self):
        env = MultiEchelonSupplyChainEnv(mode="multi_agent")
        obs_dict, info = env.reset(seed=42)
        
        # Verify agents
        self.assertEqual(len(env.possible_agents), 1 + env.num_retailers)
        self.assertIn("warehouse", obs_dict)
        self.assertIn("retailer_0", obs_dict)
        
        # Verify step with dictionary actions
        actions = {
            "warehouse": [0.5],
            "retailer_0": [0.3],
            "retailer_1": [0.3],
            "retailer_2": [0.3]
        }
        obs_dict, rewards, terms, truncs, infos = env.step(actions)
        
        self.assertIn("warehouse", obs_dict)
        self.assertIn("retailer_0", obs_dict)
        self.assertIn("warehouse", rewards)
        self.assertIn("warehouse", terms)

if __name__ == "__main__":
    unittest.main()
