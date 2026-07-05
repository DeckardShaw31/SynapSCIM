import unittest
import torch
import sys
import os

# Add src folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bdh import BDH_GPU

class TestBDHGPUPolicy(unittest.TestCase):
    def test_forward_pass_shape(self):
        obs_dim = 25
        act_dim = 4
        T_context = 10
        batch_size = 4
        
        # Instantiate model
        model = BDH_GPU(obs_dim=obs_dim, act_dim=act_dim, D=32, H=2, N=256, L=2)
        model.eval()
        
        # Mock input sequence of shape [B, T_context, obs_dim]
        mock_input = torch.randn(batch_size, T_context, obs_dim)
        
        # Run forward pass
        with torch.no_grad():
            mu, std, val = model(mock_input)
            
        # Verify output dimensions
        self.assertEqual(mu.shape, (batch_size, act_dim))
        self.assertEqual(std.shape, (batch_size, act_dim))
        self.assertEqual(val.shape, (batch_size, 1))
        
        # Verify bounds on actions (sigmoid outputs in [0, 1])
        self.assertTrue(torch.all(mu >= 0.0))
        self.assertTrue(torch.all(mu <= 1.0))
        
        self.assertTrue(torch.all(std > 0.0))

    def test_recurrent_hebbian_mode(self):
        obs_dim = 25
        act_dim = 4
        batch_size = 1
        
        model = BDH_GPU(obs_dim=obs_dim, act_dim=act_dim, D=32, H=2, N=256, L=2)
        model.eval()
        
        # Initialize recurrent states
        states = model.init_recurrent_states(batch_size, "cpu")
        self.assertEqual(len(states), model.L)
        self.assertEqual(states[0].shape, (batch_size, model.H, model.N // model.H, model.D))
        
        # Mock step input: shape [B, obs_dim]
        mock_step_input = torch.randn(batch_size, obs_dim)
        
        with torch.no_grad():
            mu, std, val, next_states = model.forward_recurrent(mock_step_input, step_idx=0, recurrent_states=states)
            
        self.assertEqual(mu.shape, (batch_size, act_dim))
        self.assertEqual(std.shape, (batch_size, act_dim))
        self.assertEqual(val.shape, (batch_size, 1))
        self.assertEqual(len(next_states), model.L)

if __name__ == "__main__":
    unittest.main()
