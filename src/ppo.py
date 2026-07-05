import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.hist_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.value_targets = []

    def clear(self):
        self.hist_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.advantages.clear()
        self.value_targets.clear()

    def compute_gae(self, next_value, gamma=0.99, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE) and Value Targets.
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values + [next_value], dtype=np.float32)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
            
        self.advantages = list(advantages)
        self.value_targets = list(advantages + np.array(self.values, dtype=np.float32))

def get_history(episode_obs, t, T_context):
    """
    Construct a history window of size T_context ending at time step t.
    Pads with the first observation if t < T_context - 1.
    """
    history = []
    for idx in range(t - T_context + 1, t + 1):
        if idx < 0:
            history.append(episode_obs[0])
        else:
            history.append(episode_obs[idx])
    return np.array(history, dtype=np.float32)

class PPOAgent:
    def __init__(self, model, lr=5e-5, clip_eps=0.2, c_value=0.5, c_entropy=0.01, max_grad_norm=0.5):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        self.clip_eps = clip_eps
        self.c_value = c_value
        self.c_entropy = c_entropy
        self.max_grad_norm = max_grad_norm
        
    def update(self, buffer, batch_size=64, epochs=10):
        if len(buffer.hist_states) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
            
        hist_states = torch.tensor(np.array(buffer.hist_states), dtype=torch.float32)  # [B, T_context, obs_dim]
        actions = torch.tensor(np.array(buffer.actions), dtype=torch.float32)          # [B, act_dim]
        old_log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32)  # [B]
        advantages = torch.tensor(np.array(buffer.advantages), dtype=torch.float32)    # [B]
        value_targets = torch.tensor(np.array(buffer.value_targets), dtype=torch.float32) # [B]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset_size = len(hist_states)
        
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        total_losses = []
        
        device = next(self.model.parameters()).device
        hist_states = hist_states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        advantages = advantages.to(device)
        value_targets = value_targets.to(device)
        
        for _ in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                b_hist_states = hist_states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_value_targets = value_targets[batch_idx]
                
                action_mu, action_std, state_value = self.model(b_hist_states)
                
                dist = torch.distributions.Normal(action_mu, action_std)
                new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratios = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = nn.MSELoss()(state_value.squeeze(-1), b_value_targets)
                loss = actor_loss + self.c_value * critic_loss - self.c_entropy * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy.item())
                total_losses.append(loss.item())
                
        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy": np.mean(entropy_losses),
            "total_loss": np.mean(total_losses)
        }

class MultiAgentPPOAgent:
    """
    Cooperative Heterogeneous Multi-Agent PPO (MAPPO) Optimizer.
    Manages separate policies for the Warehouse agent and the Retailer agents (using parameter sharing).
    """
    def __init__(self, model_warehouse, model_retailer, lr=5e-5):
        self.wh_agent = PPOAgent(model_warehouse, lr=lr)
        self.ret_agent = PPOAgent(model_retailer, lr=lr)

    def update(self, buffer_wh, buffer_ret, batch_size=64, epochs=10):
        wh_info = self.wh_agent.update(buffer_wh, batch_size, epochs)
        ret_info = self.ret_agent.update(buffer_ret, batch_size, epochs)
        
        return {
            "wh_actor_loss": wh_info["actor_loss"],
            "wh_critic_loss": wh_info["critic_loss"],
            "wh_entropy": wh_info["entropy"],
            "ret_actor_loss": ret_info["actor_loss"],
            "ret_critic_loss": ret_info["critic_loss"],
            "ret_entropy": ret_info["entropy"]
        }
