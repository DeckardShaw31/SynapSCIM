import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RoPE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # Parallel mode
        # x shape: [B, H, T, d_model]
        B, H, T, d = x.size()
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [T, d//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, d]
        
        cos = emb.cos().unsqueeze(0).unsqueeze(1)
        sin = emb.sin().unsqueeze(0).unsqueeze(1)
        
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + x_rotated * sin

    def forward_step(self, x, step):
        # Recurrent step-by-step mode
        # x shape: [B, H, 1, d_model]
        B, H, _, d = x.size()
        t = torch.tensor([step], device=x.device, dtype=x.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [1, d//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [1, d]
        
        cos = emb.cos().unsqueeze(0).unsqueeze(1) # [1, 1, 1, d]
        sin = emb.sin().unsqueeze(0).unsqueeze(1) # [1, 1, 1, d]
        
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + x_rotated * sin

class LinearAttention(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.rope = RoPE(d_head)

    def forward(self, Q, K, V):
        # Q: [B, H, T, N//H]
        # K: [B, H, T, N//H]
        # V: [B, 1, T, D]
        Qr = self.rope(Q)
        Kr = self.rope(K)
        
        # Compute causal attention weights: [B, H, T, T]
        attn_weights = Qr @ Kr.mT
        mask = torch.tril(torch.ones_like(attn_weights), diagonal=-1)
        attn_weights = attn_weights * mask
        return attn_weights @ V

class BDH_GPU(nn.Module):
    def __init__(self, obs_dim, act_dim, D=64, H=2, N=1024, L=2, dropout=0.05):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.D = D
        self.H = H
        self.N = N
        self.L = L
        
        # Layer norm and observation encoder
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.wte = nn.Linear(obs_dim, D)
        self.drop = nn.Dropout(dropout)
        
        # BDH scale-free graph parameter matrices
        self.encoder = nn.Parameter(
            torch.zeros((N, D)).normal_(std=0.02)
        )
        self.decoder_x = nn.Parameter(
            torch.zeros((H, D, N // H)).normal_(std=0.02)
        )
        self.decoder_y = nn.Parameter(
            torch.zeros((H, D, N // H)).normal_(std=0.02)
        )
        
        self.attn = LinearAttention(N // H)
        
        # Actor-Critic Heads
        self.actor_mu = nn.Linear(D, act_dim)
        self.actor_log_std = nn.Linear(D, act_dim)
        self.critic = nn.Linear(D, 1)

    def get_latent_sequence(self, obs_seq):
        """
        Process the sequence of observations through the BDH layers (parallel mode).
        obs_seq shape: [B, T, D_obs]
        Returns: [B, T, D]
        """
        B, T, _ = obs_seq.size()
        v_ast = self.ln(self.wte(obs_seq)).unsqueeze(1) # [B, 1, T, D]
        
        for _ in range(self.L):
            x = F.relu(v_ast @ self.decoder_x.unsqueeze(0))
            a_ast = self.attn(Q=x, K=x, V=v_ast)
            y = F.relu(self.ln(a_ast) @ self.decoder_y.unsqueeze(0)) * x
            
            y = y.transpose(1, 2).reshape(B, 1, T, self.N)
            y = self.drop(y)
            v_ast = v_ast + self.ln(y @ self.encoder)
            v_ast = self.ln(v_ast)
            
        return v_ast.squeeze(1)

    def forward(self, obs_seq):
        """
        Actor-Critic forward pass (parallel mode).
        """
        v_seq = self.get_latent_sequence(obs_seq)
        v_last = v_seq[:, -1, :]  # Shape: [B, D]
        
        action_mu = torch.sigmoid(self.actor_mu(v_last))  # Map to [0, 1]
        log_std = torch.clamp(self.actor_log_std(v_last), min=-20.0, max=2.0)
        action_std = torch.exp(log_std)
        state_value = self.critic(v_last)
        
        return action_mu, action_std, state_value

    def init_recurrent_states(self, batch_size, device):
        """
        Initializes the list of synaptic fast-weight matrices for recurrent mode.
        Each layer l has state rho_l of shape [B, H, N//H, D]
        """
        return [
            torch.zeros(batch_size, self.H, self.N // self.H, self.D, device=device)
            for _ in range(self.L)
        ]

    def forward_recurrent(self, obs, step_idx, recurrent_states):
        """
        Actor-Critic forward pass in step-by-step Recurrent Hebbian mode.
        Args:
            obs: Observations for the current step. Shape: [B, D_obs]
            step_idx: Current temporal position index (int).
            recurrent_states: List of synaptic states (L elements of shape [B, H, N//H, D]).
        Returns:
            action_mu: [B, act_dim]
            action_std: [B, act_dim]
            state_value: [B, 1]
            next_recurrent_states: Updated list of synaptic states.
        """
        B = obs.size(0)
        v_ast = self.ln(self.wte(obs)).unsqueeze(1).unsqueeze(2) # [B, 1, 1, D]
        
        next_states = []
        for l in range(self.L):
            # 1. Project to scale-free neuron particles: [B, H, 1, N//H]
            x = F.relu(v_ast @ self.decoder_x.unsqueeze(0))
            
            # 2. Apply RoPE step rotation to Q and K
            Q = self.attn.rope.forward_step(x, step_idx)
            K = self.attn.rope.forward_step(x, step_idx)
            
            # 3. Retrieve past synaptic fast-weight state: [B, H, N//H, D]
            rho_prev = recurrent_states[l]
            
            # 4. Compute Linear Attention: [B, H, 1, D]
            # Q is [B, H, 1, N//H], rho_prev is [B, H, N//H, D]
            a_ast = Q @ rho_prev
            
            # 5. Hebbian fast-weight update: rho_t = rho_{t-1} + K_t^T @ V_t
            # K.mT is [B, H, N//H, 1], v_ast is [B, 1, 1, D]
            # Batch matrix multiply: [B, H, N//H, 1] @ [B, 1, 1, D] -> [B, H, N//H, D]
            rho_next = rho_prev + K.mT @ v_ast
            next_states.append(rho_next)
            
            # 6. Apply Hebbian feedback projection
            y = F.relu(self.ln(a_ast) @ self.decoder_y.unsqueeze(0)) * x
            
            # 7. Project back to embedding dimension D
            y = y.transpose(1, 2).reshape(B, 1, 1, self.N)
            y = self.drop(y)
            v_ast = v_ast + self.ln(y @ self.encoder)
            v_ast = self.ln(v_ast)
            
        v_last = v_ast.squeeze(1).squeeze(1) # Shape: [B, D]
        
        # Policy output
        action_mu = torch.sigmoid(self.actor_mu(v_last))
        log_std = torch.clamp(self.actor_log_std(v_last), min=-20.0, max=2.0)
        action_std = torch.exp(log_std)
        state_value = self.critic(v_last)
        
        return action_mu, action_std, state_value, next_states

class MLP_GPU(nn.Module):
    """
    Standard MLP Feed-Forward Deep Reinforcement Learning baseline architecture.
    Receives a flattened history context window of states and maps to actions.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Flatten history window of size 10 (T_context) * obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 10, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.actor_mu = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq):
        # obs_seq shape: [B, T_context, obs_dim]
        B, T, D = obs_seq.size()
        x = obs_seq.reshape(B, T * D) # Flatten history window
        feat = self.net(x)
        
        action_mu = torch.sigmoid(self.actor_mu(feat))
        log_std = torch.clamp(self.actor_log_std(feat), min=-20.0, max=2.0)
        action_std = torch.exp(log_std)
        state_value = self.critic(feat)
        
        return action_mu, action_std, state_value


class GNN_PPO_Model(nn.Module):
    """
    Graph Neural Network (GNN) policy network baseline using Graph Convolutional Networks (GCN).
    Uses message passing via a dynamically constructed adjacency matrix to coordinate decisions
    across supply chain echelons.
    """
    def __init__(self, obs_dim, act_dim, num_nodes=None, hidden_dim=64):
        super().__init__()
        # In centralized mode, act_dim is 1 (Warehouse) + num_retailers
        # So the total number of nodes is act_dim
        self.num_nodes = act_dim if num_nodes is None else num_nodes
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Construct adjacency matrix: Node 0 (Warehouse) connects to all echelons
        # Adding self-loops on diagonal
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.num_nodes):
            adj[i, i] = 1.0
        adj[0, :] = 1.0
        adj[:, 0] = 1.0
        
        # Normalize the adjacency matrix (row normalization)
        row_sum = adj.sum(dim=1, keepdim=True)
        self.register_buffer("norm_adj", adj / row_sum)
        
        # Determine node feature dimension with padding if not perfectly divisible
        total_features = 10 * obs_dim
        rem = total_features % self.num_nodes
        if rem != 0:
            total_features += (self.num_nodes - rem)
        node_feature_dim = total_features // self.num_nodes
        
        self.gcn_layer1 = nn.Linear(node_feature_dim, hidden_dim)
        self.gcn_layer2 = nn.Linear(hidden_dim, hidden_dim)
        
        flat_dim = self.num_nodes * hidden_dim
        
        self.actor_mu = nn.Linear(flat_dim, act_dim)
        self.actor_log_std = nn.Linear(flat_dim, act_dim)
        self.critic = nn.Linear(flat_dim, 1)

    def forward(self, obs_seq):
        # obs_seq shape: [B, T_context, obs_dim]
        B, T, D = obs_seq.size()
        
        # Flatten temporal window: [B, T * D]
        x = obs_seq.reshape(B, T * D)
        
        # Pad flat context dimension if not divisible by self.num_nodes
        total_features = T * D
        rem = total_features % self.num_nodes
        if rem != 0:
            pad_size = self.num_nodes - rem
            x = F.pad(x, (0, pad_size), "constant", 0.0)
            
        # Reshape to (Batch, Nodes, Node_Features)
        x = x.reshape(B, self.num_nodes, -1)
        
        # Ensure adjacency matrix is on correct device and matches batch size
        adj = self.norm_adj.unsqueeze(0).expand(B, -1, -1).to(obs_seq.device)
        
        # GCN Layer 1: Message Passing + ReLU
        # H^(1) = ReLU( A * X * W_1 )
        x = torch.bmm(adj, x)
        x = F.relu(self.gcn_layer1(x))
        
        # GCN Layer 2: Message Passing + ReLU
        # H^(2) = ReLU( A * H^(1) * W_2 )
        x = torch.bmm(adj, x)
        x = F.relu(self.gcn_layer2(x))
        
        # Flatten graph representation: [B, Nodes * Hidden_Dim]
        flat_x = x.reshape(B, -1)
        
        # Actor and Critic heads
        action_mu = torch.sigmoid(self.actor_mu(flat_x))
        log_std = torch.clamp(self.actor_log_std(flat_x), min=-20.0, max=2.0)
        action_std = torch.exp(log_std)
        state_value = self.critic(flat_x)
        
        return action_mu, action_std, state_value



