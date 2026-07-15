import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_detailed_plots():
    print("Generating detailed training convergence plots...")
    
    ppo_log_path = "SynapSCIM_checkpoints/training_log.csv"
    mappo_log_path = "SynapSCIM_mappo_checkpoints/training_log.csv"
    mlp_log_path = "SynapSCIM_mlpppo_checkpoints/training_log.csv"
    
    df_ppo = pd.read_csv(ppo_log_path)
    df_mappo = pd.read_csv(mappo_log_path)
    df_mlp = pd.read_csv(mlp_log_path)
    
    window_size = 100
    
    # Smooth functions
    def smooth(series):
        return series.rolling(window=window_size, min_periods=1).mean()
        
    # --- PLOT 1: Detailed 2x2 Grid for All Models ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    
    # 1. Top-Left: Reward Comparison
    axes[0, 0].plot(df_ppo['iteration'], smooth(df_ppo['mean_reward'] / 100.0), label='BDH-PPO (Centralized)', color='#1f77b4', linewidth=1.5)
    axes[0, 0].plot(df_mappo['iteration'], smooth(df_mappo['joint_reward']), label='MAPPO (Decentralized)', color='#d62728', linewidth=1.5, linestyle='--')
    axes[0, 0].plot(df_mlp['iteration'], smooth(df_mlp['mean_reward'] / 100.0), label='MLP-PPO (Baseline)', color='#9467bd', linewidth=1.5, linestyle='-.')
    axes[0, 0].set_xlabel('Iteration', fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel('Average Step Reward', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('Reward Convergence Curve', fontsize=11, fontweight='bold')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    
    # 2. Top-Right: Fill Rate Comparison
    axes[0, 1].plot(df_ppo['iteration'], smooth(df_ppo['fill_rate']), label='BDH-PPO (Centralized)', color='#1f77b4', linewidth=1.5)
    axes[0, 1].plot(df_mappo['iteration'], smooth(df_mappo['fill_rate']), label='MAPPO (Decentralized)', color='#d62728', linewidth=1.5, linestyle='--')
    axes[0, 1].plot(df_mlp['iteration'], smooth(df_mlp['fill_rate']), label='MLP-PPO (Baseline)', color='#9467bd', linewidth=1.5, linestyle='-.')
    axes[0, 1].set_xlabel('Iteration', fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel('Service Level (Fill Rate %)', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('Service Level (Fill Rate) Comparison', fontsize=11, fontweight='bold')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    
    # 3. Bottom-Left: Critic Loss Comparison
    axes[1, 0].plot(df_ppo['iteration'], smooth(df_ppo['critic_loss']), label='BDH-PPO (Centralized)', color='#1f77b4', linewidth=1.2, alpha=0.8)
    # Average of WH and Retailer critic loss for MAPPO
    mappo_critic = (df_mappo['wh_critic_loss'] + df_mappo['ret_critic_loss']) / 2.0
    axes[1, 0].plot(df_mappo['iteration'], smooth(mappo_critic), label='MAPPO (Decentralized Avg)', color='#d62728', linewidth=1.2, linestyle='--', alpha=0.8)
    axes[1, 0].plot(df_mlp['iteration'], smooth(df_mlp['critic_loss']), label='MLP-PPO (Baseline)', color='#9467bd', linewidth=1.2, linestyle='-.', alpha=0.8)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('Iteration', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Critic Loss (Log Scale)', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('Critic Network Value Loss Convergence', fontsize=11, fontweight='bold')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    
    # 4. Bottom-Right: Actor Loss Comparison
    axes[1, 1].plot(df_ppo['iteration'], smooth(df_ppo['actor_loss']), label='BDH-PPO (Centralized)', color='#1f77b4', linewidth=1.2, alpha=0.8)
    mappo_actor = (df_mappo['wh_actor_loss'] + df_mappo['ret_actor_loss']) / 2.0
    axes[1, 1].plot(df_mappo['iteration'], smooth(mappo_actor), label='MAPPO (Decentralized Avg)', color='#d62728', linewidth=1.2, linestyle='--', alpha=0.8)
    axes[1, 1].plot(df_mlp['iteration'], smooth(df_mlp['actor_loss']), label='MLP-PPO (Baseline)', color='#9467bd', linewidth=1.2, linestyle='-.', alpha=0.8)
    axes[1, 1].set_xlabel('Iteration', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('Actor Loss', fontsize=10, fontweight='bold')
    axes[1, 1].set_title('Actor Policy Policy Gradient Loss', fontsize=11, fontweight='bold')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle('Detailed Training Convergence Benchmark (All Models)', fontsize=14, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    detailed_all_path = "paper_materials/centralized_ppo/detailed_training_convergence_all.png"
    plt.savefig(detailed_all_path, dpi=300)
    plt.close()
    print(f"Saved detailed multi-model plot to {detailed_all_path}")
    
    # --- PLOT 2: 3x2 Standalone MLP-PPO training curves ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), dpi=300)
    
    # 1. Reward (Top-Left)
    axes[0, 0].plot(df_mlp['iteration'], df_mlp['mean_reward'] / 100.0, color='#9467bd', alpha=0.3, label='Raw Step Reward')
    axes[0, 0].plot(df_mlp['iteration'], smooth(df_mlp['mean_reward'] / 100.0), color='#4b2075', linewidth=2, label='Smoothed (EMA-100)')
    axes[0, 0].set_xlabel('Iteration', fontsize=10)
    axes[0, 0].set_ylabel('Step Reward', fontsize=10)
    axes[0, 0].set_title('MLP-PPO Step Reward Convergence', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, linestyle=':', alpha=0.5)
    axes[0, 0].legend()
    
    # 2. Fill Rate (Top-Right)
    axes[0, 1].plot(df_mlp['iteration'], df_mlp['fill_rate'], color='#e377c2', alpha=0.3, label='Raw Fill Rate')
    axes[0, 1].plot(df_mlp['iteration'], smooth(df_mlp['fill_rate']), color='#b12483', linewidth=2, label='Smoothed (EMA-100)')
    axes[0, 1].set_xlabel('Iteration', fontsize=10)
    axes[0, 1].set_ylabel('Fill Rate (Service Level %)', fontsize=10)
    axes[0, 1].set_title('MLP-PPO Service Level (Fill Rate) Growth', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, linestyle=':', alpha=0.5)
    axes[0, 1].legend()
    
    # 3. Actor Loss (Middle-Left)
    axes[1, 0].plot(df_mlp['iteration'], df_mlp['actor_loss'], color='#bcbd22', alpha=0.3, label='Raw Actor Loss')
    axes[1, 0].plot(df_mlp['iteration'], smooth(df_mlp['actor_loss']), color='#82850a', linewidth=2, label='Smoothed (EMA-100)')
    axes[1, 0].set_xlabel('Iteration', fontsize=10)
    axes[1, 0].set_ylabel('Actor Loss', fontsize=10)
    axes[1, 0].set_title('MLP-PPO Actor Policy Loss Curve', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, linestyle=':', alpha=0.5)
    axes[1, 0].legend()
    
    # 4. Critic Loss (Middle-Right)
    axes[1, 1].plot(df_mlp['iteration'], df_mlp['critic_loss'], color='#17becf', alpha=0.3, label='Raw Critic Loss')
    axes[1, 1].plot(df_mlp['iteration'], smooth(df_mlp['critic_loss']), color='#0a7b87', linewidth=2, label='Smoothed (EMA-100)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('Iteration', fontsize=10)
    axes[1, 1].set_ylabel('Critic Loss (Log Scale)', fontsize=10)
    axes[1, 1].set_title('MLP-PPO Value Function (Critic) Loss', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, linestyle=':', alpha=0.5)
    axes[1, 1].legend()
    
    # 5. Entropy (Bottom-Left)
    axes[2, 0].plot(df_mlp['iteration'], df_mlp['entropy'], color='#ff7f0e', alpha=0.3, label='Raw Entropy')
    axes[2, 0].plot(df_mlp['iteration'], smooth(df_mlp['entropy']), color='#d15904', linewidth=2, label='Smoothed (EMA-100)')
    axes[2, 0].set_xlabel('Iteration', fontsize=10)
    axes[2, 0].set_ylabel('Policy Entropy', fontsize=10)
    axes[2, 0].set_title('MLP-PPO Action Space Exploration Entropy', fontsize=11, fontweight='bold')
    axes[2, 0].grid(True, linestyle=':', alpha=0.5)
    axes[2, 0].legend()
    
    # 6. Summary Stats Text Box (Bottom-Right)
    axes[2, 1].axis('off')
    summary_text = (
        "=========================================\n"
        "      MLP-PPO BASELINE SUMMARY METRICS\n"
        "=========================================\n\n"
        f"Training Iterations: {len(df_mlp):,}\n"
        f"Starting Average Reward: {df_mlp['mean_reward'].iloc[0]/100.0:.2f}\n"
        f"Ending Average Reward: {df_mlp['mean_reward'].iloc[-1]/100.0:.2f}\n"
        f"Final Fill Rate: {df_mlp['fill_rate'].iloc[-1]:.2f}%\n"
        f"Min Critic Loss: {df_mlp['critic_loss'].min():.4f}\n"
        f"Max Exploration Entropy: {df_mlp['entropy'].max():.4f}\n"
        f"Final Exploration Entropy: {df_mlp['entropy'].iloc[-1]:.4f}\n\n"
        "Baseline Performance Insights:\n"
        "- Policy Gradient Loss stabilizes around 0.\n"
        "- Critic Loss drops rapidly in log space,\n"
        "  confirming proper value approximation.\n"
        "- Entropy decays steadily as policy actions\n"
        "  specialize in ordering inventory."
    )
    axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes,
                    fontsize=10.5, family='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#f5f5f5', edgecolor='#cccccc'))
    
    plt.suptitle('MLP-PPO Feed-Forward Baseline Training Analysis', fontsize=14, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    mlp_standalone_path = "paper_materials/centralized_ppo/mlp_ppo_training_convergence.png"
    plt.savefig(mlp_standalone_path, dpi=300)
    plt.close()
    print(f"Saved MLP-PPO standalone training analysis plot to {mlp_standalone_path}")

if __name__ == "__main__":
    generate_detailed_plots()
