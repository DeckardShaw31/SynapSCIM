import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("paper_materials/centralized_ppo", exist_ok=True)
os.makedirs("paper_materials/decentralized_mappo", exist_ok=True)

def generate_convergence_plots():
    print("Generating training convergence line charts...")
    
    ppo_log_path = "SynapSCIM_checkpoints/training_log.csv"
    mappo_log_path = "SynapSCIM_mappo_checkpoints/training_log.csv"
    mlp_log_path = "SynapSCIM_mlpppo_checkpoints/training_log.csv"
    
    if not os.path.exists(ppo_log_path) or not os.path.exists(mappo_log_path):
        print(f"[Error] Core training logs not found at {ppo_log_path} or {mappo_log_path}")
        return
        
    df_ppo = pd.read_csv(ppo_log_path)
    df_mappo = pd.read_csv(mappo_log_path)
    
    # Smooth metrics using rolling average for academic publication clarity
    window_size = 100
    
    ppo_reward_smoothed = (df_ppo['mean_reward'] / 100.0).rolling(window=window_size, min_periods=1).mean()
    mappo_reward_smoothed = df_mappo['joint_reward'].rolling(window=window_size, min_periods=1).mean()
    
    ppo_fill_smoothed = df_ppo['fill_rate'].rolling(window=window_size, min_periods=1).mean()
    mappo_fill_smoothed = df_mappo['fill_rate'].rolling(window=window_size, min_periods=1).mean()
    
    has_mlp = os.path.exists(mlp_log_path)
    if has_mlp:
        df_mlp = pd.read_csv(mlp_log_path)
        mlp_reward_smoothed = (df_mlp['mean_reward'] / 100.0).rolling(window=window_size, min_periods=1).mean()
        mlp_fill_smoothed = df_mlp['fill_rate'].rolling(window=window_size, min_periods=1).mean()
    
    # 1. Plot Convergence Figure (2-panel: Reward and Fill Rate)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Left Panel: Reward Convergence
    axes[0].plot(df_ppo['iteration'], ppo_reward_smoothed, label='BDH-PPO (Centralized)', color='#1f77b4', linewidth=1.8)
    axes[0].plot(df_mappo['iteration'], mappo_reward_smoothed, label='MAPPO (Decentralized)', color='#d62728', linewidth=1.8, linestyle='--')
    if has_mlp:
        axes[0].plot(df_mlp['iteration'], mlp_reward_smoothed, label='MLP-PPO (DRL Baseline)', color='#9467bd', linewidth=1.8, linestyle='-.')
    axes[0].set_xlabel('Iteration', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Average Step Reward (Smoothed)', fontsize=11, fontweight='bold')
    axes[0].set_title('Training Reward Convergence Curve', fontsize=12, fontweight='bold', pad=10)
    axes[0].legend(loc='lower right', framealpha=0.9, edgecolor='#cccccc')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Right Panel: Fill Rate Convergence
    axes[1].plot(df_ppo['iteration'], ppo_fill_smoothed, label='BDH-PPO (Centralized)', color='#1f77b4', linewidth=1.8)
    axes[1].plot(df_mappo['iteration'], mappo_fill_smoothed, label='MAPPO (Decentralized)', color='#d62728', linewidth=1.8, linestyle='--')
    if has_mlp:
        axes[1].plot(df_mlp['iteration'], mlp_fill_smoothed, label='MLP-PPO (DRL Baseline)', color='#9467bd', linewidth=1.8, linestyle='-.')
    axes[1].set_xlabel('Iteration', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Fill Rate (Service Level %)', fontsize=11, fontweight='bold')
    axes[1].set_title('Service Level (Fill Rate) Growth during Training', fontsize=12, fontweight='bold', pad=10)
    axes[1].legend(loc='lower right', framealpha=0.9, edgecolor='#cccccc')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    conv_path = "paper_materials/centralized_ppo/training_convergence.png"
    plt.savefig(conv_path, dpi=300)
    plt.close()
    
    # Also copy to decentralized mapping folder for parity
    shutil.copy(conv_path, "paper_materials/decentralized_mappo/training_convergence.png")
    print("Convergence curves saved to 'paper_materials/'!")

def generate_benchmark_bar_charts():
    print("Generating benchmark comparison bar charts...")
    
    policies = [
        'BDH-PPO\n(Centralized)', 
        'MAPPO\n(Decentralized)', 
        'MLP-PPO\n(DRL Baseline)', 
        'Base-Stock\n(Baseline)', 
        's, Q Policy\n(Baseline)'
    ]
    total_costs = [475841.00, 481119.06, 1335293.12, 312541.12, 672098.04]
    fill_rates = [26.33, 25.65, 35.27, 13.87, 6.61]
    
    colors = ['#1f77b4', '#d62728', '#9467bd', '#2ca02c', '#8c564b']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Left Panel: Total Cost comparison
    bars_cost = axes[0].bar(policies, total_costs, color=colors, edgecolor='black', linewidth=0.7, width=0.55)
    axes[0].set_ylabel('Total Operational Cost ($)', fontsize=11, fontweight='bold', labelpad=8)
    axes[0].set_title('Operational Cost Comparison (Lower is Better)', fontsize=12, fontweight='bold', pad=12)
    axes[0].grid(axis='y', linestyle=':', alpha=0.5)
    axes[0].set_ylim(0, max(total_costs) * 1.15)
    axes[0].bar_label(bars_cost, fmt='$%d', padding=4, fontsize=9, fontweight='bold')
    
    # Right Panel: Service Level (Fill Rate) comparison
    bars_fill = axes[1].bar(policies, fill_rates, color=colors, edgecolor='black', linewidth=0.7, width=0.55)
    axes[1].set_ylabel('Service Level (Fill Rate %)', fontsize=11, fontweight='bold', labelpad=8)
    axes[1].set_title('Service Level (Fill Rate) Comparison (Higher is Better)', fontsize=12, fontweight='bold', pad=12)
    axes[1].grid(axis='y', linestyle=':', alpha=0.5)
    axes[1].set_ylim(0, max(fill_rates) * 1.15)
    axes[1].bar_label(bars_fill, fmt='%.2f%%', padding=4, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    bar_chart_path = "paper_materials/centralized_ppo/benchmark_comparison_bar_chart.png"
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()
    
    # Copy to decentralized folder for parity
    shutil.copy(bar_chart_path, "paper_materials/decentralized_mappo/benchmark_comparison_bar_chart.png")
    print("Benchmark bar charts saved to 'paper_materials/'!")

if __name__ == "__main__":
    generate_convergence_plots()
    generate_benchmark_bar_charts()
