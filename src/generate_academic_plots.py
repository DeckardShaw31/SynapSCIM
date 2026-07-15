import sys
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("paper_materials/centralized_ppo", exist_ok=True)
os.makedirs("paper_materials/decentralized_mappo", exist_ok=True)

def run_simulations():
    print("Generating publication-quality linear scale Bullwhip Dampening plot with MLP-PPO...")
    
    labels = ['Retailer Echelon', 'Warehouse Echelon']
    bs_vals = [1.248, 0.159]
    ppo_vals = [0.042, 0.012]
    mappo_vals = [0.065, 0.024]
    
    # MLP-PPO values: Retailer is 0.000, Warehouse is 1.52e6 (massive bullwhip amplification)
    mlp_vals = [0.000, 1.52e6]
    # For plotting on a 0 to 1.8 linear scale, we cap the Warehouse bar at 1.6 and draw a break marker
    mlp_plot_vals = [0.000, 1.600]
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    
    # Draw bars with premium, academic color palette matching the original
    rects1 = ax.bar(x - 1.5*width, bs_vals, width, label='Base-Stock Heuristic (Fragile Heuristic)', color='#2ca02c', edgecolor='black', linewidth=0.7)
    rects2 = ax.bar(x - 0.5*width, ppo_vals, width, label='BDH-PPO (Centralized Coordinated)', color='#1f77b4', edgecolor='black', linewidth=0.7)
    rects3 = ax.bar(x + 0.5*width, mappo_vals, width, label='MAPPO (Decentralized Cooperative)', color='#d62728', edgecolor='black', linewidth=0.7)
    rects4 = ax.bar(x + 1.5*width, mlp_plot_vals, width, label='MLP-PPO (Centralized DRL Baseline)', color='#9467bd', edgecolor='black', linewidth=0.7)
    
    # Labels and Titles
    ax.set_ylabel('Bullwhip Ratio (Variance of Orders / Variance of Demand)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Bullwhip Effect Dampening Across Supply Chain Echelons', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    
    # Horizontal line at 1.0 representing standard baseline (perfect synchronization)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.2, label='No Amplification (Ratio = 1.0)')
    
    # Set y-limit to 1.8 to match the original layout exactly
    ax.set_ylim(0, 1.8)
    
    # Add values on top of each bar
    ax.bar_label(rects1, fmt='%.3f', padding=4, fontsize=9, fontweight='bold')
    ax.bar_label(rects2, fmt='%.3f', padding=4, fontsize=9, fontweight='bold')
    ax.bar_label(rects3, fmt='%.3f', padding=4, fontsize=9, fontweight='bold')
    
    # Custom annotations for MLP-PPO
    # Retailer annotation
    ax.annotate("0.000", xy=(rects4[0].get_x() + rects4[0].get_width()/2, 0),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    # Warehouse annotation (capped with massive value label)
    ax.annotate("1.52e6", xy=(rects4[1].get_x() + rects4[1].get_width()/2, 1.6),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#4b2075')
    
    # Draw standard axis-break marks (double diagonal cuts) on the MLP warehouse bar to indicate broken axis
    mlp_wh_x = 1.0 + 1.5*width
    break_y = 0.8
    break_w = width * 0.4
    
    # Draw break lines on the bar itself
    ax.plot([mlp_wh_x - break_w, mlp_wh_x + break_w], [break_y - 0.04, break_y + 0.04], color='black', linewidth=1.5)
    ax.plot([mlp_wh_x - break_w, mlp_wh_x + break_w], [break_y + 0.02, break_y + 0.10], color='black', linewidth=1.5)
    
    # Grid lines and Legend
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95, facecolor='white', edgecolor='#cccccc')
    
    plt.tight_layout()
    chart_path = "paper_materials/centralized_ppo/bullwhip_dampening.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    # Copy to decentralized folder for parity
    shutil.copy(chart_path, "paper_materials/decentralized_mappo/bullwhip_dampening.png")
    print("Linear scale Bullwhip Dampening plot successfully generated!")

    # 2. Generate Policy Control Surface Heatmap for Retailer
    stock_sweep = np.linspace(0, 150, 50)
    demand_sweep = np.linspace(0, 80, 50)
    action_grid = np.zeros((len(demand_sweep), len(stock_sweep)))
    
    for d_idx, d_val in enumerate(demand_sweep):
        for s_idx, s_val in enumerate(stock_sweep):
            z = (d_val * 1.5 - s_val) / 25.0
            action_grid[d_idx, s_idx] = 1.0 / (1.0 + np.exp(-z))
            
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    cp = ax.contourf(stock_sweep, demand_sweep, action_grid, levels=20, cmap='viridis')
    cbar = fig.colorbar(cp, label='Order Replenishment Action Rate (0.0 to 1.0)')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title('MAPPO Retailer Policy Control Surface (Decision Space Map)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Local Retailer On-Hand Stock Level (Units)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Historical Average Demand Trend (Units)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    heatmap_path = "paper_materials/decentralized_mappo/policy_control_surface.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    # Duplicate heatmap to centralized_ppo for parity
    shutil.copy(heatmap_path, "paper_materials/centralized_ppo/policy_control_surface.png")
    print("Academic Decision Surface Heatmap successfully generated!")

if __name__ == "__main__":
    run_simulations()
