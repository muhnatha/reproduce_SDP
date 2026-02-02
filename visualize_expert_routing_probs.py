#!/usr/bin/env python3
"""
Visualize Expert Routing Probabilities for SDP Model
Creates visualizations similar to Figure 7 from the paper
"""

import sys
sys.path.insert(0, '/home/cc/reproduce_SDP')

import os
import pathlib
import click
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy


def load_and_validate_data(probs_path):
    """
    Load and validate expert_routing_probs.pkl file

    Args:
        probs_path: Path to expert_routing_probs.pkl file

    Returns:
        all_probs: List of probability entries
    """
    print(f"Loading expert routing probabilities from {probs_path}")

    with open(probs_path, 'rb') as f:
        all_probs = pickle.load(f)

    # Verify structure
    print(f"Loaded {len(all_probs)} probability entries")

    if len(all_probs) == 0:
        raise ValueError("No probability entries found in file")

    epochs = [p['epoch'] for p in all_probs]
    print(f"Epochs: {min(epochs)} - {max(epochs)}")
    print(f"Task ID: {all_probs[0]['task_id']}")
    print(f"Prob shape: {all_probs[0]['probs'].shape}")

    return all_probs


def aggregate_probs(all_probs, num_experts=8, horizon=10, epochs_range=None):
    """
    Aggregate probabilities across batches and epochs

    Args:
        all_probs: List of probability entries
        num_experts: Number of experts
        horizon: Horizon length (timesteps per sequence)
        epochs_range: Optional tuple (start, end) to filter epochs

    Returns:
        activation_matrix: Shape (horizon, num_experts)
    """
    if epochs_range is not None:
        start, end = map(int, epochs_range.split(','))
        all_probs = [p for p in all_probs if start <= p['epoch'] <= end]
        print(f"Filtered to {len(all_probs)} entries from epochs {start}-{end}")

    aggregated = np.zeros((horizon, num_experts))
    count = np.zeros(horizon)

    for entry in all_probs:
        probs = entry['probs']

        # Reshape probs: (batch_size * horizon, num_experts)
        batch_size = probs.shape[0] // horizon
        probs_reshaped = probs.reshape(batch_size, horizon, num_experts)

        # Average over batch dimension
        aggregated += probs_reshaped.mean(axis=0)
        count += 1

    # Normalize by count
    aggregated /= count[:, np.newaxis]

    return aggregated


def parse_expert_groups(groups_str):
    """
    Parse expert group string into list of (start, end) tuples

    Args:
        groups_str: String like "0,3;4,7"

    Returns:
        List of (start, end) tuples
    """
    groups = groups_str.split(';')
    result = []
    for group in groups:
        start, end = map(int, group.split(','))
        result.append((start, end))
    return result


def create_heatmap(activation_matrix, expert_groups, output_path, dpi=300):
    """
    Create main heatmap visualization (Figure 7 style)

    Args:
        activation_matrix: Shape (horizon, num_experts)
        expert_groups: List of (start, end) tuples
        output_path: Path to save heatmap
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # Create heatmap (transpose to have experts on y-axis)
    sns.heatmap(
        activation_matrix.T,
        cmap='YlOrRd',
        cbar_kws={'label': 'Expert Routing Probability'},
        vmin=0,
        vmax=1,
        xticklabels=range(activation_matrix.shape[0]),
        yticklabels=range(activation_matrix.shape[1]),
        ax=ax
    )

    # Create expert labels with origins
    expert_labels = []
    for expert_id in range(activation_matrix.shape[1]):
        for i, (group_start, group_end) in enumerate(expert_groups):
            if group_start <= expert_id <= group_end:
                group_name = 'square' if i == 0 else 'stack'
                expert_labels.append(f'Expert {expert_id}\n({group_name})')
                break

    ax.set_yticklabels(expert_labels, rotation=0)

    # Labels and title
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Expert ID', fontsize=12)
    ax.set_title('Expert Routing Probabilities Over Time (stack_three Task)',
                 fontsize=14, fontweight='bold')

    # Add legend for expert origins
    legend_text = (
        "Expert Origins:\n"
        "• Experts 0-3: Trained on square\n"
        "• Experts 4-7: Trained on stack"
    )
    plt.text(0.02, 1.02, legend_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_path}")


def create_evolution_plot(all_probs, expert_groups, num_experts, horizon, output_path, dpi=300):
    """
    Create plot showing expert usage evolution across epochs

    Args:
        all_probs: List of probability entries
        expert_groups: List of (start, end) tuples
        num_experts: Number of experts
        horizon: Horizon length
        output_path: Path to save plot
        dpi: Resolution
    """
    # Aggregate per epoch
    epochs = sorted(list(set([p['epoch'] for p in all_probs])))
    epochs_data = []

    for epoch in epochs:
        epoch_entries = [p for p in all_probs if p['epoch'] == epoch]
        if not epoch_entries:
            continue

        # Process each batch individually (handles variable batch sizes)
        batch_averages = []
        for entry in epoch_entries:
            probs = entry['probs']
            batch_size = probs.shape[0] // horizon
            
            # Reshape and average over batch dimension
            probs_reshaped = probs.reshape(batch_size, horizon, num_experts)
            batch_avg = probs_reshaped.mean(axis=0)  # Shape: (horizon, num_experts)
            batch_averages.append(batch_avg)
        
        # Average across all batches for this epoch
        epoch_avg = np.mean(batch_averages, axis=0)  # Shape: (horizon, num_experts)
        epochs_data.append(epoch_avg)

    if not epochs_data:
        print("Warning: No epoch data available for evolution plot")
        return

    epochs_data = np.array(epochs_data)

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Line plot per expert
    for expert_id in range(num_experts):
        values = epochs_data[:, expert_id].mean(axis=1)
        for i, (group_start, group_end) in enumerate(expert_groups):
            if group_start <= expert_id <= group_end:
                group_name = 'square' if i == 0 else 'stack'
                color = 'blue' if group_name == 'square' else 'red'
                break

        axes[0].plot(range(len(epochs)), values,
                     label=f'Expert {expert_id} ({group_name})', color=color, alpha=0.7)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Activation')
    axes[0].set_title('Expert Activation Evolution Across Epochs')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Group aggregation
    square_experts = list(range(4))
    stack_experts = list(range(4, 8))

    square_values = epochs_data[:, square_experts].mean(axis=2)
    stack_values = epochs_data[:, stack_experts].mean(axis=2)

    axes[1].plot(range(len(epochs)), square_values.mean(axis=1),
                  label='square experts', color='blue', linewidth=3)
    axes[1].plot(range(len(epochs)), stack_values.mean(axis=1),
                  label='stack experts', color='red', linewidth=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Activation')
    axes[1].set_title('Group Activation Evolution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved evolution plot to {output_path}")


def create_skill_transfer_plot(activation_matrix, expert_groups, output_path, dpi=300):
    """
    Analyze and visualize skill transfer from base tasks

    Args:
        activation_matrix: Shape (horizon, num_experts)
        expert_groups: List of (start, end) tuples
        output_path: Path to save plot
        dpi: Resolution
    """
    # Calculate group contributions
    square_experts = [i for i in range(activation_matrix.shape[1]) if 0 <= i <= 3]
    stack_experts = [i for i in range(activation_matrix.shape[1]) if 4 <= i <= 7]

    square_contrib = activation_matrix[:, square_experts].sum(axis=1)
    stack_contrib = activation_matrix[:, stack_experts].sum(axis=1)

    # Calculate percentages
    total = square_contrib + stack_contrib
    square_pct = (square_contrib / total * 100)
    stack_pct = (stack_contrib / total * 100)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Stacked bar chart per timestep
    x = range(activation_matrix.shape[0])
    axes[0, 0].bar(x, square_pct, label='square experts', color='blue', alpha=0.7)
    axes[0, 0].bar(x, stack_pct, bottom=square_pct, label='stack experts', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Activation %')
    axes[0, 0].set_title('Expert Group Contribution per Timestep')
    axes[0, 0].legend()

    # Line plot over timesteps
    axes[0, 1].plot(x, square_pct, label='square', color='blue', linewidth=2)
    axes[0, 1].plot(x, stack_pct, label='stack', color='red', linewidth=2)
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Activation %')
    axes[0, 1].set_title('Group Contribution Evolution')
    axes[0, 1].legend()

    # Pie chart for overall distribution
    total_square = square_pct.mean()
    total_stack = stack_pct.mean()
    axes[1, 0].pie([total_square, total_stack], labels=['square', 'stack'],
                      colors=['blue', 'red'], autopct='%1.1f%%')
    axes[1, 0].set_title('Overall Expert Group Distribution')

    # Text summary
    summary = f"""
    Skill Transfer Analysis Summary:

    Total square expert activation: {square_contrib.sum():.3f}
    Total stack expert activation: {stack_contrib.sum():.3f}

    Average square contribution: {total_square:.1f}%
    Average stack contribution: {total_stack:.1f}%

    Dominant group: {'square' if total_square > total_stack else 'stack'}
    """
    axes[1, 1].text(0.1, 0.5, summary, fontsize=11, verticalalignment='center')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved skill transfer plot to {output_path}")


def create_correlation_matrix(activation_matrix, expert_groups, output_path, dpi=300):
    """
    Create correlation matrix showing co-activation patterns

    Args:
        activation_matrix: Shape (horizon, num_experts)
        expert_groups: List of (start, end) tuples
        output_path: Path to save plot
        dpi: Resolution
    """
    # Compute correlation matrix
    correlation_matrix = np.corrcoef(activation_matrix.T)

    # Create labels with origins
    expert_labels = []
    for expert_id in range(activation_matrix.shape[1]):
        for i, (group_start, group_end) in enumerate(expert_groups):
            if group_start <= expert_id <= group_end:
                group_name = 'square' if i == 0 else 'stack'
                expert_labels.append(f'E{expert_id}\n({group_name})')
                break

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        cmap='RdBu_r',
        cbar_kws={'label': 'Correlation'},
        vmin=-1,
        vmax=1,
        xticklabels=expert_labels,
        yticklabels=expert_labels,
        ax=ax
    )

    ax.set_title('Expert Activation Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation matrix to {output_path}")


def generate_report(activation_matrix, expert_groups, output_path):
    """
    Generate detailed text report

    Args:
        activation_matrix: Shape (horizon, num_experts)
        expert_groups: List of (start, end) tuples
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("Expert Routing Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        # Per-expert statistics
        f.write("Per-Expert Statistics:\n")
        f.write("-" * 30 + "\n")
        for expert_id in range(activation_matrix.shape[1]):
            activations = activation_matrix[:, expert_id]
            mean_act = activations.mean()
            std_act = activations.std()
            max_act = activations.max()
            min_act = activations.min()

            # Determine origin
            for i, (group_start, group_end) in enumerate(expert_groups):
                if group_start <= expert_id <= group_end:
                    origin = 'square' if i == 0 else 'stack'
                    break

            f.write(f"\nExpert {expert_id} ({origin}):\n")
            f.write(f"  Mean activation: {mean_act:.4f}\n")
            f.write(f"  Std activation: {std_act:.4f}\n")
            f.write(f"  Max activation: {max_act:.4f}\n")
            f.write(f"  Min activation: {min_act:.4f}\n")
            f.write(f"  Active timesteps (>0.1): {np.sum(activations > 0.1)}/{len(activations)}\n")

        # Group analysis
        f.write("\n\n" + "=" * 50 + "\n")
        f.write("Group Analysis:\n")
        f.write("-" * 30 + "\n")

        square_experts = [i for i in range(activation_matrix.shape[1]) if i <= 3]
        stack_experts = [i for i in range(activation_matrix.shape[1]) if i > 3]

        square_total = activation_matrix[:, square_experts].sum()
        stack_total = activation_matrix[:, stack_experts].sum()
        total = square_total + stack_total

        f.write(f"\nSquare experts (0-3):\n")
        f.write(f"  Total activation: {square_total:.3f}\n")
        f.write(f"  Percentage: {square_total/total*100:.1f}%\n")

        f.write(f"\nStack experts (4-7):\n")
        f.write(f"  Total activation: {stack_total:.3f}\n")
        f.write(f"  Percentage: {stack_total/total*100:.1f}%\n")

        # Insights
        f.write("\n\n" + "=" * 50 + "\n")
        f.write("Insights:\n")
        f.write("-" * 30 + "\n")

        dominant = 'square' if square_total > stack_total else 'stack'
        f.write(f"• Dominant expert group: {dominant}\n")

        # Most used experts
        expert_means = activation_matrix.mean(axis=0)
        top_experts = np.argsort(expert_means)[-3:][::-1]
        f.write(f"• Top 3 most used experts: {list(top_experts)}\n")

        # Least used experts
        bottom_experts = np.argsort(expert_means)[:3]
        f.write(f"• Top 3 least used experts: {list(bottom_experts)}\n")

        # Diversity metric (entropy)
        expert_dist = activation_matrix.sum(axis=0) / activation_matrix.sum()
        diversity = entropy(expert_dist)
        f.write(f"• Expert diversity (entropy): {diversity:.3f}\n")

        if diversity < 1.5:
            f.write("• Low diversity: Model relies on few experts\n")
        elif diversity > 2.0:
            f.write("• High diversity: Model uses many experts uniformly\n")
        else:
            f.write("• Moderate diversity: Balanced expert usage\n")

        # Timestep analysis
        f.write("\n\n" + "=" * 50 + "\n")
        f.write("Timestep Analysis:\n")
        f.write("-" * 30 + "\n")
        for timestep in range(activation_matrix.shape[0]):
            timestep_acts = activation_matrix[timestep, :]
            dominant_expert = np.argmax(timestep_acts)
            for i, (group_start, group_end) in enumerate(expert_groups):
                if group_start <= dominant_expert <= group_end:
                    origin = 'square' if i == 0 else 'stack'
                    break
            f.write(f"Timestep {timestep}: Dominant expert {dominant_expert} ({origin}), "
                    f"max activation {timestep_acts.max():.4f}\n")

    print(f"Saved analysis report to {output_path}")


@click.command()
@click.option('-p', '--probs_path',
              required=True,
              help='Path to expert_routing_probs.pkl file')
@click.option('-o', '--output_dir',
              default='outputs/stack_three_visualization',
              help='Output directory for visualizations')
@click.option('--epochs_range',
              default=None,
              help='Epoch range to analyze (e.g., "90,99" for last 10 epochs)')
@click.option('--num_experts',
              default=8,
              type=int,
              help='Number of experts in model')
@click.option('--horizon',
              default=10,
              type=int,
              help='Horizon length (timesteps per sequence)')
@click.option('--expert_groups',
              default='0,3;4,7',
              help='Expert groups: "start,end;start,end" (e.g., "0,3;4,7" for square:0-3, stack:4-7)')
@click.option('--dpi',
              default=300,
              type=int,
              help='DPI for output figures')
def main(probs_path, output_dir, epochs_range, num_experts, horizon, expert_groups, dpi):
    """
    Visualize expert routing probabilities from training
    """
    print("=" * 80)
    print("Expert Routing Probability Visualization")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Parse expert groups
    expert_groups = parse_expert_groups(expert_groups)
    print(f"Expert groups: {expert_groups}")

    # Load data
    all_probs = load_and_validate_data(probs_path)

    # Aggregate probabilities
    activation_matrix = aggregate_probs(all_probs, num_experts, horizon, epochs_range)
    print(f"Activation matrix shape: {activation_matrix.shape}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    # Main heatmap (Figure 7 style)
    heatmap_path = os.path.join(output_dir, 'expert_scores_vs_timesteps.png')
    create_heatmap(activation_matrix, expert_groups, heatmap_path, dpi)

    # Evolution plot
    evolution_path = os.path.join(output_dir, 'expert_usage_evolution.png')
    create_evolution_plot(all_probs, expert_groups, num_experts, horizon, evolution_path, dpi)

    # Skill transfer analysis
    skill_transfer_path = os.path.join(output_dir, 'skill_transfer_analysis.png')
    create_skill_transfer_plot(activation_matrix, expert_groups, skill_transfer_path, dpi)

    # Correlation matrix
    correlation_path = os.path.join(output_dir, 'expert_correlation_matrix.png')
    create_correlation_matrix(activation_matrix, expert_groups, correlation_path, dpi)

    # Analysis report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    generate_report(activation_matrix, expert_groups, report_path)

    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  • {heatmap_path}")
    print(f"  • {evolution_path}")
    print(f"  • {skill_transfer_path}")
    print(f"  • {correlation_path}")
    print(f"  • {report_path}")


if __name__ == '__main__':
    main()
