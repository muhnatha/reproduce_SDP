#!/usr/bin/env python3
"""
Visualize Stack Three Task - Robot Only
Creates video showing robot execution only (no expert heatmap)
"""

import sys
sys.path.insert(0, '/home/cc/reproduce_SDP')
sys.path.insert(0, '/home/cc/reproduce_SDP/mimicgen_environments')

# Import mimicgen environments to register them
import mimicgen.envs
from mimicgen.envs.robosuite.stack import StackThree_D0

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import click
import torch
import dill
import hydra
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from tqdm import tqdm

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from omegaconf import OmegaConf
import yaml


class ExpertActivationTracker:
    """Track expert activations in TaskMoE layers"""
    
    def __init__(self, name):
        self.name = name
        self.activations = []
        self.current_timestep = 0
        
    def __call__(self, module, input, output):
        """Called during forward pass"""
        if not hasattr(module, 'batch_index') or module.batch_index is None:
            return
            
        # Get expert routing information
        expert_indices = module.batch_index.cpu().numpy()
        gate_values = module.batch_gates.cpu().numpy() if hasattr(module, 'batch_gates') else np.ones(len(expert_indices))
        
        # Aggregate by expert
        expert_activations = defaultdict(float)
        for idx, gate in zip(expert_indices, gate_values):
            expert_activations[idx] += float(gate)
        
        # Store as array of 8 values (one per expert)
        activation_array = np.zeros(8)
        for expert_id, activation in expert_activations.items():
            if 0 <= expert_id < 8:
                activation_array[expert_id] = activation
        
        # Normalize
        if activation_array.sum() > 0:
            activation_array = activation_array / activation_array.sum()
        
        self.activations.append({
            'timestep': self.current_timestep,
            'activations': activation_array,
            'expert_indices': expert_indices,
            'gate_values': gate_values
        })
        self.current_timestep += 1
    
    def reset(self):
        self.activations = []
        self.current_timestep = 0


def load_finetuned_checkpoint(checkpoint_path, device='cuda'):
    """Load finetuned checkpoint with 3 normalizers"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    
    # Check if this is a finetune_router_only checkpoint (no 'cfg' key)
    if 'cfg' not in payload:
        print("Note: Loading finetune_router_only checkpoint format")
        print(f"Checkpoint keys: {list(payload.keys())}")
        
        # Manually construct model parameters (from base.yaml structure)
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        
        shape_meta = {
            'action': {'shape': [7]},
            'obs': {
                'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
                'robot0_eef_pos': {'shape': [3]},
                'robot0_eef_quat': {'shape': [4]},
                'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
                'robot0_gripper_qpos': {'shape': [2]},
            }
        }
        
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        # Instantiate model directly
        from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
        
        policy = DiffusionTransformerHybridImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=10,
            n_tasks=3,
            n_action_steps=8,
            n_obs_steps=2,
            num_inference_steps=100,
            crop_shape=(80, 80),
            obs_encoder_group_norm=True,
            eval_fixed_crop=True,
            n_layer=12,
            n_cond_layers=0,
            n_head=4,
            n_emb=512,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False
        )
        
        # Load model weights
        policy.load_state_dict(payload['model'])
        policy.to(device)
        policy.eval()
        
        print(f"✓ Model loaded from state_dict")
        print(f"  Epoch: {payload.get('epoch', 'unknown')}")
        print(f"  Loss: {payload.get('loss', 0):.4f}")
        
        # Return None for cfg and workspace since we don't have them
        return policy, None, None
    else:
        # Original workspace-based loading
        cfg = payload['cfg']
        
        # Load workspace
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir='tmp_workspace')
        workspace.load_payload(payload)
        
        # Get policy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        
        policy.to(device)
        policy.eval()
        
        print(f"Model loaded with {len(policy.normalizers)} normalizers")
        
        return policy, cfg, workspace


def load_base_task_normalizers(device='cpu'):
    """Load square and stack normalizers"""
    base_normalizers = []
    
    shape_meta = {
        'action': {'shape': [7]},
        'obs': {
            'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
            'robot0_eef_pos': {'shape': [3]},
            'robot0_eef_quat': {'shape': [4]},
            'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
            'robot0_gripper_qpos': {'shape': [2]},
        }
    }
    
    # Load square
    try:
        square_dataset = RobomimicReplayImageDataset(
            dataset_path='datasets/core/square_d0.hdf5',
            horizon=10, n_obs_steps=2, pad_before=1, pad_after=7,
            rotation_rep='rotation_6d', seed=42, shape_meta=shape_meta,
            use_cache=True, val_ratio=0.02
        )
        square_norm = square_dataset.get_normalizer()
        square_norm.to(device)
        base_normalizers.append(square_norm)
        print("✓ Loaded square normalizer")
    except Exception as e:
        print(f"Warning: Could not load square normalizer: {e}")
    
    # Load stack
    try:
        stack_dataset = RobomimicReplayImageDataset(
            dataset_path='datasets/core/stack_d0.hdf5',
            horizon=10, n_obs_steps=2, pad_before=1, pad_after=7,
            rotation_rep='rotation_6d', seed=42, shape_meta=shape_meta,
            use_cache=True, val_ratio=0.02
        )
        stack_norm = stack_dataset.get_normalizer()
        stack_norm.to(device)
        base_normalizers.append(stack_norm)
        print("✓ Loaded stack normalizer")
    except Exception as e:
        print(f"Warning: Could not load stack normalizer: {e}")
    
    return base_normalizers


def setup_stack_three_env(output_dir):
    """Setup stack_three environment for visualization"""
    # Load config
    cfg_path = 'config/tasks/stack_three_d0.yaml'
    task_cfg = yaml.safe_load(open(cfg_path))
    
    # Create output directory
    media_dir = os.path.join(output_dir, 'media')
    os.makedirs(media_dir, exist_ok=True)
    
    # Extract required parameters from config
    dataset_path = task_cfg['dataset']['dataset_path']
    shape_meta = task_cfg['dataset']['shape_meta']
    
    # Instantiate env runner with all required parameters
    env_runner = RobomimicImageRunner(
        output_dir=output_dir,
        dataset_path=dataset_path,
        shape_meta=shape_meta,
        fps=10,
        n_train=1,
        n_test=0,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key='agentview_image',
        crf=22
    )
    
    return env_runner, task_cfg


def register_hooks(policy):
    """Register forward hooks on TaskMoE layers"""
    trackers = {}
    hooks = []
    
    # Hook on model's TaskMoE layers
    for name, module in policy.named_modules():
        if 'task_moe' in name.lower() or 'moe' in name.lower():
            tracker = ExpertActivationTracker(name)
            hook = module.register_forward_hook(tracker)
            trackers[name] = tracker
            hooks.append(hook)
    
    if not trackers:
        print("Warning: No TaskMoE layers found for tracking")
    else:
        print(f"Registered {len(trackers)} expert trackers")
    
    return trackers, hooks


def create_expert_heatmap(activations_history, current_timestep, max_timesteps=100):
    """Create 2D heatmap of expert activations"""
    # Prepare data: 8 experts x timesteps
    num_experts = 8
    window_size = min(max_timesteps, len(activations_history))
    
    # Get last window_size timesteps
    start_idx = max(0, len(activations_history) - window_size)
    window_activations = activations_history[start_idx:]
    
    # Create matrix
    heatmap_data = np.zeros((num_experts, len(window_activations)))
    for i, act in enumerate(window_activations):
        heatmap_data[:, i] = act['activations']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        cmap='YlOrRd',
        cbar_kws={'label': 'Activation'},
        vmin=0,
        vmax=1,
        ax=ax,
        xticklabels=False
    )
    
    # Labels
    ax.set_ylabel('Expert ID')
    ax.set_xlabel('Recent Timesteps')
    ax.set_title('Expert Activation Heatmap')
    
    # Add expert origin labels
    yticklabels = []
    for i in range(8):
        origin = 'square' if i < 4 else 'stack'
        yticklabels.append(f'{i} ({origin})')
    ax.set_yticklabels(yticklabels, rotation=0)
    
    # Mark current position
    if len(window_activations) > 0:
        current_in_window = len(window_activations) - 1
        ax.axvline(x=current_in_window + 0.5, color='blue', linewidth=2, linestyle='--')
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    heatmap_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # buffer_rgba returns RGBA (4 channels), convert to RGB (3 channels)
    heatmap_img = heatmap_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    heatmap_img = heatmap_img[:, :, :3]  # Drop alpha channel, keep RGB
    plt.close(fig)
    
    return heatmap_img


def create_robot_frame(robot_img, timestep, total_timesteps, active_experts):
    """Create frame showing only robot view with text overlay"""
    # Resize robot image for better visibility
    robot_h, robot_w = robot_img.shape[:2]
    target_h = 400
    scale = target_h / robot_h
    robot_resized = cv2.resize(robot_img, (int(robot_w * scale), target_h))
    
    # Use robot image directly (no heatmap)
    frame = robot_resized
    
    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title
    cv2.putText(frame, "Stack Three Demonstration", (10, 30), 
                font, 0.7, (255, 255, 255), 2)
    
    # Timestep info
    cv2.putText(frame, f"Timestep: {timestep//5}/{total_timesteps}", (10, 60),
                font, 0.6, (0, 255, 0), 2)
    
    # Active experts
    if active_experts:
        expert_labels = []
        for e in active_experts[:3]:
            origin = "square" if e < 4 else "stack"
            expert_labels.append(f"{e}({origin})")
        expert_text = "Active: " + ", ".join(expert_labels)
        cv2.putText(frame, expert_text, (10, 90),
                    font, 0.5, (0, 255, 255), 2)
    
    # Legend
    cv2.putText(frame, "Experts 0-3: square | Experts 4-7: stack", (10, frame.shape[0] - 20),
                font, 0.5, (255, 255, 255), 1)
    
    return frame


def run_episode(env_runner, policy, trackers, task_id=2, device='cuda', max_steps=400):
    """Run single episode and collect data"""
    # Reset
    for tracker in trackers.values():
        tracker.reset()
    
    obs = env_runner.env.reset()
    policy.reset()
    
    frames = []
    activations_history = []
    done = False
    timestep = 0
    
    with tqdm(total=max_steps, desc="Running episode") as pbar:
        while timestep < max_steps:  # Run all 400 frames, ignore done signal
            # Store current observation - convert to torch dict
            # obs from VectorEnv has shape (n_envs, n_obs_steps, ...), but predict_action expects (batch, n_obs_steps, ...)
            # So we need to pass it as-is, the model handles the batching
            obs_dict = {k: torch.from_numpy(v).to(device) 
                       for k, v in obs.items() if isinstance(v, np.ndarray)}
            
            # Get robot camera image for visualization (most recent frame from the observation window)
            if 'agentview_image' in obs:
                robot_img = obs['agentview_image']
                # Handle batched observation from VectorEnv: (n_envs, n_obs_steps, C, H, W)
                # We need to extract: env 0, last observation step -> (C, H, W)
                if robot_img.ndim == 5:
                    # Shape: (n_envs, n_obs_steps, C, H, W)
                    robot_img = robot_img[0, -1]  # First env, last timestep
                elif robot_img.ndim == 4:
                    # Shape: (n_obs_steps, C, H, W) - single env case
                    robot_img = robot_img[-1]  # Last timestep
                # Now robot_img should be (C, H, W)
                if robot_img.shape[0] == 3:
                    # Convert from CHW to HWC
                    robot_img = np.transpose(robot_img, (1, 2, 0))
                # Normalize to 0-255
                robot_img = ((robot_img - robot_img.min()) / 
                            (robot_img.max() - robot_img.min() + 1e-8) * 255).astype(np.uint8)
            else:
                robot_img = np.zeros((84, 84, 3), dtype=np.uint8)
            
            # Predict action with expert tracking
            with torch.no_grad():
                action_dict = policy.predict_action(
                    obs_dict, 
                    task_id=torch.tensor([task_id]).to(device)
                )
            
            # Collect activations from all trackers
            timestep_activations = np.zeros(8)
            for tracker in trackers.values():
                if tracker.activations:
                    # Get latest activation
                    latest = tracker.activations[-1]
                    timestep_activations += latest['activations']
            
            # Normalize
            if timestep_activations.sum() > 0:
                timestep_activations = timestep_activations / len(trackers)
            
            activations_history.append({
                'timestep': timestep,
                'activations': timestep_activations
            })
            
            # Find most active experts
            active_experts = np.where(timestep_activations > 0.1)[0].tolist()
            
            # Create robot frame (no heatmap)
            robot_frame = create_robot_frame(
                robot_img, timestep, max_steps//5, active_experts
            )
            frames.append(robot_frame)
            
            # Step environment
            action_np = action_dict['action'].detach().cpu().numpy()  # Keep batched shape for VectorEnv
            obs, reward, done, info = env_runner.env.step(action_np)
            
            timestep += 1
            pbar.update(1)
    
    return frames, activations_history


def save_video(frames, output_path, fps=10):
    """Save frames as MP4 video"""
    if not frames:
        print("Warning: No frames to save")
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"✓ Video saved to {output_path}")


def create_summary_heatmap(all_episodes_data, output_path):
    """Create summary heatmap of all episodes"""
    # Combine all episodes
    all_activations = []
    for episode_data in all_episodes_data:
        for act in episode_data:
            all_activations.append(act['activations'])
    
    if not all_activations:
        return
    
    # Create matrix: 8 experts x timesteps
    activation_matrix = np.array(all_activations).T  # (8, timesteps)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        activation_matrix,
        cmap='YlOrRd',
        cbar_kws={'label': 'Activation'},
        vmin=0,
        vmax=1,
        ax=ax
    )
    
    ax.set_ylabel('Expert ID')
    ax.set_xlabel('Timestep (across all episodes)')
    ax.set_title('Expert Activation Timeline - All Episodes')
    
    # Ytick labels with origins
    yticklabels = []
    for i in range(8):
        origin = 'square' if i < 4 else 'stack'
        yticklabels.append(f'{i} ({origin})')
    ax.set_yticklabels(yticklabels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Summary heatmap saved to {output_path}")


def generate_statistics(all_episodes_data, output_path):
    """Generate statistics report"""
    # Aggregate across all episodes
    all_activations = []
    for episode_data in all_episodes_data:
        for act in episode_data:
            all_activations.append(act['activations'])
    
    if not all_activations:
        return
    
    activation_matrix = np.array(all_activations)
    
    with open(output_path, 'w') as f:
        f.write("Stack Three Expert Activation Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Per-Expert Statistics:\n")
        f.write("-" * 30 + "\n")
        
        for expert_id in range(8):
            activations = activation_matrix[:, expert_id]
            mean_act = np.mean(activations)
            max_act = np.max(activations)
            active_count = np.sum(activations > 0.1)
            
            origin = 'square' if expert_id < 4 else 'stack'
            f.write(f"\nExpert {expert_id} ({origin}):\n")
            f.write(f"  Mean activation: {mean_act:.4f}\n")
            f.write(f"  Max activation: {max_act:.4f}\n")
            f.write(f"  Active timesteps: {active_count}/{len(activations)}\n")
        
        # Group analysis
        square_mean = np.mean(activation_matrix[:, :4])
        stack_mean = np.mean(activation_matrix[:, 4:])
        
        f.write("\n\nGroup Analysis:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Square experts (0-3) average: {square_mean:.4f}\n")
        f.write(f"Stack experts (4-7) average: {stack_mean:.4f}\n")
        f.write(f"Dominant group: {'stack' if stack_mean > square_mean else 'square'}\n")
        
        # Top experts
        expert_means = np.mean(activation_matrix, axis=0)
        top_3 = np.argsort(expert_means)[-3:][::-1]
        f.write(f"\nTop 3 most used experts: {list(top_3)}\n")
        
        bottom_3 = np.argsort(expert_means)[:3]
        f.write(f"Bottom 3 least used experts: {list(bottom_3)}\n")
    
    print(f"✓ Statistics saved to {output_path}")


@click.command()
@click.option('-c', '--checkpoint',
              default='outputs/stack_three_finetune/epoch=0100.ckpt',
              help='Path to finetuned checkpoint')
@click.option('-o', '--output_dir',
              default='outputs/stack_three_video_with_experts',
              help='Output directory')
@click.option('-n', '--num_episodes',
              default=3,
              type=int,
              help='Number of episodes to record')
@click.option('-d', '--device',
              default='cuda:0',
              help='Device to use')
@click.option('--max_steps',
              default=50,
              type=int,
              help='Maximum steps per episode')
def main(checkpoint, output_dir, num_episodes, device, max_steps):
    """Generate stack three video with expert activation overlay"""
    
    print("=" * 80)
    print("Stack Three Expert Activation Video Generator")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    media_dir = os.path.join(output_dir, 'media')
    os.makedirs(media_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Episodes: {num_episodes}")
    print(f"Device: {device}")
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    policy, cfg, workspace = load_finetuned_checkpoint(checkpoint, device)
    
    # Load base normalizers and add to model
    print("\nLoading base task normalizers...")
    base_normalizers = load_base_task_normalizers(device)
    
    # Add stack_three normalizer
    try:
        stack_three_dataset = RobomimicReplayImageDataset(
            dataset_path='datasets/core/stack_three_d0.hdf5',
            horizon=10, n_obs_steps=2, pad_before=1, pad_after=7,
            rotation_rep='rotation_6d', seed=42,
            shape_meta={
                'action': {'shape': [7]},
                'obs': {
                    'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
                    'robot0_eef_pos': {'shape': [3]},
                    'robot0_eef_quat': {'shape': [4]},
                    'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
                    'robot0_gripper_qpos': {'shape': [2]},
                }
            },
            use_cache=True, val_ratio=0.02
        )
        stack_three_norm = stack_three_dataset.get_normalizer()
        stack_three_norm.to(device)
        
        # Set normalizers: [square, stack, stack_three]
        policy.normalizers = base_normalizers + [stack_three_norm]
        print(f"✓ Model has {len(policy.normalizers)} normalizers ready")
    except Exception as e:
        print(f"Warning: Could not load stack_three normalizer: {e}")
        print("Using existing normalizers from checkpoint...")
    
    # Register expert tracking hooks
    print("\nRegistering expert tracking hooks...")
    trackers, hooks = register_hooks(policy)
    
    # Setup environment
    print("\nSetting up stack_three environment...")
    env_runner, task_cfg = setup_stack_three_env(output_dir)
    
    # Run episodes
    print("\n" + "=" * 80)
    print(f"Recording {num_episodes} episodes")
    print("=" * 80)
    
    all_episodes_data = []
    video_paths = []
    
    for episode_idx in range(num_episodes):
        print(f"\nEpisode {episode_idx + 1}/{num_episodes}")
        
        # Run episode
        frames, activations = run_episode(
            env_runner, policy, trackers, 
            task_id=2, device=device, max_steps=max_steps
        )
        
        # Save video
        video_path = os.path.join(media_dir, f'episode_{episode_idx:02d}.mp4')
        save_video(frames, video_path, fps=10)
        video_paths.append(video_path)
        
        # Store data
        all_episodes_data.append(activations)
        
        print(f"✓ Episode {episode_idx + 1} complete: {len(frames)} frames")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Create summary visualizations
    print("\n" + "=" * 80)
    print("Creating summary visualizations")
    print("=" * 80)
    
    # Summary heatmap
    heatmap_path = os.path.join(output_dir, 'expert_timeline.png')
    create_summary_heatmap(all_episodes_data, heatmap_path)
    
    # Statistics report
    stats_path = os.path.join(output_dir, 'activation_statistics.txt')
    generate_statistics(all_episodes_data, stats_path)
    
    # Metadata
    metadata = {
        'checkpoint': checkpoint,
        'num_episodes': num_episodes,
        'max_steps': max_steps,
        'fps': 10,
        'videos': video_paths,
        'total_frames': sum(len(frames) for frames in all_episodes_data)
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Video Generation Complete!")
    print("=" * 80)
    print(f"\nOutput files in: {output_dir}")
    print(f"  Videos: {media_dir}/")
    for i, path in enumerate(video_paths):
        print(f"    - episode_{i:02d}.mp4")
    print(f"  Summary: {heatmap_path}")
    print(f"  Statistics: {stats_path}")
    print(f"  Metadata: {metadata_path}")


if __name__ == '__main__':
    main()
