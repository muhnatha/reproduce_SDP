"""
Usage:
python eval_stack_three_only.py --checkpoint /path/to/ckpt -o /path/to/output_dir
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["MUJOCO_GL"]="osmesa"
os.environ["PYOPENGL_PLATFORM"]="osmesa"
sys.path.insert(0, '/home/cc/reproduce_SDP/mimicgen_environments')
import mimicgen.envs

# Import the specific environment we need
from mimicgen.envs.robosuite.stack import StackThree_D0

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import copy
from omegaconf.omegaconf import open_dict
import yaml


@click.command()
@click.option('-c', '--checkpoint', default='outputs/stack_three_finetune/latest.ckpt')
@click.option('-o', '--output_dir', default='stack_three_eval_only')
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load stack_three config
    stack_three_cfg_path = "config/tasks/stack_three_d0.yaml"
    with open(stack_three_cfg_path, "r") as f:
        stack_three_task_cfg = yaml.safe_load(f)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint}")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    
    # Check if this is a finetune_router_only checkpoint (no 'cfg' key)
    if 'cfg' not in payload:
        print("Loading finetune_router_only checkpoint format")
        
        # Manually construct model parameters
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
    
    # Load all 3 normalizers that the model expects
    print("Loading all normalizers...")
    from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
    
    normalizers = []
    
    # Load square normalizer (task_id=0)
    try:
        square_cfg_path = "config/tasks/square_d0.yaml"
        with open(square_cfg_path, "r") as f:
            square_task_cfg = yaml.safe_load(f)
        
        square_dataset = hydra.utils.instantiate(square_task_cfg['dataset'])
        square_normalizer = square_dataset.get_normalizer()
        square_normalizer.to(device)
        normalizers.append(square_normalizer)
        print("✓ Square normalizer loaded")
    except Exception as e:
        print(f"Warning: Could not load square normalizer: {e}")
        # Create a minimal normalizer to avoid crashes
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        dummy_normalizer = LinearNormalizer()
        dummy_normalizer.to(device)
        normalizers.append(dummy_normalizer)
    
    # Load stack normalizer (task_id=1)
    try:
        stack_cfg_path = "config/tasks/stack_d0.yaml"
        with open(stack_cfg_path, "r") as f:
            stack_task_cfg = yaml.safe_load(f)
        
        stack_dataset = hydra.utils.instantiate(stack_task_cfg['dataset'])
        stack_normalizer = stack_dataset.get_normalizer()
        stack_normalizer.to(device)
        normalizers.append(stack_normalizer)
        print("✓ Stack normalizer loaded")
    except Exception as e:
        print(f"Warning: Could not load stack normalizer: {e}")
        # Create a minimal normalizer to avoid crashes
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        dummy_normalizer = LinearNormalizer()
        dummy_normalizer.to(device)
        normalizers.append(dummy_normalizer)
    
    # Load stack_three normalizer (task_id=2)
    try:
        stack_three_dataset = hydra.utils.instantiate(stack_three_task_cfg['dataset'])
        stack_three_normalizer = stack_three_dataset.get_normalizer()
        stack_three_normalizer.to(device)
        normalizers.append(stack_three_normalizer)
        print(f"✓ Stack three normalizer loaded")
    except Exception as e:
        print(f"Error: Could not load stack_three normalizer: {e}")
        # Create a minimal normalizer to avoid crashes
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        dummy_normalizer = LinearNormalizer()
        dummy_normalizer.to(device)
        normalizers.append(dummy_normalizer)
        print(f"✓ Created dummy stack_three normalizer")
    
    # Set all normalizers
    policy.normalizers = normalizers
    print(f"✓ Total normalizers loaded: {len(policy.normalizers)}")
    
    # Setup stack_three environment runner
    print("Setting up stack_three environment...")
    env_runner = hydra.utils.instantiate(
        stack_three_task_cfg['env_runner'], 
        output_dir=output_dir
    )
    
    # Run evaluation on stack_three only (task_id=2)
    print("Running evaluation on stack_three...")
    runner_log = env_runner.run(policy, task_id=torch.tensor([2], dtype=torch.int64).to(device))
    
    # Add task prefix to keys
    runner_log = {key + '_stack_three': value for key, value in runner_log.items()}
    
    # Save evaluation log
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    
    out_path = os.path.join(output_dir, 'eval_log_stack_three.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    
    print(f"\n✓ Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Evaluation log: {out_path}")
    
    # Print metrics summary
    if 'mean_score_stack_three' in json_log:
        print(f"Mean Score: {json_log['mean_score_stack_three']:.4f}")
    
    # Count videos
    video_count = sum(1 for key, value in json_log.items() 
                     if 'sim_video' in key and value is not None)
    if video_count > 0:
        print(f"Videos generated: {video_count}")
        print(f"Video location: {output_dir}/media/")


if __name__ == '__main__':
    os.environ["MUJOCO_GL"]="osmesa"
    main()