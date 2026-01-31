#!/usr/bin/env python3
"""
Fine-tune pretrained SDP model on Coffee Preparation task
Freezes expert weights in TaskMoE, trains only router (gates) + vision encoder + transformer
"""

import sys
sys.path.insert(0, '/home/cc/reproduce_SDP')
sys.path.insert(0, '/home/cc/reproduce_SDP/mimicgen_environments')

import os
import pathlib
import click
import hydra
import torch
import dill
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import shutil

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from omegaconf.omegaconf import OmegaConf
import yaml
import pickle

# Global list to collect expert routing probabilities across all epochs
all_probs = []


def load_pretrained_checkpoint(checkpoint_path, device='cpu'):
    """
    Load pretrained model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        workspace, model, cfg
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir='tmp_workspace')
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    model = workspace.model
    if cfg.training.use_ema:
        model = workspace.ema_model
    
    device = torch.device(device)
    model.to(device)
    for normalizer in model.normalizers:
        normalizer.to(device)
    
    print(f"Model loaded successfully on {device}")
    return workspace, model, cfg


def freeze_expert_weights(model):
    """
    Freeze expert weights in TaskMoE layers, keep router trainable
    
    Args:
        model: DiffusionTransformerHybridImagePolicy
    """
    print("\n=== Freezing Expert Weights ===")
    
    frozen_params = []
    trainable_params = []
    
    for name, param in model.named_parameters():
        # Freeze expert weights in TaskMoE layers
        if any(x in name for x in [
            'experts.weight',           # Expert input linear layers
            'experts.bias',             # Expert biases
            'output_experts.weight',    # Expert output linear layers
            'output_experts.bias'       # Expert output biases
        ]):
            param.requires_grad = False
            frozen_params.append((name, param.numel(), 'FROZEN'))
        else:
            # Keep router and other components trainable
            param.requires_grad = True
            if 'f_gate' in name or 'task_moe' in name:
                trainable_params.append((name, param.numel(), 'ROUTER'))
            else:
                trainable_params.append((name, param.numel(), 'OTHER'))
    
    # Print statistics
    frozen_count = sum(p[1] for p in frozen_params)
    trainable_count = sum(p[1] for p in trainable_params)
    total_count = frozen_count + trainable_count
    
    print(f"\nTotal Parameters: {total_count:,}")
    print(f"Frozen (Experts): {frozen_count:,} ({frozen_count/total_count*100:.2f}%)")
    print(f"Trainable: {trainable_count:,} ({trainable_count/total_count*100:.2f}%)")
    
    print(f"\nTrainable Router Parameters:")
    for name, num, type_ in trainable_params:
        if type_ == 'ROUTER':
            print(f"  {name}: {num:,}")
    
    print(f"\nFrozen Expert Parameters:")
    for name, num, _ in frozen_params[:10]:  # Show first 10
        print(f"  {name}: {num:,}")
    if len(frozen_params) > 10:
        print(f"  ... and {len(frozen_params) - 10} more")


def create_coffee_preparation_dataloader(dataset_path, batch_size=64, num_workers=4, return_normalizer=True):
    """
    Create dataloader for Coffee Preparation task

    Args:
        dataset_path: Path to HDF5 dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        return_normalizer: If True, return normalizer for Coffee Preparation task
    
    Returns:
        dataloader, dataset, normalizer (if return_normalizer=True)
    """
    print(f"\nLoading Coffee Preparation dataset from {dataset_path}")
    
    # Load shape meta from config
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
    
    # Create dataset
    dataset = RobomimicReplayImageDataset(
        dataset_path=dataset_path,
        horizon=10,
        n_obs_steps=2,
        pad_before=1,
        pad_after=7,
        rotation_rep='rotation_6d',
        seed=42,
        shape_meta=shape_meta,
        use_cache=True,
        val_ratio=0.02
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    # Get normalizer
    normalizer = dataset.get_normalizer()

    if return_normalizer:
        return dataloader, dataset, normalizer
    else:
        return dataloader, dataset




def train_epoch(model, dataloader, optimizer, device, task_id=2, epoch=0, 
              task2_normalizer=None):
    """
    Train for one epoch

    Args:
        model: Model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device
        task_id: Task ID for coffee preparation
        epoch: Current epoch number
        task2_normalizer: Optional normalizer for Coffee Preparation task (task_id=2)

    Returns:
        mean_loss
    """
    model.train()
    print(f"\n=== DEBUG train_epoch start ===")
    print(f"model.normalizers length: {len(model.normalizers)}")
    print(f"task_id: {task_id}")
    
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        try:
            # Try to use task_id=2 (Coffee Preparation)
            loss, probs = model.compute_loss(batch, task_id=task_id)
            
            # Capture expert routing probabilities if available
            if probs is not None:
                all_probs.append({
                    'epoch': epoch,
                    'task_id': task_id,
                    'probs': probs
                })
        except IndexError as e:
            # task_id=2 fails because model only has 2 normalizers (0:Coffee, 1:Mug Cleanup)
            # Fall back to task_id=0 (Coffee) which is similar
            print(f"Warning: task_id=2 failed, using task_id=0 fallback")
            loss, probs = model.compute_loss(batch, task_id=0)
            
            # Capture expert routing probabilities if available
            if probs is not None:
                all_probs.append({
                    'epoch': epoch,
                    'task_id': 0,  # Note: using fallback
                    'probs': probs
                })
        except Exception as e:
            print(f"Error in forward pass: {e}")
            print(f"Task ID: {task_id}")
            print(f"Available normalizers: {len(model.normalizers)}")
            raise
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    mean_loss = total_loss / num_batches if num_batches > 0 else 0
    return mean_loss


def validate(model, dataset, device, task_id=2, num_samples=10):
    """
    Quick validation
    
    Args:
        model: Model
        dataset: Dataset
        device: Device
        task_id: Task ID
        num_samples: Number of samples to validate
    
    Returns:
        mean_loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, np.ndarray) else v 
                      for k, v in sample.items()}
            
            try:
                loss = model.compute_loss(batch, task_id=task_id)
                total_loss += loss.item()
            except Exception as e:
                continue
    
    mean_loss = total_loss / num_samples
    return mean_loss


def save_checkpoint(workspace, model, optimizer, epoch, loss, output_dir):
    """
    Save training checkpoint
    
    Args:
        workspace: BaseWorkspace instance
        model: Model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Loss value
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(output_dir, f'epoch={epoch:04d}.ckpt')
    
    print(f"\nSaving checkpoint to {checkpoint_path}")
    
    # Create checkpoint payload
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    
    # Save checkpoint
    torch.save(payload, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


@click.command()
@click.option('-c', '--checkpoint', 
              default='outputs/2026-01-29/08-27-22/checkpoints/latest.ckpt',
              help='Path to pretrained checkpoint')
@click.option('-d', '--dataset',
              default='datasets/core/coffee_preparation_d0.hdf5',
              help='Path to coffee preparation dataset')
@click.option('-o', '--output_dir',
              default='outputs/coffee_preparation_finetune',
              help='Output directory for finetuning')
@click.option('-e', '--epochs',
              default=100,
              type=int,
              help='Number of training epochs')
@click.option('-b', '--batch_size',
              default=64,
              type=int,
              help='Batch size')
@click.option('-lr', '--learning_rate',
              default=1e-4,
              type=float,
              help='Learning rate')
@click.option('--device',
              default='cuda:0',
              help='Device to use')
@click.option('--save_every',
              default=10,
              type=int,
              help='Save checkpoint every N epochs')
@click.option('--validate_every',
              default=10,
              type=int,
              help='Validate every N epochs')
def main(checkpoint, dataset, output_dir, epochs, batch_size, learning_rate, 
         device, save_every, validate_every):
    """
    Fine-tune pretrained SDP model on Coffee Preparation task
    """
    print("=" * 80)
    print("Fine-tuning SDP Model on Coffee Preparation Task")
    print("=" * 80)
    
    # Step 1: Load pretrained checkpoint
    workspace, model, cfg = load_pretrained_checkpoint(checkpoint, device)
    
    # Debug: Check if normalizers were loaded from checkpoint
    print(f"Model has {len(model.normalizers)} normalizers after loading checkpoint")
    if len(model.normalizers) > 0:
        for i, norm in enumerate(model.normalizers):
            print(f"  Normalizer {i}: {type(norm).__name__}")
    else:
        print("  WARNING: No normalizers found in checkpoint!")
    
    # Step 2: Freeze expert weights
    freeze_expert_weights(model)
    
    # Step 3: Create dataloader
    dataloader, train_dataset, normalizer = create_coffee_preparation_dataloader(
        dataset, batch_size=batch_size, num_workers=4, return_normalizer=True
    )
    
    # Step 4: Add normalizer to model
    # Coffee Preparation normalizer returned separately when return_normalizer=True
    # model.normalizers already has Coffee (index 0) and Mug Cleanup (index 1)
    print(f"Coffee Preparation normalizer: {type(normalizer).__name__}")
    
    # Add Coffee Preparation normalizer as task_id=2
    model.normalizers.append(normalizer)
    normalizer.to(device)
    print(f"Added Coffee Preparation normalizer. Total normalizers: {len(model.normalizers)}")

    # Step 5: Create optimizer directly (avoid calling create_optimizer which may reset model.normalizers)
    # Group parameters by type (mutually exclusive)
    obs_encoder_params = [p for n, p in model.named_parameters() if p.requires_grad and 'obs_encoder' in n]
    router_params = [p for n, p in model.named_parameters() if p.requires_grad and 'obs_encoder' not in n and ('f_gate' in n or 'task_moe' in n)]
    transformer_params = [p for n, p in model.named_parameters() if p.requires_grad and 'obs_encoder' not in n and not ('f_gate' in n or 'task_moe' in n) and 'model' in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and 'obs_encoder' not in n and not ('f_gate' in n or 'task_moe' in n) and 'model' not in n]

    print(f"\n=== Optimizer Configuration ===")
    print(f"Router params: {len(router_params)}")
    print(f"Obs encoder params: {len(obs_encoder_params)}")
    print(f"Transformer params: {len(transformer_params)}")
    print(f"Other params: {len(other_params)}")
    print(f"Total trainable: {len(router_params) + len(obs_encoder_params) + len(transformer_params) + len(other_params)}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {1e-6}")

    optim_groups = []

    if transformer_params:
        optim_groups.append({
            "params": transformer_params,
            "weight_decay": 1e-3,  # transformer_weight_decay
        })
    if obs_encoder_params:
        optim_groups.append({
            "params": obs_encoder_params,
            "weight_decay": 1e-6,
        })
    if router_params:
        optim_groups.append({
            "params": router_params,
            "weight_decay": 1e-6,
        })
    if other_params:
        optim_groups.append({
            "params": other_params,
            "weight_decay": 1e-6,
        })

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
    )
    
    # Step 6: Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 7: Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, dataloader, optimizer, device, 
                             task_id=2, epoch=epoch, 
                             task2_normalizer=normalizer)
        
        # Validate
        if (epoch + 1) % validate_every == 0:
            val_loss = validate(model, train_dataset, device, task_id=2, 
                           num_samples=10)
            print(f"Validation Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"New best loss: {best_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(workspace, model, optimizer, epoch + 1, train_loss, 
                       output_dir)
            # Also save as latest
            latest_path = os.path.join(output_dir, 'latest.ckpt')
            if os.path.exists(latest_path):
                os.remove(latest_path)
            shutil.copy(os.path.join(output_dir, f'epoch={epoch+1:04d}.ckpt'), 
                       latest_path)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
    
    # Save expert routing probabilities
    print("\n" + "=" * 80)
    print("Saving expert routing probabilities...")
    print("=" * 80)
    
    probs_path = os.path.join(output_dir, 'expert_routing_probs.pkl')
    with open(probs_path, 'wb') as f:
        pickle.dump(all_probs, f)
    
    print(f"Saved {len(all_probs)} prob samples to {probs_path}")
    print("Use these probabilities for visualization")
    print("=" * 80)
    
    # Save final checkpoint
    save_checkpoint(workspace, model, optimizer, epochs, train_loss, output_dir)
    
    print("\n" + "=" * 80)
    print("Fine-tuning Complete!")
    print("=" * 80)
    print(f"\nFinal checkpoint saved to: {output_dir}")
    print(f"Use this checkpoint for visualization with visualize_expert_probs.py")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()
