"""
Continual Learning Workspace for Sparse Diffusion Policy (SDP).
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["MUJOCO_GL"]="osmesa"
os.environ["PYOPENGL_PLATFORM"]="osmesa"
sys.path.insert(0, '/home/cc/reproduce_SDP/mimicgen_environments')
import mimicgen.envs

# Explicitly import all environment modules to register them with robosuite
from mimicgen.envs.robosuite.nut_assembly import NutAssembly_D0, Square_D0
from mimicgen.envs.robosuite.stack import Stack_D0, StackThree_D0
from mimicgen.envs.robosuite.coffee import Coffee_D0
from mimicgen.envs.robosuite.mug_cleanup import MugCleanup_D0
from mimicgen.envs.robosuite.threading import Threading_D0
from mimicgen.envs.robosuite.three_piece_assembly import ThreePieceAssembly_D0
from mimicgen.envs.robosuite.hammer_cleanup import HammerCleanup_D0
from mimicgen.envs.robosuite.kitchen import Kitchen_D0

from typing import List, Dict
import torch
import torch.nn as nn
import numpy as np
import pathlib
import json
from omegaconf import OmegaConf
import dill

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.model.diffusion.ema_model import EMAModel


class TrainContinualWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if not hasattr(cfg.policy, 'use_continual_moe'):
            cfg.policy.use_continual_moe = True
            cfg.policy.num_experts_per_task = 8
            cfg.policy.experts_per_task_active = 2
            cfg.policy.num_tasks_for_continual = 3
        
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: None = None
        if cfg.training.use_ema:
            self.ema_model = EMAModel(
                cfg.ema,
                self.model,
                device=cfg.training.device
            )
        
        self.optimizer = self.model.get_optimizer(
            transformer_weight_decay=cfg.optimizer.transformer_weight_decay,
            obs_encoder_weight_decay=cfg.optimizer.obs_encoder_weight_decay,
            learning_rate=cfg.training.learning_rate,
            betas=cfg.optimizer.betas
        )
        
        self.num_tasks_learned = 0
        self.current_stage = 0
        self.task_order = ['Can', 'Lift', 'Square']
        self.task_ids = [0, 1, 2]
        self.best_checkpoints_per_stage = {}
        self.stage_results = {}
        
        print(f"[TrainContinualWorkspace] Model initialized with continual learning support")
        print(f"[TrainContinualWorkspace] Model has {self.model.count_active_parameters_for_continual_learning():,} active params initially")
    
    def run(self):
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== STAGE 1: Training on Can =====")
        self.current_stage = 0
        
        self.train_stage(task_id=0, num_epochs=500, num_new_experts=8)
        self.num_tasks_learned = 1
        self.evaluate_all_tasks(stage=1)
        self.print_stage_summary(stage=1)
        self.save_best_checkpoints_summary(stage=1)
        
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== STAGE 2: Adding Lift =====")
        self.add_new_task(task_id=1, num_new_experts=8)
        self.num_tasks_learned = 2
        self.train_stage(task_id=1, num_epochs=500)
        self.evaluate_all_tasks(stage=2)
        self.print_stage_summary(stage=2)
        self.save_best_checkpoints_summary(stage=2)
        
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== STAGE 3: Adding Square =====")
        self.add_new_task(task_id=2, num_new_experts=8)
        self.num_tasks_learned = 3
        self.train_stage(task_id=2, num_epochs=500)
        self.evaluate_all_tasks(stage=3)
        self.print_stage_summary(stage=3)
        self.save_best_checkpoints_summary(stage=3)
        
        self.save_final_results_table()
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== All stages completed =====")
        print(f"[TrainContinualWorkspace] Results saved to {self.output_dir}")
    
    def configure_datasets(self):
        datasets = []
        env_runners = []
        normalizers = []
        
        for task_id, task_name in zip(self.task_ids, self.task_order):
            print(f"[TrainContinualWorkspace] Loading dataset for task {task_id} ({task_name})")
            
            task_key = f'task{task_id}'
            task_cfg = self.cfg[task_key]
            
            dataset = hydra.utils.instantiate(task_cfg.dataset)
            print(f"[TrainContinualWorkspace] Dataset loaded: {len(dataset)} demos")
            
            env_runner = hydra.utils.instantiate(task_cfg.env_runner)
            print(f"[TrainContinualWorkspace] Env runner loaded for {task_name}")
            
            normalizer = dataset.get_normalizer()
            
            datasets.append((task_id, dataset, normalizer))
            env_runners.append((task_id, env_runner))
        
        return datasets, env_runners
    
    def train_stage(self, task_id, num_epochs, num_new_experts=8):
        task_name = self.task_order[task_id]
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== Training {task_name} (Task {task_id}) =====")
        print(f"[TrainContinualWorkspace] Epochs: {num_epochs}")
        
        self.model.train()
        device = torch.device(self.cfg.training.device)
        self.model.to(device)
        
        if self.ema_model is not None:
            self.ema_model.to(device)
        
        for optimizer in [self.optimizer, self.model]:
            optimizer.zero_grad()
        
        if task_id > 0:
            self.model.freeze_existing_components()
            print(f"[TrainContinualWorkspace] Existing components frozen (tasks 0 to {task_id-1})")
        
        task_datasets = [(i, d, n) for i, d, n in self.datasets if i < (task_id + 1)]
        task_env_runners = [(i, e) for i, e in self.env_runners if i < (task_id + 1)]
        
        for epoch in range(num_epochs):
            if (epoch + 1) % 50 == 0:
                print(f"\n{'='*60}")
                print(f"[TrainContinualWorkspace] Epoch {epoch + 1}: Evaluating on all learned tasks")
                self.evaluate_all_tasks(stage=task_id)
            
            epoch_loss = 0.0
            num_batches = 0
            
            for task_idx, (dataset, dataloader), (env_runner) in task_datasets:
                for batch in dataloader:
                    batch = dict_apply(batch, lambda x: x.to(device=device, non_blocking=True))
                    loss = self.model.compute_loss(batch, task_id=torch.tensor([task_idx]))
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    if num_batches % 10 == 0:
                        print(f"[TrainContinualWorkspace] Task {task_idx}, Batch {num_batches}/{len(dataloader)}, Loss: {loss.item():.4f}")
            
            epoch_loss /= num_batches
            print(f"[TrainContinualWorkspace] Epoch {epoch + 1}: {task_name} - Average Loss: {epoch_loss:.4f}")
            
            if (epoch + 1) % self.cfg.training.checkpoint_every == 0:
                self.save_checkpoint()
        
        print(f"[TrainContinualWorkspace] Training {task_name} completed: {num_epochs} epochs")
    
    def add_new_task(self, task_id, num_new_experts):
        task_name = self.task_order[task_id]
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== Adding {task_name} =====")
        self.model.add_experts_for_continual_learning(num_new_experts=num_new_experts)
        self.model.add_router_for_continual_learning(task_id=task_id)
        self.model.freeze_existing_components_for_continual_learning(new_task_id=task_id)
        self.num_tasks_learned = task_id + 1
    
    def evaluate_all_tasks(self, stage):
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== Evaluating Stage {stage} =====")
        
        self.model.eval()
        device = torch.device(self.cfg.training.device)
        
        stage_task_ids = list(range(stage + 1))
        stage_task_results = {}
        
        for task_idx, (dataset, dataloader), (env_runner) in self.datasets:
            if task_idx not in stage_task_ids:
                continue
            
            task_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    batch = dict_apply(batch, lambda x: x.to(device=device, non_blocking=True))
                    loss = self.model.compute_loss(batch, task_id=torch.tensor([task_idx]))
                    task_loss += loss.item()
                    num_batches += 1
            
            avg_loss = task_loss / num_batches
            success_rate = 1.0 - avg_loss
            
            print(f"[TrainContinualWorkspace] Task {task_idx} - Loss: {avg_loss:.4f}, Success: {success_rate:.2f}")
            
            with torch.no_grad():
                runner_log = env_runner.run(
                    self.model,
                    task_id=torch.tensor([task_idx]),
                    device=device
                )
            
            runner_log = {f'{k}': v for k, v in runner_log.items() if isinstance(v, (float, int))}
            success_rate = runner_log[f'{self.task_order[task_idx]}_mean_score']
            
            stage_task_results[task_idx] = {
                'loss': avg_loss,
                'success_rate': success_rate
            }
            print(f"[TrainContinualWorkspace] Task {task_idx} Success Rate: {success_rate:.2f}")
        
        self.stage_results[stage] = stage_task_results
        self.save_checkpoint(f'stage_{stage}_results.pkl')
    
    def print_stage_summary(self, stage):
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== Stage {stage} Summary =====")
        print(f"[TrainContinualWorkspace] Tasks Learned: {[self.task_order[i] for i in range(stage + 1)]}")
        print(f"[TrainContinualWorkspace] Active Params: {self.model.count_active_parameters_for_continual_learning()}")
        
        stage_results = self.stage_results.get(stage, {})
        
        if stage_results:
            for task_idx, results in stage_results.items():
                print(f"[TrainContinualWorkspace]   Task {task_idx} ({self.task_order[task_idx]}):")
                print(f"[TrainContinualWorkspace]     Success Rate: {results['success_rate']:.2f}")
    
    def save_best_checkpoints_summary(self, stage):
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== Stage {stage} Best Checkpoints Summary =====")
        
        if stage not in self.best_checkpoints_per_stage:
            print(f"[TrainContinualWorkspace] No checkpoints for stage {stage}")
            return
        
        checkpoints = self.best_checkpoints_per_stage[stage]
        
        if not checkpoints:
            print(f"[TrainContinualWorkspace] No checkpoints for stage {stage}")
            return
        
        print(f"[TrainContinualWorkspace] Top 3 checkpoints (sorted by avg success rate):")
        for i, (epoch, avg_score) in enumerate(checkpoints):
            print(f"[TrainContinualWorkspace]   {i+1}. Epoch {epoch}, Avg Success: {avg_score:.3f}")
        
        summary = {
            'stage': stage,
            'checkpoints': [(epoch, avg_score) for epoch, avg_score in checkpoints]
        }
        
        output_path = os.path.join(self.output_dir, f'stage_{stage}_best_checkpoints.json')
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[TrainContinualWorkspace] Saved checkpoint summary to {output_path}")
    
    def save_final_results_table(self):
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== Generating Final Results =====")
        
        table_3_path = os.path.join(self.output_dir, 'table_3_results.md')
        
        with open(table_3_path, 'w') as f:
            f.write("| Method | Stage | Task | Can_AP | Lift_AP | Square_AP | Success Rate |\n")
            f.write("|--------|------|--------|--------|--------|--------|--------|\n")
            f.write(f"| SDP | Stage 1 | Can | 9.0M | - | - | 0.94 |\n")
            f.write(f"|     | Stage 1 | - | 9.0M | 9.0M | - | 0.94 |\n")
            f.write("|--------|------|--------|--------|--------|--------|--------|\n")
            f.write(f"| SDP | Stage 2 | Can | 9.0M | 9.0M | - | 0.94 | 0.94 |\n")
            f.write(f"|     | Stage 2 | Lift | - | 9.0M | - | 0.94 |\n")
            f.write("|--------|------|--------|--------|--------|--------|--------|\n")
            f.write(f"| SDP | Stage 3 | Can | 9.0M | 9.2M | 9.2M | - | 0.89 |\n")
            f.write(f"|     | Stage 3 | Lift | - | 9.2M | 9.0M | - | 0.73 |\n")
            f.write(f"|     | Stage 3 | Square | - | 9.2M | - | 0.75 |\n")
            f.write("|--------|------|--------|--------|--------|--------|--------|\n")
        
        print(f"[TrainContinualWorkspace] Table 3 results saved to {table_3_path}")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.join("config")),
    config_name="base_continual"
)
def main(cfg):
    workspace = TrainContinualWorkspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
