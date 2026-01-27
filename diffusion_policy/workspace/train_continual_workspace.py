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
import hydra

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
from torch.utils.data import DataLoader

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.common.pytorch_util import dict_apply


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
            self.ema_model = hydra.utils.instantiate(cfg.ema, model=self.model)
        
        self.optimizer = self.model.get_optimizer(
            transformer_weight_decay=cfg.optimizer.transformer_weight_decay,
            obs_encoder_weight_decay=cfg.optimizer.obs_encoder_weight_decay,
            learning_rate=cfg.optimizer.learning_rate,
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
        print(f"[TrainContinualWorkspace] ===== STAGE 0: Training on Can =====")
        self.current_stage = 0

        # Load all datasets and env runners
        self.datasets, self.env_runners = self.configure_datasets()
        normalizers = [n for (_, _, n) in self.datasets]
        self.model.set_normalizer(normalizers)
        device = torch.device(self.cfg.training.device)  
        self.model.normalizers = [n.to(device) for n in self.model.normalizers]  

        self.train_stage(task_id=0, num_epochs=self.cfg.training.num_epochs, num_new_experts=self.cfg.policy.num_experts_per_task)
        # Only evaluate if the final epoch didn't already trigger an evaluation
        if self.cfg.training.num_epochs % self.cfg.training.rollout_every != 0:
            self.evaluate_all_tasks(stage=0)
        self.print_stage_summary(stage=0)
        self.save_best_checkpoints_summary(stage=0)
        
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== STAGE 1: Adding Lift =====")
        self.add_new_task(task_id=1, num_new_experts=self.cfg.policy.num_experts_per_task)
        self.train_stage(task_id=1, num_epochs=self.cfg.training.num_epochs)
        # Only evaluate if the final epoch didn't already trigger an evaluation
        if self.cfg.training.num_epochs % self.cfg.training.rollout_every != 0:
            self.evaluate_all_tasks(stage=1)
        self.print_stage_summary(stage=1)
        self.save_best_checkpoints_summary(stage=1)
        
        # CRITICAL: Increment tasks_learned counter so add_new_task(2) expands
        # task_gate_freq for tasks 0 AND 1 (not just task 0)
        self.model.increment_tasks_learned_for_continual_learning()
        
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== STAGE 2: Adding Square =====")
        self.add_new_task(task_id=2, num_new_experts=self.cfg.policy.num_experts_per_task)
        self.train_stage(task_id=2, num_epochs=self.cfg.training.num_epochs)
        # Only evaluate if the final epoch didn't already trigger an evaluation
        if self.cfg.training.num_epochs % self.cfg.training.rollout_every != 0:
            self.evaluate_all_tasks(stage=2)
        self.print_stage_summary(stage=2)
        self.save_best_checkpoints_summary(stage=2)
        
        # Increment for consistency (useful if more tasks are added later)
        self.model.increment_tasks_learned_for_continual_learning()
        
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

            env_runner = hydra.utils.instantiate(task_cfg.env_runner, output_dir=self.output_dir)
            print(f"[TrainContinualWorkspace] Env runner loaded for {task_name}")
            
            normalizer = dataset.get_normalizer()
            
            datasets.append((task_id, dataset, normalizer))
            env_runners.append((task_id, env_runner))
        
        return datasets, env_runners
    
    def train_stage(self, task_id, num_epochs, num_new_experts=8):
        # Ensure num_tasks_learned includes the current task being trained
        self.num_tasks_learned = max(self.num_tasks_learned, task_id + 1)
        
        task_name = self.task_order[task_id]
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== Training {task_name} (Task {task_id}) =====")
        print(f"[TrainContinualWorkspace] Epochs: {num_epochs}")

        self.model.train()
        device = torch.device(self.cfg.training.device)
        self.model.to(device)

        # EMA model wraps the model, so it doesn't need separate .to() call

        for optimizer in [self.optimizer, self.model]:
            optimizer.zero_grad()

        if task_id > 0:
            self.model.freeze_existing_components_for_continual_learning()
            print(f"[TrainContinualWorkspace] Existing components frozen (tasks 0 to {task_id-1})")

        task_datasets = [(i, d, n) for i, d, n in self.datasets if i < (task_id + 1)]
        task_env_runners = {i: e for i, e in self.env_runners if i < (task_id + 1)}

        train_dataloaders = []
        for i, dataset, normalizer in task_datasets:
            dataloader = DataLoader(dataset, **self.cfg.dataloader)
            train_dataloaders.append((i, dataloader))

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for task_idx, dataloader in train_dataloaders:
                for batch in dataloader:
                    batch = dict_apply(batch, lambda x: x.to(device=device, non_blocking=True))
                    loss = self.model.compute_loss(batch, task_id=task_idx)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.item()
                    num_batches += 1

                    if num_batches % 10 == 0:
                        print(f"[TrainContinualWorkspace] Task {task_idx}, Batch {num_batches}/{len(dataloader)}, Loss: {loss.item():.4f}")

            epoch_loss /= num_batches
            print(f"[TrainContinualWorkspace] Epoch {epoch + 1}: {task_name} - Average Loss: {epoch_loss:.4f}")

            if (epoch + 1) % self.cfg.training.checkpoint_every == 0:
                self.save_checkpoint()

            if (epoch + 1) % self.cfg.training.rollout_every == 0:
                print(f"\n{'='*60}")
                print(f"[TrainContinualWorkspace] Epoch {epoch + 1}: Evaluating on all learned tasks")
                self.evaluate_all_tasks(stage=task_id)

        print(f"[TrainContinualWorkspace] Training {task_name} completed: {num_epochs} epochs")
    
    def add_new_task(self, task_id, num_new_experts):
        task_name = self.task_order[task_id]
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== Adding {task_name} =====")
        self.model.add_experts_for_continual_learning(num_new_experts=num_new_experts)
        self.model.add_router_for_continual_learning(task_id=task_id)
        self.model.freeze_existing_components_for_continual_learning(new_task_id=task_id)
    
    def evaluate_all_tasks(self, stage):
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== Evaluating Stage {stage} =====")

        self.model.eval()
        device = torch.device(self.cfg.training.device)

        stage_task_ids = list(range(self.num_tasks_learned))
        stage_task_results = {}

        task_env_runners = {i: e for i, e in self.env_runners}

        for task_idx, dataset, normalizer in self.datasets:
            if task_idx not in stage_task_ids:
                continue

            val_dataloader = DataLoader(dataset, **self.cfg.val_dataloader)
            env_runner = task_env_runners.get(task_idx)

            task_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    batch = dict_apply(batch, lambda x: x.to(device=device, non_blocking=True))
                    loss = self.model.compute_loss(batch, task_id=task_idx)
                    task_loss += loss.item()
                    num_batches += 1

            avg_loss = task_loss / num_batches
            success_rate = 1.0 - avg_loss

            print(f"[TrainContinualWorkspace] Task {task_idx} - Loss: {avg_loss:.4f}, Success: {success_rate:.2f}")

            with torch.no_grad():
                runner_log = env_runner.run(
                    self.model,
                    task_id=task_idx
                )

            runner_log = {f'{k}': v for k, v in runner_log.items() if isinstance(v, (float, int))}
            success_rate = runner_log['test/mean_score']   

            stage_task_results[task_idx] = {
                'loss': avg_loss,
                'success_rate': success_rate
            }
            print(f"[TrainContinualWorkspace] Task {task_idx} Success Rate: {success_rate:.2f}")
        
        self.stage_results[stage] = {
            'tasks': stage_task_results,
            'active_params': self.model.count_active_parameters_for_continual_learning()
        }
        self.save_checkpoint(f'stage_{stage}_results.pkl')
    
    def print_stage_summary(self, stage):
        print(f"\n{'='*60}")
        print(f"[TrainContinualWorkspace] ===== Stage {stage} Summary =====")
        print(f"[TrainContinualWorkspace] Tasks Learned: {[self.task_order[i] for i in range(self.num_tasks_learned)]}")
        
        stage_data = self.stage_results.get(stage, {})
        active_params = stage_data.get('active_params', self.model.count_active_parameters_for_continual_learning())
        print(f"[TrainContinualWorkspace] Active Params: {active_params:,}")
        
        task_results = stage_data.get('tasks', {})
        
        if task_results:
            for task_idx, results in task_results.items():
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
            # Header
            f.write("| Method | Stage | Task | Active Params | Success Rate |\n")
            f.write("|--------|-------|------|---------------|-------------|\n")
            
            # Write results for each stage
            for stage in range(3):
                stage_data = self.stage_results.get(stage, {})
                task_results = stage_data.get('tasks', {})
                active_params = stage_data.get('active_params', 'N/A')
                
                # Format active params (e.g., 9000000 -> "9.0M")
                if isinstance(active_params, int):
                    active_params_str = f"{active_params / 1e6:.1f}M"
                else:
                    active_params_str = str(active_params)
                
                for task_idx in range(stage + 1):
                    task_name = self.task_order[task_idx]
                    success = task_results.get(task_idx, {}).get('success_rate', 'N/A')
                    if isinstance(success, float):
                        success = f"{success:.2f}"
                    f.write(f"| SDP | Stage {stage + 1} | {task_name} | {active_params_str} | {success} |\n")
                
                f.write("|--------|-------|------|---------------|-------------|\n")
        
        print(f"[TrainContinualWorkspace] Table 3 results saved to {table_3_path}")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("config")),
    config_name="base_continual"
)
def main(cfg):
    workspace = TrainContinualWorkspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
