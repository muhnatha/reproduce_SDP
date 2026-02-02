"""
Continual TaskMoE: Extended TaskMoE with support for expert expansion and freezing
for continual learning experiments (Can → Lift → Square).

This module extends the original TaskMoE to support:
1. Dynamically adding new experts as new tasks are learned
2. Adding new routers for each new task
3. Freezing existing experts and routers to prevent catastrophic forgetting
4. Counting active (trainable) parameters for AP reporting

Key concepts from paper Appendix C.1:
- 8 experts per new task
- Activate 2 experts per task
- Freeze all existing components when learning new task
"""

import math
from typing import List, Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.cuda.amp import custom_fwd, custom_bwd
from torch import Tensor

# Import parent class
from mixture_of_experts.task_moe import TaskMoE, compute_gating


class ContinualTaskMoE(TaskMoE):
    """
    TaskMoE with continual learning support.
    
    Key additions over TaskMoE:
    - add_experts(num_new_experts): Dynamically expand expert capacity
    - add_router(): Add new router for new task
    - freeze_existing_experts(): Freeze old expert parameters
    - freeze_existing_routers(): Freeze old router parameters
    - get_active_parameter_count(): Count trainable parameters only
    - get_expert_assignment(): Return expert indices per task
    
    Usage pattern for continual learning:
    - Stage 1 (Can): Initialize with 8 experts, 1 router
    - Stage 2 (Lift): Add 8 experts (total 16), add 1 router (total 2), freeze all old components
    - Stage 3 (Square): Add 8 experts (total 24), add 1 router (total 3), freeze all old components
    """
    
    def __init__(self, input_size, head_size, num_experts, k, w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, 
                 task_num=9, noisy_gating=True, gating_activation=None, num_tasks_for_continual=1, 
                 num_experts_per_task=8, experts_per_task_active=2, **kwargs):
        """
        Initialize ContinualTaskMoE.
        
        Args:
            input_size: Size of input features
            head_size: Hidden size of experts
            num_experts: Initial number of experts (typically 8 for first task)
            k: Number of experts to activate per forward pass (typically 2)
            w_MI, w_H, w_finetune_MI: Loss weights
            limit_k: Limit on top-k experts
            w_topk_loss: Weight for top-k loss
            task_num: Total number of tasks in dataset (not number of continual stages)
            noisy_gating: Whether to use noisy gating
            gating_activation: Activation function for gating
            num_tasks_for_continual: Number of continual learning stages (e.g., 3 for Can→Lift→Square)
            num_experts_per_task: Number of new experts to add per task (default 8 per paper)
            experts_per_task_active: Number of experts to activate per task (default 2 per paper)
        """
        # MODIFIED: Only initialize with first task's experts
        # num_experts should be the initial experts for task 0 (e.g., 8)
        super(ContinualTaskMoE, self).__init__(
            input_size=input_size, 
            head_size=head_size, 
            num_experts=num_experts,
            k=k,
            w_MI=w_MI,
            w_H=w_H,
            w_finetune_MI=w_finetune_MI,
            limit_k=limit_k,
            w_topk_loss=w_topk_loss,
            task_num=task_num,  # Total tasks in dataset
            noisy_gating=noisy_gating, 
            gating_activation=gating_activation,
            **kwargs
        )
        
        # Continual learning specific state tracking
        self.num_tasks_for_continual = num_tasks_for_continual
        self.num_experts_per_task = num_experts_per_task
        self.experts_per_task_active = experts_per_task_active
        
        # Track which experts belong to which task
        # Format: list of lists, where task_experts[task_id] = [expert_start_idx, expert_end_idx)
        # Example: [[0, 1], [2, 3], [4, 5]] for 3 tasks with 2 experts each
        self.task_expert_assignment = []
        
        # FIXED: Initialize assignment for first task consistently with subsequent tasks
        # Now Task 0 uses experts_per_task_active experts, same as other tasks
        first_task_experts = list(range(min(self.experts_per_task_active, num_experts)))
        self.task_expert_assignment.append(first_task_experts)
        
        # Track frozen state
        # frozen_expert_masks: List of tensors, one per task, indicating which experts are frozen
        # frozen_router_masks: List of tensors, one per router, indicating which parameters are frozen
        self.frozen_expert_masks = []
        self.frozen_router_masks = []
        
        # FIXED: Initialize frozen mask for task 0 correctly
        # Only the experts assigned to task 0 should be active, others should be frozen
        # This ensures Stage 0 has the same number of active parameters as other stages
        frozen_mask = torch.ones(num_experts, dtype=torch.bool)
        frozen_mask[first_task_experts] = False  # Mark assigned experts as active
        self.frozen_expert_masks.append(frozen_mask)
        
        # Count frozen vs trainable experts for task 0
        num_frozen = frozen_mask.sum().item()
        num_trainable = (~frozen_mask).sum().item()
        print(f"[ContinualTaskMoE] Task 0: Frozen {num_frozen} experts, {num_trainable} trainable (experts {first_task_experts})")
        
        # FIXED: Set up gradient hooks for task 0 to enforce freezing from the start
        # This ensures only the assigned experts for task 0 can be trained
        def create_expert_grad_hook(mask):
            def hook(grad):
                # mask shape: (num_experts,), grad shape: (num_experts, input_size, output_size)
                # Expand mask to match grad dimensions and zero out frozen expert gradients
                expanded_mask = mask.view(-1, 1, 1).expand_as(grad)
                return grad * (~expanded_mask).float()
            return hook
        
        self._expert_grad_hooks = []
        
        # Register hooks for expert weights and biases
        hook = self.experts.weight.register_hook(create_expert_grad_hook(frozen_mask))
        self._expert_grad_hooks.append(hook)
        
        if self.experts.bias is not None:
            # Bias has shape (num_experts, output_size)
            def create_bias_grad_hook(mask):
                def hook(grad):
                    expanded_mask = mask.view(-1, 1).expand_as(grad)
                    return grad * (~expanded_mask).float()
                return hook
            hook = self.experts.bias.register_hook(create_bias_grad_hook(frozen_mask))
            self._expert_grad_hooks.append(hook)
        
        hook = self.output_experts.weight.register_hook(create_expert_grad_hook(frozen_mask))
        self._expert_grad_hooks.append(hook)
        
        if self.output_experts.bias is not None:
            def create_bias_grad_hook(mask):
                def hook(grad):
                    expanded_mask = mask.view(-1, 1).expand_as(grad)
                    return grad * (~expanded_mask).float()
                return hook
            hook = self.output_experts.bias.register_hook(create_bias_grad_hook(frozen_mask))
            self._expert_grad_hooks.append(hook)
        
        # Initialize frozen mask for initial router(s) in f_gate
        # This ensures frozen_router_masks[task_id] is valid when freeze_existing_routers() is called
        for router in self.f_gate:
            num_params = sum(p.numel() for p in router.parameters())
            self.frozen_router_masks.append(torch.zeros(num_params, dtype=torch.bool))
        
        self.num_tasks_learned = 1  # Start with 1 task learned
        
        print(f"[ContinualTaskMoE] Initialized with {num_experts} experts for task 0")
        print(f"[ContinualTaskMoE] Expert assignment: {self.task_expert_assignment}")
    
    def add_experts(self, num_new_experts, task_id=None):
        """
        Dynamically expand the expert capacity by adding new experts.
        
        Args:
            num_new_experts: Number of new experts to add (typically 8)
            task_id: ID of the new task (e.g., 1 for second task)
        
        This method:
        1. Saves existing expert weights
        2. Creates new expert tensors with expanded capacity
        3. Initializes new expert weights (preserves old weights)
        4. Replaces the expert parameters in-place
        """
        if task_id is None:
            task_id = self.num_tasks_learned
        
        old_num_experts = self.num_experts
        new_num_experts = old_num_experts + num_new_experts
        
        print(f"[ContinualTaskMoE] Adding {num_new_experts} experts for task {task_id}")
        print(f"[ContinualTaskMoE] Expanding from {old_num_experts} to {new_num_experts} experts")
        
        # Save existing expert weights and biases
        old_expert_weights = self.experts.weight.data.clone()
        if self.experts.bias is not None:
            old_expert_biases = self.experts.bias.data.clone()
        else:
            old_expert_biases = None
        
        old_output_expert_weights = self.output_experts.weight.data.clone()
        if self.output_experts.bias is not None:
            old_output_expert_biases = self.output_experts.bias.data.clone()
        else:
            old_output_expert_biases = None
        
        # Create new expert tensors with expanded capacity
        # Shape: [new_num_experts, input_size, head_size] for experts
        # Shape: [new_num_experts, head_size, input_size] for output experts
        new_expert_weights = torch.zeros(
            new_num_experts, 
            old_expert_weights.size(1), 
            old_expert_weights.size(2),
            device=old_expert_weights.device,
            dtype=old_expert_weights.dtype
        )
        
        # Copy old expert weights to new tensors
        new_expert_weights[:old_num_experts] = old_expert_weights
        
        # Initialize new expert weights
        # Use same initialization as parent class: uniform(-1/input_size, 1/input_size)
        std_expert = 1.0 / math.sqrt(old_expert_weights.size(1))
        new_expert_weights[old_num_experts:] = torch.empty(
            num_new_experts, 
            old_expert_weights.size(1), 
            old_expert_weights.size(2),
            device=old_expert_weights.device,
            dtype=old_expert_weights.dtype
        ).uniform_(-std_expert, std_expert)
        
        # Create new output expert tensors
        new_output_expert_weights = torch.zeros(
            new_num_experts, 
            old_output_expert_weights.size(1), 
            old_output_expert_weights.size(2),
            device=old_output_expert_weights.device,
            dtype=old_output_expert_weights.dtype
        )
        new_output_expert_weights[:old_num_experts] = old_output_expert_weights
        
        # Initialize new output expert weights
        std_output = 1.0 / math.sqrt(old_output_expert_weights.size(1))
        new_output_expert_weights[old_num_experts:] = torch.empty(
            num_new_experts, 
            old_output_expert_weights.size(1), 
            old_output_expert_weights.size(2),
            device=old_output_expert_weights.device,
            dtype=old_output_expert_weights.dtype
        ).uniform_(-std_output, std_output)
        
        # Handle biases
        if old_expert_biases is not None:
            new_expert_biases = torch.zeros(
                new_num_experts,
                old_expert_biases.size(1),
                device=old_expert_biases.device,
                dtype=old_expert_biases.dtype
            )
            new_expert_biases[:old_num_experts] = old_expert_biases
            # Initialize new biases to zeros
            new_expert_biases[old_num_experts:] = torch.zeros(
                num_new_experts,
                old_expert_biases.size(1),
                device=old_expert_biases.device,
                dtype=old_expert_biases.dtype
            )
        else:
            new_expert_biases = None
        
        if old_output_expert_biases is not None:
            new_output_expert_biases = torch.zeros(
                new_num_experts,
                old_output_expert_biases.size(1),
                device=old_output_expert_biases.device,
                dtype=old_output_expert_biases.dtype
            )
            new_output_expert_biases[:old_num_experts] = old_output_expert_biases
            new_output_expert_biases[old_num_experts:] = torch.zeros(
                num_new_experts,
                old_output_expert_biases.size(1),
                device=old_output_expert_biases.device,
                dtype=old_output_expert_biases.dtype
            )
        else:
            new_output_expert_biases = None
        
        # Update num_experts in parent class
        self.num_experts = new_num_experts
        
        # Replace expert parameters in-place
        # We need to create new nn.Parameter objects
        del self.experts.weight
        del self.experts.bias
        del self.output_experts.weight
        del self.output_experts.bias
        
        self.experts.weight = nn.Parameter(new_expert_weights)
        if new_expert_biases is not None:
            self.experts.bias = nn.Parameter(new_expert_biases)
        else:
            self.experts.bias = None
            
        self.output_experts.weight = nn.Parameter(new_output_expert_weights)
        if new_output_expert_biases is not None:
            self.output_experts.bias = nn.Parameter(new_output_expert_biases)
        else:
            self.output_experts.bias = None
        
        # Expand PTE (task-expert distribution) buffer
        # Shape: [task_num, new_num_experts]
        old_PTE = self.PTE.data.clone()
        new_PTE = torch.zeros(
            self.task_num,
            new_num_experts,
            device=old_PTE.device,
            dtype=old_PTE.dtype
        )
        new_PTE[:, :old_num_experts] = old_PTE[:, :old_num_experts]
        del self.PTE
        self.register_buffer('PTE', new_PTE)
        
        # Expand PE (expert usage) buffer
        # Shape: [new_num_experts]
        old_PE = self.PE.data.clone()
        new_PE = torch.zeros(
            new_num_experts,
            device=old_PE.device,
            dtype=old_PE.dtype
        )
        new_PE[:old_num_experts] = old_PE[:old_num_experts]
        del self.PE
        self.register_buffer('PE', new_PE)
        
        # Expand task_gate_freq for all previously learned tasks
        # These start as scalar 0 but become tensors of shape [num_experts] during training
        for t in range(task_id):
            if isinstance(self.task_gate_freq[t], torch.Tensor):
                old_tensor = self.task_gate_freq[t]
                new_tensor = torch.zeros(new_num_experts, device=old_tensor.device, dtype=old_tensor.dtype)
                new_tensor[:old_num_experts] = old_tensor
                self.task_gate_freq[t] = new_tensor
        
        # Expand topk_acc_probs for all previously learned tasks
        # These start as scalar 0 but become tensors of shape [num_experts] during training
        for t in range(task_id):
            if isinstance(self.topk_acc_probs[t], torch.Tensor):
                old_tensor = self.topk_acc_probs[t]
                new_tensor = torch.zeros(new_num_experts, device=old_tensor.device, dtype=old_tensor.dtype)
                new_tensor[:old_num_experts] = old_tensor
                self.topk_acc_probs[t] = new_tensor
        
        # Note: token_probs has shape [k] where k is fixed, so no expansion needed
        
        # Update task expert assignment
        # New task uses next available experts
        new_task_start_idx = old_num_experts
        new_task_end_idx = min(old_num_experts + self.experts_per_task_active, new_num_experts)
        self.task_expert_assignment.append(list(range(new_task_start_idx, new_task_end_idx)))
        
        # Initialize frozen masks for new experts
        # Initially, all experts are trainable (mask = False means not frozen)
        # We'll set frozen=True for old experts when calling freeze_existing_experts()
        self.frozen_expert_masks.append(torch.zeros(new_num_experts, dtype=torch.bool, device=self.experts.weight.device))
        
        print(f"[ContinualTaskMoE] Expert assignment after adding task {task_id}: {self.task_expert_assignment}")
        print(f"[ContinualTaskMoE] New expert shape: {self.experts.weight.shape}")
    
    def add_router(self, task_id=None):
        """
        Expand all existing routers and create a new router for the specified task.
        
        Args:
            task_id: ID of the new task (e.g., 1 for second task)
        
        This method:
        1. Expands all existing routers (tasks 0 to task_id-1) to output logits for
           the current number of experts, preserving learned weights
        2. Creates a fresh router for the new task_id
        
        Note: TaskMoE.__init__ creates routers for all tasks upfront, but they are sized
        for the initial num_experts. When we add experts, we need to expand all routers
        so they output the correct size (2 * new_num_experts for noisy_gating).
        """
        if task_id is None:
            task_id = self.num_tasks_learned
        
        print(f"[ContinualTaskMoE] Adding/expanding routers for {self.num_experts} experts")
        print(f"[ContinualTaskMoE] Current number of routers: {len(self.f_gate)}")
        
        new_output_size = 2 * self.num_experts if self.noisy_gating else self.num_experts
        
        # Step 1: Expand all existing routers (tasks 0 to task_id-1) to preserve learned routing
        for existing_task_id in range(task_id):
            old_router = self.f_gate[existing_task_id]
            
            # Get the final linear layer of the router
            old_final_layer = old_router[-1]
            old_output_size = old_final_layer.out_features
            
            # Skip if already correctly sized
            if old_output_size == new_output_size:
                print(f"[ContinualTaskMoE] Router {existing_task_id} already sized for {self.num_experts} experts, skipping")
                continue
            
            print(f"[ContinualTaskMoE] Expanding router {existing_task_id} from {old_output_size} to {new_output_size} outputs")
            
            # Create new final layer with expanded output size
            new_final_layer = nn.Linear(
                old_final_layer.in_features,
                new_output_size,
                bias=old_final_layer.bias is not None
            )
            
            # Copy old weights to new layer (preserve learned routing for existing experts)
            with torch.no_grad():
                new_final_layer.weight.data[:old_output_size] = old_final_layer.weight.data
                # Initialize new outputs to zero so softmax gives ~equal small probabilities to new experts
                new_final_layer.weight.data[old_output_size:] = 0.0
                
                if old_final_layer.bias is not None:
                    new_final_layer.bias.data[:old_output_size] = old_final_layer.bias.data
                    new_final_layer.bias.data[old_output_size:] = 0.0
            
            # Move to same device as old layer
            new_final_layer = new_final_layer.to(old_final_layer.weight.device)
            
            # Replace final layer in the router
            old_router[-1] = new_final_layer
            
            # Update frozen mask for this router (keep existing mask pattern, just expand size)
            num_params = sum(p.numel() for p in old_router.parameters())
            device = next(old_router.parameters()).device
            self.frozen_router_masks[existing_task_id] = torch.zeros(num_params, dtype=torch.bool, device=device)
        
        # Step 2: Create a fresh router for the new task_id
        print(f"[ContinualTaskMoE] Creating new router for task {task_id} (sized for {self.num_experts} experts)")
        
        if self.w_finetune_MI < -100:
            # Use simple linear layer (same as finetune mode)
            new_router = nn.Sequential(
                nn.Linear(
                    self.input_size,
                    new_output_size,
                    bias=False
                )
            )
            nn.init.zeros_(new_router[-1].weight)
        else:
            # Use two-layer network with activation (same as normal mode)
            gating_activation = nn.GELU()
            new_router = nn.Sequential(
                nn.Linear(self.input_size, self.input_size // 4),
                gating_activation,
                nn.Linear(
                    self.input_size // 4,
                    new_output_size,
                    bias=True
                )
            )
            nn.init.zeros_(new_router[-1].weight)
        
        # Replace the router at task_id
        self.f_gate[task_id] = new_router
        
        # Update frozen mask for new router (all trainable)
        num_params = sum(p.numel() for p in new_router.parameters())
        self.frozen_router_masks[task_id] = torch.zeros(num_params, dtype=torch.bool, device=next(new_router.parameters()).device)
        
        print(f"[ContinualTaskMoE] Router at index {task_id} created (total routers: {len(self.f_gate)})")
    
    def freeze_existing_experts(self, new_task_id=None):
        """
        Freeze all existing expert parameters (before the new task).
        
        Args:
            new_task_id: ID of the new task being learned (e.g., 1 for Lift after Can)
        
        This method sets requires_grad=False for all experts that don't belong to the new task,
        preventing catastrophic forgetting of previously learned tasks.
        """
        if new_task_id is None:
            new_task_id = self.num_tasks_learned
        
        print(f"[ContinualTaskMoE] Freezing experts for tasks 0 to {new_task_id-1}")
        
        # Get expert indices for new task only
        new_task_expert_indices = self.task_expert_assignment[new_task_id]
        
        # Freeze all experts that are not in the new task
        # Create mask: True = frozen, False = trainable
        frozen_mask = torch.ones(self.num_experts, dtype=torch.bool, device=self.experts.weight.device)
        frozen_mask[new_task_expert_indices] = False
        
        # Store frozen mask for this task (used by gradient hook)
        self.frozen_expert_masks[new_task_id] = frozen_mask.clone()
        
        # Register gradient hooks to zero out gradients for frozen experts
        # This effectively freezes them while keeping the tensor as a single Parameter
        def create_expert_grad_hook(mask):
            def hook(grad):
                # mask shape: (num_experts,), grad shape: (num_experts, input_size, output_size)
                # Expand mask to match grad dimensions and zero out frozen expert gradients
                expanded_mask = mask.view(-1, 1, 1).expand_as(grad)
                return grad * (~expanded_mask).float()
            return hook
        
        # Remove existing hooks if any
        if hasattr(self, '_expert_grad_hooks'):
            for hook_handle in self._expert_grad_hooks:
                hook_handle.remove()
        
        self._expert_grad_hooks = []
        
        # Register hooks for expert weights and biases
        hook = self.experts.weight.register_hook(create_expert_grad_hook(frozen_mask))
        self._expert_grad_hooks.append(hook)
        
        if self.experts.bias is not None:
            # Bias has shape (num_experts, output_size)
            def create_bias_grad_hook(mask):
                def hook(grad):
                    expanded_mask = mask.view(-1, 1).expand_as(grad)
                    return grad * (~expanded_mask).float()
                return hook
            hook = self.experts.bias.register_hook(create_bias_grad_hook(frozen_mask))
            self._expert_grad_hooks.append(hook)
        
        hook = self.output_experts.weight.register_hook(create_expert_grad_hook(frozen_mask))
        self._expert_grad_hooks.append(hook)
        
        if self.output_experts.bias is not None:
            def create_bias_grad_hook(mask):
                def hook(grad):
                    expanded_mask = mask.view(-1, 1).expand_as(grad)
                    return grad * (~expanded_mask).float()
                return hook
            hook = self.output_experts.bias.register_hook(create_bias_grad_hook(frozen_mask))
            self._expert_grad_hooks.append(hook)
        
        # Count frozen vs trainable experts
        num_frozen = frozen_mask.sum().item()
        num_trainable = (~frozen_mask).sum().item()
        print(f"[ContinualTaskMoE] Frozen {num_frozen} experts, {num_trainable} trainable")
    
    def freeze_existing_routers(self, new_task_id=None):
        """
        Freeze all existing router parameters (before the new task).
        
        Args:
            new_task_id: ID of the new task being learned
        """
        if new_task_id is None:
            new_task_id = self.num_tasks_learned
        
        print(f"[ContinualTaskMoE] Freezing routers for tasks 0 to {new_task_id-1}")
        
        # Freeze all routers except the new one
        for task_idx in range(new_task_id):
            router = self.f_gate[task_idx]
            for param in router.parameters():
                param.requires_grad = False
            self.frozen_router_masks[task_idx].fill_(True)  # Mark as frozen
            print(f"[ContinualTaskMoE] Frozen router {task_idx}")
        
        # Ensure new router is trainable
        new_router = self.f_gate[new_task_id]
        for param in new_router.parameters():
            param.requires_grad = True
        self.frozen_router_masks[new_task_id].fill_(False)  # Mark as not frozen
        print(f"[ContinualTaskMoE] Router {new_task_id} is trainable")
    
    def get_active_parameter_count(self):
        """
        Count the number of active (trainable) parameters.
        
        FIXED: Now uses frozen_expert_masks instead of requires_grad to accurately
        count only the truly trainable parameters, not just those with gradients enabled.
        
        Returns:
            int: Total number of parameters that are actually trainable (not frozen)
        
        This is used to compute the Active Parameters (AP) metric in Table 3.
        Only counts parameters that are currently trainable (i.e., not frozen via gradient hooks).
        """
        # FIXED: Count expert parameters using frozen masks, not requires_grad
        expert_param_count = 0
        
        # Check if we have frozen masks to use for accurate counting
        if hasattr(self, 'frozen_expert_masks') and len(self.frozen_expert_masks) > 0:
            # Use the most recent frozen mask to determine which experts are active
            frozen_mask = self.frozen_expert_masks[-1]
            num_active_experts = (~frozen_mask).sum().item()
            
            # Count only active expert parameters
            expert_weight_params_per_expert = self.experts.weight.shape[1] * self.experts.weight.shape[2]
            expert_param_count += num_active_experts * expert_weight_params_per_expert
            
            if self.experts.bias is not None:
                expert_bias_params_per_expert = self.experts.bias.shape[1]
                expert_param_count += num_active_experts * expert_bias_params_per_expert
            
            output_expert_weight_params_per_expert = self.output_experts.weight.shape[1] * self.output_experts.weight.shape[2]
            expert_param_count += num_active_experts * output_expert_weight_params_per_expert
            
            if self.output_experts.bias is not None:
                output_expert_bias_params_per_expert = self.output_experts.bias.shape[1]
                expert_param_count += num_active_experts * output_expert_bias_params_per_expert
                
        else:
            # Fallback: use the original (incorrect) method if no frozen masks exist
            if self.experts.weight.requires_grad:
                expert_param_count += self.experts.weight.numel()
            
            if self.experts.bias is not None and self.experts.bias.requires_grad:
                expert_param_count += self.experts.bias.numel()
            
            if self.output_experts.weight.requires_grad:
                expert_param_count += self.output_experts.weight.numel()
            
            if self.output_experts.bias is not None and self.output_experts.bias.requires_grad:
                expert_param_count += self.output_experts.bias.numel()
        
        # Count trainable router parameters (router freezing works correctly with requires_grad)
        router_param_count = 0
        for task_idx, router in enumerate(self.f_gate):
            if task_idx < self.num_tasks_learned:
                # Only count router parameters for learned tasks
                for param in router.parameters():
                    if param.requires_grad:
                        router_param_count += param.numel()
        
        total_active_params = expert_param_count + router_param_count
        
        # Also count other trainable parameters (activation, gating, etc.)
        # These are typically small compared to expert weights
        other_param_count = 0
        for name, param in self.named_parameters():
            if 'experts' not in name and 'f_gate' not in name and param.requires_grad:
                other_param_count += param.numel()
        
        total_active_params += other_param_count
        
        print(f"[ContinualTaskMoE] FIXED - Active parameter count: {total_active_params:,} (experts: {expert_param_count:,}, routers: {router_param_count:,}, other: {other_param_count:,})")
        
        return total_active_params
    
    def get_expert_assignment(self, task_id):
        """
        Get the expert indices assigned to a specific task.
        
        Args:
            task_id: ID of the task (0, 1, 2, ...)
        
        Returns:
            list: List of expert indices for this task
        """
        if task_id < len(self.task_expert_assignment):
            return self.task_expert_assignment[task_id]
        else:
            print(f"[ContinualTaskMoE] Warning: Task {task_id} not found in assignment")
            return []
    
    def get_total_experts(self):
        """Get total number of experts (frozen + trainable)."""
        return self.num_experts
    
    def get_num_tasks_learned(self):
        """Get number of tasks learned so far."""
        return self.num_tasks_learned
    
    def increment_tasks_learned(self):
        """Increment the counter of learned tasks."""
        self.num_tasks_learned += 1
        print(f"[ContinualTaskMoE] Tasks learned: {self.num_tasks_learned}")


# Test function to verify ContinualTaskMoE works correctly
if __name__ == '__main__':
    # Test basic initialization
    print("Testing ContinualTaskMoE...")
    
    model = ContinualTaskMoE(
        input_size=256,
        head_size=1024,  # n_emb * 4 = 256 * 4
        num_experts=8,  # Initial experts for first task
        k=2,  # Activate 2 experts per task
        task_num=9,  # Total tasks in dataset (for original TaskMoE compatibility)
        num_tasks_for_continual=3,  # We'll learn 3 tasks
        num_experts_per_task=8,  # Add 8 experts per task
        experts_per_task_active=2,  # Use 2 experts per task
        w_MI=0.0005,
        w_H=0.0005,
        gating_activation=nn.GELU(),
        noisy_gating=False
    )
    
    print(f"\nInitial state:")
    print(f"  Total experts: {model.get_total_experts()}")
    print(f"  Tasks learned: {model.get_num_tasks_learned()}")
    print(f"  Active params: {model.get_active_parameter_count()}")
    print(f"  Task 0 experts: {model.get_expert_assignment(0)}")
    
    print(f"\nAdding task 1 (Lift)...")
    model.add_experts(num_new_experts=8, task_id=1)
    model.add_router(task_id=1)
    model.freeze_existing_experts(new_task_id=1)
    model.freeze_existing_routers(new_task_id=1)
    model.increment_tasks_learned()
    
    print(f"\nAfter adding task 1:")
    print(f"  Total experts: {model.get_total_experts()}")
    print(f"  Tasks learned: {model.get_num_tasks_learned()}")
    print(f"  Active params: {model.get_active_parameter_count()}")
    print(f"  Task 0 experts: {model.get_expert_assignment(0)}")
    print(f"  Task 1 experts: {model.get_expert_assignment(1)}")
    
    print(f"\nAdding task 2 (Square)...")
    model.add_experts(num_new_experts=8, task_id=2)
    model.add_router(task_id=2)
    model.freeze_existing_experts(new_task_id=2)
    model.freeze_existing_routers(new_task_id=2)
    model.increment_tasks_learned()
    
    print(f"\nAfter adding task 2:")
    print(f"  Total experts: {model.get_total_experts()}")
    print(f"  Tasks learned: {model.get_num_tasks_learned()}")
    print(f"  Active params: {model.get_active_parameter_count()}")
    print(f"  Task 0 experts: {model.get_expert_assignment(0)}")
    print(f"  Task 1 experts: {model.get_expert_assignment(1)}")
    print(f"  Task 2 experts: {model.get_expert_assignment(2)}")
    
    print("\nTest completed successfully!")
