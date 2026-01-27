from typing import Union, Optional, Tuple, Callable
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import torch.nn.functional as F
from torch import nn, Tensor
import copy
from mixture_of_experts.moe import MoE
from mixture_of_experts.task_moe import TaskMoE
from mixture_of_experts.continual_task_moe import ContinualTaskMoE  # MODIFIED: Import ContinualTaskMoE for continual learning
logger = logging.getLogger(__name__)

class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers) 
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt: Tensor, task_id, memory: Tensor, tgt_mask:Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        output = tgt
        loss = 0.0
        for mod in self.layers:
            output, aux_loss, probs = mod(output, task_id, memory, tgt_mask=tgt_mask,
                                          memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
            loss += aux_loss
        if self.norm is not None:
            output = self.norm(output)

        return output, loss, probs
    
class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    
    MODIFIED for Continual Learning: Added support for ContinualTaskMoE to enable
    dynamic expert expansion and freezing for continual learning experiments.
    
    Args:
        d_model: the number of expected features in the decoder (required).
        n_tasks: the number of tasks in the dataset (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).
        use_continual_moe: If True, use ContinualTaskMoE instead of TaskMoE (for continual learning).
        num_experts_per_task: Number of experts to add per task (e.g., 8 for new task).
        experts_per_task_active: Number of experts to activate per task (e.g., 2).
        num_tasks_for_continual: Total number of continual learning stages (e.g., 3 for Can→Lift→Square).
    
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    
    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first', 'use_continual_moe']  # MODIFIED: Added use_continual_moe
    
    def __init__(self, d_model: int, n_tasks: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,norm_first: bool = False,
                 device=None, dtype=None, 
                 use_continual_moe: bool = False,  # MODIFIED: Added for continual learning
                 num_experts_per_task: int = 8,  # MODIFIED: Default 8 experts per task
                 experts_per_task_active: int = 2,  # MODIFIED: Default 2 active experts per task
                 num_tasks_for_continual: int = 3,  # MODIFIED: Default 3 continual learning stages
                 **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                    **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        
        # MODIFIED: Choose between TaskMoE and ContinualTaskMoE based on use_continual_moe flag
        if use_continual_moe:
            # MODIFIED: Use ContinualTaskMoE for continual learning with expert expansion support
            self.task_moe_layer = ContinualTaskMoE(
                d_model, 
                dim_feedforward//2, 
                8,  # num_experts (initial for first task)
                2,  # k (experts to activate)
                bias=True, 
                acc_aux_loss = True, 
                w_MI=0.0005, 
                w_finetune_MI=0,
                task_num=n_tasks,  # Total tasks in dataset
                activation=nn.Sequential(nn.GELU()),
                noisy_gating=False,
                # MODIFIED: Pass continual learning specific parameters
                num_tasks_for_continual=num_tasks_for_continual,
                num_experts_per_task=num_experts_per_task,
                experts_per_task_active=experts_per_task_active
            )
        else:
            # Use original TaskMoE
            self.task_moe_layer = TaskMoE(d_model, dim_feedforward//2, 8, 2, bias=True, 
                                    acc_aux_loss = True, w_MI=0.0005, w_finetune_MI=0,
                                    task_num=n_tasks, activation=nn.Sequential(nn.GELU()),
                                    noisy_gating=False)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # legecalize the activation function
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
    
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)
    
    def forward(self, tgt: Tensor, task_id, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the encoder layer (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            output, aux_loss, probs = self._ff_block(self.norm3(x), task_id)
            x = x + output
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x, task_id))
        
        return x, aux_loss, probs

    # self attention block
    def _sa_block(self, x: Tensor,
                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)
    
    # multi-head attention block
    def _mha_block(self, x:Tensor, mem:Tensor,
                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)
    
    def _ff_block(self, x:Tensor, task_id) -> Tensor:
        x, aux_loss, probs = self.task_moe_layer(x, task_id)
        return self.dropout3(x), aux_loss, probs
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

class TransformerForDiffusion(ModuleAttrMixin):
    # MODIFIED: Added parameters and methods for continual learning support
    def __init__(self, input_dim: int, output_dim: int, horizon: int,
                  n_obs_steps: int = None, cond_dim: int = 0, n_tasks: int = 1,
                  n_layer: int = 12, n_head: int =12, n_emb: int = 768,
                  p_drop_emb: float = 0.1, p_drop_attn: float = 0.1,
                  causal_attn: bool = False, time_as_cond: bool = True, 
                  obs_as_cond: bool = False, n_cond_layers : int =0,
                  use_continual_moe: bool = False,  # MODIFIED: Enable continual learning mode
                  num_experts_per_task: int = 8,  # MODIFIED: Default 8 experts per new task
                  experts_per_task_active: int = 2,  # MODIFIED: Default 2 active experts per task
                  num_tasks_for_continual: int = 3) -> None:  # MODIFIED: Default 3 continual learning stages
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim>0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps
        
        # MODIFIED: Store continual learning parameters for delegation
        self.use_continual_moe = use_continual_moe  # Whether to use ContinualTaskMoE
        self.num_experts_per_task = num_experts_per_task  # Experts to add per task
        self.num_tasks_for_continual = num_tasks_for_continual  # Total continual learning stages
        self.experts_per_task_active = experts_per_task_active  # Active experts per task
        
        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        
        self.cond_pos_emb = None
        
        if T_cond>0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb)) 
        
        # MODIFIED: Check if using continual learning mode
        if self.use_continual_moe:
            print("[TransformerForDiffusion] Continual learning mode enabled")
            print(f"[TransformerForDiffusion] Number of continual stages: {self.num_tasks_for_continual}")
            print(f"[TransformerForDiffusion] Experts per task: {self.num_experts_per_task}, active: {self.experts_per_task_active}")

        self.input_emb = nn.Linear(input_dim, n_emb)

        self.drop = nn.Dropout(p_drop_emb)
        
        self.encoder_only = not time_as_cond
        self.obs_as_cond = obs_as_cond

        if self.encoder_only:
             # Main backbone is encoder
             encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
             self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )
             self.decoder = nn.Identity()
        else:
             # Main backbone is decoder
             # Encoder is for condition
             if n_cond_layers > 0:
                 encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                 self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
             else:
                 self.encoder = nn.Identity()

             decoder_layer = TransformerDecoderLayer(
                d_model=n_emb,
                n_tasks=n_tasks,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                use_continual_moe=use_continual_moe,
                num_experts_per_task=num_experts_per_task,
                experts_per_task_active=experts_per_task_active,
                num_tasks_for_continual=num_tasks_for_continual
            )
             self.decoder = TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )

        if causal_attn:
             mask = nn.Transformer.generate_square_subsequent_mask(T)
             self.register_buffer('mask', mask)
        else:
             self.mask = None
        
        self.memory_mask = None
        
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
    
    def add_experts_for_continual_learning(self, num_new_experts, task_id=None):
        """
        MODIFIED: Delegate expert expansion to ContinualTaskMoE layers.
        
        This method adds new experts to all decoder layers in the transformer.
        Used when learning a new task in continual learning.
        
        Args:
            num_new_experts: Number of new experts to add (typically 8)
            task_id: ID of the new task (e.g., 1 for Lift after Can)
        """
        if not self.use_continual_moe:
            print("[add_experts_for_continual_learning] Warning: Continual learning mode not enabled")
            return
        
        if task_id is None:
            task_id = self.decoder.layers[0].task_moe_layer.get_num_tasks_learned()
        
        print(f"[TransformerForDiffusion] Adding {num_new_experts} experts for task {task_id}")
        for layer in self.decoder.layers:
            if hasattr(layer, 'task_moe_layer') and hasattr(layer.task_moe_layer, 'add_experts'):
                layer.task_moe_layer.add_experts(num_new_experts, task_id=task_id)
    
    def add_router_for_continual_learning(self, task_id=None):
        """
        MODIFIED: Delegate router addition to ContinualTaskMoE layers.
        
        Args:
            task_id: ID of the new task (e.g., 1 for Lift after Can)
        """
        if not self.use_continual_moe:
            print("[add_router_for_continual_learning] Warning: Continual learning mode not enabled")
            return
        
        if task_id is None:
            task_id = self.decoder.layers[0].task_moe_layer.get_num_tasks_learned()
        
        print(f"[TransformerForDiffusion] Adding router for task {task_id}")
        for layer in self.decoder.layers:
            if hasattr(layer, 'task_moe_layer') and hasattr(layer.task_moe_layer, 'add_router'):
                layer.task_moe_layer.add_router(task_id=task_id)
    
    def freeze_existing_experts_for_continual_learning(self, new_task_id=None):
        """
        MODIFIED: Delegate expert freezing to ContinualTaskMoE layers.
        
        Args:
            new_task_id: ID of the new task being learned (e.g., 1 for Lift)
        """
        if not self.use_continual_moe:
            print("[freeze_existing_experts_for_continual_learning] Warning: Continual learning mode not enabled")
            return
        
        if new_task_id is None:
            new_task_id = self.decoder.layers[0].task_moe_layer.get_num_tasks_learned()
        
        print(f"[TransformerForDiffusion] Freezing experts for tasks 0 to {new_task_id-1}")
        for layer in self.decoder.layers:
            if hasattr(layer, 'task_moe_layer') and hasattr(layer.task_moe_layer, 'freeze_existing_experts'):
                layer.task_moe_layer.freeze_existing_experts(new_task_id=new_task_id)
    
    def freeze_existing_routers_for_continual_learning(self, new_task_id=None):
        """
        MODIFIED: Delegate router freezing to ContinualTaskMoE layers.
        
        Args:
            new_task_id: ID of the new task being learned
        """
        if not self.use_continual_moe:
            print("[freeze_existing_routers_for_continual_learning] Warning: Continual learning mode not enabled")
            return
        
        if new_task_id is None:
            new_task_id = self.decoder.layers[0].task_moe_layer.get_num_tasks_learned()
        
        print(f"[TransformerForDiffusion] Freezing routers for tasks 0 to {new_task_id-1}")
        for layer in self.decoder.layers:
            if hasattr(layer, 'task_moe_layer') and hasattr(layer.task_moe_layer, 'freeze_existing_routers'):
                layer.task_moe_layer.freeze_existing_routers(new_task_id=new_task_id)
    
    def get_num_tasks_learned(self):
        """
        Get the number of tasks learned so far in continual learning mode.
        
        Returns:
            int: Number of tasks learned (1 for initial task, 2 after first continual task, etc.)
        """
        if not self.use_continual_moe:
            print("[get_num_tasks_learned] Warning: Continual learning mode not enabled")
            return 1
        
        return self.decoder.layers[0].task_moe_layer.get_num_tasks_learned()
    
    def count_active_parameters_for_continual_learning(self):
        """
        MODIFIED: Count active (trainable) parameters in ContinualTaskMoE mode.
        
        Returns:
            int: Total number of parameters with requires_grad=True
        
        This is used to compute the Active Parameters (AP) metric in Table 3.
        """
        if not self.use_continual_moe:
            print("[count_active_parameters_for_continual_learning] Warning: Continual learning mode not enabled")
            return 0
        
        total_active = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total_active += param.numel()
        
        print(f"[TransformerForDiffusion] Active parameter count: {total_active}")
        return total_active
    
    def increment_tasks_learned_for_continual_learning(self):
        """
        MODIFIED: Increment counter of learned tasks in ContinualTaskMoE mode.
        
        This should be called after successfully learning a new task.
        """
        if not self.use_continual_moe:
            print("[increment_tasks_learned_for_continual_learning] Warning: Continual learning mode not enabled")
            return
        
        for layer in self.decoder.layers:
            if hasattr(layer, 'task_moe_layer') and hasattr(layer.task_moe_layer, 'increment_tasks_learned'):
                layer.task_moe_layer.increment_tasks_learned()
        
        print(f"[TransformerForDiffusion] Tasks learned: {self.decoder.layers[0].task_moe_layer.get_num_tasks_learned() if self.use_continual_moe else 'N/A'}")
    
    def get_expert_assignment_for_continual_learning(self, task_id):
        """
        MODIFIED: Get expert assignment for a specific task in ContinualTaskMoE mode.
        
        Args:
            task_id: ID of the task (0, 1, 2, ...)
        
        Returns:
            list: Expert indices for this task
        """
        if not self.use_continual_moe:
            print("[get_expert_assignment_for_continual_learning] Warning: Continual learning mode not enabled")
            return []
        
        for layer in self.decoder.layers:
            if hasattr(layer, 'task_moe_layer') and hasattr(layer.task_moe_layer, 'get_expert_assignment'):
                return layer.task_moe_layer.get_expert_assignment(task_id)
        
        return []
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # seperate out all parametes to those that will and won't experience weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('experts.weight'):
                    no_decay.add(fpn)
        
        # special case the position embedding parameter in the root module
        no_decay.add('pos_emb')
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add('cond_pos_emb')
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def configure_optimizers(
            self,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.95),
        ) :
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
    
    def forward(
            self,
            sample: Tensor,
            timestep: Union[torch.Tensor, float, int],
            cond: Optional[torch.Tensor] = None,
            task_id: Union[torch.Tensor, int] = None,
            **kwargs,
        ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        task_id: int
        """
        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) ==0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that compatible with ONNX/coreml
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B,1,n_emb)

        input_emb = self.input_emb(sample)  # (B,T,n_emb)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,n_emb)
            x = x[:,1:,:]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                # (B,To,n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)
            
            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x,loss,probs = self.decoder(
                tgt=x,
                task_id=task_id,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask,
            )
            # (B,T,n_emb)
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,input_dim)
        return x, loss, probs
    
def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
    

    # GPT with time embedding and obs cond
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        # causal_attn=True,
        time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
 
        
