# reproduce_SDP: Skill Diffusion Policy for Multi-Task Robotic Manipulation

A research implementation of **Skill Diffusion Policy (SDP)** that combines diffusion models with Mixture of Experts (MoE) for efficient multi-task robotic manipulation and continual learning. This work presents a novel architecture that enables skill specialization and transfer across diverse robotic manipulation tasks while maintaining performance through continual learning.

## 🎯 Research Contributions

This work introduces three key innovations for multi-task robotics:

### 1. **Diffusion-MoE Integration**
- Novel combination of diffusion-based action generation with mixture of experts architecture
- Enables task-specialized skill representation while maintaining diffusion policy benefits
- Supports complex action sequences through conditional diffusion models

### 2. **Efficient Continual Learning**
- Router-only fine-tuning that freezes expert weights (99.5% parameters frozen, 0.5% trainable)
- Prevents catastrophic forgetting while enabling rapid adaptation to new tasks
- Maintains expert specialization across sequential task learning

### 3. **Multi-Task Skill Transfer**
- Expert specialization learns distinct manipulation skills across diverse tasks
- Natural skill transfer through expert routing mechanisms
- Comprehensive analysis of expert activation patterns and skill composition

## 🏗️ Technical Architecture

### Core Components
- **DiffusionTransformerHybrid Policy**: Transformer-based diffusion model for action generation
- **Mixture of Experts**: Task-specific expert networks with learned routing
- **Hybrid Observation Encoder**: Multi-modal observation processing (RGB images + proprioception)
- **Router Network**: Task-conditioned expert selection and weighting

### Architecture Details
```
Observations → Encoder → Router + Experts → Diffusion Model → Actions
     ↓              ↓           ↓               ↓             ↓
Multi-modal    Feature     Expert           Conditional    Action
Sensors       Embeddings  Specialization   Denoising      Sequences
```

### Key Implementation Features
- **Hybrid Diffusion Policy**: Combines transformer architecture with diffusion sampling
- **Expert Routing**: Learned gating mechanism for expert selection
- **Continual Learning**: Frozen expert weights with router adaptation
- **Multi-Task Training**: Simultaneous training across 8+ manipulation tasks

## 🤖 Supported Manipulation Tasks

The system supports diverse robotic manipulation tasks of varying complexity:

### **Core Tasks**
1. **Square Manipulation** (`square_d0`) - Basic shape manipulation
2. **Object Stacking** (`stack_d0`) - Stacking objects
3. **Multi-Object Stacking** (`stack_three_d0`) - Complex stacking scenarios
4. **Coffee Preparation** (`coffee_d0`) - Sequential manipulation tasks
5. **Coffee Preparation Extended** (`coffee_preparation_d0`) - Complex coffee workflows
6. **Mug Cleanup** (`mug_cleanup_d0`) - Tool-assisted cleaning tasks
7. **Nut Assembly** (`nut_assembly_d0`) - Precision assembly tasks
8. **Threading Operations** (`threading_d0`) - Fine motor manipulation
9. **Hammer Cleanup** (`hammer_cleanup_d0`) - Tool manipulation tasks
10. **Kitchen Tasks** (`kitchen_d0`) - Multi-step household tasks

### **Task Complexity Categories**
- **Basic**: Shape manipulation, simple stacking
- **Intermediate**: Tool use, multi-object coordination
- **Advanced**: Sequential operations, precision assembly
- **Complex**: Multi-step workflows, tool switching

## 🛠️ Installation

### System Requirements
- **OS**: Linux (Ubuntu 18.04+)
- **GPU**: NVIDIA GPU with CUDA 11.6 support
- **Memory**: 16GB+ RAM (32GB+ recommended for multi-task training)
- **Storage**: 50GB+ for datasets and models

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd reproduce_SDP

# Create conda environment
conda env create -f conda_environment.yaml
conda activate sdp

# Install additional dependencies
pip install -r requirements.txt

# Set up environment variables
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
```

### Dataset Requirements
Download Robomimic datasets for supported tasks:
```bash
# Expected directory structure
datasets/
├── core/
│   ├── square_d0.hdf5
│   ├── stack_d0.hdf5
│   ├── coffee_d0.hdf5
│   └── ... (other task datasets)
└── processed/ (optional preprocessed data)
```

## 🚀 Quick Start

### Multi-Task Training
```bash
# Train on all 8 tasks simultaneously
python train.py --config-name=train_diffusion_transformer_hybrid_workspace
```

### Single Task Training
```bash
# Train on individual task
python train.py --config-name=train_diffusion_transformer_hybrid_workspace \
    task_num=1 task0.dataset.dataset_path=datasets/core/square_d0.hdf5
```

### Evaluation
```bash
# Evaluate trained model
python eval.py --checkpoint /path/to/checkpoint.ckpt -o /path/to/output
```

## 📊 Research Usage Examples

### 1. Multi-Task Training Experiments
```yaml
# Configuration for 8-task training
task_num: 8
n_tasks: 8
# Each task configured in task0, task1, ..., task7
```

### 2. Continual Learning Scenarios
```bash
# Sequential task learning with router-only fine-tuning
python train.py --config-name=continual_learning_config \
    training.resume=true training.freeze_experts=true
```

### 3. Expert Analysis
```python
# Analyze expert activation patterns
python analyze_expert_activations.py \
    --checkpoint model.ckpt \
    --output_dir analysis_results/
```

### 4. Skill Transfer Experiments
```bash
# Test skill transfer between related tasks
python transfer_experiment.py \
    --source_task coffee_d0 \
    --target_task coffee_preparation_d0
```

## ⚙️ Configuration System

### YAML Configuration Structure
The system uses Hydra for flexible configuration management:

```yaml
# Core configuration sections
policy:                 # Diffusion-MoE architecture
training:              # Training parameters
task0, task1, ...:     # Individual task configurations
dataset:              # Dataset loading and preprocessing
logging:              # Experiment tracking
```

### Key Research Parameters

#### Expert Configuration
```yaml
policy:
  n_tasks: 8                    # Number of experts
  n_emb: 512                   # Embedding dimension
  n_layer: 12                  # Transformer layers
  n_head: 4                    # Attention heads
```

#### Diffusion Parameters
```yaml
policy:
  noise_scheduler:
    num_train_timesteps: 100   # Diffusion steps
    beta_schedule: "squaredcos_cap_v2"
    prediction_type: "epsilon"
```

#### Training Parameters
```yaml
training:
  max_epochs: 300              # Training duration
  learning_rate: 0.0001        # Optimizer learning rate
  use_ema: true               # Exponential moving average
```

### Task Configuration Pattern
Each task follows a consistent structure:
```yaml
taskX:
  name: "task_name"
  dataset_path: "path/to/dataset.hdf5"
  shape_meta:                 # Observation/action dimensions
  env_runner:                 # Evaluation settings
```

## 📈 Analysis & Visualization Tools

### Expert Analysis Tools
- **Expert Activation Visualization**: Analyze which experts activate for specific task phases
- **Skill Composition Analysis**: Understand expert combinations for complex behaviors
- **Routing Dynamics**: Study how routing patterns evolve during training

### Performance Monitoring
- **Multi-Task Metrics**: Track performance across all tasks simultaneously
- **Continual Learning Analysis**: Monitor forgetting and transfer effects
- **Expert Specialization**: Measure expert specialization over time

### Visualization Scripts
```bash
# Expert activation heatmap
python visualize_expert_activations.py --checkpoint model.ckpt

# Skill transfer visualization
python visualize_skill_transfer.py --tasks task1 task2

# Training dynamics
python plot_training_curves.py --log_dir logs/
```

## 📁 File Structure for Research

```
reproduce_SDP/
├── config/                    # Configuration files
│   ├── tasks/                # Individual task configs
│   └── tmp/                  # Training configurations
├── diffusion_policy/          # Core policy implementation
│   ├── policy/              # Policy architectures
│   ├── model/               # Model components
│   └── dataset/             # Dataset loaders
├── mixture_of_experts/       # MoE implementations
├── scripts/                  # Analysis and utility scripts
├── datasets/                 # Dataset storage
├── data/                     # Training outputs and checkpoints
├── train.py                  # Multi-task training script
├── eval.py                   # Evaluation script
└── conda_environment.yaml     # Environment specification
```

### Key Research Components
- **`diffusion_policy/policy/`**: Core diffusion-MoE policy implementations
- **`mixture_of_experts/`**: Expert architectures and routing mechanisms
- **`config/tasks/`**: Task-specific configurations for experiments
- **`scripts/`**: Analysis tools for expert behavior and performance

## 🔬 Research Methodology Notes

### Reproducibility Guidelines
1. **Seed Management**: All experiments use fixed seeds (default: 42)
2. **Configuration Tracking**: Complete experiment configurations logged
3. **Checkpoint Management**: Regular model and optimizer state saving
4. **Evaluation Protocol**: Standardized evaluation across all tasks

### Experimental Setup
- **Training Split**: 98% training, 2% validation (per task)
- **Evaluation Episodes**: 50 test episodes per task
- **Observation Processing**: RGB (84×84) + proprioception
- **Action Space**: 7-DoF robot commands + gripper

### Implementation Details
- **Diffusion Sampling**: 100 timesteps with DDPM scheduler
- **Expert Routing**: Learned gating with softmax activation
- **Continual Learning**: Router-only fine-tuning with frozen experts
- **Multi-Task Balancing**: Equal data sampling across tasks

### Hyperparameter Configuration
```python
# Key hyperparameters for reproduction
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EXPERT_EMBEDDING_DIM = 512
DIFFUSION_TIMESTEPS = 100
ROUTER_TEMPERATURE = 1.0
```

## 🐛 Troubleshooting for Researchers

### Common Experimental Issues

#### Memory Optimization
```bash
# Reduce batch size for memory constraints
python train.py dataloader.batch_size=32

# Enable gradient accumulation
python train.py training.gradient_accumulate_every=2
```

#### Multi-Task Training Issues
- **Dataset Loading**: Ensure all task datasets are accessible
- **Memory Requirements**: Monitor GPU memory during multi-task training
- **Expert Balancing**: Check expert utilization across tasks

#### Continual Learning Problems
```bash
# Verify expert freezing
python check_expert_frozen.py --checkpoint model.ckpt

# Monitor router adaptation
python analyze_router_weights.py --log_dir logs/
```

#### Debugging Expert Routing
```python
# Expert activation diagnostics
python debug_expert_routing.py \
    --checkpoint model.ckpt \
    --task coffee_d0 \
    --num_episodes 10
```

### Performance Optimization
- **Data Loading**: Use persistent workers and caching
- **Mixed Precision**: Enable automatic mixed precision training
- **Gradient Checkpointing**: Reduce memory for large models

## 📄 Citation & Acknowledgments

If you use this code or find this research implementation helpful, please cite the relevant papers and acknowledge the dependencies:

### Key Dependencies
- **Robomimic**: Imitation learning framework and datasets
- **MimicGen**: Dataset generation for manipulation tasks
- **Robosuite**: Simulation environments for manipulation
- **Diffusers**: Diffusion model implementations
- **Hydra**: Configuration management

### Related Work
- Diffusion Policy for robotic manipulation
- Mixture of Experts for multi-task learning
- Continual learning in robotics
- Skill discovery and transfer

### License
This implementation is provided for research purposes. Please refer to individual dependency licenses for usage terms.

## 🤝 Contributing

We welcome research contributions and improvements. Key areas of interest:
- New expert architectures and routing mechanisms
- Additional manipulation tasks and benchmarks
- Advanced continual learning algorithms
- Improved analysis and visualization tools

For research collaboration or questions about the implementation, please open an issue or contact the maintainers.

---

**Note**: This research implementation focuses on methodology and architectural contributions rather than specific performance benchmarks. The codebase is designed to enable reproducible research into diffusion-MoE integration and continual learning for robotic manipulation.