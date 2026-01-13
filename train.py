"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
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

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'config', 'tmp')),
    config_name="full.yaml",
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    from utils.recursive_yaml import read_yaml, write_yaml
    data = read_yaml('config/base.yaml')
    write_yaml(data, 'config/tmp/full.yaml')
    main()
