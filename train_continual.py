"""
Continual learning training scripts

Implement 3-step continual learning pipeline  based on the paper:
    stage 1: Train on can 
    stage 2: add 2 expert and a router for lift, freeze can
    stage 3: add 2 expert and a router for square, freeze can & lift
Each stage train for 500 epochs.
"""
import sys                                                                                                                                                                               
import os                                                                                                                                                                                
os.environ["CUDA_VISIBLE_DEVICES"]='0'                                                                                                                                                   
os.environ["MUJOCO_GL"]="osmesa"                                                                                                                                                         
os.environ["PYOPENGL_PLATFORM"]="osmesa"                                                                                                                                                 
sys.path.insert(0, '/home/cc/reproduce_SDP/mimicgen_environments')                                                                                                                       
import mimicgen.envs    
import hydra
# Register !include constructor with OmegaConf's YAML loader                                                                                                                                                                             
import yaml                                                                                                                                                                                                                              
from yamlinclude import YamlIncludeConstructor                                                                                                                                                                                           
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.SafeLoader)
# Import standard robosuite environments for Can and Lift                                                                                                                                
from robosuite.environments.manipulation.pick_place import PickPlaceCan                                                                                                                  
from robosuite.environments.manipulation.lift import Lift                                                                                                                                
                                                                                                                                                                                         
# Import custom mimicgen environment for Square                                                                                                                                          
from mimicgen.envs.robosuite.nut_assembly import NutAssemblySquare  # or Square_D0

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.train_continual_workspace import TrainContinualWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('config')),
    config_name="base_continual",
)

def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = TrainContinualWorkspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()

