from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer

class BaseImagePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml
    """Base class for image-based policies. to accommodate different action/image shapes,
    all shape-related info are passed in shape_meta dict.
    shape_meta example:
        shape_meta = {
            'obs_image_shape': (C,H,W),  # image observation shape
            'action_image_shape': (C,H,W),  # image action shape
            'action_dim': Da,  # action vector dimension
            ...
        }
    """

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()