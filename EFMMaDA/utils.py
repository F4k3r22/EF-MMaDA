from omegaconf import OmegaConf
import torch
from typing import Any, Optional, Union
from omegaconf import DictConfig
from diffusers.utils.hub_utils import PushToHubMixin

CONFIG_NAME = "config.json"

def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        torch.distributed.broadcast(tensor, src=src)
        return tensor

def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    # added by Xavier -- delete '--local-rank' in multi-nodes training, don't know why there is such a keyword
    if '--local-rank' in cfg:
        del cfg['--local-rank']
    # added by Xavier -- delete '--local-rank' in multi-nodes training, don't know why there is such a keyword
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg

class ModelMixin(torch.nn.Module, PushToHubMixin):
    r"""
    Base class for all models.

    [`ModelMixin`] takes care of storing the model configuration and provides methods for loading, downloading and
    saving models.

        - **config_name** ([`str`]) -- Filename to save a model to when calling [`~models.ModelMixin.save_pretrained`].
    """

    config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    _supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = None
    _no_split_modules = None

    def __init__(self):
        super().__init__()