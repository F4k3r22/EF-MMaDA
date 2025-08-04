from dataclasses import dataclass, field
import numpy as np 
import torch
import torch.nn as nn
from commons import *
from utils import ModelMixin
import math
from typing import List
from diffusers.configuration_utils import ConfigMixin

class Updateable:
    def do_update_step(
            self, epoch: int, global_step: int, on_load_weights: bool = False
    ):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step(
                    epoch, global_step, on_load_weights=on_load_weights
                )
        self.update_step(epoch, global_step, on_load_weights=on_load_weights)

    def do_update_step_end(self, epoch: int, global_step: int):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step_end(epoch, global_step)
        self.update_step_end(epoch, global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass

    def update_step_end(self, epoch: int, global_step: int):
        pass

class VQGANEncoder(ModelMixin, ConfigMixin):
    @dataclass
    class Config:
        ch: int = 128
        ch_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 4, 4])
        num_res_blocks: List[int] = field(default_factory=lambda: [4, 3, 4, 3, 4])
        attn_resolutions: List[int] = field(default_factory=lambda: [5])
        dropout: float = 0.0
        in_ch: int = 3
        out_ch: int = 3
        resolution: int = 256
        z_channels: int = 13
        double_z: bool = False