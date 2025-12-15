#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation adapted and modified from SDCodec: https://github.com/XiaoyuBIE1994/SDCodec

"""

import os
import sys
import shutil
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

import torch

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:

    from src.trainer_codecsep_dnr import Trainer
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()