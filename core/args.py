import argparse
import ruamel.yaml as yaml
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser


@dataclass
class TrainerArguments:
    """
    Trainer related arguments, e.g., learning rate, batch size
    """
    exp_group:       str = field(default=None)
    run_name:        str = field(default=None)
    dataset_name:    str = field(default=None)

    max_input_len:   int = field(default=96)
    max_output_len:  int = field(default=32)

    epochs:          int = field(default=1)
    lr:            float = field(default=5e-5)
    train_bz:        int = field(default=8)
    eval_bz:         int = field(default=8)
    fp16:           bool = field(default=False)
    device_name:     str = field(default='cuda:0')
    seed:            int = field(default=42)
    train_steps:     int = field(default=6000)
    
    use_scheduler:  bool = field(default=False)
    scheduler_type:  str = field(default=None)
    warmup_steps:    int = field(default=500)
    decay_ratio:   float = field(default=0.)

    eval_steps:      int = field(default=200)
    patience:        int = field(default=200)
    max_grad_norm: float = field(default=1.0)

    use_actual_num: bool = field(default=True)
    output_as_code: bool = field(default=True)
    add_replacement: bool = field(default=True)

    model_init_pth: float = field(default='random')
    remove_weights_after_train: bool = field(default=False)
    save_at_last_step: bool = field(default=False)

    
    def __post_init__(self):
        self.device = torch.device(self.device_name)


def get_data_trainer_args():
    """
    This function obtains data and trainer arguments, 
        also returns whole args_dict for later model args
    """
    config_parser = argparse.ArgumentParser(description='YAML configuration file')
    config_parser.add_argument('-c', type=str, required=True)
    config = config_parser.parse_args()
    parser = HfArgumentParser(TrainerArguments)
    with open(config.c, 'r', encoding='utf-8') as f:
        args_dict = yaml.load(f.read(), Loader=yaml.Loader)
    trainer_args = parser.parse_dict(args=args_dict)[0]
    return trainer_args, args_dict


def get_args_in_dict():
    config_parser = argparse.ArgumentParser(description='YAML configuration file')
    config_parser.add_argument('-c', type=str, required=True)
    config = config_parser.parse_args()
    with open(config.c, 'r', encoding='utf-8') as f:
        args_dict = yaml.load(f.read(), Loader=yaml.Loader)
    return args_dict
