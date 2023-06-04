import os
import sys
sys.path.append('.')
import argparse
import json
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d %H:%M",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")
import ruamel.yaml as yaml
import transformers
transformers.utils.logging.set_verbosity("WARNING")
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
from transformers import (
    AutoConfig,
    RobertaTokenizer,
    set_seed,
    HfArgumentParser
)
import datasets
datasets.utils.logging.set_verbosity("WARNING")

from core.args import TrainerArguments
from deductreasoner.prepare_dataset import DecoderTokenizer, get_dataset, get_train_dataloader
from deductreasoner.model import DeductReasoner
from deductreasoner.trainer import Trainer


def main(args_dict):
    parser = HfArgumentParser(TrainerArguments)
    trainer_args = parser.parse_dict(args=args_dict)[0]

    """Log on device information"""
    logger.info(f"Device: {trainer_args.device}, 16-bits training: {trainer_args.fp16}")
    logger.info(f"Trainer parameters {trainer_args}")

    """Set seed before initializing model"""
    set_seed(trainer_args.seed)

    """Load dataset"""
    train_dataset, eval_dataset = get_dataset(trainer_args)

    """Build encoder/decoder tokenizer"""
    enc_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dec_tokenizer = DecoderTokenizer(trainer_args)
    logger.info(f"Decoder vocab: \n\t{dec_tokenizer.w2id.keys()}")

    """Build model"""
    rb_config = AutoConfig.from_pretrained("roberta-base")
    model_args = DeductReasoner.parse_model_args(args_dict)
    model_args.num_const = dec_tokenizer.nwords
    model = DeductReasoner(model_args)

    """Build trainer"""
    trainer = Trainer(
        trainer_args=trainer_args, 
        model=model,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        enc_tokenizer=enc_tokenizer, 
        dec_tokenizer=dec_tokenizer,
        logger=logger,
        )

    trainer.train()

    """Record results"""
    res_folder = os.path.join("results", trainer_args.exp_group, trainer_args.run_name)
    os.makedirs(res_folder, exist_ok=True)
    res_json_file = os.path.join(res_folder, "res.json")
    with open(res_json_file, "w") as f:
        json.dump(trainer.result_dict, f)

    """Record hyperparameters"""
    args_json_file = os.path.join(res_folder, "args.json")
    with open(args_json_file, "w") as f:
        json.dump(args_dict, f)



if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(description='YAML configuration file')
    config_parser.add_argument('-c', type=str, required=True)
    config = config_parser.parse_args()
    with open(config.c, 'r', encoding='utf-8') as f:
        args_dict = yaml.load(f.read(), Loader=yaml.Loader)
    main(args_dict)