import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import sys
sys.path.append('.')
import logging
import json
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d %H:%M",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")
import transformers
transformers.utils.logging.set_verbosity("ERROR")
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
from transformers import (
    AutoConfig,
    HfArgumentParser,
    RobertaTokenizer,
    set_seed,
)
import datasets
datasets.utils.logging.set_verbosity("ERROR")
from datasets import load_dataset

from core.args import TrainerArguments
from robertagen.prepare_dataset import DecoderTokenizer, get_dataset, get_train_dataloader, get_eval_dataloader, get_eval_dataset
from robertagen.model import RoBERTaGen, RoBERTaGenConfig
from robertagen.trainer import Trainer


def main(hf_config, model_name, eval_dataset_name):

    args_dict_hf = hf_config.to_diff_dict()
    args_dict_hf['dataset_name'] = config.d

    parser = HfArgumentParser(TrainerArguments)
    trainer_args = parser.parse_dict(args=args_dict_hf)[0]

    """Set seed before initializing model"""
    set_seed(trainer_args.seed)

    """Load dataset"""
    eval_dataset = get_eval_dataset(trainer_args, eval_dataset_name)

    """Build encoder/decoder tokenizer"""
    enc_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dec_tokenizer = DecoderTokenizer(trainer_args)
    
    """Build model"""
    model_args = RoBERTaGen.parse_model_args(args_dict_hf)
    model_args.decode_nwords = dec_tokenizer.nwords
    model = RoBERTaGen(model_args)

    """Build trainer"""
    trainer = Trainer(
        trainer_args=trainer_args, 
        model=model,
        train_dataset=None, 
        eval_dataset=eval_dataset, 
        enc_tokenizer=enc_tokenizer, 
        dec_tokenizer=dec_tokenizer,
        logger=logger,
        )
    trainer.load_model_from_hf(hf_config, model_name)
    print('\nEvaluating...')
    res = trainer.evaluate()
    acc = res['accuracy']
    acc = f'{acc*100:.1f}%'
    print(f'\nAccuracy: {acc}\n')


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(description='YAML configuration file')
    config_parser.add_argument('-m', default=None, type=str, help='model name')
    config_parser.add_argument('-d', default=None, type=str, help='dataset')
    config = config_parser.parse_args()
    
    hf_config = RoBERTaGenConfig.from_pretrained(config.m)
    main(hf_config, config.m, config.d)


