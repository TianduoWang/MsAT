import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import Sampler

from transformers import BertTokenizer, BertTokenizerFast
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data import default_data_collator
from transformers import RobertaModel, RobertaTokenizer
from datasets import load_dataset

from core.utils import from_prefix_to_infix, from_prefix_to_code


class DecoderTokenizer:
    def __init__(self, args):
        if args.use_actual_num:
            self.w2id = {
                '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, 
                '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '.': 10,
                '<s>':  11, '</s>': 12, 
                '+'  :  13, '-'   : 14, '*': 15, '/': 16, '=': 17, 'unk': 18,
                'NUM0': 19, 'NUM1': 20, 'NUM2': 21, 'NUM3': 22, 'NUM4': 23,
                'NUM5': 24, 'NUM6': 25, 'NUM7': 26, 'NUM8': 27, 'NUM9': 28, 
                'm_1': 29, 'm_2': 30, 'm_3': 31, 'm_4': 32, 'm_5': 33, 'm_6': 34,
                'm_7': 35, 'm_8': 36, 'm_9': 37, 'RES_': 38, '<SEP>':39,
                '0.01': 40, '0.05': 41, '0.1': 42, '0.25': 43, '0.5': 44, 
                '10': 45, '12': 46, '25': 47, '60': 48, '100': 49}

            self.id2w = {
                0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
                5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '.',
                11: '<s>', 12: '</s>', 
                13: '+', 14: '-', 15: '*', 16: '/', 17: '=', 18: 'unk',
                19: 'NUM0', 20: 'NUM1', 21: 'NUM2', 22: 'NUM3', 23: 'NUM4', 
                24: 'NUM5', 25: 'NUM6', 26: 'NUM7', 27: 'NUM8', 28: 'NUM9', 
                29: 'm_1', 30: 'm_2', 31: 'm_3', 32: 'm_4', 33: 'm_5', 
                34: 'm_6', 35: 'm_7', 36: 'm_8', 37: 'm_9', 38: 'RES_', 39: '<SEP>',
                40: '0.01', 41: '0.05', 42: '0.1', 43: '0.25', 44: '0.5', 
                45: '10', 46: '12', 47: '25', 48: '60', 49: '100'}

            self.nwords = 50

        else:
            self.w2id = {
                '<s>': 11, '</s>': 1, 
                '+'  :  2, '-'   : 3, '*': 4, '/': 5, 
                'number0': 6, 'number1': 7, 'number2': 8, 'number3': 9, 'number4': 10, 'number5': 0, 
                'unk': 12}
            self.id2w = {
                11: '<s>', 1: '</s>', 2: 
                '+', 3: '-', 4: '*', 5: '/', 
                6: 'number0', 7: 'number1', 8: 'number2', 9: 'number3', 10: 'number4', 0: 'number5', 
                12: 'unk'}
            self.w2c = {
                '+': 0, '-': 0, '*': 0, '/': 0, 
                'number0': 0, 'number1': 0, 'number2': 0, 'number3': 0, 'number4': 0, 'number5': 0, 
                'unk': 0}
            self.nwords = 13

    def get_id(self, word):
        if re.match(r"\d+\.?\d*", word) and word.endswith(".0"):
            word = word[:-2]
        if word in self.w2id:
            return self.w2id[word]
        else:
            return self.w2id['unk']

    def get_word(self, idx):
        return self.id2w[idx]


def get_dataset(args):
    data_files = {"train": "train.csv", "eval": "dev.csv"}

    dataset_path = os.path.join("data/", args.dataset_name)
    dataset = load_dataset(dataset_path, data_files=data_files, cache_dir="./data/")
    if "asdiv" in args.dataset_name:
        dataset = dataset.remove_columns(['group_nums', 'Grade', 'Type', 'Body', 'Ques_Statement'])

    def preprocess_func(example):

        def process_num(num_str):
            """ 
            Transform number to the most suitable precision and split by digit
            """
            num_float = float(num_str)
            if abs(int(num_float) - num_float) < 1e-4:
                num_str = str(int(num_float))
            elif abs(round(num_float, 1)- num_float) < 1e-4:
                num_str = f'{num_float:.1f}'
            elif abs(round(num_float, 2) - num_float) < 1e-4:
                num_str = f'{num_float:.2f}'
            else:
                num_str = f'{num_float:.3f}'

            digit_ls = list()
            for digit in num_str:
                digit_ls.append(digit)
            out_str = ' '.join(digit_ls)
            out_str = '#' + out_str + '#'
            return out_str

        """ process question"""
        if args.use_actual_num:
            ques, num_list = example['Question'], example['Numbers'].split()
            for i, num_str in enumerate(num_list):
                ques = re.sub(f'number{i}', process_num(num_str), ques)
            example['Question'] = ques

        """ process equation"""
        eqn_tokens = example['Equation'].split()
        if args.output_as_code:
            example['Equation'] = ' '.join(from_prefix_to_code(eqn_tokens))
        else:
            example['Equation'] = ' '.join(from_prefix_to_infix(eqn_tokens))
        if args.use_actual_num:
            equ = example['Equation']
            for i, num_str in enumerate(num_list):
                equ = re.sub(f'number{i}', process_num(num_str), equ)
                equ = re.sub(r'#', '', equ)
            
            if args.output_as_code:
                executable_code = re.sub('<SEP>', '\n', ''.join(equ.split()))
                exec(executable_code)
                exec_res = float(locals()['RES_'])
                assert abs(exec_res - float(example['Answer'])) < 0.01
            else:
                assert abs(eval(''.join(equ.split())) - float(example['Answer'])) < 0.01
            
            example['Equation'] = equ
        return example

    dataset = dataset.map(preprocess_func, remove_columns='Answer')
    
    return dataset['train'], dataset['eval']


def get_eval_dataset(args, dataset_name):
    data_files = {"eval": "dev.csv"}

    dataset_path = os.path.join("data/", dataset_name)
    dataset = load_dataset(dataset_path, data_files=data_files, cache_dir="./data/")
    if "asdiv" in args.dataset_name:
        dataset = dataset.remove_columns(['group_nums', 'Grade', 'Type', 'Body', 'Ques_Statement'])

    def preprocess_func(example):

        def process_num(num_str):
            """ 
            Transform number to the most suitable precision and split by digit
            """
            num_float = float(num_str)
            if abs(int(num_float) - num_float) < 1e-4:
                num_str = str(int(num_float))
            elif abs(round(num_float, 1)- num_float) < 1e-4:
                num_str = f'{num_float:.1f}'
            elif abs(round(num_float, 2) - num_float) < 1e-4:
                num_str = f'{num_float:.2f}'
            else:
                num_str = f'{num_float:.3f}'

            digit_ls = list()
            for digit in num_str:
                digit_ls.append(digit)
            out_str = ' '.join(digit_ls)
            out_str = '#' + out_str + '#'
            return out_str

        """ process question"""
        if args.use_actual_num:
            ques, num_list = example['Question'], example['Numbers'].split()
            for i, num_str in enumerate(num_list):
                ques = re.sub(f'number{i}', process_num(num_str), ques)
            example['Question'] = ques

        """ process equation"""
        eqn_tokens = example['Equation'].split()
        if args.output_as_code:
            example['Equation'] = ' '.join(from_prefix_to_code(eqn_tokens))
        else:
            example['Equation'] = ' '.join(from_prefix_to_infix(eqn_tokens))
        if args.use_actual_num:
            equ = example['Equation']
            for i, num_str in enumerate(num_list):
                equ = re.sub(f'number{i}', process_num(num_str), equ)
                equ = re.sub(r'#', '', equ)
            
            if args.output_as_code:
                executable_code = re.sub('<SEP>', '\n', ''.join(equ.split()))
                exec(executable_code)
                exec_res = float(locals()['RES_'])
                assert abs(exec_res - float(example['Answer'])) < 0.01
            else:
                assert abs(eval(''.join(equ.split())) - float(example['Answer'])) < 0.01
            
            example['Equation'] = equ
        return example

    dataset = dataset.map(preprocess_func, remove_columns='Answer')
    
    return dataset['eval']


@dataclass
class DataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_mtp (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
    """

    tokenizers: Tuple[PreTrainedTokenizerBase, DecoderTokenizer]
    padding: Union[bool, str, PaddingStrategy] = True
    max_in_len: Optional[int] = None
    max_out_len: Optional[int] = None
    pad_mtp: Optional[int] = 8

    def __call__(self, features):
        """ tokenize Questions """
        questions = [f['Question'] for f in features]
        
        all_tokens  = [['<s>'] + self.tokenizers[0].tokenize(sent) + ['</s>'] for sent in questions]

        scale_ids = None

        """ check max input length and pad """
        input_lengths = [len(tokens) for tokens in all_tokens]
        max_length    = max(input_lengths)
        if self.pad_mtp and max_length % self.pad_mtp > 0:
            max_length += self.pad_mtp - max_length % self.pad_mtp
            assert max_length % self.pad_mtp == 0
        assert max_length <= self.max_in_len
        padded_tokens = [tokens + ['<pad>' for _ in range(max_length - len(tokens))] for tokens in all_tokens]
        token_ids = torch.tensor([self.tokenizers[0].convert_tokens_to_ids(tokens) for tokens in padded_tokens])

        
        """ attention mask """
        pad_token = self.tokenizers[0].convert_tokens_to_ids('<pad>')
        attn_masks = (token_ids != pad_token).long()

        """ pad Numbers with zeros"""
        batch_nums_str = [f['Numbers'] for f in features]

        """ pad Equations"""
        eqns = [f['Equation'] for f in features]

        encoded_eqns = []

        max_out_len = 0
        for eq in eqns:
            encoded_eq = [self.tokenizers[1].get_id('<s>')]
            for token in eq.split():
                encoded_eq.append(self.tokenizers[1].get_id(token))
            encoded_eq.append(self.tokenizers[1].get_id('</s>'))
            max_out_len = len(encoded_eq) if len(encoded_eq) > max_out_len else max_out_len
            encoded_eqns.append(encoded_eq)

        try:
            assert max_out_len <= self.max_out_len
        except:
            for i in eqns:
                print(i)
            exit('I dont know why I am here.')

        end_idx = self.tokenizers[1].get_id('</s>')
        pad_eqns = [eq + [end_idx for _ in range(max_out_len - len(eq))] for eq in encoded_eqns]
        pad_eqns = torch.LongTensor(pad_eqns)

        return {'input_ids': token_ids, 'scale_ids': scale_ids, 'attention_mask': attn_masks, 'output_seq': pad_eqns, 
        'numbers': batch_nums_str, 'eqns': eqns, 'current_bz': str(len(features)), 'target_len': max_out_len}


class DataSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_samples (int): number of training steps
    """

    def __init__(self, dataset, num_samples=None):


        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(num_samples))
        self.dataset = dataset
        self.epoch = 0
        self.total_size = num_samples
        self.num_samples = num_samples

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        n = len(self.dataset)
        n_repeats = self.total_size // n
        n_remain = self.total_size % n

        indices = [torch.randperm(n, generator=g) for _ in range(n_repeats)]
        indices.append(torch.randperm(n, generator=g)[:n_remain])
        indices = torch.cat(indices, dim=0).tolist()

        assert len(indices) == self.total_size
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_train_dataloader(args, dataset, tokenizers):
    num_samples = int(args.train_steps) * args.train_bz
    return DataLoader(
        dataset, 
        batch_size=args.train_bz, 
        collate_fn=DataCollator(tokenizers, 
            max_in_len=args.max_input_len,
            max_out_len=args.max_output_len),
        num_workers=0, 
        drop_last=False, 
        pin_memory=True,
        sampler=DataSampler(dataset, num_samples=num_samples),
        )


def get_eval_dataloader(args, dataset, tokenizers):
    return DataLoader(
        dataset, 
        batch_size=args.eval_bz, 
        shuffle=False, 
        collate_fn=DataCollator(tokenizers, 
            max_in_len=args.max_input_len,
            max_out_len=args.max_output_len), 
        num_workers=4,
        )

