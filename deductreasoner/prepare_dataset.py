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
from datasets import load_dataset
from transformers import RobertaModel, RobertaTokenizer

from core.utils import from_prefix_to_infix, from_prefix_to_code, from_prefix_to_deductive


class DecoderTokenizer:
    def __init__(self, args):
        self.const_list = ['1.0', '0.1',  '3.0',  '5.0',  '0.5', 
                     '12.0', '4.0', '60.0', '25.0', '0.01', 
                     '0.05', '2.0', '10.0', '0.25',  '8.0', '7.0', '100.0']
        self.const_list_float = [1.0, 0.1, 3.0, 5.0, 0.5, 12.0, 4.0, 60.0, 25.0, 0.01, 
                     0.05, 2.0, 10.0, 0.25,  8.0, 7.0, 100.0]
        self.w2id = {c: idx for idx, c in enumerate(self.const_list)}
        self.op2id = {'+':  0, '-': 1, '-_rev': 2, '*': 3, '/': 4, '/_rev': 5}
        self.nwords = len(self.const_list)


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
        else:
            ques, num_list = example['Question'], example['Numbers'].split()
            for i, num_str in enumerate(num_list):
                ques = re.sub(f'number{i}', f'#number{i}#', ques)
            example['Question'] = ques

        """ process equation"""
        eqn_tokens = example['Equation'].split()
        example['Equation'] = ' '.join(from_prefix_to_deductive(eqn_tokens))

        if not args.add_replacement and re.search(r'(number\d)\s\1', example['Equation']):
           example['Equation'] = ''
           return example

        executable_code = list()
        for idx, num_str in enumerate(example['Numbers'].split()):
            executable_code.extend([f'number{idx}', '=', num_str, '\n'])

        eqn = example['Equation'].split()
        step_cnt = 1
        for j in range(0, len(eqn), 3):
            operator = eqn[j+2]
            if operator.endswith('_rev'):
                l, r = eqn[j+1], eqn[j]
                operator = operator[0]
            else:
                l, r = eqn[j], eqn[j+1]
            executable_code.extend([f'm_{step_cnt}', '=', l, operator, r, '\n'])
            step_cnt += 1
        executable_code[-6] = 'RES_'
        executable_code = ''.join(executable_code)
        exec(executable_code)
        exec_res = float(locals()['RES_'])
        assert abs(exec_res - float(example['Answer'])) < 0.01
        return example

    dataset = dataset.map(preprocess_func, remove_columns='Answer')
    dataset = dataset.filter(lambda x: len(x['Equation']) > 1)
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
        example['Equation'] = ' '.join(from_prefix_to_deductive(eqn_tokens))

        if re.search(r'(number\d)\s\1', example['Equation']):
            example['Equation'] = ''
            return example

        executable_code = list()
        for idx, num_str in enumerate(example['Numbers'].split()):
            executable_code.extend([f'number{idx}', '=', num_str, '\n'])

        eqn = example['Equation'].split()
        step_cnt = 1
        for j in range(0, len(eqn), 3):
            operator = eqn[j+2]
            if operator.endswith('_rev'):
                l, r = eqn[j+1], eqn[j]
                operator = operator[0]
            else:
                l, r = eqn[j], eqn[j+1]
            executable_code.extend([f'm_{step_cnt}', '=', l, operator, r, '\n'])
            step_cnt += 1
        executable_code[-6] = 'RES_'
        executable_code = ''.join(executable_code)
        exec(executable_code)
        exec_res = float(locals()['RES_'])

        assert abs(exec_res - float(example['Answer'])) < 0.01
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

    def get_label_ids_incremental(self, equation):
        """
            Args:
                equation (str): code-style equation in string
            Returns:
                output sequence
        """
        label_ids = []
        num_constant = self.tokenizers[1].nwords
        
        eqn_tokens = equation.split()
        assert len(eqn_tokens) % 3 == 0
        for idx in range(0, len(eqn_tokens), 3):
            left_var, right_var, operator = eqn_tokens[idx: idx+3]
            step_cnt = idx // 3
            assert idx % 3 == 0
            is_stop = 1 if idx+3==len(eqn_tokens) else 0

            if left_var in self.tokenizers[1].w2id:
                left_var_idx = self.tokenizers[1].w2id[left_var] + step_cnt
            elif left_var+'.0' in self.tokenizers[1].w2id:
                left_var_idx = self.tokenizers[1].w2id[left_var+'.0'] + step_cnt
            elif left_var.startswith('number'):
                left_var_idx = (int(left_var[6:]) + num_constant + step_cnt)
            else:
                assert left_var.startswith('m_')
                m_idx = int(left_var[2:])
                left_var_idx = step_cnt - m_idx

            if right_var in self.tokenizers[1].w2id:
                right_var_idx = self.tokenizers[1].w2id[right_var] + step_cnt
            elif right_var+'.0' in self.tokenizers[1].w2id:
                right_var_idx = self.tokenizers[1].w2id[right_var+'.0'] + step_cnt
            elif right_var.startswith('number'):
                right_var_idx = (int(right_var[6:]) + num_constant + step_cnt)
            else:
                assert right_var.startswith('m_')
                m_idx = int(right_var[2:])
                right_var_idx = step_cnt - m_idx
            assert left_var_idx >= 0 and right_var_idx >= 0

            if left_var_idx < right_var_idx:
                op_idx = self.tokenizers[1].op2id[operator]
                label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
            else:
                try:
                    assert not operator.endswith('_rev')
                except:
                    print(equation)
                    exit()
                # assert left_var_idx != right_var_idx
                if operator in ['-', '/']:
                    op_idx = self.tokenizers[1].op2id[operator+'_rev']
                else:
                    op_idx = self.tokenizers[1].op2id[operator]
                label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])

            assert label_ids[-1][0] <= label_ids[-1][1]

        return label_ids

    def __call__(self, features):
        """ tokenize Questions """
        questions = [f['Question'] for f in features]
        all_tokens  = [['<s>'] + self.tokenizers[0].tokenize(sent) + ['</s>'] for sent in questions]

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

        """ prepare Numbers"""
        batch_nums_str = [f['Numbers'] for f in features]

        """ variable start positions """
        var_starts_batch, var_mask_batch = [], []
        var_ends_batch = []
        for tokens in token_ids:
            var_starts, var_ends = [], []
            is_in_number = False
            for pos_id, t_id in enumerate(tokens):
                
                if t_id == self.tokenizers[0].convert_tokens_to_ids('Ġ#')\
                or (pos_id==1 and t_id == self.tokenizers[0].convert_tokens_to_ids('#')):
                    var_starts.append(pos_id)
                    is_in_number = True
                elif t_id == self.tokenizers[0].convert_tokens_to_ids('#'):
                    assert is_in_number
                    var_ends.append(pos_id)
                    is_in_number = False

            var_starts_batch.append(var_starts)
            var_ends_batch.append(var_ends)
            var_mask_batch.append([1]*len(var_starts))

        num_vars_batch, max_num_var = [], 0
        for batch_id, batch in enumerate(var_starts_batch):
            assert len(batch) == len(batch_nums_str[batch_id].split())
            num_vars_batch.append(len(batch))
            if len(batch) > max_num_var:
                max_num_var = len(batch)

        for batch_id, (var_starts, var_ends, var_mask) in enumerate(zip(var_starts_batch, var_ends_batch, var_mask_batch)):
            var_starts.extend([0] * (max_num_var - num_vars_batch[batch_id]))
            var_ends.extend([0] * (max_num_var - num_vars_batch[batch_id]))
            var_mask.extend([0] * (max_num_var - num_vars_batch[batch_id]))

        var_starts_batch = torch.tensor(var_starts_batch)
        var_ends_batch = torch.tensor(var_ends_batch)
        num_vars_batch = torch.tensor(num_vars_batch)
        var_mask = torch.tensor(var_mask_batch)

        """ pad Equations"""
        eqns = [f['Equation'] for f in features]
        
        encoded_eqns, eqn_masks, max_out_len = [], [], 0
        for eq in eqns:
            encoded_eq = self.get_label_ids_incremental(eq)
            max_out_len = len(encoded_eq) if len(encoded_eq) > max_out_len else max_out_len
            encoded_eqns.append(encoded_eq)
            eqn_masks.append([1] * len(encoded_eq))
        assert max_out_len <= self.max_out_len

        pad_eqns = []

        for eq in encoded_eqns:
            padded_eq = deepcopy(eq)
            if max_out_len > len(eq):
                for _ in range(max_out_len-len(eq)):
                    padded_eq.append([0,1,0,0])
            pad_eqns.append(padded_eq)

        try:
            pad_eqns = torch.LongTensor(pad_eqns)
        except:
            for eq in pad_eqns:
                print(eq)
            exit()

        pad_eqn_masks = [mask + [0 for _ in range(max_out_len - len(mask))] for mask in eqn_masks]
        pad_eqn_masks = torch.LongTensor(pad_eqn_masks)

        out = {
        'input_ids': token_ids, 'attention_mask': attn_masks, 
        'var_starts': var_starts_batch, 'var_ends': var_ends_batch, 
        'var_mask': var_mask, 'num_vars': num_vars_batch,
        'output_seq': pad_eqns, 'output_mask': pad_eqn_masks,
        'numbers': batch_nums_str, 'eqns': eqns, 'current_bz': str(len(features)), 'target_len': max_out_len
        }

        return out


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
        num_workers=4, 
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

