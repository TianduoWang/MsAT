import argparse
import os
import sys
import math
import logging
import pdb
import random

from copy import deepcopy
from collections import OrderedDict
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, HoulsbyConfig, PretrainedConfig, PreTrainedModel
from transformers.adapters import RobertaAdapterModel
from transformers.optimization import get_scheduler



def get_combination_mask(batched_num_variables: torch.Tensor, combination: torch.Tensor):
    """

    :param batched_num_variables: (batch_size)
    :param combination: (num_combinations, 2) 6,2
    :return: batched_comb_mask: (batch_size, num_combinations)
    """
    batch_size, = batched_num_variables.size()
    num_combinations, _ = combination.size()
    batched_num_variables = batched_num_variables.unsqueeze(1).unsqueeze(2).expand(batch_size, num_combinations, 2)
    batched_combination = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
    batched_comb_mask = torch.lt(batched_combination, batched_num_variables)

    return batched_comb_mask[:,:, 0] * batched_comb_mask[:,:, 1]


class DeductReasoner(nn.Module):

    @dataclass
    class ModelArguments:
        d_model: int = field(default=768)
        dropout: float = field(default=0.1)
        num_const:  int = field(default=17)
        lr: float = field(default=1e-5)
        add_replacement: bool = field(default=False)

        """ adapter arguments """
        use_adapter:    bool = field(default=False)
        adapter_tuning: bool = field(default=False)
        bn_dim:          int = field(default=64)

    @staticmethod
    def parse_model_args(args_dict):
        parser = HfArgumentParser(DeductReasoner.ModelArguments)
        model_args = parser.parse_dict(args=args_dict)[0]
        return model_args
    
    def __init__(self, config):
        super(DeductReasoner, self).__init__()
        self.config = config
        
        """ encoder modules"""
        self.roberta = RobertaAdapterModel.from_pretrained('roberta-base')
        if self.config.use_adapter:
            task_name = 'math'
            redu_factor = self.config.d_model // self.config.bn_dim
            adapter_config = HoulsbyConfig(reduction_factor=redu_factor)
            self.roberta.add_adapter(task_name, config=adapter_config)
            
            if self.config.adapter_tuning:
                self.roberta.train_adapter('math')
            else:
                self.roberta.active_adapters = "math"

        """ decoder modules"""
        self.num_labels = 6
        self.label_rep2label = nn.Linear(self.config.d_model, 1)
        
        self.linears = nn.ModuleList()
        for i in range(6):
            self.linears.append(nn.Sequential(
                nn.Linear(3 * config.d_model, config.d_model),
                nn.ReLU(),
                nn.LayerNorm(config.d_model, eps=1e-12),
                nn.Dropout(0.1)
            ))

        self.stopper_transformation = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.LayerNorm(config.d_model, eps=1e-12),
            nn.Dropout(0.1)
            )
        self.stopper = nn.Linear(self.config.d_model, 2)
        self.variable_gru = nn.GRUCell(self.config.d_model, self.config.d_model)
        self.const_rep = nn.Parameter(torch.randn(self.config.num_const, self.config.d_model))
        self.constant_num = self.config.num_const
        self.variable_scorer = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.config.d_model, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(self.config.d_model, 1)
            )
        self.add_replacement = self.config.add_replacement
        self.consider_multiple_m0 = True
        self._initialize_optimizer()

    def _initialize_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.config.lr)
    
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        variable_indexs_start: torch.Tensor = None,
        variable_indexs_end: torch.Tensor = None,
        num_variables: torch.Tensor = None,
        variable_index_mask:torch.Tensor = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_height_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_eval=False,
        ):
        return_dict = True
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        batch_size, sent_len, hidden_size = outputs.last_hidden_state.size()
        if labels is not None and not is_eval:
            _, max_height, _ = labels.size()
        else:
            max_height = 8

        _, max_num_variable = variable_indexs_start.size()

        var_sum = (variable_indexs_start - variable_indexs_end).sum()
        var_start_hidden_states = torch.gather(outputs.last_hidden_state, 
            1, variable_indexs_start.unsqueeze(-1).expand(batch_size, max_num_variable, hidden_size))
        if var_sum != 0:
            var_end_hidden_states = torch.gather(outputs.last_hidden_state, 
                1, variable_indexs_end.unsqueeze(-1).expand(batch_size, max_num_variable, hidden_size))
            var_hidden_states = var_start_hidden_states + var_end_hidden_states
        else:
            var_hidden_states = var_start_hidden_states
        if self.constant_num > 0:
            constant_hidden_states = self.const_rep.unsqueeze(0).expand(batch_size, self.constant_num, hidden_size)
            var_hidden_states = torch.cat([constant_hidden_states, var_hidden_states], dim=1)
            num_variables = num_variables + self.constant_num
            max_num_variable = max_num_variable + self.constant_num
            const_idx_mask = torch.ones((batch_size, self.constant_num), device=variable_indexs_start.device)
            variable_index_mask = torch.cat([const_idx_mask, variable_index_mask], dim = 1)

        best_mi_label_rep = None
        loss = 0
        all_logits = []
        best_mi_scores = None
        for i in range(max_height):
            linear_modules = self.linears
            if i == 0:
                num_var_range = torch.arange(0, max_num_variable, device=variable_indexs_start.device)
                combination = torch.combinations(num_var_range, r=2, with_replacement=self.add_replacement)
                num_combinations, _ = combination.size()
                batched_combination_mask = get_combination_mask(batched_num_variables=num_variables,
                                                                combination=combination)
                var_comb_hidden_states = torch.gather(var_hidden_states, 1, 
                    combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_combinations * 2, hidden_size)
                    )

                expanded_var_comb_hidden_states = \
                    var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size)

                m0_hidden_states = torch.cat(
                    [expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :],
                     expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]], dim=-1)

                m0_label_rep = torch.stack([layer(m0_hidden_states) for layer in linear_modules], dim=2)
                m0_logits = self.label_rep2label(m0_label_rep).expand(batch_size, num_combinations, self.num_labels, 2)
                m0_logits = m0_logits + batched_combination_mask\
                    .unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2).log()
                m0_stopper_logits = self.stopper(self.stopper_transformation(m0_label_rep))

                var_scores = self.variable_scorer(var_hidden_states).squeeze(-1)
                expanded_var_scores = torch.gather(
                    var_scores, 1, combination.unsqueeze(0).expand(batch_size, num_combinations, 2).view(batch_size, -1))\
                    .unsqueeze(-1).view(batch_size, num_combinations, 2)
                expanded_var_scores = expanded_var_scores.sum(dim=-1)\
                    .unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2)
                m0_combined_logits = m0_logits + m0_stopper_logits + expanded_var_scores

                all_logits.append(m0_combined_logits)
                best_temp_logits, best_stop_label = m0_combined_logits.max(dim=-1)
                best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)
                best_m0_score, best_comb = best_temp_score.max(dim=-1)
                best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)

                b_idxs = [k for k in range(batch_size)]
                if labels is not None and not is_eval:
                    m0_gold_labels = labels[:, i, :]
                    m0_gold_comb = m0_gold_labels[:, :2].unsqueeze(1).expand(batch_size, num_combinations, 2)
                    batched_comb = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
                    judge = m0_gold_comb == batched_comb
                    judge = judge[:, :, 0] * judge[:, :, 1]
                    judge = judge.nonzero()[:, 1]

                    m0_gold_scores = m0_combined_logits[
                        b_idxs, judge, m0_gold_labels[:, 2], m0_gold_labels[:, 3]]
                    loss = loss + (best_m0_score - m0_gold_scores).sum()

                    best_mi_label_rep = m0_label_rep[b_idxs, judge, m0_gold_labels[:, 2]]
                    best_mi_scores = m0_logits[b_idxs, judge, m0_gold_labels[:, 2]][:, 0]
                else:
                    best_m0_label_rep = m0_label_rep[b_idxs, best_comb, best_label]
                    best_mi_label_rep = best_m0_label_rep
                    best_mi_scores = m0_logits[b_idxs, best_comb, best_label][:, 0]
            else:
                init_h = best_mi_label_rep.unsqueeze(1).expand(batch_size, max_num_variable + i - 1,
                                                               hidden_size).contiguous().view(-1, hidden_size)
                gru_inputs = var_hidden_states.view(-1, hidden_size)
                var_hidden_states = self.variable_gru(gru_inputs, init_h).view(batch_size,
                                                                               max_num_variable + i - 1,
                                                                               hidden_size)

                num_var_range = torch.arange(0, max_num_variable + i, device=variable_indexs_start.device)
                combination = torch.combinations(num_var_range, r=2, with_replacement=self.add_replacement)
                num_combinations, _ = combination.size()
                batched_combination_mask = get_combination_mask(batched_num_variables=num_variables+i,
                                                                combination=combination)

                var_hidden_states = torch.cat([best_mi_label_rep.unsqueeze(1), var_hidden_states], dim=1)

                var_comb_hidden_states = torch.gather(var_hidden_states, 1,
                    combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_combinations * 2, hidden_size))
                
                expanded_var_comb_hidden_states = \
                    var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size)

                mi_hidden_states = torch.cat(
                    [expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :],
                     expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]],
                    dim=-1)
                mi_label_rep = torch.stack([layer(mi_hidden_states) for layer in linear_modules], dim=2)
                mi_logits = self.label_rep2label(mi_label_rep).expand(batch_size, num_combinations, self.num_labels,
                                                                      2)
                mi_logits = mi_logits + batched_combination_mask.unsqueeze(-1)\
                    .unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2).log()

                mi_stopper_logits = self.stopper(self.stopper_transformation(mi_label_rep))
                var_scores = self.variable_scorer(var_hidden_states).squeeze(-1)
                expanded_var_scores = torch.gather(var_scores, 1, combination.unsqueeze(0)\
                    .expand(batch_size, num_combinations, 2)\
                    .view(batch_size,-1))\
                    .unsqueeze(-1)\
                    .view(batch_size, num_combinations, 2)
                
                expanded_var_scores = expanded_var_scores.sum(dim=-1)\
                    .unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2)

                mi_combined_logits = mi_logits + mi_stopper_logits + expanded_var_scores
                all_logits.append(mi_combined_logits)
                best_temp_logits, best_stop_label = mi_combined_logits.max(dim=-1)
                best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)
                best_mi_score, best_comb = best_temp_score.max(dim=-1)
                best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)

                if labels is not None and not is_eval:
                    mi_gold_labels = labels[:, i, :]
                    mi_gold_comb = mi_gold_labels[:, :2].unsqueeze(1).expand(batch_size, num_combinations, 2)
                    batched_comb = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
                    judge = mi_gold_comb == batched_comb
                    judge = judge[:, :, 0] * judge[:, :, 1]
                    judge = judge.nonzero()[:, 1]

                    mi_gold_scores = mi_combined_logits[
                        b_idxs, judge, mi_gold_labels[:, 2], mi_gold_labels[:, 3]]
                    height_mask = label_height_mask[:, i]
                    current_loss = (best_mi_score - mi_gold_scores) * height_mask
                    loss = loss + current_loss.sum()
                    best_mi_label_rep = mi_label_rep[b_idxs, judge, mi_gold_labels[:, 2]]
                    best_mi_scores = mi_logits[b_idxs, judge, mi_gold_labels[:, 2]][:, 0]
                else:
                    best_mi_label_rep = mi_label_rep[b_idxs, best_comb, best_label]
                    best_mi_scores = mi_logits[b_idxs, best_comb, best_label][:, 0]

        return loss, all_logits
 
    def training_step(self, 
        input_ids:      torch.Tensor, # size: batch_size x input_len
        attention_mask: torch.Tensor, # size: batch_size x input_len
        output_seq:     torch.Tensor, # size: batch_size x out_len
        var_starts:     torch.Tensor, # size: batch_size x num_vars
        var_ends:       torch.Tensor, # size: batch_size x num_vars
        var_mask:       torch.Tensor, # size: batch_size x num_vars
        num_vars:       torch.Tensor, # size: batch_size x 1
        output_mask:    torch.Tensor, # size: batch_size x out_len
        ):

        loss, _ = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            variable_indexs_start = var_starts,
            variable_indexs_end = var_ends,
            num_variables = num_vars,
            variable_index_mask = var_mask,
            head_mask=None,
            inputs_embeds=None,
            labels=output_seq,
            label_height_mask = output_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            is_eval=False)
        return loss

    def greedy_decode(self, 
        input_ids:      torch.Tensor, # size: batch_size x input_len
        attention_mask: torch.Tensor, # size: batch_size x input_len
        output_seq:     torch.Tensor, # size: batch_size x out_len
        var_starts:     torch.Tensor, # size: batch_size x num_vars
        var_ends:       torch.Tensor, # size: batch_size x num_vars
        var_mask:       torch.Tensor, # size: batch_size x num_vars
        num_vars:       torch.Tensor, # size: batch_size x 1
        output_mask:    torch.Tensor, # size: batch_size x out_len
        ):
        with torch.no_grad():
            _, all_logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            variable_indexs_start = var_starts,
            variable_indexs_end = var_ends,
            num_variables = num_vars,
            variable_index_mask = var_mask,
            head_mask=None,
            inputs_embeds=None,
            labels=output_seq,
            label_height_mask = output_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            is_eval=True)

        cpu_logits = []
        for item in all_logits:
            cpu_logits.append(item.detach().cpu())
        return cpu_logits


class DeductReasonerConfig(PretrainedConfig):

    def __init__(self, 
        d_model:          int = 768,
        dropout:        float = 0.1,
        num_const:        int = 7,
        lr:             float = 1e-5,
        add_replacement: bool = False,
        use_adapter:     bool = False,
        adapter_tuning:  bool = False,
        bn_dim:           int = 64,
        **kwargs):
        self.d_model = d_model
        self.dropout = dropout
        self.num_const = num_const
        self.lr = lr
        self.add_replacement = add_replacement
        self.use_adapter = use_adapter
        self.adapter_tuning = adapter_tuning
        self.bn_dim = bn_dim
        super().__init__(**kwargs)


class HFDeductReasoner(PreTrainedModel):
    def __init__(self, config, pytorch_model):
        super().__init__(config)
        self.model = pytorch_model

""" End here """
