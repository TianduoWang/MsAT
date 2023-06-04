from dataclasses import dataclass, field, asdict
from copy import deepcopy
import argparse
import os
import sys
import math
import logging
import pdb
import random
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, HfArgumentParser, PreTrainedModel, HoulsbyConfig, PretrainedConfig
from transformers.adapters import RobertaAdapterModel
from transformers.optimization import get_scheduler

from collections import OrderedDict


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1)*0.5)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
            Args:
                x (tensor): embeddings | size : [max_len x batch_size x d_model]
            Returns:
                z (tensor) : embeddings with positional encoding | size : [max_len x batch_size x d_model]
        '''
        
        x = x + self.scale * self.pe[:x.size(0), :]
        z = self.dropout(x)
        return z


class RoBERTaGen(nn.Module):

    @dataclass
    class ModelArguments:
        d_model: int = field(default=768)
        dropout: float = field(default=0.1)
        lr: float = field(default=1e-5)

        """ decoder-specific args"""
        decoder_layers: int = field(default=2)
        decoder_heads:  int = field(default=8)
        decoder_dim_ff: int = field(default=1024)
        decode_nwords:  int = field(default=50)
        label_smooth: float = field(default=0.05)

        """ adapter arguments """
        use_adapter:    bool = field(default=False)
        adapter_tuning: bool = field(default=False)
        bn_dim:          int = field(default=64)

    @staticmethod
    def parse_model_args(args_dict):
        parser = HfArgumentParser(RoBERTaGen.ModelArguments)
        model_args = parser.parse_dict(args=args_dict)[0]
        return model_args
    
    def __init__(self, config):
        super(RoBERTaGen, self).__init__()
        self.config = config
        
        """ encoder modules"""
        self.roberta = RobertaAdapterModel.from_pretrained('roberta-base')
        if self.config.use_adapter:
            task_name = 'math'
            redu_factor = self.config.d_model // self.config.bn_dim
            adapter_config = HoulsbyConfig(reduction_factor=redu_factor)
            self.roberta.add_adapter(task_name, config=adapter_config)
            
            #---------------------------------------
            if self.config.adapter_tuning:
                self.roberta.train_adapter('math')
            else:
                self.roberta.active_adapters = "math"
            #---------------------------------------

        """ decoder modules"""
        self.embedding2  = nn.Embedding(self.config.decode_nwords, self.config.d_model)
        nn.init.uniform_(self.embedding2.weight, -0.02, 0.02)
        
        self.pos_embedding2 = PositionalEncoding(self.config.d_model, self.config.dropout)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.config.d_model, 
                nhead=self.config.decoder_heads,
                dim_feedforward=self.config.decoder_dim_ff, 
                dropout=self.config.dropout),
            num_layers=self.config.decoder_layers,
            )
        self.fc_out = nn.Linear(self.config.d_model, self.config.decode_nwords)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smooth)

        self._initialize_optimizer()

    def _initialize_optimizer(self):

        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optim.AdamW(grouped_parameters, lr=self.config.lr)

    def training_step(self, 
        input_ids:      torch.Tensor, # size: batch_size x input_len
        attention_mask: torch.Tensor, # size: batch_size x input_len
        output_seq:     torch.Tensor, # size: batch_size x (out_len - 1)
        tgt_mask:       torch.Tensor, # size: (out_len - 1) x (out_len - 1)
        tgt_pad_mask:   torch.Tensor, # size: (out_len - 1) x batch_size
        mem_pad_mask:   torch.Tensor, # size: (out_len - 1) x batch_size
        ):

        src = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        src = src.transpose(0, 1)

        output_seq = output_seq.transpose(0, 1)
        tgt = output_seq[:-1, :]
        tgt = self.embedding2(tgt)
        tgt = self.pos_embedding2(tgt)

        output = self.transformer_decoder(
            tgt=tgt, memory=src,
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_pad_mask, 
            memory_key_padding_mask=mem_pad_mask)
        
        output = self.fc_out(output)
        output_dim = output.shape[-1]
        loss = self.criterion(output.reshape(-1, output_dim), output_seq[1:,:].reshape(-1))
        return loss

    def greedy_decode(self, input_ids, attention_mask, decode_starter, target_len, mem_pad_mask):
        with torch.no_grad():

            src = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            memory = src.transpose(0,1)
            
            input_list = deepcopy(decode_starter)
            decoded_words = [[] for i in range(input_ids.size(0))]

            for step in range(target_len):
                decoder_input = torch.LongTensor(input_list).to(input_ids.device)
                tgt_emb = self.pos_embedding2(self.embedding2(decoder_input))
                d_out = self.transformer_decoder(tgt_emb, memory, memory_key_padding_mask=mem_pad_mask)
                decoder_output = self.fc_out(d_out)
                out_tokens = decoder_output.argmax(2)[-1,:]
                input_list.append(out_tokens.detach().tolist())
            return input_list


class RoBERTaGenConfig(PretrainedConfig):

    def __init__(self, 
        d_model:         int = 768,
        dropout:       float = 0.1,
        lr:            float = 1e-5,
        decoder_layers:  int = 2,
        decoder_heads:   int = 8,
        decoder_dim_ff:  int = 1024,
        decode_nwords:   int = 17,
        label_smooth:  float = 0.05,
        use_adapter:    bool = True,
        adapter_tuning: bool = False,
        bn_dim:          int = 64,
        **kwargs):

        self.d_model = d_model
        self.dropout = dropout
        self.lr = lr
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.decoder_dim_ff = decoder_dim_ff
        self.decode_nwords = decode_nwords
        self.label_smooth = label_smooth
        self.use_adapter = use_adapter
        self.adapter_tuning = adapter_tuning
        self.bn_dim = bn_dim
        self.max_input_len = 128
        self.max_output_len = 64
        self.eval_bz = 100
        self.output_as_code = True
        super().__init__(**kwargs)


class HFRoBERTaGen(PreTrainedModel):
    def __init__(self, config, pytorch_model):
        super().__init__(config)
        self.model = pytorch_model

""" End here """
