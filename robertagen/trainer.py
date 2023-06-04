import logging
import os
import contextlib
import pickle
import numpy as np
from dataclasses import asdict
from argparse import Namespace
from inspect import signature
from collections import OrderedDict
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_ as clip_gn
from transformers import AutoTokenizer, AutoConfig
from transformers.optimization import get_scheduler
import wandb

from core.hooks import Hook, EvaluationHook, CheckpointHook, EarlyStopHook
from core.utils import count_parameters, calculate_acc, calculate_acc_code
from robertagen.prepare_dataset import get_train_dataloader, get_eval_dataloader
from robertagen.model import HFRoBERTaGen, RoBERTaGenConfig


class Trainer:

    def __init__(self, trainer_args, model, 
        data_collator=None, train_dataset=None, eval_dataset=None, enc_tokenizer=None, dec_tokenizer=None,
        logger=None, **kwargs):

        self.args = trainer_args
        self.device = self.args.device
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.enc_tokenizer, self.dec_tokenizer = enc_tokenizer, dec_tokenizer

        self.model = model.to(self.device)
        if self.args.model_init_pth != 'random':
            if self.args.model_init_pth.startswith('checkpoints'):
                self.load_model(load_path=self.args.model_init_pth)
            else:
                hf_config = DeductReasonerConfig.from_pretrained(self.args.model_init_pth)
                self.load_model_from_hf(hf_config, self.args.model_init_pth)

        self.build_dataloader()
        if train_dataset:
            self.init_wandb()
            self.logger = logger
            self.train_steps = self.args.train_steps
            self.eval_steps = self.args.eval_steps
            self.log_steps = self.args.eval_steps
            self.hook_queue = list()
            self.register_hooks()
            self.set_optimizer_scheduler()
            self.should_stop = False
            self.current_step = 0
            self.state_dict = {
                'latest_valid_acc': 0.,
                'best_valid_acc': 0.,
                'current_step': 0,
                'best_step': 0,
            }
            self.result_dict = {
                'final_acc': 0.,
            }
            self.train_loss = None
            self.loss_record, self.grad_norm_record = list(), list()

    def init_wandb(self):
        wandb.init(
            mode="disabled",
            project = "MsAT",
            name = self.args.exp_group+'-'+self.args.run_name,
            tags = [self.args.exp_group],
            config = {
                "train_step": self.args.train_steps,
            })
        wandb.run.summary['best_acc'] = 0.
        wandb.run.summary['best_step'] = 0
        # wandb.define_metric("acc", step_metric="eval_step")

    def build_dataloader(self):
        self.train_loader = get_train_dataloader(
            args=self.args, 
            dataset=self.train_dataset,
            tokenizers=(self.enc_tokenizer, self.dec_tokenizer),
            ) if self.train_dataset else None
        self.eval_loader = get_eval_dataloader(
            self.args, 
            self.eval_dataset,
            tokenizers=(self.enc_tokenizer, self.dec_tokenizer),
            )
        
    def set_optimizer_scheduler(self):
        self.optimizer = self.model.optimizer
        lr_decay_steps = int(self.args.train_steps*(1/self.args.decay_ratio))

        assert self.args.scheduler_type in ['constant_with_warmup', 'linear']
        self.lr_scheduler = get_scheduler(
            self.args.scheduler_type, optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps, 
            num_training_steps=lr_decay_steps,
        )

    def register_hooks(self):
        self.hook_queue.append(EvaluationHook())
        self.hook_queue.append(CheckpointHook())
        self.hook_queue.append(EarlyStopHook(self.args.patience))

    def process_batch(self, raw_batch_data, eval_data=False):
        """ process batch data, send data to cuda """

        def make_tgt_mask(sz):
            '''
                Args:
                    sz (integer): max_len of sequence in target without EOS i.e. (T-1)
                Returns:
                    mask (tensor) : square mask | size : [T-1 x T-1]
            '''
            mask = torch.triu(torch.ones(sz, sz), 1)
            mask = mask.masked_fill(mask==1, float('-inf'))
            return mask

        def make_len_mask(input_ids, pad_id):
            '''
                Args:
                    input_ids (tensor) | size : (seqence_len, bz)
                    pad_id (int)
                Returns:
                    mask (tensor) : pad mask | size : (bz, seqence_len)
            '''
            mask = (input_ids == pad_id).transpose(0, 1)
            return mask

        batch_data = {}
        for key, val in raw_batch_data.items():
            if isinstance(val, torch.Tensor):
                batch_data[key] = val.to(self.device)

        encode_pad_id = self.enc_tokenizer.pad_token_id
        if eval_data:
            dec_start_token = self.dec_tokenizer.get_id('<s>')
            current_bz = int(raw_batch_data['current_bz'])
            batch_data['decode_starter'] = [[dec_start_token for i in range(current_bz)]]
            encoder_input = batch_data['input_ids'].transpose(0, 1)
            batch_data['mem_pad_mask'] = make_len_mask(encoder_input, encode_pad_id).to(self.device)
            del batch_data['output_seq']
        else:
            encoder_input = batch_data['input_ids'].transpose(0, 1)
            batch_data['mem_pad_mask'] = make_len_mask(encoder_input, encode_pad_id).to(self.device)

            decoder_input = batch_data['output_seq'].transpose(0, 1)[:-1, :]
            decode_pad_id = self.dec_tokenizer.get_id('</s>')
            batch_data['tgt_pad_mask'] = make_len_mask(decoder_input, decode_pad_id).to(self.device)
            batch_data['tgt_mask'] = make_tgt_mask(len(decoder_input)).to(self.device)
        return batch_data

    def train(self):
        self.model.train()
        self.call_hook("before_run")

        self.logger.info(f"")
        self.logger.info(f"***** Running training *****")
        self.logger.info(f"  Dataset name: {self.args.dataset_name}")
        self.logger.info(f"  Num train examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num eval examples = {len(self.eval_dataset)}")
        self.logger.info(f"  Training batch size = {self.args.train_bz}")
        self.logger.info(f"  Total optimization steps = {len(self.train_loader)}")
        self.logger.info(f"  Trainable params: {count_parameters(self.model)}\n")
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            assert self.args.train_steps == len(self.train_loader)
            for batch_data in tqdm(self.train_loader):
                assert self.current_step < self.args.train_steps
                self.optimizer.zero_grad()
                loss = self.model.training_step(**self.process_batch(batch_data))
                self.loss_record.append(loss.item())
                loss.backward()
                
                if self.args.max_grad_norm > 0:
                    gn = clip_gn(self.model.parameters(), self.args.max_grad_norm)
                    self.state_dict['grad_norm'] = gn

                self.optimizer.step()
                self.lr_scheduler.step()
                self.current_step += 1
                self.call_hook("after_train_step")

                if self.should_stop:
                    self.logger.warning('Training stops due to early-stop')
                    break

            self.call_hook("after_train_epoch")
        self.call_hook("after_run")

    def evaluate(self, eval_mode='valid'):
        """Evaluate loop: validation and test
            
            Args:
                eval_mode (str, optional): 'valid' or 'test'
            Returns:
                Dict: result dictionary
        """
        self.model.eval()
        total_examples, correct_examples = 0, 0

        for batch_data in self.eval_loader:
            proc_batch_data = self.process_batch(batch_data, eval_data=True)
            proc_batch_data['target_len'] = self.args.max_output_len
            pred_token_ids = self.model.greedy_decode(**proc_batch_data)
            current_bz = int(batch_data['current_bz'])
            batch_decoded_tokens = [[] for i in range(current_bz)]
            for i in range(len(pred_token_ids)):
                for j in range(current_bz):
                    if pred_token_ids[i][j] == self.dec_tokenizer.get_id('<s>'):
                        continue
                    if pred_token_ids[i][j] == self.dec_tokenizer.get_id('</s>'):
                        continue
                    decoded_token = self.dec_tokenizer.get_word(pred_token_ids[i][j])
                    batch_decoded_tokens[j].append(decoded_token)

            numbers = [nums.split() for nums in batch_data['numbers']]
            gold_eqns = [eqn.split() for eqn in batch_data['eqns']]
            pred_eqns = batch_decoded_tokens

            if self.args.output_as_code:
                num_correct, num_examples = calculate_acc_code(gold_eqns, pred_eqns)
            else:
                num_correct, num_examples = calculate_acc(gold_eqns, pred_eqns, numbers)

            correct_examples += num_correct
            total_examples += num_examples
        
        self.model.train()
        return {'accuracy': correct_examples / total_examples}
    
    def save_model(self):
        """
        save best model checkpoints on validation data
        """
        save_folder = os.path.join("checkpoints", self.args.exp_group, self.args.run_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)        
        torch.save(self.model.state_dict(), os.path.join(save_folder, 'model.pt'))

    def load_model(self, load_path=None):
        """
        load model and tokenizers for final evaluation
        """
        if load_path is None:
            load_path = os.path.join("checkpoints", self.args.exp_group, self.args.run_name)
        model_path = os.path.join(load_path, 'model.pt')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def load_model_from_hf(self, config, model_name):
        hf_model = HFRoBERTaGen.from_pretrained(model_name, config=config, pytorch_model=self.model)
        self.model.load_state_dict(hf_model.model.state_dict())
    
    def upload(self):
        config = RoBERTaGenConfig(d_model=self.model.config.d_model)
        hf_model = HFRoBERTaGen(config, self.model)
        hf_model.push_to_hub('Tianduo/MsAT-RoBERTaGen-SVAMP')
        
    def call_hook(self, fn_name, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self.hook_queue:
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)


"""end"""



