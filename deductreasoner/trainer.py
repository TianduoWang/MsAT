import logging
import os
import contextlib
import pickle
import numpy as np
from dataclasses import asdict
from argparse import Namespace
from inspect import signature
from collections import OrderedDict, Counter
from tqdm import tqdm
import wandb
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers.optimization import get_scheduler

from core.hooks import Hook, EvaluationHook, CheckpointHook, EarlyStopHook
from core.utils import count_parameters, compute_value_for_incremental_equations
from deductreasoner.prepare_dataset import get_train_dataloader, get_eval_dataloader
from deductreasoner.model import HFDeductReasoner, DeductReasonerConfig


def get_batched_prediction_consider_multiple_m0(
    var_starts, all_logits, constant_num, add_replacement=False
    ):
    batch_size, max_num_variable = var_starts.size()
    batched_prediction = [[] for _ in range(batch_size)]
    for k, logits in enumerate(all_logits):
        current_max_num_variable = max_num_variable + constant_num + k
        num_var_range = torch.arange(0, current_max_num_variable)
        combination = torch.combinations(num_var_range, r=2, with_replacement=add_replacement)
        num_combinations, _ = combination.size()

        best_temp_logits, best_temp_stop_label = logits.max(dim=-1)
        best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)
        best_m0_score, best_comb = best_temp_score.max(dim=-1)
        best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)
        b_idxs = [bidx for bidx in range(batch_size)]
        best_stop_label = best_temp_stop_label[b_idxs, best_comb, best_label]

        best_comb_var_idxs = torch.gather(combination.unsqueeze(0).expand(batch_size, num_combinations, 2), 1,
                                          best_comb.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, 2)).squeeze(1)
        best_comb_var_idxs = best_comb_var_idxs.cpu().numpy()
        best_labels = best_label.cpu().numpy()
        curr_best_stop_labels = best_stop_label.cpu().numpy()
        for b_idx, (best_comb_idx, best_label, stop_label) in \
            enumerate(zip(best_comb_var_idxs, best_labels, curr_best_stop_labels)):
            left, right = best_comb_idx
            curr_label = [left, right, best_label, stop_label]
            batched_prediction[b_idx].append(curr_label)
    return batched_prediction


class Trainer:

    def __init__(self, trainer_args, model, 
        data_collator=None, train_dataset=None, eval_dataset=None, enc_tokenizer=None, dec_tokenizer=None,
        tb_log=None, logger=None, **kwargs):

        self.args = trainer_args
        self.device = self.args.device
        self.model = model.to(self.device)
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
        batch_data = {}
        for key, val in raw_batch_data.items():
            if isinstance(val, torch.Tensor):
                batch_data[key] = val.to(self.device)
        
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
                    gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.state_dict['grad_norm'] = gn
                self.optimizer.step()
                if self.lr_scheduler:
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

        correct_examples, total_examples = 0, 0
        corrects_by_step, total_by_step = Counter(), Counter()
        for batch_data in self.eval_loader:
            proc_batch_data = self.process_batch(batch_data, eval_data=True)

            logits = self.model.greedy_decode(**proc_batch_data) # size: Seq_len x Bz
            batch_prediction = get_batched_prediction_consider_multiple_m0(
                var_starts=batch_data['var_starts'], all_logits=logits, 
                constant_num=self.model.constant_num, add_replacement=self.model.add_replacement
                )
            for b, inst_predictions in enumerate(batch_prediction):
                for p, prediction_step in enumerate(inst_predictions):
                    left, right, op_id, stop_id = prediction_step
                    if stop_id == 1:
                        batch_prediction[b] = batch_prediction[b][:(p+1)]
                        break
            batch_labels = batch_data['output_seq'].cpu().numpy().tolist()
            for b, inst_labels in enumerate(batch_labels):
                for p, label_step in enumerate(inst_labels):
                    left, right, op_id, stop_id = label_step
                    if stop_id == 1:
                        batch_labels[b] = batch_labels[b][:(p+1)]
                        break

            batch_num_list = []
            for nums_str in batch_data['numbers']:
                batch_num_list.append([float(s) for s in nums_str.split()])

            for inst_predictions, inst_labels, num_list in zip(batch_prediction, batch_labels, batch_num_list):
                num_constant = self.model.constant_num
                uni_labels = list(self.dec_tokenizer.op2id.keys())
                constant_values = self.dec_tokenizer.const_list_float

                pred_val, _ = compute_value_for_incremental_equations(
                    inst_predictions, num_list, num_constant, uni_labels, constant_values)
                gold_val, _ = compute_value_for_incremental_equations(
                    inst_labels, num_list, num_constant, uni_labels, constant_values)
                                
                if abs((gold_val- pred_val)) < 1e-4:
                    correct_examples += 1
                    corrects_by_step[len(inst_labels)] += 1
                
                total_examples += 1
                total_by_step[len(inst_labels)] += 1
        
        self.model.train()

        eval_result = {'accuracy': correct_examples / total_examples}
        for step, _ in sorted(total_by_step.items()):
            eval_result[f'step{step}'] = (corrects_by_step[step], total_by_step[step])
        return eval_result
    
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
        hf_model = HFDeductReasoner.from_pretrained(model_name, config=config, pytorch_model=self.model)
        self.model.load_state_dict(hf_model.model.state_dict())

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



