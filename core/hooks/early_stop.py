
import wandb
from .hook import Hook

class EarlyStopHook(Hook):
    def __init__(self, patience:int) -> None:
        super().__init__()
        self.max_patience = patience
        self.cur_patience = self.max_patience

    def after_train_step(self, trainer):
        """must be called after evaluation"""
        if self.every_n_iters(trainer, trainer.eval_steps):
            if wandb.run.summary['best_step'] == trainer.current_step:
                self.cur_patience = self.max_patience
            else:
                self.cur_patience -= 1
            if self.cur_patience <= 0:
                trainer.should_stop = True
