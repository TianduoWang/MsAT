import os
import wandb
from .hook import Hook

class CheckpointHook(Hook):
    def __init__(self):
        super().__init__()
    
    def after_train_step(self, trainer):
        """ should be called after evaluation hook """
        if trainer.args.save_at_last_step:
            if self.is_last_iter(trainer):
                trainer.logger.info("  Saving model checkpoint at last step.....\n")
                trainer.save_model()
        elif self.every_n_iters(trainer, trainer.eval_steps) or self.is_last_iter(trainer):
            if wandb.run.summary['best_step'] == trainer.current_step:
                trainer.logger.info("  Saving model checkpoint.....\n")
                trainer.save_model()

        