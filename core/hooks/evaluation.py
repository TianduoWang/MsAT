import os
import statistics as stat
import wandb
from .hook import Hook



class EvaluationHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train_step(self, trainer):
        if trainer.args.warmup_steps-1 < trainer.current_step and \
            (self.every_n_iters(trainer, trainer.eval_steps) or self.is_last_iter(trainer)):
            trainer.logger.info("  Validating.....")
            eval_dict = trainer.evaluate('valid')
            acc = eval_dict['accuracy']

            # update best metrics
            if acc > wandb.run.summary['best_acc']:
                wandb.run.summary['best_acc'] = acc
                wandb.run.summary['best_step'] = trainer.current_step
                
                trainer.state_dict['best_valid_acc'] = acc
                trainer.state_dict['best_step'] = trainer.current_step

            trainer.logger.info(f"  This step: {trainer.current_step},\t"
                + f"this acc: {acc*100:.1f}%")
            trainer.logger.info(f"  Best step: {wandb.run.summary['best_step']},\t"
                + f"best acc: {wandb.run.summary['best_acc']*100:.1f}%")

            wandb.log({"accuracy": acc}, step=trainer.current_step//100)
        
        if self.every_n_iters(trainer, 100):
            loss_value = sum(trainer.loss_record)
            trainer.loss_record.clear()

            trainer.logger.info(f"  Loss: {loss_value:.5f}")
            lr = trainer.lr_scheduler.get_last_lr()[0]
            wandb.log({'loss': loss_value, 'lr': lr}, step=trainer.current_step//100)
    

    def after_run(self, trainer):
        # load best model and re-evaluate on valid set or test set
        trainer.load_model()
        eval_dict = trainer.evaluate('test') # here should be test
        trainer.logger.info(f"  Final accuracy: {eval_dict['accuracy']*100:.1f}\n")
        trainer.result_dict['final_acc'] = f"{eval_dict['accuracy']*100:.3f}"

        # Remove unnecessary model weights
        if trainer.args.remove_weights_after_train:
            trainer.logger.info('Removing model weights...')
            model_weights_path = os.path.join("checkpoints", 
                trainer.args.exp_group, trainer.args.run_name, "model.pt")
            os.remove(model_weights_path)
