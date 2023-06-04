
class Hook:
    stages = ('before_run', 
              'before_train_epoch', 'before_train_step', 'after_train_step', 'after_train_epoch',
              'after_run')

    def before_train_epoch(self, trainer):
        pass

    def after_train_epoch(self, trainer):
        pass

    def before_train_step(self, trainer):
        pass

    def after_train_step(self, trainer):
        pass
    
    def before_run(self, trainer):
        pass

    def after_trun(self, trainer):
        pass

    def every_n_epochs(self, trainer, n):
        return (trainer.epoch + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, trainer, n):
        return trainer.current_step % n == 0 if n > 0 else False

    def end_of_epoch(self, trainer):
        return trainer.current_step % len(trainer.data_loader['train_lb']) == 0

    def is_last_epoch(self, trainer):
        return trainer.epoch + 1 == trainer.epochs

    def is_last_iter(self, trainer):
        return trainer.current_step == trainer.train_steps