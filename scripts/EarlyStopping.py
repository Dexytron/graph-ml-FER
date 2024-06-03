import torch


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.val_loss_min = None
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_name=""):
        torch.save(model.state_dict(), f'{model_name}-checkpoint.pt')
        self.val_loss_min = val_loss

