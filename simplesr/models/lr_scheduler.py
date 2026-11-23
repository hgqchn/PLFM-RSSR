
from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineAnnealingHoldLR(CosineAnnealingLR):

    def get_lr(self):
        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.optimizer.param_groups]
        return super().get_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]
        return super()._get_closed_form_lr()
