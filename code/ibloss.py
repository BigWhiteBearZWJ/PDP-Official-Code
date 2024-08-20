import torch
import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC

@LOSSFUNC.register_module(module_name="information_bottleneck")
class IBLoss(AbstractLossClass):
    def __init__(self, method='mi', mi_calculator='kl', temp1=1.5, temp2=0.5):
        super().__init__()

        self.method=method
        if mi_calculator == "kl":
            self.mi_calculator = torch.nn.KLDivLoss()
        self.temp1 =temp1
        self.temp2 =temp2
        self.softmax = torch.nn.Softmax(dim=1)
 
    def forward(self, pred_dict):
        # p_catart = pred_dict['p_catart']
        p_tid = pred_dict['p_tid']
        p_sid = pred_dict['p_sid']
        p_tart = pred_dict['p_tart']
        p_sart = pred_dict['p_sart']
        p_tart_pure = pred_dict['p_tart_pure']
        p_sart_pure = pred_dict['p_sart_pure']

        # Local Information Loss
        local_mi_loss = (self.mi_calculator((self.softmax(p_tart.detach() / self.temp1)+1e-8).log(), self.softmax(p_tid / self.temp1)) + 
                      self.mi_calculator((self.softmax(p_sart.detach() / self.temp1)+1e-8).log(),self.softmax(p_sid / self.temp1)))
        local_mi_loss = local_mi_loss.abs()

        #Global Information Loss
        global_mi_loss = (self.mi_calculator((self.softmax(p_tart.detach() / self.temp2)+1e-8).log(),self.softmax(p_tart_pure / self.temp2)) +
                          self.mi_calculator((self.softmax(p_sart.detach() / self.temp2)+1e-8).log(),self.softmax(p_sart_pure / self.temp2)))
        global_mi_loss = global_mi_loss.abs()
        # global_mi_loss = torch.tensor(0.0001)

        loss_ib = torch.exp(-local_mi_loss) + 0.5*torch.exp(global_mi_loss)
        # loss_ib=0
        return loss_ib, local_mi_loss, global_mi_loss