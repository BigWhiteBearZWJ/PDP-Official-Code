import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC

def swap_spe_features(type_list, value_list):
    type_list = type_list.cpu().numpy().tolist()
    # get index
    index_list = list(range(len(type_list)))

    # init a dict, where its key is the type and value is the index
    spe_dict = defaultdict(list)

    # do for-loop to get spe dict
    for i, one_type in enumerate(type_list):
        spe_dict[one_type].append(index_list[i])

    # shuffle the value list of each key
    for keys in spe_dict.keys():
        random.shuffle(spe_dict[keys])
    
    # generate a new index list for the value list
    new_index_list = []
    for one_type in type_list:
        value = spe_dict[one_type].pop()
        new_index_list.append(value)

    # swap the value_list by new_index_list
    value_list_new = value_list[new_index_list]

    return value_list_new


@LOSSFUNC.register_module(module_name="contrastive_regularization")
class ContrastiveLoss(AbstractLossClass):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def id_contrastive_loss(self, art, id1, id2):
        dist_id = F.cosine_similarity(art, id1) + F.cosine_similarity(art, id2)
        # Compute loss as the distance between anchor and negative minus the distance between anchor and positive
        loss = torch.mean(torch.clamp(dist_id, min=0, max=2))
        return loss

    # def forward(self, common, specific, spe_label):
    def forward(self, common, tar, src):
        # prepare
        bs = common.shape[0]
        real_common, fake_common = common.chunk(2)
        real_tar, fake_tar = tar.chunk(2)
        real_src, fake_src = src.chunk(2)
        ### common real
        idx_list = list(range(0, bs//2))
        random.shuffle(idx_list)
        real_common_anchor = common[idx_list]
        ### common fake
        idx_list = list(range(bs//2, bs))
        random.shuffle(idx_list)
        fake_common_anchor = common[idx_list]

        loss_id = (2-self.id_contrastive_loss(real_common, real_tar, real_src)) + self.id_contrastive_loss(fake_common, fake_tar, fake_src)
        loss = loss_id
        return loss
