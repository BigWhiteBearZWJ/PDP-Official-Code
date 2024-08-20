'''
# author: Weijie Zhouw
# date: 2024-7
# description: Class for the PDPDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

'''

import os
import datetime
import logging
import random
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from efficientnet_pytorch import EfficientNet
from metrics.base_metrics_class import calculate_metrics_for_train
from networks.iresnet import iresnet100
from networks.igam import IGAM
from networks.AdaAttn import AdaAttN

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='mydetector')
class MyDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config['backbone_config']['num_classes']
        self.encoder_feat_dim = config['encoder_feat_dim']
        self.id_dim = self.encoder_feat_dim//2
        self.fingerprint_dim = self.encoder_feat_dim//2

        self.encoder_tid = self.init_efficient()
        self.encoder_sid = self.init_efficient()
        logger.info('Load pretrained model successfully!')

        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
        # basic function
        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # conditional gan
        self.con_gan = Conditional_UNet()

        # head
        self.head_art = Art_Head(
            in_f=self.fingerprint_dim,
            hidden_dim=self.encoder_feat_dim, 
            out_f=self.num_classes
        )
        self.head_tart = Art_Head(
            in_f=self.fingerprint_dim,
            hidden_dim=self.encoder_feat_dim, 
            out_f=self.num_classes
        )
        self.head_sart = Art_Head(
            in_f=self.fingerprint_dim,
            hidden_dim=self.encoder_feat_dim, 
            out_f=self.num_classes
        )

        # for original id classify
        self.head_tarid_org = Id_Head(
            in_f=self.id_dim,
            hidden_dim=self.encoder_feat_dim, 
            out_f=self.num_classes
        )
        self.head_srcid_org = Id_Head(
            in_f=self.id_dim,
            hidden_dim=self.encoder_feat_dim, 
            out_f=self.num_classes
        )

        # for pure art classify
        self.head_tart_pure = Id_Head(
            in_f=self.fingerprint_dim,
            hidden_dim=self.encoder_feat_dim, 
            out_f=self.num_classes
        )
        self.head_sart_pure = Id_Head(
            in_f=self.fingerprint_dim,
            hidden_dim=self.encoder_feat_dim, 
            out_f=self.num_classes
        )
        # for pure id classify
        self.head_tarid_pure = Id_Head(
            in_f=self.id_dim,
            hidden_dim=self.encoder_feat_dim, 
            out_f=self.num_classes
        )
        self.head_srcid_pure = Id_Head(
            in_f=self.id_dim,
            hidden_dim=self.encoder_feat_dim, 
            out_f=self.num_classes
        )

        # disentangle block
        self.block_tarid = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.id_dim, 
            out_f=self.id_dim
        )
        self.block_srcid = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.id_dim, 
            out_f=self.id_dim
        )
        self.block_tarart = Conv2d1x1(
            in_f=self.encoder_feat_dim, 
            hidden_dim=self.fingerprint_dim, 
            out_f=self.fingerprint_dim
        )
        self.block_srcart = Conv2d1x1(
            in_f=self.encoder_feat_dim, 
            hidden_dim=self.fingerprint_dim, 
            out_f=self.fingerprint_dim
        )

        # art feat compresss
        self.block_mam = Conv2d1x1(
            in_f=self.encoder_feat_dim, 
            hidden_dim=self.fingerprint_dim, 
            out_f=self.fingerprint_dim
        )
        
        # Identity Artifact Correlation Compression
        self.block_IACC = IACC(in_channels1=256, in_channels2=256, k_dim=8, v_dim=8, num_heads=8)


    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        # logger.info('Load pretrained model successfully!')
        return backbone
    
    def init_efficient(self):
        model = efficientnet(pretrain='efficientnet-b4')
        return model

    def build_loss(self, config):
        cls_loss_class = LOSSFUNC[config['loss_func']['cls_loss']]
        con_loss_class = LOSSFUNC[config['loss_func']['con_loss']]
        rec_loss_class = LOSSFUNC[config['loss_func']['rec_loss']]
        ib_loss_class = LOSSFUNC[config['loss_func']['ib_loss']]
        cls_loss_func = cls_loss_class()
        con_loss_func = con_loss_class(margin=3.0)
        rec_loss_func = rec_loss_class()
        ib_loss_func = ib_loss_class()
        loss_func = {
            'cls': cls_loss_func, 
            'con': con_loss_func,
            'rec': rec_loss_func,
            'ib': ib_loss_func,
        }
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        cat_data = data_dict['image']
        # encoder
        tar_all = self.encoder_tid.features(cat_data)
        src_all = self.encoder_sid.features(cat_data) 
        feat_dict = {'tar': tar_all, 'src': src_all}
        return feat_dict

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)

    def classifier_target(self, features: torch.tensor) -> torch.tensor:
        # classification, multi-task
        # split the features into the content and target
        f_tar = self.block_tarid(features)
        f_tart = self.block_tarart(features)
        return f_tar, f_tart

    def classifier_source(self, features: torch.tensor) -> torch.tensor:
        # classification, multi-task
        # split the features into the art and source
        f_src = self.block_srcid(features)
        f_sart = self.block_srcart(features)
        return f_src, f_sart

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # if 'label_spe' in data_dict and 'recontruction_imgs' in pred_dict:
        if 'recontruction_imgs' in pred_dict:
            return self.get_train_losses(data_dict, pred_dict)
        else:  # test mode
            return self.get_test_losses(data_dict, pred_dict)

    def get_train_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # get combined, real, fake imgs
        cat_data = data_dict['image']
        real_img, fake_img = cat_data.chunk(2, dim=0)
        # get the reconstruction imgs
        reconstruction_image_1, \
        reconstruction_image_2, \
        self_reconstruction_image_1, \
        self_reconstruction_image_2 \
            = pred_dict['recontruction_imgs']
        # get label
        label = data_dict['label']
        # get pred
        pred = pred_dict['cls']

        # 1. classification loss for common features
        loss_sha = self.loss_func['cls'](pred, label)

        # 2. reconstruction loss
        self_loss_reconstruction_1 = self.loss_func['rec'](fake_img, self_reconstruction_image_1)
        self_loss_reconstruction_2 = self.loss_func['rec'](real_img, self_reconstruction_image_2)
        cross_loss_reconstruction_1 = self.loss_func['rec'](fake_img, reconstruction_image_2)
        cross_loss_reconstruction_2 = self.loss_func['rec'](real_img, reconstruction_image_1)
        loss_reconstruction = \
            self_loss_reconstruction_1 + self_loss_reconstruction_2 + \
            cross_loss_reconstruction_1 + cross_loss_reconstruction_2

        # 3. constrative loss
        art_feat = pred_dict['feat_art']
        tar_feat = pred_dict['feat_tar']
        src_feat = pred_dict['feat_src']
        loss_con = self.loss_func['con'](art_feat, tar_feat, src_feat)

        # 4. ib loss
        loss_ib = self.loss_func['ib'](pred_dict)
        loss_ib, _, _= self.loss_func['ib'](pred_dict)

        # 5. total loss
        loss = 5*loss_sha + 0.1*loss_reconstruction + 0.5*loss_ib + 0.5*loss_con
        loss_dict = {
            'overall': loss,
            'common': loss_sha,
            'reconstruction': loss_reconstruction,
            'contrastive': loss_con,
            'ib': loss_ib,
        }
        return loss_dict

    def get_test_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # get label
        label = data_dict['label']
        # get pred
        pred = pred_dict['cls']
        # for test mode, only classification loss for common features
        loss = self.loss_func['cls'](pred, label)
        loss_dict = {'common': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        def get_accracy(label, output):
            _, prediction = torch.max(output, 1)    # argmax
            correct = (prediction == label).sum().item()
            accuracy = correct / prediction.size(0)
            return accuracy
        
        # get pred and label
        label = data_dict['label']
        pred = pred_dict['cls']

        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        y_true = np.where(y_true!=0, 1, 0)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true,y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        return {'acc':acc, 'auc':auc, 'eer':eer, 'ap':ap, 'pred':y_pred, 'label':y_true}

    def forward(self, data_dict: dict, inference=False) -> dict:
        # split the features into the content and forgery
        features = self.features(data_dict)
        tar_features, src_features = features['tar'], features['src'] #[2*bs, 512, 8, 8]

        # get the prediction by classifier
        id_tar, f_tart= self.classifier_target(tar_features) #[2*bs, 256, 8, 8]
        id_src, f_sart= self.classifier_source(src_features)

        # get the mean and variance of the identity to generate the corresponding noise
        id_tar_mean, id_tar_std = calc_mean_std(id_tar)
        id_src_mean, id_src_std = calc_mean_std(id_src)
        id_tar_noise = torch.randn(id_tar.shape).cuda() * id_tar_std + id_tar_mean
        id_src_noise = torch.randn(id_src.shape).cuda() * id_src_std + id_src_mean
        tart_mean, tart_std = calc_mean_std(f_tart)
        sart_mean, sart_std = calc_mean_std(f_sart)
        tart_noise = torch.randn(f_tart.shape).cuda() * tart_std + tart_mean
        sart_noise = torch.randn(f_sart.shape).cuda() * sart_std + sart_mean

        # identity forgery information compress
        pure_tar_id, pure_tar_art = self.block_IACC(id_tar, f_tart, id_tar_noise, tart_noise) #[2*bs, 256, 8, 8]
        pure_src_id, pure_src_art = self.block_IACC(id_src, f_sart, id_src_noise, sart_noise)

        mix_art_cat = torch.cat((pure_tar_art, pure_src_art), dim=1) #[2*bs, 512, 8, 8]
        mix_art_cls = self.block_mam(mix_art_cat) #[2*bs, 256, 8, 8]

        if inference:
            # inference only consider share loss
            out_art, art_feat = self.head_art(mix_art_cls)
            _, tar_feat = self.head_tarid_pure(pure_tar_id)
            _, src_feat = self.head_srcid_pure(pure_src_id)
            prob_art = torch.softmax(out_art, dim=1)[:, 1]
            self.prob.append(
                prob_art
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(out_art, 1)
            common_label = (data_dict['label'] >= 1)
            correct = (prediction_class == common_label).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)

            pred_dict = {'cls': out_art, 'prob': prob_art, 'feat': art_feat, 'feat_tar': tar_feat, 'feat_src': src_feat,}
            return pred_dict

        bs = self.config['train_batchSize']
        # using idx aug in the training mode
        aug_idx = random.random()
        if aug_idx < 0.7:
            # real
            idx_list = list(range(0, bs))
            random.shuffle(idx_list)
            mix_art_cls[0: bs] = mix_art_cls[idx_list]
            # fake
            idx_list = list(range(bs, bs*2))
            random.shuffle(idx_list)
            mix_art_cls[bs: bs*2] = mix_art_cls[idx_list]

        # concat spe and share to obtain new_id_all
        new_identity_all = torch.cat((pure_tar_id, pure_src_id), dim=1) #[2*bs, 512, 8, 8]

        # reconstruction loss
        f2, f1 = mix_art_cat.chunk(2, dim=0) #[bs, 512, 8, 8]
        c2, c1 = new_identity_all.chunk(2, dim=0)

        # ==== self reconstruction ==== #
        # f1 + c1 -> f11, f11 + c1 -> near~I1
        self_reconstruction_image_1 = self.con_gan(f1, c1)

        # f2 + c2 -> f2, f2 + c2 -> near~I2
        self_reconstruction_image_2 = self.con_gan(f2, c2)

        # ==== cross combine ==== #
        reconstruction_image_1 = self.con_gan(f1, c2)
        reconstruction_image_2 = self.con_gan(f2, c1)

        # head for id and art
        _, tar_feat = self.head_tarid_pure(pure_tar_id)
        _, src_feat = self.head_srcid_pure(pure_src_id)

        out_art, art_feat = self.head_art(mix_art_cls)

        # compute possibility for ibloss
        p_tid, _ = self.head_tarid_org(id_tar)
        p_sid, _ = self.head_srcid_org(id_src)
        p_tart, _ = self.head_tart(f_tart)
        p_sart, _ = self.head_sart(f_sart)
        p_tart_pure, _ = self.head_tart_pure(pure_tar_art)
        p_sart_pure, _ = self.head_sart_pure(pure_src_art)

        # get the probability of the pred
        prob_art = torch.softmax(out_art, dim=1)[:, 1]

        # build the prediction dict for each output
        pred_dict = {
            'cls': out_art, 
            'prob': prob_art, 
            'feat_art': art_feat,
            'feat_tar': tar_feat,
            'feat_src': src_feat,
            'p_tid': p_tid,
            'p_sid': p_sid,
            'p_tart': p_tart,
            'p_sart': p_sart,
            'p_tart_pure': p_tart_pure,
            'p_sart_pure': p_sart_pure,
            'recontruction_imgs': (
                reconstruction_image_1,
                reconstruction_image_2,
                self_reconstruction_image_1, 
                self_reconstruction_image_2
            )
        }
        return pred_dict

def r_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

class SANet(nn.Module):

    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O

class Conditional_UNet(nn.Module):

    def init_weight(self, std=0.2):
        for m in self.modules():
            cn = m.__class__.__name__
            if cn.find('Conv') != -1:
                m.weight.data.normal_(0., std)
            elif cn.find('Linear') != -1:
                m.weight.data.normal_(1., std)
                m.bias.data.fill_(0)

    def __init__(self):
        super(Conditional_UNet, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv2d(512, 512, 1)
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(512, 512, (3, 3))
        
        self.sa3 = SANet(in_planes=512)

        self.dconv_up3 = r_double_conv(512, 256)
        self.dconv_up2 = r_double_conv(256, 128)
        self.dconv_up1 = r_double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)
        self.up_last = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.activation = nn.Tanh()
        #self.init_weight() 
        
    def forward(self, c, x):  # c is the style and x is the content [bs, 512, 8, 8]
        # new decoder
        cs = self.sa3(x, c) # [bs, 512, 8, 8]
        cs = self.conv1(cs)
        cs = cs + x
        cs = self.merge_conv(self.merge_conv_pad(cs))
        cs = self.upsample(cs)
        cs = self.dropout(cs)
        cs = self.dconv_up3(cs) # [bs, 256, 16, 16]
        cs = self.upsample(cs)
        cs = self.dropout(cs)
        cs = self.dconv_up2(cs) # [bs, 128, 32, 32]
        cs = self.upsample(cs)
        cs = self.dropout(cs)
        cs = self.dconv_up1(cs) # [bs, 64, 64, 64]
        cs = self.conv_last(cs)
        out = self.up_last(cs)  # [bs, 3, 256, 256]
        
        return self.activation(out)

class MLP(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(MLP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        x = self.pool(x)
        x = self.mlp(x)
        return x

class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(hidden_dim, out_f, 1, 1),)

    def forward(self, x):
        x = self.conv2d(x)
        return x

class Art_Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Art_Head, self).__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        bs = x.size()[0]
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat
    
class Id_Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Id_Head, self).__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        bs = x.size()[0]
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat
    
class efficientnet(nn.Module):
    def __init__(self, pretrain='efficientnet-b4'):
        super(efficientnet, self).__init__()
        self.model = EfficientNet.from_pretrained(pretrain,weights_path='/mnt/raid1/zwj22/paper_models/DeepfakeBench-v2-main/training/pretrained/efficientnet-b4-6ed6700e.pth')

        if pretrain == 'efficientnet-b4':
            self.conv = nn.Conv2d(1792, 512, 1)
        elif pretrain == 'efficientnet-b1':
            self.conv = nn.Conv2d(1280, 512, 1)
        elif pretrain == 'efficientnet-b3':
            self.conv = nn.Conv2d(1536, 512, 1)
        elif pretrain == 'efficientnet-b5':
            self.conv = nn.Conv2d(2048, 512, 1)
        elif pretrain == 'efficientnet-b6':
            self.conv = nn.Conv2d(2304, 512, 1)
        else:
            raise ValueError('pretrain is not supported')

        # self.channel_adjust_conv = nn.Conv2d(2424, 512, 1)
    
    def features(self, x):
        x = self.model.extract_features(x)
        x = self.conv(x)

        return x
    
    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.conv(x)

        return x

class CrossAttention(nn.Module):
    def __init__(self, in_channels1, in_channels2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        # 修改Linear层的输入维度为通道数
        self.proj_q = nn.Linear(in_channels1, k_dim * num_heads, bias=False)
        self.proj_k = nn.Linear(in_channels2, k_dim * num_heads, bias=False)
        self.proj_v = nn.Linear(in_channels2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_channels1)

    def forward(self, x, y, mask=None):
        batch_size, channels1, height1, width1 = x.size()
        _, channels2, height2, width2 = y.size()

        # 将输入从 [batch_size, channels, height, width] 转换为 [batch_size, height * width, channels]
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, height1 * width1, channels1)
        y = y.permute(0, 2, 3, 1).contiguous().view(batch_size, height2 * width2, channels2)

        # 计算两次attention权重
        q = self.proj_q(x).view(batch_size, height1 * width1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k = self.proj_k(y).view(batch_size, height2 * width2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v = self.proj_v(y).view(batch_size, height2 * width2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k) / self.k_dim**0.5
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch_size, height1 * width1, -1)
        output = self.proj_o(output)

        # 在注意力计算后，将输出重新转换回 [batch_size, channels, height, width]
        output = output.view(batch_size, height1, width1, channels1).permute(0, 3, 1, 2).contiguous()
        # weight = self.sigmoid(output)

        return output
    
# Identity Artifact Correlation Compression
class IACC(nn.Module):
    def __init__(self, in_channels1, in_channels2, k_dim, v_dim, num_heads):
        super(IACC, self).__init__()
        self.id_crossattn = CrossAttention(in_channels1, in_channels2, k_dim, v_dim, num_heads)
        self.art_crossattn = CrossAttention(in_channels1, in_channels2, k_dim, v_dim, num_heads)
        self.sigmoid = nn.Sigmoid()
        
        # 修改Linear层的输入维度为通道数
        self.conv512 = nn.Conv2d(in_channels1*2, in_channels1, 1, bias=True)
        self.conv256 = nn.Conv2d(in_channels1, in_channels1, 1, bias=True)

    def forward(self, id, art, idnoise, artnoise, mask=None):
        # 计算交互权重
        id_weight = self.id_crossattn(id, art)
        art_weight = self.id_crossattn(art, id)
        mix_weight = self.conv512(torch.cat((id_weight, art_weight), dim=1))
        mix_weight = self.sigmoid(mix_weight)

        # 利用权重进行信息压缩
        pure_id = (1-mix_weight) * id + mix_weight * idnoise
        pure_id = self.conv256(pure_id)
        pure_art = (1-mix_weight) * art + mix_weight * artnoise
        pure_art = self.conv256(pure_art)

        return pure_id, pure_art
