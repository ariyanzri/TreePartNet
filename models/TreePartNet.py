import torch
import numpy as np
import math
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule,  PointnetSAModuleMSG, build_shared_mlp
from torch.utils.data import DataLoader
import sys
import os
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet'))
sys.path.append(os.path.join(ROOT_DIR, 'dgcnn'))
sys.path.append(ROOT_DIR)
from TreeDataset import TreeDataset,SorghumDataset
from modules import PointNetfeat
from dgcnn.modules import DGCNN_cls
from collections import namedtuple

lr_clip = 1e-5
bnm_clip = 1e-2

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn

class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)

class ScaledDot(nn.Module):
    '''
    Scaled Dot product
    :parameter
        input: BxCxN
        output: input*weight*transpose(input)
    '''
    def __init__(self, d_model):
        super(ScaledDot, self).__init__()
        self.d_model = d_model
        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256)
        )
        #self.linear = nn.Linear(256,256)

    def forward(self, input):
        projection = self.fc_lyaer(input)
        projection = torch.nn.functional.normalize(projection,p=2,dim=1)
        dot_product = torch.matmul(projection.permute(0,2,1),projection)
        return torch.abs(dot_product)

class FocalLoss(nn.Module):
    '''
    Focal Loss
        FL=alpha*(1-p)^gamma*log(p) where p is the probability of ground truth class
    Parameters:
        alpha(1D tensor): weight for positive
        gamma(1D tensor):
    '''
    def __init__(self, alpha=1, gamma=2, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha=torch.tensor(alpha)
        self.gamma=gamma
        self.reduce=reduce

    def forward(self, input, target):
        BCE_Loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_Loss)
        Focal_Loss = torch.pow((1-pt), self.gamma) * F.binary_cross_entropy_with_logits(
            input, target, pos_weight=self.alpha, reduction='none')

        if self.reduce=='none':
            return Focal_Loss
        elif self.reduce=='sum':
            return torch.sum(Focal_Loss)
        else:
            return torch.mean(Focal_Loss)

class DynamicInstanceSegmentationLoss(nn.Module):

    def dirac(self,array,a=-0.565,b=4):
        return 1/(a*1.772453851)*torch.exp(-(b*array/a)**2)+1

    def forward(self,input,target):
        input = torch.unsqueeze(input.float(),dim=-1)
        target = torch.unsqueeze(target.float(),dim=-1)
        distances_pred = torch.cdist(input,input)
        distances_gt = torch.cdist(target,target)
        normalized_pred = self.dirac(distances_pred)
        normalized_gt = self.dirac(distances_gt)
        # loss = F.binary_cross_entropy(normalized_pred, normalized_gt)
        loss = torch.mean(torch.pow(normalized_pred-normalized_gt,2))

        return loss

class SpaceSimilarityLoss(nn.Module):

    def __init__(self,foc_ind=None,knn=None):
        super().__init__()
        self.one_t = 1#torch.Tensor(1).to("cuda")
        self.zero_t = -1#torch.Tensor(0).to("cuda")
        self.pred_is_focal_indices = foc_ind
        # self.knn = knn
    
    def forward(self,input,target):
        
        # target = torch.unsqueeze(target.float(),dim=-1)
        # distances_gt = torch.cdist(target,target)
        # normalized_distance_gt = torch.zeros(distances_gt.size())
        # normalized_distance_gt = torch.where(distances_gt==0,self.one_t,self.zero_t)
        # # normalized_distance_gt = normalized_distance_gt.to("cuda")

        # distance_pred = torch.cdist(input,input)
        
        # loss = torch.mean(torch.mul(distance_pred,normalized_distance_gt))

        # return loss

        if self.pred_is_focal_indices is not None:
            target = target[:,self.pred_is_focal_indices[1]]

        distance_pred = torch.cdist(input,input)
        inv_distance_pred = distance_pred.div(distance_pred**2+0.00001)
        
        target = torch.unsqueeze(target.float(),dim=-1)
        distances_gt = torch.cdist(target,target)
        normalized_distance_gt = torch.where(distances_gt==0,distance_pred,inv_distance_pred)
        
        # if self.knn is not None:
            
        #     knn_loss = torch.mean(distance_pred[self.knn])
            
        #     return torch.mean(normalized_distance_gt) + knn_loss

        return torch.mean(normalized_distance_gt)

class SpaceSimilarityLossV2(nn.Module):

    def __init__(self,M1=1,M2=10):
        super().__init__()
        self.M1 = M1
        self.M2 = M2
    
    def forward(self,input,target):
        
        distance_pred = torch.cdist(input,input)
        
        target = torch.unsqueeze(target.float(),dim=-1)
        distances_gt = torch.cdist(target,target)
        
        normalized_distance_gt = torch.where(distances_gt==0,torch.clamp(distance_pred-self.M1,min=0),torch.clamp(self.M2-distance_pred,min=0))
        
        return torch.mean(normalized_distance_gt)

class LeafMetrics(nn.Module):

    def __init__(self,dist):
        super().__init__()
        self.threshold = dist
        
    def forward(self,input,target):
        
        cluster_pred = torch.cdist(input,input)

        target = torch.unsqueeze(target.float(),dim=-1)
        cluster_gt = torch.cdist(target,target)
        ones = torch.ones(cluster_pred.shape).cuda()
        zeros = torch.zeros(cluster_pred.shape).cuda()
        
        
        TP = torch.sum(torch.where((cluster_gt == 0) & (cluster_pred < 5), ones, zeros))
        TN = torch.sum(torch.where((cluster_gt > 0) & (cluster_pred >= 5), ones, zeros))
        FP = torch.sum(torch.where((cluster_gt > 0) & (cluster_pred < 5), ones, zeros))
        FN = torch.sum(torch.where((cluster_gt == 0) & (cluster_pred >= 5), ones, zeros))
        
        Acc = (TP+TN)/(TP+FP+TN+FN)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN) 
        F = 2*(Precision*Recall)/(Precision+Recall)

        return Acc, Precision, Recall, F

class TreePartNet(pl.LightningModule):
        
    def __init__(self, hparams):
        '''
        Parameters
        ----------
        hparams: hyper parameters
        '''
        super(TreePartNet,self).__init__()
        # self._hparams = hparams

        self.hparams.update(hparams)
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=hparams['lc_count'],
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[hparams['input_channels'], 16, 16, 16],
                      [hparams['input_channels'], 32, 32, 32]],
                use_xyz=hparams['use_xyz'],
            )
        )

        c_out_0 = 16 + 32

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_out_0, 64, 64, 128],
                      [c_out_0, 64, 64, 128]],
                use_xyz=hparams['use_xyz'],
            )
        )

        c_out_1 = 128 + 128

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[64, 256, 64]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_0+c_out_1, 256, 64]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 2, kernel_size=1),
        )
        
        self.lp_fc_layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 3, kernel_size=1),
        )

        self.sharedMLP_layer = build_shared_mlp([64, 64, 32, 1])
        self.dot = ScaledDot(64)
        self.scale = nn.Parameter(torch.tensor(10.0, dtype=torch.float), requires_grad=True)
        #init_threshold = math.cos(math.pi / 9) * self._hparams['ED_weight']  # 20 degree
        #self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float), requires_grad=True)
        self.save_hyperparameters()

    def forward(self, xyz):
        num_point = xyz.shape[1]

        # PointNet SA Module
        l_xyz, l_features, l_s_idx = [xyz], [None], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_s_idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_s_idx.append(li_s_idx)

        # PointNet FP Module
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        # Semantic Label Prediction
        is_focal_plant_pred = self.fc_layer(l_features[0])
        leaf_part_pred = self.lp_fc_layer(l_features[0])

        # Local Context Label Prediction
        point_feat = torch.unsqueeze(l_features[0],dim=-2)
        point_feat = point_feat.repeat(1,1,self.hparams['lc_count'],1)
        lc_feat = torch.unsqueeze(l_features[-2],dim=-1)
        lc_feat = lc_feat.repeat(1,1,1,num_point)
        per_point_feat = point_feat - lc_feat

        initial_fine_cluster_pred = self.sharedMLP_layer(per_point_feat)
        initial_fine_cluster_pred = initial_fine_cluster_pred.squeeze(dim=1)

        # Tree Edge Prediction
        dot = self.dot(l_features[-2])
        batch_idx = torch.tensor(range(xyz.shape[0]))
        batch_idx = batch_idx.unsqueeze(-1)
        batch_idx = batch_idx.repeat(1, self.hparams['lc_count'])
        s_xyz = xyz[batch_idx.cuda().long(), l_s_idx[-2].long()]
        s_xyz = torch.unsqueeze(s_xyz, dim=-2)
        s_xyz = s_xyz.repeat(1, 1, self.hparams['lc_count'], 1)
        dis = s_xyz - s_xyz.permute(0, 2, 1, 3)
        dis = dis ** 2
        dis = torch.sum(dis, dim=-1)
        dis = torch.sqrt(dis)

        affinity_matrix_pred = dot-self.scale*dis

        # FPS Sample Points
        # lc_idx = l_s_idx[0]
        # lc_idx = lc_idx[:,0:self.hparams['lc_count']]

        return is_focal_plant_pred,initial_fine_cluster_pred,leaf_part_pred,affinity_matrix_pred


    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams['lr_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            lr_clip / self.hparams['lr'],
        )
        bn_lbmd = lambda _: max(
            self.hparams['bn_momentum']
            * self.hparams['bnm_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self,ds_path,shuff=True):
        dataset = TreeDataset(ds_path)
        loader = DataLoader(dataset, batch_size=self.hparams['batch_size'], num_workers=4, shuffle=shuff)
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['train_data'],shuff=True)

    def training_step(self, batch, batch_idx):
        points,is_focal_plant,initial_fine_cluster,leaf_part,affinity_matrix = batch
        pred_is_focal_plant, pred_initial_fine_cluster, pred_leaf_part, pred_affinity_matrix = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        is_focal_plant_loss = critirion(pred_is_focal_plant, is_focal_plant)
        leaf_part_loss = critirion(pred_leaf_part, leaf_part)
        initial_fine_cluster_loss = critirion(pred_initial_fine_cluster, initial_fine_cluster)

        critirion2 = FocalLoss(alpha=self.hparams['FL_alpha'], gamma=self.hparams['FL_gamma'], reduce='mean')
        affinity_matrix_loss = critirion2(pred_affinity_matrix, affinity_matrix)
        
        total_loss = is_focal_plant_loss + initial_fine_cluster_loss + leaf_part_loss + affinity_matrix_loss

        with torch.no_grad():
            is_focal_plant_acc = (torch.argmax(pred_is_focal_plant, dim=1) == is_focal_plant).float().mean()
            leaf_part_acc = (torch.argmax(pred_leaf_part, dim=1) == leaf_part).float().mean()
            initial_fine_cluster_acc = (torch.argmax(pred_initial_fine_cluster, dim=1) == initial_fine_cluster).float().mean()
            # affinity_matrix prediction
            o = (pred_affinity_matrix > 0)
            o = o.int()
            # affinity_matrix_acc = (o == affinity_matrix).float().mean()
            tp = torch.sum((affinity_matrix == 1) & (o == 1))
            pp = o.sum()
            ap = affinity_matrix.sum()
            affinity_matrix_recall = tp.float() / ap.float()
            affinity_matrix_precision = tp.float() / pp.float()
            f1_score = 2 * affinity_matrix_recall * affinity_matrix_precision / (affinity_matrix_recall + affinity_matrix_precision)

        tensorboard_logs = {
            'train_loss': total_loss, 
            'is_focal_plant_loss': is_focal_plant_loss, 
            'leaf_part_loss': leaf_part_loss,
            'leaf_part_acc':leaf_part_acc,
            'initial_fine_cluster_loss': initial_fine_cluster_loss,
            'affinity_matrix_loss': affinity_matrix_loss, 
            'affinity_matrix_f1_score': f1_score,
            'affinity_matrix_recall': affinity_matrix_recall, 
            'affinity_matrix_precision': affinity_matrix_precision,
            'is_focal_plant_acc': is_focal_plant_acc, 
            'initial_fine_cluster_acc': initial_fine_cluster_acc,
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss, 'log': tensorboard_logs}


    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['val_data'],shuff=False)

    def validation_step(self, batch, batch_idx):
        points, is_focal_plant, initial_fine_cluster, leaf_part, affinity_matrix = batch
        pred_is_focal_plant, pred_initial_fine_cluster, pred_leaf_part, pred_affinity_matrix = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        is_focal_plant_loss = critirion(pred_is_focal_plant, is_focal_plant)
        leaf_part_loss = critirion(pred_leaf_part, leaf_part)
        initial_fine_cluster_loss = critirion(pred_initial_fine_cluster, initial_fine_cluster)
        
        critirion2 = FocalLoss(alpha=self.hparams['FL_alpha'], gamma=self.hparams['FL_gamma'], reduce='mean')
        affinity_matrix_loss = critirion2(pred_affinity_matrix, affinity_matrix)
        
        total_loss = is_focal_plant_loss + initial_fine_cluster_loss + leaf_part_loss + affinity_matrix_loss

        is_focal_plant_acc = (torch.argmax(pred_is_focal_plant, dim=1) == is_focal_plant).float().mean()
        leaf_part_acc = (torch.argmax(pred_leaf_part, dim=1) == leaf_part).float().mean()
        initial_fine_cluster_acc = (torch.argmax(pred_initial_fine_cluster, dim=1) == initial_fine_cluster).float().mean()
        o = (pred_affinity_matrix > 0)
        o = o.int()
        # affinity_matrix_acc = (o == affinity_matrix).float().mean()
        tp = torch.sum((affinity_matrix == 1) & (o == 1))
        pp = o.sum()
        ap = affinity_matrix.sum()
        affinity_matrix_recall = tp.float() / ap.float()
        affinity_matrix_precision = tp.float() / pp.float()
        f1_score = 2 * affinity_matrix_recall * affinity_matrix_precision / (affinity_matrix_recall + affinity_matrix_precision)

        tensorboard_logs = {
            'val_loss': total_loss, 
            'val_is_focal_plant_acc': is_focal_plant_acc,
            'val_leaf_part_acc': leaf_part_acc, 
            'val_initial_fine_cluster_acc': initial_fine_cluster_acc, 
            'val_initial_fine_cluster_loss': initial_fine_cluster_loss,
            'val_affinity_matrix_f1_score': f1_score, 
            'val_affinity_matrix_recall': affinity_matrix_recall, 
            'val_affinity_matrix_loss': affinity_matrix_loss, 
            'val_affinity_matrix_precision': affinity_matrix_precision
            }

        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sem_acc = torch.stack([x['val_is_focal_plant_acc'] for x in outputs]).mean()
        leaf_part_acc = torch.stack([x['val_leaf_part_acc'] for x in outputs]).mean()
        lc_acc = torch.stack([x['val_initial_fine_cluster_acc'] for x in outputs]).mean()
        lc_loss = torch.stack([x['val_initial_fine_cluster_loss'] for x in outputs]).mean()
        f1_score = torch.stack([x['val_affinity_matrix_f1_score'] for x in outputs]).mean()
        affinity_matrix_recall = torch.stack([x['val_affinity_matrix_recall'] for x in outputs]).mean()
        affinity_matrix_precision = torch.stack([x['val_affinity_matrix_precision'] for x in outputs]).mean()
        affinity_matrix_loss = torch.stack([x['val_affinity_matrix_loss'] for x in outputs]).mean()
        
        tensorboard_logs = {
            'val_total_loss': avg_loss, 
            'val_is_focal_plant_acc': sem_acc,
            'val_leaf_part_acc': leaf_part_acc, 
            'val_initial_fine_cluster_acc': lc_acc,
            'val_affinity_matrix_f1_score': f1_score, 
            'val_affinity_matrix_recall': affinity_matrix_recall,
            'val_initial_fine_cluster_loss': lc_loss,
            'val_affinity_matrix_loss': affinity_matrix_loss, 
            'val_affinity_matrix_precision': affinity_matrix_precision
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

class SorghumPartNet(pl.LightningModule):
        
    def __init__(self, hparams):
        '''
        Parameters
        ----------
        hparams: hyper parameters
        '''
        super(SorghumPartNet,self).__init__()
        # self._hparams = hparams

        self.hparams.update(hparams)

        if self.hparams['feed_leaf_part'] and self.hparams['feed_is_focal']:
            input_dim = 8
        elif self.hparams['feed_leaf_part']:
            input_dim = 6
        elif self.hparams['feed_is_focal']:
            input_dim = 5
        else:
            input_dim = 3

        self.PN_module = PointNetfeat(global_feat=False,feature_transform=True,feature_space_dim=hparams['feature_space_dim'],input_dim=input_dim,deep=hparams['use_deep'])

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=hparams['lc_count'],
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[hparams['input_channels'], 16, 16, 16],
                      [hparams['input_channels'], 32, 32, 32]],
                use_xyz=hparams['use_xyz'],
            )
        )

        c_out_0 = 16 + 32

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_out_0, 64, 64, 128],
                      [c_out_0, 64, 64, 128]],
                use_xyz=hparams['use_xyz'],
            )
        )

        c_out_1 = 128 + 128

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[64, 256, 64]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_0+c_out_1, 256, 64]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 2, kernel_size=1),
        )
        
        self.lp_fc_layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 3, kernel_size=1),
        )

        self.sharedMLP_layer = build_shared_mlp([64, 64, 32, 1])
        self.dot = ScaledDot(64)
        self.scale = nn.Parameter(torch.tensor(10.0, dtype=torch.float), requires_grad=True)
        #init_threshold = math.cos(math.pi / 9) * self._hparams['ED_weight']  # 20 degree
        #self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float), requires_grad=True)
        self.save_hyperparameters()
        self.knn = None

    def forward(self, xyz):
        
        # if self.knn is None:
        #     point_distance = torch.cdist(xyz,xyz)
        #     _,self.knn = torch.topk(point_distance,self.hparams['k'],largest=False)

        # PointNet SA Module
        l_xyz, l_features, l_s_idx = [xyz], [None], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_s_idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_s_idx.append(li_s_idx)

        # PointNet FP Module
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        # Semantic Label Prediction
        is_focal_plant_pred = self.fc_layer(l_features[0])
        leaf_part_pred = self.lp_fc_layer(l_features[0])

        # PointNet Main Module
        if self.hparams['feed_leaf_part'] and self.hparams['feed_is_focal']:
            points = torch.cat([xyz,leaf_part_pred.transpose(1, 2).contiguous(),is_focal_plant_pred.transpose(1, 2).contiguous()],dim=2)
        elif self.hparams['feed_leaf_part']:
            points = torch.cat([xyz,leaf_part_pred.transpose(1, 2).contiguous()],dim=2)
        elif self.hparams['feed_is_focal']:
            points = torch.cat([xyz,is_focal_plant_pred.transpose(1, 2).contiguous()],dim=2)
        else:
            points = xyz

        pointnet_features = self.PN_module(points)

        return is_focal_plant_pred,leaf_part_pred,pointnet_features

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams['lr_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            lr_clip / self.hparams['lr'],
        )
        bn_lbmd = lambda _: max(
            self.hparams['bn_momentum']
            * self.hparams['bnm_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self,ds_path,shuff=True):
        dataset = TreeDataset(ds_path)
        loader = DataLoader(dataset, batch_size=self.hparams['batch_size'], num_workers=4, shuffle=shuff)
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['train_data'],shuff=True)

    def training_step(self, batch, batch_idx):
        points,is_focal_plant,leaf,leaf_part,leaf_part_full = batch
        pred_is_focal_plant, pred_leaf_part, pred_pointnet_feat = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        is_focal_plant_loss = critirion(pred_is_focal_plant, is_focal_plant)
        leaf_part_loss = critirion(pred_leaf_part, leaf_part)

        criterion_cluster = SpaceSimilarityLoss()
        if 'use_fine_leaf_index' in self.hparams and self.hparams['use_fine_leaf_index']:
            leaf_loss = criterion_cluster(pred_pointnet_feat, leaf_part_full)
        else:
            leaf_loss = criterion_cluster(pred_pointnet_feat, leaf)

        total_loss = is_focal_plant_loss + leaf_loss + leaf_part_loss

        with torch.no_grad():
            is_focal_plant_acc = (torch.argmax(pred_is_focal_plant, dim=1) == is_focal_plant).float().mean()
            leaf_part_acc = (torch.argmax(pred_leaf_part, dim=1) == leaf_part).float().mean()
            

        tensorboard_logs = {
            'train_total_loss': total_loss, 
            'train_is_focal_plant_loss': is_focal_plant_loss, 
            'train_leaf_part_loss': leaf_part_loss,
            'train_leaf_loss': leaf_loss,
            'train_leaf_part_acc':leaf_part_acc,
            'train_is_focal_plant_acc': is_focal_plant_acc,
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss, 'log': tensorboard_logs}


    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['val_data'],shuff=False)

    def validation_step(self, batch, batch_idx):
        points, is_focal_plant, leaf, leaf_part, leaf_part_full = batch
        pred_is_focal_plant, pred_leaf_part,pred_pointnet_feat  = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        is_focal_plant_loss = critirion(pred_is_focal_plant, is_focal_plant)
        leaf_part_loss = critirion(pred_leaf_part, leaf_part)
        
        criterion_cluster = SpaceSimilarityLoss()
        if 'use_fine_leaf_index' in self.hparams and self.hparams['use_fine_leaf_index']:
            leaf_loss = criterion_cluster(pred_pointnet_feat, leaf_part_full)
        else:
            leaf_loss = criterion_cluster(pred_pointnet_feat, leaf)
        
        total_loss = is_focal_plant_loss + leaf_loss + leaf_part_loss

        is_focal_plant_acc = (torch.argmax(pred_is_focal_plant, dim=1) == is_focal_plant).float().mean()
        leaf_part_acc = (torch.argmax(pred_leaf_part, dim=1) == leaf_part).float().mean()

        tensorboard_logs = {
            'val_total_loss': total_loss, 
            'val_is_focal_plant_loss': is_focal_plant_loss,
            'val_leaf_part_loss': leaf_part_loss,
            'val_leaf_loss': leaf_loss,
            'val_leaf_part_acc': leaf_part_acc, 
            'val_is_focal_plant_acc': is_focal_plant_acc,
            }

        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        val_total_loss = torch.stack([x['val_total_loss'] for x in outputs]).mean()
        is_focal_plant_loss = torch.stack([x['val_is_focal_plant_loss'] for x in outputs]).mean()
        leaf_part_loss = torch.stack([x['val_leaf_part_loss'] for x in outputs]).mean()
        val_leaf_loss = torch.stack([x['val_leaf_loss'] for x in outputs]).mean()
        leaf_part_acc = torch.stack([x['val_leaf_part_acc'] for x in outputs]).mean()
        is_focal_plant_acc = torch.stack([x['val_is_focal_plant_acc'] for x in outputs]).mean()

        tensorboard_logs = {
            'val_total_loss': val_total_loss, 
            'val_is_focal_plant_loss': is_focal_plant_loss,
            'val_leaf_part_loss': leaf_part_loss,
            'val_leaf_loss': val_leaf_loss,
            'val_leaf_part_acc': leaf_part_acc, 
            'val_is_focal_plant_acc': is_focal_plant_acc,
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': val_total_loss, 'log': tensorboard_logs}

class SorghumPartNetDGCNN(pl.LightningModule):
        
    def __init__(self, hparams):
        '''
        Parameters
        ----------
        hparams: hyper parameters
        '''
        super(SorghumPartNetDGCNN,self).__init__()
        # self._hparams = hparams

        self.hparams.update(hparams)

        if self.hparams['feed_leaf_part'] and self.hparams['feed_is_focal']:
            input_dim = 8
        elif self.hparams['feed_leaf_part']:
            input_dim = 6
        elif self.hparams['feed_is_focal']:
            input_dim = 5
        else:
            input_dim = 3

        MyStruct = namedtuple('args', 'k')
        args = MyStruct(k=15)

        self.DGCNN_module = DGCNN_cls(args)

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=hparams['lc_count'],
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[hparams['input_channels'], 16, 16, 16],
                      [hparams['input_channels'], 32, 32, 32]],
                use_xyz=hparams['use_xyz'],
            )
        )

        c_out_0 = 16 + 32

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_out_0, 64, 64, 128],
                      [c_out_0, 64, 64, 128]],
                use_xyz=hparams['use_xyz'],
            )
        )

        c_out_1 = 128 + 128

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[64, 256, 64]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_0+c_out_1, 256, 64]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 2, kernel_size=1),
        )
        
        self.lp_fc_layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 3, kernel_size=1),
        )

        self.sharedMLP_layer = build_shared_mlp([64, 64, 32, 1])
        self.dot = ScaledDot(64)
        self.scale = nn.Parameter(torch.tensor(10.0, dtype=torch.float), requires_grad=True)
        #init_threshold = math.cos(math.pi / 9) * self._hparams['ED_weight']  # 20 degree
        #self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float), requires_grad=True)
        self.save_hyperparameters()
        self.knn = None

    def forward(self, xyz):
        
        # if self.knn is None:
        #     point_distance = torch.cdist(xyz,xyz)
        #     _,self.knn = torch.topk(point_distance,self.hparams['k'],largest=False)

        # PointNet SA Module
        l_xyz, l_features, l_s_idx = [xyz], [None], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_s_idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_s_idx.append(li_s_idx)

        # PointNet FP Module
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        # Semantic Label Prediction
        is_focal_plant_pred = self.fc_layer(l_features[0])
        leaf_part_pred = self.lp_fc_layer(l_features[0])

        # PointNet Main Module
        if self.hparams['feed_leaf_part'] and self.hparams['feed_is_focal']:
            points = torch.cat([xyz,leaf_part_pred.transpose(1, 2).contiguous(),is_focal_plant_pred.transpose(1, 2).contiguous()],dim=2)
        elif self.hparams['feed_leaf_part']:
            points = torch.cat([xyz,leaf_part_pred.transpose(1, 2).contiguous()],dim=2)
        elif self.hparams['feed_is_focal']:
            points = torch.cat([xyz,is_focal_plant_pred.transpose(1, 2).contiguous()],dim=2)
        else:
            points = xyz

        dgcnn_features = self.DGCNN_module(points)

        # # PointNet Main Module
        # is_focal_plant_pred_indices = F.softmax(is_focal_plant_pred,dim=1)
        # is_focal_indices = (is_focal_plant_pred_indices[0,:]>is_focal_plant_pred_indices[1,:])
        
        # if not torch.any(is_focal_indices):
        #     is_focal_plant_pred_indices[0] = 1
        #     is_focal_indices = (is_focal_plant_pred_indices[0,:]>is_focal_plant_pred_indices[1,:])

        
        # points = xyz[:,is_focal_indices[1],:]

        # print(xyz.shape)
        # print(points.shape)

        # dgcnn_features = self.PN_module(points)

        return is_focal_plant_pred,leaf_part_pred,dgcnn_features

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams['lr_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            lr_clip / self.hparams['lr'],
        )
        bn_lbmd = lambda _: max(
            self.hparams['bn_momentum']
            * self.hparams['bnm_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self,ds_path,shuff=True):
        dataset = TreeDataset(ds_path)
        loader = DataLoader(dataset, batch_size=self.hparams['batch_size'], num_workers=4, shuffle=shuff)
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['train_data'],shuff=True)

    def training_step(self, batch, batch_idx):
        points,is_focal_plant,leaf,leaf_part,leaf_part_full = batch
        pred_is_focal_plant, pred_leaf_part, pred_pointnet_feat = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        is_focal_plant_loss = critirion(pred_is_focal_plant, is_focal_plant)
        leaf_part_loss = critirion(pred_leaf_part, leaf_part)

        criterion_cluster = SpaceSimilarityLossV2()
        if 'use_fine_leaf_index' in self.hparams and self.hparams['use_fine_leaf_index']:
            leaf_loss = criterion_cluster(pred_pointnet_feat, leaf_part_full)
        else:
            leaf_loss = criterion_cluster(pred_pointnet_feat, leaf)

        total_loss = is_focal_plant_loss + leaf_loss + leaf_part_loss

        with torch.no_grad():
            is_focal_plant_acc = (torch.argmax(pred_is_focal_plant, dim=1) == is_focal_plant).float().mean()
            leaf_part_acc = (torch.argmax(pred_leaf_part, dim=1) == leaf_part).float().mean()
            

        tensorboard_logs = {
            'train_total_loss': total_loss, 
            'train_is_focal_plant_loss': is_focal_plant_loss, 
            'train_leaf_part_loss': leaf_part_loss,
            'train_leaf_loss': leaf_loss,
            'train_leaf_part_acc':leaf_part_acc,
            'train_is_focal_plant_acc': is_focal_plant_acc,
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss, 'log': tensorboard_logs}


    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['val_data'],shuff=False)

    def validation_step(self, batch, batch_idx):
        points, is_focal_plant, leaf, leaf_part, leaf_part_full = batch
        pred_is_focal_plant, pred_leaf_part,pred_pointnet_feat  = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        is_focal_plant_loss = critirion(pred_is_focal_plant, is_focal_plant)
        leaf_part_loss = critirion(pred_leaf_part, leaf_part)
        
        criterion_cluster = SpaceSimilarityLossV2()
        if 'use_fine_leaf_index' in self.hparams and self.hparams['use_fine_leaf_index']:
            leaf_loss = criterion_cluster(pred_pointnet_feat, leaf_part_full)
        else:
            leaf_loss = criterion_cluster(pred_pointnet_feat, leaf)
        
        total_loss = is_focal_plant_loss + leaf_loss + leaf_part_loss

        is_focal_plant_acc = (torch.argmax(pred_is_focal_plant, dim=1) == is_focal_plant).float().mean()
        leaf_part_acc = (torch.argmax(pred_leaf_part, dim=1) == leaf_part).float().mean()

        tensorboard_logs = {
            'val_total_loss': total_loss, 
            'val_is_focal_plant_loss': is_focal_plant_loss,
            'val_leaf_part_loss': leaf_part_loss,
            'val_leaf_loss': leaf_loss,
            'val_leaf_part_acc': leaf_part_acc, 
            'val_is_focal_plant_acc': is_focal_plant_acc,
            }

        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        val_total_loss = torch.stack([x['val_total_loss'] for x in outputs]).mean()
        is_focal_plant_loss = torch.stack([x['val_is_focal_plant_loss'] for x in outputs]).mean()
        leaf_part_loss = torch.stack([x['val_leaf_part_loss'] for x in outputs]).mean()
        val_leaf_loss = torch.stack([x['val_leaf_loss'] for x in outputs]).mean()
        leaf_part_acc = torch.stack([x['val_leaf_part_acc'] for x in outputs]).mean()
        is_focal_plant_acc = torch.stack([x['val_is_focal_plant_acc'] for x in outputs]).mean()

        tensorboard_logs = {
            'val_total_loss': val_total_loss, 
            'val_is_focal_plant_loss': is_focal_plant_loss,
            'val_leaf_part_loss': leaf_part_loss,
            'val_leaf_loss': val_leaf_loss,
            'val_leaf_part_acc': leaf_part_acc, 
            'val_is_focal_plant_acc': is_focal_plant_acc,
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': val_total_loss, 'log': tensorboard_logs}

class SorghumPartNetGroundDGCNN(pl.LightningModule):
        
    def __init__(self, hparams):
        '''
        Parameters
        ----------
        hparams: hyper parameters
        '''
        super(SorghumPartNetGroundDGCNN,self).__init__()
        # self._hparams = hparams

        self.hparams.update(hparams)

        MyStruct = namedtuple('args', 'k')
        args = MyStruct(k=15)

        self.DGCNN_module = DGCNN_cls(args)

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=hparams['lc_count'],
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[hparams['input_channels'], 16, 16, 16],
                      [hparams['input_channels'], 32, 32, 32]],
                use_xyz=hparams['use_xyz'],
            )
        )

        c_out_0 = 16 + 32

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_out_0, 64, 64, 128],
                      [c_out_0, 64, 64, 128]],
                use_xyz=hparams['use_xyz'],
            )
        )

        c_out_1 = 128 + 128

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[64, 256, 64]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_0+c_out_1, 256, 64]))

        self.semantic_fc_layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 3, kernel_size=1),
        )

        self.save_hyperparameters()
        self.knn = None

    def forward(self, xyz, input_semantic_label=None):
        
        # Normalization (min max)
        mins,_ = torch.min(xyz,axis=1)
        maxs,_ = torch.max(xyz,axis=1)
        xyz = (xyz-mins)/(maxs-mins) - 0.5

        # PointNet SA Module
        l_xyz, l_features, l_s_idx = [xyz], [None], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_s_idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_s_idx.append(li_s_idx)

        # PointNet FP Module
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        # Semantic Label Prediction
        semantic_label_pred = self.semantic_fc_layer(l_features[0])
        
        if input_semantic_label is None:
            input_semantic_label = torch.clone(semantic_label_pred)

            input_semantic_label = F.softmax(input_semantic_label,dim=1)
            input_semantic_label = input_semantic_label[0].cpu().detach().numpy().T
            input_semantic_label = np.argmax(input_semantic_label,1)
            
            # input_semantic_label[input_semantic_label == 2] = 0
            # input_semantic_label[input_semantic_label == 1] = 3
            # input_semantic_label[input_semantic_label == 0] = 1
            # input_semantic_label[input_semantic_label == 3] = 0
            
            points = xyz[:,input_semantic_label==1,:]

        else:
            if len(input_semantic_label.shape) == 1:
                points = xyz[:,input_semantic_label==1,:]
            else:
                points = xyz[:,input_semantic_label[0]==1,:]
        
        dgcnn_features = self.DGCNN_module(points)

        return semantic_label_pred,dgcnn_features

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams['lr_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            lr_clip / self.hparams['lr'],
        )
        bn_lbmd = lambda _: max(
            self.hparams['bn_momentum']
            * self.hparams['bnm_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self,ds_path,shuff=True):
        dataset = SorghumDataset(ds_path)
        loader = DataLoader(dataset, batch_size=self.hparams['batch_size'], num_workers=4, shuffle=shuff)
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['train_data'],shuff=True)

    def training_step(self, batch, batch_idx):
        points,_,semantic_label,_,leaf = batch
        
        leaf = leaf[:,semantic_label[0]==1]
        
        pred_semantic_label, pred_leaf_features = self(points,semantic_label)

        critirion = torch.nn.CrossEntropyLoss()
        semantic_label_loss = critirion(pred_semantic_label, semantic_label)

        criterion_cluster = SpaceSimilarityLossV2()
        leaf_loss = criterion_cluster(pred_leaf_features, leaf)

        total_loss = semantic_label_loss + leaf_loss

        leaf_metrics = LeafMetrics(self.hparams['leaf_space_threshold'])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features,leaf)
        
        with torch.no_grad():
            semantic_label_acc = (torch.argmax(pred_semantic_label, dim=1) == semantic_label).float().mean()
        
        # print(Acc,Prec,Rec,F)
        
        tensorboard_logs = {
            'train_total_loss': total_loss, 
            'train_semantic_label_loss': semantic_label_loss, 
            'train_leaf_loss': leaf_loss,
            'train_semantic_label_acc': semantic_label_acc,
            'train_leaf_accuracy': Acc,
            'train_leaf_precision': Prec,
            'train_leaf_recall': Rec,
            'train_leaf_f1': F
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss, 'log': tensorboard_logs}


    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['val_data'],shuff=False)

    def validation_step(self, batch, batch_idx):
        points,_,semantic_label,_,leaf = batch
        leaf = leaf[:,semantic_label[0]==1]

        pred_semantic_label, pred_leaf_features = self(points,semantic_label)

        critirion = torch.nn.CrossEntropyLoss()
        semantic_label_loss = critirion(pred_semantic_label, semantic_label)

        criterion_cluster = SpaceSimilarityLossV2()
        leaf_loss = criterion_cluster(pred_leaf_features, leaf)

        total_loss = semantic_label_loss + leaf_loss

        leaf_metrics = LeafMetrics(self.hparams['leaf_space_threshold'])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features,leaf)

        semantic_label_acc = (torch.argmax(pred_semantic_label, dim=1) == semantic_label).float().mean()
        
        tensorboard_logs = {
            'val_total_loss': total_loss, 
            'val_semantic_label_loss': semantic_label_loss,
            'val_leaf_loss': leaf_loss,
            'val_semantic_label_acc': semantic_label_acc,
            'val_leaf_accuracy': Acc,
            'val_leaf_precision': Prec,
            'val_leaf_recall': Rec,
            'val_leaf_f1': F
            }

        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        val_total_loss = torch.stack([x['val_total_loss'] for x in outputs]).mean()
        semantic_label_loss = torch.stack([x['val_semantic_label_loss'] for x in outputs]).mean()
        val_leaf_loss = torch.stack([x['val_leaf_loss'] for x in outputs]).mean()
        semantic_label_acc = torch.stack([x['val_semantic_label_acc'] for x in outputs]).mean()
        Acc = torch.stack([x['val_leaf_accuracy'] for x in outputs]).mean()
        Prec = torch.stack([x['val_leaf_precision'] for x in outputs]).mean()
        Rec = torch.stack([x['val_leaf_recall'] for x in outputs]).mean()
        F = torch.stack([x['val_leaf_f1'] for x in outputs]).mean()

        tensorboard_logs = {
            'val_total_loss': val_total_loss, 
            'val_semantic_label_loss': semantic_label_loss,
            'val_leaf_loss': val_leaf_loss,
            'val_semantic_label_acc': semantic_label_acc,
            'val_leaf_accuracy': Acc,
            'val_leaf_precision': Prec,
            'val_leaf_recall': Rec,
            'val_leaf_f1': F
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': val_total_loss, 'log': tensorboard_logs}


if __name__ == "__main__":
    #device = torch.device('cuda:0')
    net = TreePartNet(output_lc=256)
    #print(net)
    train_dataloader = net.train_dataloader()
    net = net.cuda()
    i = 0
    for points, leaf_indices, leaf_part_indices in train_dataloader:
        i = i + 1
        '''transform_p = points.float()
        out,sem,idxs=net(transform_p.cuda())
        print(out.size())
        print(sem.size())
        print(idxs.size())'''

        pred_isfork, pred_lcl, fn_pred, idx = net(points.float().cuda())
        # critirion = torch.nn.CrossEntropyLoss()
        # sem_loss = critirion(pred_isfork, isfork.cuda())
        # lc_loss = critirion(pred_lcl, lcl.cuda())
        # total_loss = sem_loss + lc_loss
        # print(total_loss.cpu())
        if i == 1:
            break
