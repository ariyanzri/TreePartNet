import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, feature_space_dim=64, input_dim=3, deep=False):
        super(PointNetfeat, self).__init__()
        
        self.use_deep_net = deep
        self.feature_space_dim = feature_space_dim

        self.stn = STN3d()
        
        self.conv1 = torch.nn.Conv1d(input_dim, feature_space_dim, 1)
        
        if feature_space_dim % 8 == 0:
            self.conv1_1 = torch.nn.Conv1d(input_dim, int(feature_space_dim/8), 1).to('cuda:1')
            self.bn1_1  = nn.BatchNorm1d(int(feature_space_dim/8)).to('cuda:1')
            self.conv1_2 = torch.nn.Conv1d(int(feature_space_dim/8), int(feature_space_dim/4), 1).to('cuda:1')
            self.bn1_2 = nn.BatchNorm1d(int(feature_space_dim/4)).to('cuda:1')
            self.conv1_3 = torch.nn.Conv1d(int(feature_space_dim/4), int(feature_space_dim/2), 1).to('cuda:1')
            self.bn1_3 = nn.BatchNorm1d(int(feature_space_dim/2)).to('cuda:1')
            self.conv1_4 = torch.nn.Conv1d(int(feature_space_dim/2), feature_space_dim, 1).to('cuda:1')
            self.bn1_4 = nn.BatchNorm1d(int(feature_space_dim)).to('cuda:1')

        self.conv2 = torch.nn.Conv1d(feature_space_dim, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 1024, 1)
        self.bn1 = nn.BatchNorm1d(feature_space_dim)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=feature_space_dim)

    def forward(self, x):
        feature = x[:,:,3:]
        x = x[:,:,:3]
        x = x.transpose(1, 2).contiguous()
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = torch.cat([x,feature.transpose(1, 2).contiguous()],dim=1)
        
        if not self.use_deep_net or self.feature_space_dim % 8 != 0:
            x = F.relu(self.bn1(self.conv1(x)))
        else:
            x = F.relu(self.bn1_1(self.conv1_1(x)))
            x = F.relu(self.bn1_2(self.conv1_2(x)))
            x = F.relu(self.bn1_3(self.conv1_3(x)))
            x = F.relu(self.bn1_4(self.conv1_4(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            x = x.transpose(1, 2).contiguous()
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            # x = torch.cat([x, pointfeat], 1)
            x = pointfeat
            x = x.transpose(1, 2).contiguous()
            return x
