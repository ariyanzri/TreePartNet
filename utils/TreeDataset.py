from lib2to3.pytree import LeafPattern
import os
from random import random
import torch
import numpy as np
import torch.utils.data as data
import h5py
from pointnet2_ops import pointnet2_utils
import random

class TreeDataset(data.Dataset):
    def __init__(self, h5_filename):
        super().__init__()
        self.h5_filename=h5_filename
        self.length = -1

    def __getitem__(self, index):
        f = h5py.File(self.h5_filename,'r')
        points = f['points'][index]
        is_focal_plant = f['is_focal_plant'][index]
        leaf_index = f['leaf_index'][index]
        leaf_part_index = f['leaf_part_index'][index]
        leaf_part_full_index = f['leaf_part_full_index'][index]
        
        f.close()
        
        return torch.from_numpy(points).float(),\
            torch.from_numpy(is_focal_plant).type(torch.LongTensor),\
            torch.from_numpy(leaf_index).type(torch.LongTensor),\
            torch.from_numpy(leaf_part_index).type(torch.LongTensor),\
            torch.from_numpy(leaf_part_full_index).type(torch.LongTensor)

    def __len__(self):
        if self.length != -1:
            return self.length
        else:
            f = h5py.File(self.h5_filename,'r')
            self.length=len(f['names'])
            f.close()
            return self.length

    def get_name(self,index):
        f = h5py.File(self.h5_filename,'r')
        name = f['names'][index].decode("utf-8")
        return name

    def save_ply(self, xyz, cls, fn):
        with open(fn, 'w') as f:
            pn = xyz.shape[0]
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (pn))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar pid\n')
            f.write('end_header\n')
            for i in range(pn):
                f.write('%f %f %f %d\n' % (xyz[i][0], xyz[i][1], xyz[i][2], cls[i]))

class FoliageDataset(data.Dataset):

    def __init__(self, h5_filename):
        super().__init__()
        self.h5_filename=h5_filename
        self.length = -1

    def __len__(self):
        if self.length != -1:
            return self.length
        else:
            f = h5py.File(self.h5_filename,'r')
            self.length=len(f['names'])
            f.close()
            return self.length

    def __getitem__(self,item):
        f = h5py.File(self.h5_filename, 'r')
        points = f['points'][item]
        isfoliage = f['isfoliage'][item]
        f.close()
        return torch.from_numpy(points).float(), torch.from_numpy(isfoliage)

if __name__ == "__main__":
    ds = TreeDataset('/app/ImplicitCylinders/utils/tree_test.hdf5')
    p,i,pi=ds[3]
    samples=pointnet2_utils.furthest_point_sample(torch.Tensor(np.expand_dims(p,0)).cuda(), 256)
    pts = samples[0].cpu()
    xyz = p[pts.long()]
    sz = i.shape[0]
    cl = []
    for i in range(sz):
        xyzi=p[i]
        xyzi=xyzi.repeat(256,1)
        dis = torch.nn.functional.pairwise_distance(xyzi.cuda(),xyz.cuda(),p=2)
        cls = torch.argmin(dis)
        cl.append(cls)
    ds.save_ply(xyz=p, cls=cl,fn='fps.ply')
    #sa = pi[9]
    #sa = sa.repeat(256)
    #print(pts)
