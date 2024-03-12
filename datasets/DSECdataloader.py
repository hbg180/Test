import numpy as np
import torch
import torch.utils.data as data

import random
import os
import glob

from datasets.augument import Augumentor

class DSECdataset(data.Dataset):
    def __init__(self, args, augument=True):
        super(DSECdataset, self).__init__()
        self.init_seed = False
        self.files = []
        self.flows = []

        self.root = args.root
        self.augment = augument     # 数据增强
        if self.augment:
            self.augmentor = Augumentor(crop_size=[288, 384])
        
        self.files = glob.glob(os.path.join(self.root, 'train', '*', 'seq_*.npz'))  # 事件体:voxel_prev, voxel_curr
        self.files.sort()   # 对文件目录排序
        self.flows = glob.glob(os.path.join(self.root, 'train', '*', 'seq_*.npy'))  # optical flow
        self.flows.sort()

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()    # 返回各种有用的信息(包括工作ID,数据集副本,初始种子等)
            if worker_info is not None:
                torch.manual_seed(worker_info.id)   # 设置CPU生成随机数的种子，方便下次复现实验结果。
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        
        voxel_file = np.load(self.files[index])     # .npz文件
        voxel1 = voxel_file['voxel_prev'].transpose([1,2,0])    # [Bins,H,W]->[H,W,Bins]
        voxel2 = voxel_file['voxel_curr'].transpose([1,2,0])


        flow_16bit = np.load(self.flows[index])     # ground truth flow .npy文件
        flow_map, valid2D = flow_16bit_to_float(flow_16bit)

        voxel1, voxel2, flow_map, valid2D = self.augmentor(voxel1, voxel2, flow_map, valid2D)   # 数据增强
        
        voxel1 = torch.from_numpy(voxel1).permute(2, 0, 1).float()  # [h,w,Bin]->[Bin,h,w]
        voxel2 = torch.from_numpy(voxel2).permute(2, 0, 1).float()
        flow_map = torch.from_numpy(flow_map).permute(2, 0, 1).float()  # [h,w,2]->[2,h,w]
        valid2D = torch.from_numpy(valid2D).float()
        return voxel1, voxel2, flow_map, valid2D
    
    def __len__(self):
        return len(self.files)
    
def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1   # [...,2]表示前面的维度全选，最后一纬度选择2
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)    # all()判断整个数组中的元素是否全部满足条件，如果满足条件返回True
    valid_map = np.where(valid2D)   # 有效光流的坐标

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128   # x方向光流
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128   # y方向光流
    # flow_map[valid_map[0],valid_map[1],0]表示有效像素的x方向光流
    return flow_map, valid2D


def make_data_loader(args, batch_size, num_workers):
    dset = DSECdataset(args)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True)     # 当样本数不能被batch_size整除时，最后一批数据是否舍弃
    return loader
