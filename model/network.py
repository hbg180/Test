import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('model')
from backbone import ExtractorF, ExtractorC
from corr import CorrBlock
from aggregate import MotionFeatureEncoder, MPA
from update import UpdateBlock
from util import coords_grid
import matplotlib.pyplot as plt
from encoders import twins_svt_large
from configs.mvsec import get_cfg
# import flow_viz
from sk_update import SKMotionEncoder6_Deep_nopool_res, SKUpdateBlock6_Deep_nopoolres_AllDecoder
from cbam import CBAM


class TMA(nn.Module):
    def __init__(self, cfg, input_bins=15):
        super(TMA, self).__init__()
        self.cfg = cfg

        f_channel = 128
        self.split = 5
        # self.corr_level = 1
        # self.corr_radius = 3

        if cfg.fnet == 'twins':
            print("[Using twins as feature encoder]")
            self.fnet = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.fnet == 'basic':
            print("[Using basicencoder as feature encoder]")
            self.fnet = ExtractorF(input_channel=input_bins // self.split, outchannel=f_channel, norm='IN')

        if cfg.cnet == 'twins':
            print("[Using twins as context encoder]")
            self.cnet = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basic':
            print("[Using basicencoder as context encoder]")
            self.cnet = ExtractorC(input_channel=input_bins // self.split + input_bins, outchannel=256, norm='BN')

        if cfg.mfe == 'sk':
            print("[Using SKMotionEncoder as motion feature encoder]")
            self.mfe = SKMotionEncoder6_Deep_nopool_res(args=cfg)
        elif cfg.mfe == 'basic':
            print("[Using basic as motion feature encoder]")
            self.mfe = MotionFeatureEncoder(cfg.corr_levels, cfg.corr_radius)
        self.mpa = MPA(d_model=128)  # Motion Pattern Aggregation

        if cfg.updater == 'sk':
            print("[Using SKUpdater as motion feature encoder]")
            self.update = SKUpdateBlock6_Deep_nopoolres_AllDecoder(args=cfg, hidden_dim=128, split=self.split)
        elif cfg.updater == 'basic':
            print("[Using basic as motion feature encoder]")
            self.update = UpdateBlock(hidden_dim=128, split=self.split)

        if cfg.cbam:
            self.cbam = CBAM(256,2,3)

    def upsample_flow(self, flow, mask, scale=8):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(scale * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale * H, scale * W)

    def forward(self, x1, x2, iters=6):
        b, _, h, w = x2.shape

        # Feature maps [f_0 :: f_i :: f_g]
        voxels = x2.chunk(self.split, dim=1)
        voxelref = x1.chunk(self.split, dim=1)[-1]
        voxels = (voxelref,) + voxels  # [group+1] elements
        fmaps = self.fnet(voxels)  # Tuple(f0, f1, ..., f_g)    # 6*(2,3,288,384)->6*(2,128,36,48)

        # Context map [net, inp]
        # voxels = torch.cat([fm for fm in voxels], dim=0)
        cmap = self.cnet(torch.cat(voxels, dim=1))  # 6*(2,3,288,384)->[2,18,288,384]->[2,256,36,48]
        if self.cbam:
            cmap = self.cbam(cmap)
        net, inp = torch.split(cmap, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0 = coords_grid(b, h // 8, w // 8, device=cmap.device)
        coords1 = coords_grid(b, h // 8, w // 8, device=cmap.device)

        # MidCorr
        corr_fn_list = []
        for i in range(self.split):
            corr_fn = CorrBlock(fmaps[0], fmaps[i + 1], num_levels=self.cfg.corr_levels,
                                radius=self.cfg.corr_radius)  # [c01,c02,...,c05]
            corr_fn_list.append(corr_fn)

        flow_predictions = []
        for iter in range(iters):

            coords1 = coords1.detach()
            flow = coords1 - coords0    # 光流

            corr_map_list = []
            du = flow / self.split
            for i in range(self.split):
                coords = (coords0 + du * (i + 1)).detach()
                corr_map = corr_fn_list[i](coords)  # index correlation volume
                corr_map_list.append(corr_map)

            corr_maps = torch.cat(corr_map_list, dim=0)

            mfs = self.mfe(torch.cat([flow] * self.split, dim=0), corr_maps)    #[10,128,36,48]
            mfs = mfs.chunk(self.split, dim=0)  #[10,128,36,48]->[5*(2,128,36,48)]
            mfs = self.mpa(mfs)  # 公式（8）MFi^
            mf = torch.cat(mfs, dim=1)  # 5[2,128,36,48]->[2,640,36,48]
            net, dflow, upmask = self.update(net, inp, mf)  # [2,128,36,48],[2,128,36,48],[2,640,36,48]->[2,128,36,48],[2,2,36,48],[2,576,36,48]
            coords1 = coords1 + dflow

            if self.training:
                flow_up = self.upsample_flow(coords1 - coords0, upmask)
                flow_predictions.append(flow_up)

        # dataset = MpiSintel(aug_params=None, split='training', root='E:/Git/Papers/Datasets/Sintel',
        #                     dstype='clean')
        # plt.subplot(3, 1, 1)
        # image = plt.imread(dataset.image_list[img_id][0])
        # plt.imshow(image)
        # h, w = corr_fn.corr_pyramid[0].shape[2], corr_fn.corr_pyramid[0].shape[3]
        # plt.subplot(2, 1, 1)
        # corr = corr_fn.corr_pyramid[0].view(h, w, -1)
        # values, indexes = corr.max(dim=2)
        # plt.imshow(values.cpu().detach().numpy(), vmin=torch.min(corr), vmax=torch.max(corr))
        # plt.subplot(2, 1, 2)
        # flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
        # flo = flow_viz.flow_to_image(flo)  # map flow to rgb image
        # plt.imshow(flo[:, :, [2, 1, 0]] / 255.0)
        # # plt.suptitle(f'{dataset.extra_info[img_id]}')
        # plt.show(block=True)
        # keyboard.wait('enter')

        if self.training:
            return flow_predictions
        else:
            return self.upsample_flow(coords1 - coords0, upmask)


if __name__ == '__main__':
    input1 = torch.rand(2, 15, 288, 384)
    input2 = torch.rand(2, 15, 288, 384)
    model = TMA(get_cfg(), input_bins=15)
    # model.cuda()
    model.train()
    preds = model(input1, input2)
    print(len(preds))
    model.eval()
    pred = model(input1, input2)
    print(pred.shape)
