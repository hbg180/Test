import argparse
import torch
from torch import nn, einsum
import torch.nn.functional as F
# from gma import Aggregate
from einops import rearrange


class PCBlock4_Deep_nopool_res(nn.Module):  # SKBlock   [1,3,288,512]->[1, 16, 288, 512]
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList([  # 辅助卷积S×S 大卷积L×L
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)  # point-wise convolution 点卷积,图2(d)中的PW
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:  #####################
            x = F.gelu(x + conv(x))  # Super kernel module
        x = F.gelu(x + self.pw(x))  #####################
        x = self.ffn2(x)
        return x


class SKMotionEncoder6_Deep_nopool_res(nn.Module):  # SKMotionEncoder   [10,2,36,48] [10,49,36,48]->[10,128,36,48]
    def __init__(self, args):
        super().__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2  # 4*81
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 256, k_conv=args.k_conv)  # k_conv [1,15]
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)

        self.convf1 = nn.Conv2d(2, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64 + 192, 128 - 2, k_conv=args.k_conv)

    def forward(self, flow, corr):  # [10,2,36,48] [10,49,36,48]
        cor = F.gelu(self.convc1(corr))  # 公式（8）SKBlock(c)   [10,49,36,48]->[10,256,36,48]

        cor = self.convc2(cor)  # 公式（8）c'   [10,256,36,48]->[10,192,36,48]

        flo = self.convf1(flow)  # 公式（9）SKBlock(f) [10,2,36,48]->[10,128,36,48]
        flo = self.convf2(flo)  # 公式（9）f'   [10,128,36,48]->[10,64,36,48]

        cor_flo = torch.cat([cor, flo], dim=1)  # 公式（10）Concat(c',f')
        out = self.conv(cor_flo)  # 公式（10）SKBlock(Concat(c',f'))

        return torch.cat([out, flow], dim=1)  # 公式（10）o


class SKUpdateBlock6_Deep_nopoolres_AllDecoder(nn.Module):  # SKUpdater
    def __init__(self, args, hidden_dim, split=5):
        super().__init__()
        self.args = args
        # self.encoder = SKMotionEncoder6_Deep_nopool_res(args)   # Super kernel motion encoder
        self.gru = PCBlock4_Deep_nopool_res(hidden_dim*2+128*split, 128,
                                            k_conv=args.PCUpdater_conv)  # SKBlock
        self.flow_head = PCBlock4_Deep_nopool_res(128, 2, k_conv=args.k_conv)  # SKBlock

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

        # self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=self.args.num_heads)   # GMA-Aggregator 运动聚合

    def forward(self, net, inp, mf):
        """
        :param net: GRU隐藏状态
        :param inp:
        :param mf: 运动聚合特征
        :return:
        """
        # motion_features = self.encoder(flow, corr)      # motion_features公式（11）o
        # motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, mf], dim=1)  # 公式（11）Concat(xm,xc,xg)

        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))  # net表示隐藏状态

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, delta_flow, mask


# 来自GMA的运动聚合函数
class Aggregate(nn.Module):
    def __init__(
            self,
            args,
            dim,
            heads=4,
            dim_head=128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TMA')
    # training setting
    parser.add_argument('--k_conv', type=int, nargs='+', default=[1, 15])  # 200000
    parser.add_argument('--corr_levels', type=int, default=1)  # 200000
    parser.add_argument('--corr_radius', type=int, default=3)  # 200000
    parser.add_argument('--PCUpdater_conv', type=int, nargs='+', default=[1, 7])
    args = parser.parse_args()
    net = torch.randn([2, 128, 36, 48])
    inp = torch.randn([2, 128, 36, 48])
    mf = torch.randn([2, 640, 36, 48])
    model = SKUpdateBlock6_Deep_nopoolres_AllDecoder(args, 128, 5)
    model.train()
    preds = model(net, inp, mf)
    print(len(preds))
    model.eval()
    pred = model(net, inp, mf)
    print(pred.shape)

    # x = torch.randn([3, 16, 288, 512])
    # # pcblock = PCBlock4_Deep_nopool_res(x.shape[1], 16, [1, 15])
    # pcblock = SKMotionEncoder6_Deep_nopool_res(args)
    # x = pcblock(x)
    # print(x.shape)

    # input1 = torch.rand(2, 15, 288, 384)
    # input2 = torch.rand(2, 15, 288, 384)
    # input1 = torch.rand(10, 2, 36, 48)
    # input2 = torch.rand(10, 49, 36, 48)
    # model = SKMotionEncoder6_Deep_nopool_res(args)
    # # model.cuda()
    # model.train()
    # preds = model(input1, input2)
    # print(len(preds))
    # model.eval()
    # pred = model(input1, input2)
    # print(pred.shape)
