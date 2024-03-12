from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix = ''
_CN.cnet = 'basic'
_CN.fnet = 'basic'
_CN.k_conv = [1, 15]
_CN.corr_levels = 1
_CN.corr_radius = 3
_CN.PCUpdater_conv = [1, 7]
_CN.updater = 'sk'

#training setting
_CN.num_steps = 100
_CN.checkpoint_dir = 'ckpts/'
_CN.lr = 2e-4

#dataset setting
# _CN.root = 'E:/Git/Papers/TMA/dsec/'
_CN.root = 'dsec/'
_CN.crop_size = [288, 384]

#dataloader setting
_CN.batch_size = 1
_CN.num_workers = 1

#model setting
_CN.grad_clip = 1

#loss setting
_CN.weight = 0.8

#wandb setting
_CN.wandb = 'false'

_CN.output = 'ckpts'
_CN.print_freq = 100

_CN.pretrain = True


def get_cfg():
    return _CN.clone()

