import sys
sys.path.append('model')
import time
import os
import random
from tqdm import tqdm
import wandb
import torch
import numpy as np
from utils.file_utils import get_logger
from utils import flow_viz
from datasets.DSECdataloader import make_data_loader
from configs.mvsec import get_cfg
import cv2
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler

####Important####
from model.network import TMA
####Important####

MAX_FLOW = 400
SUM_FREQ = 100


class Loss_Tracker:
    def __init__(self, args, wandb):
        self.running_loss = {}
        self.total_steps = 0
        self.wandb = wandb

    def push(self, args, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == 0:
            if self.wandb:
                wandb.log({'EPE': self.running_loss['epe'] / SUM_FREQ}, step=self.total_steps)
                wandb.log({'1PX': self.running_loss['1px'] / SUM_FREQ}, step=self.total_steps)
                wandb.log({'3PX': self.running_loss['3px'] / SUM_FREQ}, step=self.total_steps)
                wandb.log({'5PX': self.running_loss['5px'] / SUM_FREQ}, step=self.total_steps)
            # info = f'{self.total_steps}    '
            # info += f"epe: {self.running_loss['epe'] / SUM_FREQ:7.4f}    1px: {self.running_loss['1px'] / SUM_FREQ:7.4f}    "
            # info += f"3px: {self.running_loss['3px'] / SUM_FREQ:7.4f}    5px: {self.running_loss['5px'] / SUM_FREQ:7.4f}"
            # with open(args.filename, 'a') as f:
            #     f.write(info+'\n')
            # print('\n'+info)
            # print(f"\nepe: {self.running_loss['epe'] / SUM_FREQ:7.4f}", end='    ')
            # print(f"1px: {self.running_loss['1px'] / SUM_FREQ:7.4f}", end='    ')
            # print(f"3px: {self.running_loss['3px'] / SUM_FREQ:7.4f}", end='    ')
            # print(f"5px: {self.running_loss['5px'] / SUM_FREQ:7.4f}")
            self.running_loss = {}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    def __init__(self, args):
        self.args = args

        self.model = TMA(args, input_bins=15)
        self.model = self.model.cuda()

        # Loader
        self.train_loader = make_data_loader(args, args.batch_size, args.num_workers)
        print('train_loader done!')

        # Optimizer and scheduler for training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.0001
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            total_steps=args.num_steps + 100,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy='linear')
        # Logger
        self.checkpoint_dir = args.checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.writer = get_logger(os.path.join(self.checkpoint_dir, 'train.log'))
        self.tracker = Loss_Tracker(args, args.wandb)

        self.writer.info('====A NEW TRAINING PROCESS====')
        print(f"Parameter Count: {count_parameters(self.model)}")

    def train(self):
        # self.writer.info(self.model)
        self.writer.info(self.args)
        self.model.train()

        total_steps = 0
        keep_training = True
        self.args.print_freq = len(self.train_loader)
        while keep_training:

            bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=100)
            for index, (voxel1, voxel2, flow_map, valid2D) in bar:
                self.optimizer.zero_grad()
                flow_preds = self.model(voxel1.cuda(), voxel2.cuda())
                flow_loss, loss_metrics = sequence_loss(flow_preds, flow_map.cuda(), valid2D.cuda(), self.args.weight,
                                                        MAX_FLOW)
                # 打印损失
                # current_time = time.strftime("%Y-%m-%d %H:%M:%S") + "  "
                # metrics_str = ''
                # metrics_str += 'epe:'
                # print(current_time + metrics_str)

                flow_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                bar.set_description(f'Step: {total_steps:6d}/{self.args.num_steps}')
                self.tracker.push(self.args, loss_metrics)
                total_steps += 1

                if total_steps % 10000 == 0:      # 保存模型
                    ckpt = os.path.join(self.args.checkpoint_dir, f'checkpoint_{total_steps}.pth')
                    torch.save(self.model.state_dict(), ckpt)
                if total_steps > self.args.num_steps:   # 训练结束
                    keep_training = False
                    print(f"total_steps is {total_steps}")
                    break

            time.sleep(0.03)
        ckpt_path = os.path.join(self.args.checkpoint_dir, 'checkpoint.pth')
        torch.save(self.model.state_dict(), ckpt_path)
        return ckpt_path


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # b,h,w
    valid = (valid >= 0.5) & (mag < max_flow)  # b,1,h,w

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TMA')
    # training setting
    parser.add_argument('--num_steps', type=int, default=100000)  # 200000
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default='ckpts/sk_b1/')
    parser.add_argument('--lr', type=float, default=2e-4)

    # datasets setting
    parser.add_argument('--root', type=str, default='dsec/')
    parser.add_argument('--crop_size', type=list, default=[288, 384])

    # dataloader setting                                                      #             慢 慢 快 快
    parser.add_argument('--batch_size', type=int, default=3)  # batch_size ：4 2  1 1
    parser.add_argument('--num_workers', type=int, default=4)  # num_workers：1 2  2 3

    # model setting
    parser.add_argument('--cnet', type=str, default='basic')
    parser.add_argument('--fnet', type=str, default='basic')
    parser.add_argument('--cbam', type=bool, default=True)
    parser.add_argument('--mfe', type=str, default='basic')
    parser.add_argument('--updater', type=str, default='basic')
    parser.add_argument('--corr_levels', type=int, default=1)
    parser.add_argument('--corr_radius', type=int, default=3)
    parser.add_argument('--k_conv', type=list, default=[1, 15])
    parser.add_argument('--PCUpdater_conv', type=list, default=[1, 7])
    parser.add_argument('--grad_clip', type=float, default=1)

    # loss setting
    parser.add_argument('--weight', type=float, default=0.8)
    parser.add_argument('--filename',type=str, default='train_metrics1_sk.txt')

    # wandb setting
    parser.add_argument('--wandb', action='store_true', default=True)
    args = parser.parse_args()
    set_seed(1)
    if args.wandb:
        wandb_name = args.checkpoint_dir.split('/')[-1]
        wandb.init(name=wandb_name, project='myTest')

    trainer = Trainer(args)
    res = trainer.train()
    print(res)
