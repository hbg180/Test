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

def convert_flow_to_image(image1, flow):
    flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow)
    flow_image = cv2.resize(flow_image, (image1.shape[3], image1.shape[2]))
    return flow_image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()#b,h,w
    valid = (valid >= 0.5) & (mag < max_flow)#b,1,h,w

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    # print(f"epe:{metrics['epe']:5f}  epe:{metrics['1px']:5f}  epe:{metrics['3px']:5f}  "
    #       f"5px:{metrics['epe']:5f}  loss:{flow_loss:5f}")

    return flow_loss, metrics

def print_args(args):
    print('----------- args ----------')
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    print('---------------------------')


class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_epe_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}
        self.terminal = sys.stdout
        self.log = open(self.args.output + '/log.txt', 'a')

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data[:-1])).format(*metrics_data[:-1])

        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps + 1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int32)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60,
                                                       time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        current_time = time.strftime("%Y-%m-%d %H:%M:%S") + "  "
        # print the training status
        print(current_time + training_str + metrics_str + time_left_hms)
        sys.stdout.flush()

        # logging running loss to total loss
        self.train_epe_list.append(np.mean(self.running_loss_dict['epe']))
        self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []

            self.running_loss_dict[key].append(metrics[key])

        if self.total_steps % self.args.print_freq == self.args.print_freq - 1:
            self._print_training_status()
            self.running_loss_dict = {}

    def write(self, message):
        for i in message:
            self.terminal.write(i)
            self.log.write(i)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.model = TMA(cfg, input_bins=15)
        self.model = self.model.cuda()

        # Loader
        self.train_loader = make_data_loader(cfg, cfg.batch_size, cfg.num_workers)
        print('train_loader done!')

        # Optimizer and scheduler for training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=0.0001
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.lr,
            total_steps=args.num_steps + 100,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy='linear')
        # Logger
        self.checkpoint_dir = cfg.checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.writer = get_logger(os.path.join(self.checkpoint_dir, 'train.log'))
        self.logger = Logger(self.model, self.scheduler, cfg)

    def train(self):
        # self.writer.info(self.model)
        self.logger.write(self.cfg)
        self.model.train()

        total_steps = 0
        keep_training = True
        while keep_training:

            bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=60, position=0, leave=True)
            for index, (voxel1, voxel2, flow_map, valid2D) in bar:
                self.optimizer.zero_grad()
                flow_preds = self.model(voxel1.cuda(), voxel2.cuda())
                flow_loss, loss_metrics = sequence_loss(flow_preds, flow_map.cuda(), valid2D.cuda(), self.cfg.weight,
                                                        MAX_FLOW)

                flow_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                bar.set_description(f'Step: {total_steps}/{self.cfg.num_steps}')
                self.logger.push(loss_metrics)
                total_steps += 1

                if total_steps >= self.cfg.num_steps - 10000 and total_steps % 5000 == 0:
                    ckpt = os.path.join(self.cfg.checkpoint_dir, f'checkpoint_{total_steps}.pth')
                    torch.save(self.model.state_dict(), ckpt)
                if total_steps > self.cfg.num_steps:
                    keep_training = False
                    break

            time.sleep(0.03)
        ckpt_path = os.path.join(self.cfg.checkpoint_dir, 'checkpoint.pth')
        torch.save(self.model.state_dict(), ckpt_path)
        return ckpt_path

# def validate(model, args, logger):
#     model.eval()
#     results = {}
#
#     # Evaluate results
#     for val_dataset in args.validation:
#         if val_dataset == 'chairs':
#             results.update(evaluate.validate_chairs(model.module, args.iters, args.chairs_root))
#         elif val_dataset == 'sintel':
#             results.update(evaluate.validate_sintel(model.module, args.iters, args.sintel_root, tqdm_miniters=100))
#         elif val_dataset == 'kitti':
#             results.update(evaluate.validate_kitti(model.module, args.iters, args.kitti_root))
#
#     # Record results in logger
#     for key in results.keys():
#         if key not in logger.val_results_dict.keys():
#             logger.val_results_dict[key] = []
#         logger.val_results_dict[key].append(results[key])
#
#     logger.val_steps_list.append(logger.total_steps)
#
#     model.train()


def plot_val(logger, args):
    for key in logger.val_results_dict.keys():
        # plot validation curve
        plt.figure()
        latest_x, latest_result = logger.val_steps_list[-1], logger.val_results_dict[key][-1]
        best_result = min(logger.val_results_dict[key])
        best_x = logger.val_steps_list[logger.val_results_dict[key].index(best_result)]

        plt.rc('font', family='Times New Roman')
        plt.plot(logger.val_steps_list, logger.val_results_dict[key])
        plt.annotate('(%s, %s)' % (best_x, round(best_result, 4)), xy=(best_x, best_result),
                     fontproperties='Times New Roman', size=6)
        plt.annotate('(%s, %s)' % (latest_x, round(latest_result, 4)), xy=(latest_x, latest_result),
                     fontproperties='Times New Roman', size=6)
        plt.xlabel('x_steps')
        plt.ylabel(key)
        plt.title(f'Results for {key} for the validation set')
        plt.savefig(args.output + f"/{key}.png", bbox_inches='tight')
        plt.close()


def plot_train(logger, args):
    # plot training curve
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_epe_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.output + "/train_epe.png", bbox_inches='tight')
    plt.close()
      

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


        
if __name__=='__main__':
    import argparse


    parser = argparse.ArgumentParser(description='TMA')
    #training setting
    parser.add_argument('--num_steps', type=int, default=100)  # 200000

    args = parser.parse_args()
    set_seed(1)
    cfg = get_cfg()
    cfg.update(vars(args))
    # process_cfg(cfg)
    # loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    # loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    # train(cfg)
    # if cfg.wandb:
    #     wandb_name = cfg.checkpoint_dir.split('/')[-1]
    #     wandb.init(name=wandb_name, project='Test1')

    trainer = Trainer(cfg)
    trainer.train()
    
