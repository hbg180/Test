from glob import glob

import torch
import numpy as np
from datasets.DSECdataloader import flow_16bit_to_float
from datasets.augument import Augumentor
from flow_viz import flow_to_image
from PIL import Image
import os
import imageio
from tqdm import tqdm


def flow_to_images(path:str):
    scenes = os.listdir(gt_path)
    scene = 'zurich_city_11_a'
    # path = os.path.join(r'E:\Git\Papers\TMA\dsec\test', scene, 'seq_*.npy')
    path = os.path.join(r'E:\Git\Papers\MyCode\Test\optical_flow\zurich_city_11_a\forward', 'seq_*.npy')
    flow_list = glob(path)
    num = 0
    augmentor = Augumentor(crop_size=[288, 384])
    # bar = tqdm(range(len(flow_list)), ncols=60)
    for flow in flow_list:
        flow_16bit = np.load(flow)
        flow_map, valid2D = flow_16bit_to_float(flow_16bit)     # 480,640,2

        # voxel1, voxel2, flow_map, valid2D = augmentor(voxel1, voxel2, flow_map, valid2D)   # 数据增强
        flow_img = flow_to_image(flow_map)  # 480,640,2 -> 480,640,3
        image = Image.fromarray(flow_img)
        if not os.path.exists(f'vis_result/{scene}_GTFlow_forward'):
            os.makedirs(f'vis_result/{scene}_GTFlow_forward')

        image.save(f'vis_result/{scene}_GTFlow_forward/seq_{num:06d}.png')
        num += 1


def generation_flow(gt_path: str='', dst: str='', dir: str='forward'):
    """
    :param gt_path: 真实光流路径
    :param dst: .npy文件保存路径
    :param dir: forward|backward|bidirectional
    :return: None
    """
    if dir not in ['forward', 'backward', 'bidirectional']:
        print("dir error!")

    if dir == 'bidirectional':
        dir_list = ['forward', 'backward']
    else:
        dir_list = [dir]

    scenes = os.listdir(gt_path)
    total = len(scenes)

    for direction in dir_list:
        cnt = 1
        for scene in scenes:  # e.g. zurich_city_02_a
            output_dir = os.path.join(dst, scene, direction)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # gt timestamps and gt optical flow
            timestamp_f = os.path.join(gt_path, scene, 'flow', f'{direction}_timestamps.txt')  # 光流时间戳文件
            timestamps = np.genfromtxt(timestamp_f, delimiter=',')  # 光流时间戳
            gt_flow_f = sorted(glob(os.path.join(gt_path, scene, 'flow', direction, '*.png')))  # 光流
            assert len(timestamps) == len(gt_flow_f)

            bar = tqdm(range(len(timestamps)), ncols=100)  # ncols: 调整进度条宽度
            for idx in bar:
                bar.set_description(f'{direction}:{cnt:2d}/{total}  {scene}')
                flow_16bit = imageio.imread(gt_flow_f[idx], format='PNG-FI')

                output_name = os.path.join(output_dir, 'seq_{:06d}'.format(idx))

                # save gt flow
                np.save(output_name, flow_16bit)
            cnt += 1

if __name__ == "__main__":
    gt_path = 'E:/Git/Papers/TMA/DSE/train_optical_flow'
    dst = os.path.join(os.getcwd(), 'optical_flow')
    # generation_flow(gt_path, dst, 'bidirectional')
    flow_to_images('')
