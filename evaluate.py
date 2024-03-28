from tqdm import tqdm
import numpy as np
import torch
import os
from datasets.DSEC_split_loader import DSECsplit
from model.network import TMA
from flow_viz import flow_to_image
from PIL import Image


@torch.no_grad()
def validate_DSEC(model, args):
    if args.mfe == 'basic':
        folder = 'flow_basic'
    else:
        folder = 'flow'
    model.eval()
    val_dataset = DSECsplit('test')

    epe_list = []
    out_list = []

    frame_id = 1
    bar = tqdm(enumerate(val_dataset), total=len(val_dataset), ncols=60)
    bar.set_description('Test')
    for index, (voxel1, voxel2, flow_map, valid2D) in bar:
        voxel1 = voxel1[None].cuda()
        voxel2 = voxel2[None].cuda()
        flow_pred = model(voxel1, voxel2)[0].cpu()  # [1,2,H,W]

        h = flow_pred.shape[1]
        w = flow_pred.shape[2]
        flow_pred2 = flow_pred.view(h, w, -1).numpy()
        flow_img = flow_to_image(flow_pred2)
        image = Image.fromarray(flow_img)
        if not os.path.exists(f'vis_result'):
            os.makedirs(f'vis_result/{folder}')

        image.save(f'vis_result/{folder}/{frame_id}.png')
        frame_id += 1

        epe = torch.sum((flow_pred - flow_map) ** 2, dim=0).sqrt()  # [H,W]
        mag = torch.sum(flow_map ** 2, dim=0).sqrt()  # [H,W]

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid2D.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    npx = {'1px': 0, '2px': 0, '3px': 0, '4px': 0, '5px': 0, '6px': 0, '7px': 0}
    npx['1px'] = np.mean(epe_list[epe_list < 1])
    npx['2px'] = np.mean(epe_list[epe_list < 2])
    npx['3px'] = np.mean(epe_list[epe_list < 3])
    npx['4px'] = np.mean(epe_list[epe_list < 4])
    npx['5px'] = np.mean(epe_list[epe_list < 5])
    npx['6px'] = np.mean(epe_list[epe_list < 6])
    npx['7px'] = np.mean(epe_list[epe_list < 7])
    print(npx)
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation DSEC-TEST: %f, %f" % (epe, f1))
    return {'dsec-epe': epe, 'dsec-f1': f1}


if __name__ == '__main__':
    pattern = 'basic'

    import argparse

    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--checkpoint_dir', type=str, default=f'D:/下载/浏览器下载/checkpoint_{pattern}.pth')
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
    parser.add_argument('--mfe', type=str, default=pattern)
    parser.add_argument('--updater', type=str, default=pattern)
    parser.add_argument('--corr_levels', type=int, default=1)
    parser.add_argument('--corr_radius', type=int, default=3)
    parser.add_argument('--k_conv', type=list, default=[1, 15])
    parser.add_argument('--PCUpdater_conv', type=list, default=[1, 7])
    parser.add_argument('--grad_clip', type=float, default=1)

    # loss setting
    parser.add_argument('--weight', type=float, default=0.8)

    args = parser.parse_args()

    model = TMA(args)
    model.load_state_dict(torch.load(args.checkpoint_dir))
    model.cuda()

    validate_DSEC(model, args)
