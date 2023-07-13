import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import csv

import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import Mixed_CBCT_dataset
from models.model import DIF_Net
from utils import convert_cuda, add_argument, save_nifti



def eval_one_epoch(model, loader, npoint=50000, save_dir=None, ignore_msg=True, use_tqdm=False):
    model.eval()
    results = {}
    metrics = {}
    metrics_tmp = {key:[] for key in ['psnr', 'ssim']} # , 'rmse', 'mse', 'mae']}
    if use_tqdm:
        loader = tqdm(loader, ncols=50)
    
    with torch.no_grad():
        for item in loader:
            item = convert_cuda(item)

            dst_name = item['dst_name'][0]
            name = item['name'][0]
            image = item['image'].cpu().numpy()
            image = image[0] # W, H, D

            output = model(item, is_eval=True, eval_npoint=npoint) # B, 1, N
            output = output[0, 0].data.cpu().numpy()
            output = output.reshape(image.shape)

            psnr = peak_signal_noise_ratio(image, output)
            ssim = structural_similarity(image, output)

            if not ignore_msg:
                print('{}, PSNR: {:.4}, SSIM: {:.4}'.format(
                    name, psnr, ssim
                ))

            dst_res = results.get(dst_name, [])
            dst_met = metrics.get(dst_name, deepcopy(metrics_tmp))

            dst_res.append({
                'name': name, 
                'psnr': psnr,
                'ssim': ssim,
            })
            for key in dst_met.keys():
                dst_met[key].append(dst_res[-1][key])
            
            results[dst_name] = dst_res
            metrics[dst_name] = dst_met

            if save_dir is not None:
                output = np.clip(output, 0, 1)
                output *= 255.
                output = output.astype(np.uint8)
                save_path = os.path.join(save_dir, f'{name}.nii.gz')
                save_nifti(output, save_path)

    for dst_name in metrics.keys():
        dst_met = metrics[dst_name]
        m = {key:np.mean(val) for key, val in dst_met.items()}
        metrics[dst_name] = m
    
    return metrics, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')
    parser = add_argument(parser, train=False)
    args = parser.parse_args()
    print(args)

    # -- dataloader
    eval_loader = DataLoader(
        Mixed_CBCT_dataset(
            dst_list=args.dst_list.split('+'),
            split=args.split, 
            num_views=args.num_views,
            out_res=args.out_res,
            view_offset=args.view_offset,
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )

    # -- model, load ckpt
    ckpt_path = f'./logs/{args.name}/ep_{args.epoch}.pth'
    ckpt = torch.load(ckpt_path)
    print('load ckpt from', ckpt_path)
    
    model = DIF_Net(
        num_views=args.num_views,
        combine=args.combine
    )
    model.load_state_dict(ckpt)
    model = model.cuda()

    # -- output dir
    save_dir = None
    if args.visualize:
        save_dir = f'./logs/{args.name}/results/ep_{args.epoch}/predictions'
        os.makedirs(save_dir, exist_ok=True)

    # -- evaluate
    metrics, results = eval_one_epoch(
        model, 
        eval_loader, 
        args.eval_npoint,
        save_dir=save_dir,
        ignore_msg=False,
        use_tqdm=False
    )
    print(metrics)

    # -- save results
    pred_dir = f'./logs/{args.name}/results/ep_{args.epoch}'
    os.makedirs(pred_dir, exist_ok=True)

    csv_file = open(os.path.join(pred_dir, 'results.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['dataset', 'obj_id', 'psnr', 'ssim'])

    for dst_name in results.keys():
        dst_res = results[dst_name]
        for res in dst_res:
            csv_writer.writerow([dst_name, res['name'], res['psnr'], res['ssim']])

        dst_avg = metrics[dst_name]
        csv_writer.writerow([dst_name, 'average', dst_avg['psnr'], dst_avg['ssim']])
    
    csv_file.close()

    with open(os.path.join(pred_dir, 'args.json'), 'w') as f:
        args = vars(args)
        json.dump(args, f, indent=4)
