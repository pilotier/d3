
from utils.d3_loader import D3_loader____
import torch
import torch.nn.functional as F

from tqdm import tqdm
from icecream import ic
import argparse
import os

import numpy as np

### raftstereo imports 
from models.RAFT_Stereo.core.raft_stereo import RAFTStereo


def fetch_dataloader():
    gpuargs = {'shuffle': False, 'num_workers': 12, 'drop_last' : False}
    d3_root_dir = "/home/beast/data/d3/v2/validation"
    test_dataset = D3_loader____()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, **gpuargs)
    return test_loader


def prepare_images_and_depths(self, image1, image2):
    """ padding, normalization, and scaling """
    
    ht, wd = image1.shape[-2:]

    pad_h = (-ht) % 8
    pad_w = (-wd) % 8

    image1 = F.pad(image1, [0,pad_w,0,pad_h], mode='replicate')
    image2 = F.pad(image2, [0,pad_w,0,pad_h], mode='replicate')

    image1 = self.normalize_image(image1.float())
    image2 = self.normalize_image(image2.float())

    return image1, image2 



def get_hardcoded_raftstereoargs():
    args = argparse.Namespace(
        restore_ckpt="/home/beast/pilotier/d3/models/RAFT_Stereo/models/raftstereo-middlebury.pth",
        save_numpy=True,
        left_imgs="datasets/Middlebury/MiddEval3/testH/*/im0.png",
        right_imgs="datasets/Middlebury/MiddEval3/testH/*/im1.png",
        output_directory="demo_output",
        mixed_precision=True,
        valid_iters=32,
        hidden_dims=[128, 128, 128],
        corr_implementation="alt",
        shared_backbone=True,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=True,
        n_gru_layers=3
    )
    return args


def get_metrics(predicted_disp, gt_disp):
        
    
    epe = np.mean(np.sqrt((predicted_disp - gt_disp) ** 2))
    rmse = np.sqrt(np.mean((predicted_disp - gt_disp) ** 2))
    bad_pixels_2 = (np.abs(predicted_disp - gt_disp) > 2).astype(np.float32).mean()
    bad_pixels_1 = (np.abs(predicted_disp - gt_disp) > 1).astype(np.float32).mean()

    # epe_list.append(epe)
    # rmse_list.append(rmse)
    # bad_pixel_ratio_list_2.append(bad_pixels_2)
    # bad_pixel_ratio_list_1.append(bad_pixels_1)
    return epe, rmse, bad_pixels_2, bad_pixels_1 

@torch.no_grad()
def evaluate_flow_metrics_leastereo():

    # ╦  ╔═╗╔═╗╔╦╗  ╔╦╗╔═╗╔╦╗╔═╗╦  models/LEAStereo/
    # ║  ║ ║╠═╣ ║║  ║║║║ ║ ║║║╣ ║  
    # ╩═╝╚═╝╩ ╩═╩╝  ╩ ╩╚═╝═╩╝╚═╝╩═╝
    args = get_hardcoded_raftstereoargs()
    raftstereo_model = torch.nn.DataParallel(RAFTStereo(args))
    ic(args.restore_ckpt)
    raftstereo_model.load_state_dict(torch.load(args.restore_ckpt), strict=True)

    # raftstereo_model = raftstereo_model.module
    raftstereo_model.cuda()
    raftstereo_model.eval()
    
    metrics = {}
    dstype = "simdata"
  
    d3_test_loader = fetch_dataloader()

    epe_list = []
    rmse_list = []
    bad_pixel_ratio_list_2 = []
    bad_pixel_ratio_list_1 = []


    for i_batch, d3_test_data_blob in enumerate(tqdm(d3_test_loader)):
        image1, image2, right_img, depth1, depth2, flow2d, flow3d = [data_item.cuda() for data_item in d3_test_data_blob]
        disp_gt = 800.0 * 0.12 / (depth1.squeeze(0).cpu().numpy()  )
        disp_gt /= 2.0 ## cuz im downsizing the img by 2

        ### RUN MODEL
        lst_disp = raftstereo_model(image1, right_img)
        np_lst_disp = lst_disp[0].squeeze(0).cpu().numpy()

        epe, rmse, bad_pixels_2, bad_pixels_1 = get_metrics(np_lst_disp, disp_gt)
        epe_list.append(epe)
        rmse_list.append(rmse)
        bad_pixel_ratio_list_2.append(bad_pixels_2)
        bad_pixel_ratio_list_1.append(bad_pixels_1)

        # result = [data_item for data_item in d3_test_data_blob]
        # ic(result)
        # image1_padded, image2_padded = prepare_images_and_depths(image1, image2)

        # ic(image1_padded.shape)
    metrics['epe'] = np.mean(epe_list)
    metrics['rmse'] = np.mean(rmse_list)
    metrics['bad_pixel 2.0'] = np.mean(bad_pixel_ratio_list_2)
    metrics['bad_pixel 1.0'] = np.mean(bad_pixel_ratio_list_1)

    print(metrics)
    ic(metrics)


evaluate_flow_metrics_leastereo()