
import torch
import torch.nn.functional as F

from tqdm import tqdm
from icecream import ic
import argparse
import os
from utils import d3_utils

import numpy as np

def d3_data_loop(args):

    d3_loader, d3_intrinsics = d3_utils.fetch_dataloader(root_dir=args.root_dir)

    ic(args.root_dir)
    ic(d3_intrinsics)

    ### Metrics
    # end point error
    epe_list = []
    # root mean squared error
    rmse_list = []
    # percentage of pixels that have a disparity error greater than 1
    bad_pixel_ratio_list_1 = []
    # percentage of pixels that have a disparity error greater than 2
    bad_pixel_ratio_list_2 = []

    metrics = {}

    for i_batch, d3_data_blob in enumerate(tqdm(d3_loader)):

        left_img_t, left_img_t1, right_img_t, depth_map_t, depth_map_t1, flow_map, flowxyz  = [data_item.cuda() for data_item in d3_data_blob]
        
        # Compute ground truth dipsarity from depth
        disp_gt_t = d3_utils.depth_to_disp(depth_map_t, d3_intrinsics)
        # Convert to numpy
        np_disp_gt = disp_gt_t.squeeze(0).cpu().numpy()

        ### run YOUR MODEL here
        np_disp_pred = np.random.randint(0, disp_gt_t.shape[-1], np_disp_gt.shape) # placeholder for depth prediction (in disparity)

        # compute and append metrics
        epe, rmse, bad_pixels_2, bad_pixels_1 = d3_utils.get_metrics(np_disp_pred, np_disp_gt)
        epe_list.append(epe)
        rmse_list.append(rmse)
        bad_pixel_ratio_list_2.append(bad_pixels_2)
        bad_pixel_ratio_list_1.append(bad_pixels_1)

    metrics['epe'] = np.mean(epe_list)
    metrics['rmse'] = np.mean(rmse_list)
    metrics['bad_pixel 2.0'] = np.mean(bad_pixel_ratio_list_2)
    metrics['bad_pixel 1.0'] = np.mean(bad_pixel_ratio_list_1)

    print(metrics)
    ic(metrics)


def main():

    parser = argparse.ArgumentParser(description="d3 arguments")
    parser.add_argument('-r', "--root_dir", type=str, required=True, help='the full path of validation or training splits')
    args = parser.parse_args()
    
    d3_data_loop(args)

if __name__ == "__main__":
    main()