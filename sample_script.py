
import torch
import torch.nn.functional as F

from tqdm import tqdm
from icecream import ic
import argparse
import os
from utils import d3_utils

import numpy as np

def d3_data_loop(args):

    d3_test_loader = d3_utils.fetch_dataloader(root_dir=args.root_dir)



    ### Metrics
    # end point error
    epe_list = []
    # root mean squared error
    rmse_list = []
    # percentage of pixels that have a disparity error greater than 1
    bad_pixel_ratio_list_1 = []
    # percentage of pixels that have a disparity error greater than 2
    bad_pixel_ratio_list_2 = []

    for i_batch, d3_data_blob in enumerate(tqdm(d3_test_loader)):
        image1, image2, right_img, depth1, depth2, flow2d, flow3d = [data_item.cuda() for data_item in d3_data_blob]


        disp_gt = 800.0 * 0.12 / (depth1.squeeze(0).cpu().numpy()  )
        disp_gt /= 2.0 ## cuz im downsizing the img by 2

        ic(i_batch)
    #     ### RUN MODEL
    #     lst_disp = leastereo_model(image1, right_img)
    #     np_lst_disp = lst_disp.squeeze(0).cpu().numpy()

    #     epe, rmse, bad_pixels_2, bad_pixels_1 = d3_utils.get_metrics(np_lst_disp, disp_gt)
    #     epe_list.append(epe)
    #     rmse_list.append(rmse)
    #     bad_pixel_ratio_list_2.append(bad_pixels_2)
    #     bad_pixel_ratio_list_1.append(bad_pixels_1)

    #     # result = [data_item for data_item in d3_test_data_blob]
    #     # ic(result)
    #     # image1_padded, image2_padded = prepare_images_and_depths(image1, image2)

    #     # ic(image1_padded.shape)
    # metrics['epe'] = np.mean(epe_list)
    # metrics['rmse'] = np.mean(rmse_list)
    # metrics['bad_pixel 2.0'] = np.mean(bad_pixel_ratio_list_2)
    # metrics['bad_pixel 1.0'] = np.mean(bad_pixel_ratio_list_1)

    # print(metrics)
    # ic(metrics)


def main():

    parser = argparse.ArgumentParser(description="d3 arguments")

    parser.add_argument('-r', "--root_dir", type=str, required=True, help='the full path of validation or training splits')

    args = parser.parse_args()
    
    ic(args)

    d3_data_loop(args)

if __name__ == "__main__":
    main()