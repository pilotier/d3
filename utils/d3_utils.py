
from utils.d3_reader import D3_Reader
import torch
import torch.nn.functional as F

from tqdm import tqdm
from icecream import ic
import argparse
import os

import numpy as np

def depth_to_disp(depth, intrinsics):
    disp_gt = (intrinsics["fx"] * intrinsics["baseline"]) / depth
    return disp_gt

def fetch_dataloader(root_dir, shuffle=False, num_workers=12, drop_last = False):
    gpuargs = {'shuffle': shuffle, 'num_workers': num_workers, 'drop_last' : drop_last}
    test_dataset = D3_Reader(root_dir)
    dataset_intrinsics = test_dataset.intrinsics
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, **gpuargs)
    return test_loader, dataset_intrinsics


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



def get_metrics(predicted_disp, gt_disp):
        
    epe = np.mean(np.sqrt((predicted_disp - gt_disp) ** 2))
    rmse = np.sqrt(np.mean((predicted_disp - gt_disp) ** 2))
    bad_pixels_2 = (np.abs(predicted_disp - gt_disp) > 2).astype(np.float32).mean()
    bad_pixels_1 = (np.abs(predicted_disp - gt_disp) > 1).astype(np.float32).mean()

    return epe, rmse, bad_pixels_2, bad_pixels_1 
