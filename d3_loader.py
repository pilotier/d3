# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from argparse import ArgumentParser
import mmcv
from icecream import ic
from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow, write_flow
from nets.flow_module import CreFlow
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import cv2
import math
import random
import json
import pickle
import os.path as osp
from tqdm import tqdm
import pylotier.utils.flow_viz as flow_viz

class D3_Loader(data.Dataset):
    def __init__(self, root_dir="/media/satya/Satya/Pilotier/neurips/dataset/val_subset"):
        self.fx = 800.
        self.baseline = 0.12
        self.fy = 800.0
        self.cx = 960.0
        self.cy = 540.0
        self.intrinsics = np.array([self.fx, self.fy, self.cx, self.cy])
        
        self.left_images = []
        self.right_images = []
        self.depth_maps = []
        self.flow_maps = []
        self.z_flow_maps = []
        
        for subdir in sorted(os.listdir(root_dir)):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path): 
                left_files = sorted(os.listdir(os.path.join(subdir_path, "left")))
                right_files = sorted(os.listdir(os.path.join(subdir_path, "right")))
                depth_files = sorted(os.listdir(os.path.join(subdir_path, "depth")))
                flow_files = sorted(os.listdir(os.path.join(subdir_path, "flow")))
                z_flow_files = sorted(os.listdir(os.path.join(subdir_path, "z_flow")))

                self.left_images.extend([os.path.join(subdir_path, "left", f) for f in left_files[:-1]])
                self.right_images.extend([os.path.join(subdir_path, "right", f) for f in right_files[:-1]])
                self.depth_maps.extend([os.path.join(subdir_path, "depth", f) for f in depth_files[:-1]])
                self.flow_maps.extend([os.path.join(subdir_path, "flow", f) for f in flow_files[:-1]])
                self.z_flow_maps.extend([os.path.join(subdir_path, "z_flow", f) for f in z_flow_files[:-1]])



    def __len__(self):
        return len(self.left_images) - 1

    def __getitem__(self, idx):
        left_img_path_t = self.left_images[idx]
        left_img_path_t1 = self.left_images[idx + 1]
        right_img_path_t = self.right_images[idx]
        depth_map_path_t = self.depth_maps[idx]
        depth_map_path_t1 = self.depth_maps[idx + 1]
        flow_map_path_t = self.flow_maps[idx]
        z_flow_map_path_t = self.z_flow_maps[idx]
        
        # ic(left_img_path_t)

        # try:
        left_img_t = cv2.imread(left_img_path_t)
        left_img_t1 = cv2.imread(left_img_path_t1)
        right_img_t = cv2.imread(right_img_path_t)
        depth_map_t = np.load(depth_map_path_t)['arr_0']
        depth_map_t1 = np.load(depth_map_path_t1)['arr_0']
        flow_map = np.load(flow_map_path_t)['arr_0']
        z_flow_map = np.load(z_flow_map_path_t)['arr_0']
        # except Exception as e:
        #     print(f"Error loading data: {e}")
        #     return None

        # # Optionally resize images and maps
        eval_h, eval_w = 540, 960
        left_img_t = cv2.resize(left_img_t, (eval_w, eval_h))
        left_img_t1 = cv2.resize(left_img_t1, (eval_w, eval_h))

        right_img_t = cv2.resize(right_img_t, (eval_w, eval_h))
        depth_map_t = cv2.resize(depth_map_t, (eval_w, eval_h))
        depth_map_t1 = cv2.resize(depth_map_t1, (eval_w, eval_h))

        flow_map = cv2.resize(flow_map, (eval_w, eval_h))
        z_flow_map = cv2.resize(z_flow_map, (eval_w, eval_h))

        # Convert images to PyTorch tensors
        left_img_t = torch.from_numpy(left_img_t).permute(2, 0, 1).float()
        left_img_t1 = torch.from_numpy(left_img_t1).permute(2, 0, 1).float()
        right_img_t = torch.from_numpy(right_img_t).permute(2, 0, 1).float()
        depth_map_t = torch.from_numpy(depth_map_t).float().squeeze(0)
        depth_map_t1 = torch.from_numpy(depth_map_t1).float().squeeze(0)
        flow_map = torch.from_numpy(flow_map).float().squeeze(0)
        z_flow_map = torch.from_numpy(z_flow_map).float().squeeze(0)
        
        disp1  = (self.fx * self.baseline) / depth_map_t
        
        depth12 = (self.fx / (disp1 + z_flow_map)).float()
      
        # Normalization can be applied here or in a transform
        return left_img_t, left_img_t1, right_img_t, depth_map_t, depth_map_t1, flow_map



def normalize_image(image):
    image = image[:, [2,1,0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=image.device)
    return (image/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

def prepare_images_and_depths(image1, image2):
    """ padding, normalization, and scaling """
    
    ht, wd = image1.shape[-2:]

    pad_h = (-ht) % 8
    pad_w = (-wd) % 8

    image1 = F.pad(image1, [0,pad_w,0,pad_h], mode='replicate')
    image2 = F.pad(image2, [0,pad_w,0,pad_h], mode='replicate')

    image1 = normalize_image(image1.float())
    image2 = normalize_image(image2.float())

    return image1, image2 

def fetch_dataloader():
    gpuargs = {'shuffle': False, 'num_workers': 12, 'drop_last' : False}
    test_dataset = SimDataTest()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, **gpuargs)
    return test_loader

@torch.no_grad()
def evaluate_flow_metrics():
    
    creflow = CreFlow(cuda_device = "cuda:0",
                        onnx_device_id = 0,
                        creflow_onnx_path = "/media/satya/Satya/Pilotier/projects/TensorRT-CREStereo/800000_cremos-sintel_2_test2_540_960_iters20.onnx"
                    )
    
    results = {}
    dstype = "simdata"
  
    test_loader = fetch_dataloader()
    epe_list = []
    rmse_list = []
    
    config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.py"
    checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth"
    
    # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/pwcnet/pwcnet_kitti_test.py"
    # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.pth"
    
    # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/raft/raft_kitti_test.py"
    # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/raft_8x2_50k_kitti2015_288x960.pth"
    
    # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/gma/gma_8x2_50k_kitti2015_288x960.py"
    # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/gma_8x2_50k_kitti2015_288x960.pth"
    
    # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/gma/gma_plus-p_8x2_50k_kitti2015_288x960.py"
    # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/gma_plus-p_8x2_50k_kitti2015_288x960.pth"

    # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/irr/irrpwc_ft_4x1_300k_kitti_320x896.py"
    # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/irrpwc_ft_4x1_300k_kitti_320x896.pth"
    
    # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/liteflownet2/liteflownet2_ft_4x1_500k_kitti_320x896.py"
    # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/liteflownet2_ft_4x1_600k_sintel_kitti_320x768.pth"
    
    
    model = init_model(config, checkpoint, device="cuda:0")

    for i_batch, data_blob in enumerate(tqdm(test_loader)):
      
        image1, image2, right_img, depth1, depth2, flow_gt = [data_item.cuda() for data_item in data_blob]
    
        image1_padded, image2_padded = prepare_images_and_depths(image1, image2)

        
        
        # ic(image1.shape, image2.shape, right_img.shape, depth1.shape, depth2.shape, flow_gt.shape)
        
        image1_np = image1.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image2_np = image2.squeeze(0).permute(1, 2, 0).cpu().numpy()
        flow_gt = flow_gt.squeeze(0)
        
        # flow = inference_model(model, image1_np, image2_np)
        flow = creflow.infer_onnx(image2_np, image1_np)
        
        flow = torch.from_numpy(flow).cuda()
         # ic(flow.shape)
        
        flow_vis = flow_viz.np_flow_to_image(flow.detach().cpu().numpy())
        flow_gt_vis = flow_viz.np_flow_to_image(flow_gt.detach().cpu().numpy())
        
        cat = np.vstack([image1_np, flow_vis, flow_gt_vis])
        cat = cv2.resize(cat, None, fx=0.5, fy=0.5)
        
        # ic(flow, flow_gt)
        cv2.imshow("output", cat.astype(np.uint8))
        cv2.waitKey(10)
        
        epe = torch.sum((flow - flow_gt)**2, dim=-1).sqrt()
        rmse = torch.sqrt(torch.mean(torch.sum((flow - flow_gt)**2, dim=-1)))

        # ic(epe)
   
        # if torch.mean(epe.view(-1)) < 50:
        epe_list.append(epe.view(-1).detach().cpu().numpy())
        rmse_list.append(rmse.item()) 
        # ic(epe_list)
        
        # counter += 1
        # if counter > 100:
        #     break

    epe_all = np.concatenate(epe_list)
    rmse_mean = np.mean(rmse_list)
    # ic(np.max(epe_all))
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all<1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)

    print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
    ic(rmse_mean)
    results[dstype] = np.mean(epe_list)

    return results




if __name__ == '__main__':
    evaluate_flow_metrics()
  