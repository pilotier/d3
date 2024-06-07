# # Copyright (c) OpenMMLab. All rights reserved.
# import os.path as osp

# from icecream import ic
# # from mmflow.apis import inference_model, init_model
# # from mmflow.datasets import visualize_flow, write_flow
# # from nets.flow_module import CreFlow

# from __future__ import print_function
# import argparse
# import skimage
# import skimage.io
# import skimage.transform
# from PIL import Image
# from math import log10

# import sys
# import shutil
# import os
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from retrain.LEAStereo import LEAStereo 

# from config_utils.predict_args import obtain_predict_args
# from utils.colorize import get_color_map
# from utils.multadds_count import count_parameters_in_MB, comp_multadds
# from time import time
# from struct import unpack
# import matplotlib.pyplot as plt
# import re
# import numpy as np
# import pdb
# from path import Path
# from icecream import ic
# import numpy as np
# import torch
# import torch.nn.functional as F

# import argparse

# import os
# import cv2
# import math
# import random
# import json
# import pickle
# import os.path as osp
# from tqdm import tqdm
# import pylotier.utils.flow_viz as flow_viz



# def normalize_image(image):
#     image = image[:, [2,1,0]]
#     mean = torch.as_tensor([0.485, 0.456, 0.406], device=image.device)
#     std = torch.as_tensor([0.229, 0.224, 0.225], device=image.device)
#     return (image/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

# def prepare_images_and_depths(image1, image2):
#     """ padding, normalization, and scaling """
    
#     ht, wd = image1.shape[-2:]

#     pad_h = (-ht) % 8
#     pad_w = (-wd) % 8

#     image1 = F.pad(image1, [0,pad_w,0,pad_h], mode='replicate')
#     image2 = F.pad(image2, [0,pad_w,0,pad_h], mode='replicate')

#     image1 = normalize_image(image1.float())
#     image2 = normalize_image(image2.float())

#     return image1, image2 

# def fetch_dataloader():
#     gpuargs = {'shuffle': False, 'num_workers': 12, 'drop_last' : False}
#     test_dataset = SimDataTest()
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, **gpuargs)
#     return test_loader

# def get_hardcoded_leastereo_args():
#     args = argparse.Namespace(
#         kitti2015=1,
#         maxdisp=192,
#         crop_height=384,
#         crop_width=1248,
#         data_path='./dataset/kitti2015/testing/',
#         test_list='./dataloaders/lists/kitti2015_test.list',
#         save_path='./predict/kitti2015/images/',
#         fea_num_layer=6,
#         mat_num_layers=12,
#         fea_filter_multiplier=8,
#         fea_block_multiplier=4,
#         fea_step=3,
#         mat_filter_multiplier=8,
#         mat_block_multiplier=4,
#         mat_step=3,
#         net_arch_fea='run/sceneflow/best/architecture/feature_network_path.npy',
#         cell_arch_fea='run/sceneflow/best/architecture/feature_genotype.npy',
#         net_arch_mat='run/sceneflow/best/architecture/matching_network_path.npy',
#         cell_arch_mat='run/sceneflow/best/architecture/matching_genotype.npy',
#         resume='./run/Kitti15/best/best.pth'
#     )

#     return args


# @torch.no_grad()
# def evaluate_flow_metrics():
    
#     # # ╔═╗╦ ╦ ╦╦
#     # # ╠╣ ║ ║ ║║
#     # # ╚  ╚═╝╚╝╩
#     # creflow = CreFlow(cuda_device = "cuda:0",
#     #                     onnx_device_id = 0,
#     #                     creflow_onnx_path = "/media/satya/Satya/Pilotier/projects/TensorRT-CREStereo/800000_cremos-sintel_2_test2_540_960_iters20.onnx"
#     #                 )
    
#     # ╦  ┌─┐┌─┐┌─┐┌┬┐┌─┐┬─┐┌─┐┌─┐
#     # ║  ├┤ ├─┤└─┐ │ ├┤ ├┬┘├┤ │ │
#     # ╩═╝└─┘┴ ┴└─┘ ┴ └─┘┴└─└─┘└─┘
#     ############################################
#     torch.backends.cudnn.benchmark = True
#     opt = get_hardcoded_leastereo_args()
#     cuda = opt.cuda
#     if cuda and not torch.cuda.is_available():
#         raise Exception("No GPU found, please run without --cuda")

#     print('===> Building LEAStereo model')
#     lst_model = LEAStereo(opt)

#     print('Total Params = %.2fMB' % count_parameters_in_MB(model))
#     print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
#     print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))
    
#     mult_adds = comp_multadds(model, input_size=(3,opt.crop_height, opt.crop_width)) #(3,192, 192))
#     print("compute_average_flops_cost = %.2fMB" % mult_adds)

#     if cuda:
#         model = torch.nn.DataParallel(model).cuda()

#     if opt.resume:
#         if os.path.isfile(opt.resume):
#             print("=> loading checkpoint '{}'".format(opt.resume))
#             checkpoint = torch.load(opt.resume)
#             model.load_state_dict(checkpoint['state_dict'], strict=True)      
#         else:
#             print("=> no checkpoint found at '{}'".format(opt.resume))

#     turbo_colormap_data = get_color_map()
#     ############################################


#     results = {}
#     dstype = "simdata"
  
#     test_loader = fetch_dataloader()
#     epe_list = []
#     rmse_list = []
    
#     config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.py"
#     checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth"
    
#     # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/pwcnet/pwcnet_kitti_test.py"
#     # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.pth"
    
#     # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/raft/raft_kitti_test.py"
#     # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/raft_8x2_50k_kitti2015_288x960.pth"
    
#     # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/gma/gma_8x2_50k_kitti2015_288x960.py"
#     # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/gma_8x2_50k_kitti2015_288x960.pth"
    
#     # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/gma/gma_plus-p_8x2_50k_kitti2015_288x960.py"
#     # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/gma_plus-p_8x2_50k_kitti2015_288x960.pth"

#     # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/irr/irrpwc_ft_4x1_300k_kitti_320x896.py"
#     # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/irrpwc_ft_4x1_300k_kitti_320x896.pth"
    
#     # config = "/media/satya/Satya/Pilotier/neurips/mmflow/configs/liteflownet2/liteflownet2_ft_4x1_500k_kitti_320x896.py"
#     # checkpoint = "/media/satya/Satya/Pilotier/neurips/mmflow/models/liteflownet2_ft_4x1_600k_sintel_kitti_320x768.pth"
    
    
#     # model = init_model(config, checkpoint, device="cuda:0")

#     for i_batch, data_blob in enumerate(tqdm(test_loader)):
      
#         image1, image2, right_img, depth1, depth2, flow_gt = [data_item.cuda() for data_item in data_blob]
    
#         image1_padded, image2_padded = prepare_images_and_depths(image1, image2)

        
        
#         # ic(image1.shape, image2.shape, right_img.shape, depth1.shape, depth2.shape, flow_gt.shape)
        
#         image1_np = image1.squeeze(0).permute(1, 2, 0).cpu().numpy()
#         image2_np = image2.squeeze(0).permute(1, 2, 0).cpu().numpy()
#         flow_gt = flow_gt.squeeze(0)
        
#         # flow = inference_model(model, image1_np, image2_np)
#         # flow = creflow.infer_onnx(image2_np, image1_np)
        
#         flow = torch.from_numpy(flow).cuda()
#          # ic(flow.shape)
        
#         flow_vis = flow_viz.np_flow_to_image(flow.detach().cpu().numpy())
#         flow_gt_vis = flow_viz.np_flow_to_image(flow_gt.detach().cpu().numpy())
        
#         cat = np.vstack([image1_np, flow_vis, flow_gt_vis])
#         cat = cv2.resize(cat, None, fx=0.5, fy=0.5)
        
#         # ic(flow, flow_gt)
#         cv2.imshow("output", cat.astype(np.uint8))
#         cv2.waitKey(10)
        
#         epe = torch.sum((flow - flow_gt)**2, dim=-1).sqrt()
#         rmse = torch.sqrt(torch.mean(torch.sum((flow - flow_gt)**2, dim=-1)))

#         # ic(epe)
   
#         # if torch.mean(epe.view(-1)) < 50:
#         epe_list.append(epe.view(-1).detach().cpu().numpy())
#         rmse_list.append(rmse.item()) 
#         # ic(epe_list)
        
#         # counter += 1
#         # if counter > 100:
#         #     break

#     epe_all = np.concatenate(epe_list)
#     rmse_mean = np.mean(rmse_list)
#     # ic(np.max(epe_all))
#     epe = np.mean(epe_all)
#     px1 = np.mean(epe_all<1)
#     px3 = np.mean(epe_all<3)
#     px5 = np.mean(epe_all<5)

#     print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
#     ic(rmse_mean)
#     results[dstype] = np.mean(epe_list)

#     return results




# if __name__ == '__main__':
#     evaluate_flow_metrics()
  