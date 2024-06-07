
from utils.d3_loader import D3_loader____
import torch
import torch.nn.functional as F

from tqdm import tqdm
from icecream import ic
import argparse
import os

import numpy as np

### lst imports
from models.LEAStereo.retrain.LEAStereo import LEAStereo 
from models.LEAStereo.utils.colorize import get_color_map
from models.LEAStereo.utils.multadds_count import count_parameters_in_MB, comp_multadds


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



def get_hardcoded_leastereo_args():
    args = argparse.Namespace(
        middlebury=1,
        maxdisp=192,
        crop_height=576,
        crop_width=960,
        data_path='models/LEAStereo/dataset/kitti2015/testing/',
        test_list='models/LEAStereo/dataloaders/lists/kitti2015_test.list',
        save_path='models/LEAStereo/predict/kitti2015/images/',
        fea_num_layer=6,
        mat_num_layers=12,
        fea_filter_multiplier=8,
        fea_block_multiplier=4,
        fea_step=3,
        mat_filter_multiplier=8,
        mat_block_multiplier=4,
        mat_step=3,
        net_arch_fea='models/LEAStereo/run/sceneflow/best/architecture/feature_network_path.npy',
        cell_arch_fea='models/LEAStereo/run/sceneflow/best/architecture/feature_genotype.npy',
        net_arch_mat='models/LEAStereo/run/sceneflow/best/architecture/matching_network_path.npy',
        cell_arch_mat='models/LEAStereo/run/sceneflow/best/architecture/matching_genotype.npy',
        resume='models/LEAStereo/run/Kitti15/best/best.pth',
        cuda=True,
        fea_num_layers=6,
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
    opt = get_hardcoded_leastereo_args()
    torch.backends.cudnn.benchmark = True
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print('===> Building LEAStereo model')
    leastereo_model = LEAStereo(opt)

    print('Total Params = %.2fMB' % count_parameters_in_MB(leastereo_model))
    print('Feature Net Params = %.2fMB' % count_parameters_in_MB(leastereo_model.feature))
    print('Matching Net Params = %.2fMB' % count_parameters_in_MB(leastereo_model.matching))
    
    mult_adds = comp_multadds(leastereo_model, input_size=(3,opt.crop_height, opt.crop_width)) #(3,192, 192))
    print("compute_average_flops_cost = %.2fMB" % mult_adds)
    if cuda:
        leastereo_model = torch.nn.DataParallel(leastereo_model).cuda()
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            leastereo_model.load_state_dict(checkpoint['state_dict'], strict=True)      
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    
    
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
        lst_disp = leastereo_model(image1, right_img)
        np_lst_disp = lst_disp.squeeze(0).cpu().numpy()

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