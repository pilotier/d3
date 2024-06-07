import torch.utils.data as data
import os 
import torch
import cv2
import numpy as np 
import torch.nn.functional as F

class D3_loader____(data.Dataset):
    def __init__(self, root_dir = "/home/beast/data/d3/v2/validation/"):
        self.fx = 800.
        self.baseline = 0.12
        self.fy = 800.0
        self.cx = 960.0
        self.cy = 540.0
        self.intrinsics = np.array([self.fx, self.fy, self.cx, self.cy])
        self.data_tuples = []
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        previous_folder_last_image = None

        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir, "left")
            left_files = sorted(os.listdir(subdir_path))
            if previous_folder_last_image:
                left_files = left_files[1:]
            for i in range(len(left_files) - 1):
                self.data_tuples.append((os.path.join(subdir_path, left_files[i]), os.path.join(subdir_path, left_files[i + 1])))
            previous_folder_last_image = os.path.join(subdir_path, left_files[-1])

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        
        left_img_path_t, left_img_path_t1 = self.data_tuples[idx]
        right_img_path_t = left_img_path_t.replace('left', 'right')
        depth_map_path_t = left_img_path_t.replace('left', 'depth').replace('.png', '.npz')
        depth_map_path_t1 = left_img_path_t1.replace('left', 'depth').replace('.png', '.npz')
        flow_map_path_t = left_img_path_t.replace('left', 'flow').replace('.png', '.npz')
        z_flow_map_path_t = left_img_path_t.replace('left', 'z_flow').replace('.png', '.npz')
        
        left_img_t = cv2.imread(left_img_path_t)
        left_img_t1 = cv2.imread(left_img_path_t1)
        right_img_t = cv2.imread(right_img_path_t)
        depth_map_t = np.load(depth_map_path_t)['arr_0']
        depth_map_t1 = np.load(depth_map_path_t1)['arr_0']
        flow_map = np.load(flow_map_path_t)['arr_0']
        z_flow_map = np.load(z_flow_map_path_t)['arr_0']
        
        eval_h, eval_w = 576, 960
        left_img_t = cv2.resize(left_img_t, (eval_w, eval_h))
        left_img_t1 = cv2.resize(left_img_t1, (eval_w, eval_h))
        right_img_t = cv2.resize(right_img_t, (eval_w, eval_h))
        depth_map_t = cv2.resize(depth_map_t, (eval_w, eval_h))
        depth_map_t1 = cv2.resize(depth_map_t1, (eval_w, eval_h))
        flow_map = cv2.resize(flow_map, (eval_w, eval_h))
        z_flow_map = cv2.resize(z_flow_map, (eval_w, eval_h))
        
        left_img_t = torch.from_numpy(left_img_t).permute(2, 0, 1).float()
        left_img_t1 = torch.from_numpy(left_img_t1).permute(2, 0, 1).float()
        right_img_t = torch.from_numpy(right_img_t).permute(2, 0, 1).float()
        depth_map_t = torch.from_numpy(depth_map_t).float().squeeze(0)
        depth_map_t1 = torch.from_numpy(depth_map_t1).float().squeeze(0)
        flowxy = torch.from_numpy(flow_map).float().squeeze(0)
        z_flow_map = torch.from_numpy(z_flow_map).float().squeeze(0)
        
        flowz = (1.0 / depth_map_t1 - 1.0 / depth_map_t).unsqueeze(-1)
        disp1  = (self.fx * self.baseline) / depth_map_t
        
        flowxyz = torch.cat([flowxy, flowz], dim=-1)
        
        depth12 = (self.fx / (disp1 + z_flow_map)).float()
        

        # Normalization can be applied here or in a transform
        return left_img_t, left_img_t1, right_img_t, depth_map_t, depth_map_t1, flow_map, flowxyz

        ######## only returning integers here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # return "1", "2", "3", "4", "5", "6", "7"


    # def normalize_image(self, image):
    #     image = image[:, [2,1,0]]
    #     mean = torch.as_tensor([0.485, 0.456, 0.406], device=image.device)
    #     std = torch.as_tensor([0.229, 0.224, 0.225], device=image.device)
    #     return (image/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

    # def prepare_images_and_depths(self, image1, image2):
    #     """ padding, normalization, and scaling """
        
    #     ht, wd = image1.shape[-2:]

    #     pad_h = (-ht) % 8
    #     pad_w = (-wd) % 8

    #     image1 = F.pad(image1, [0,pad_w,0,pad_h], mode='replicate')
    #     image2 = F.pad(image2, [0,pad_w,0,pad_h], mode='replicate')

    #     image1 = self.normalize_image(image1.float())
    #     image2 = self.normalize_image(image2.float())

    #     return image1, image2 