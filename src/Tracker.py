import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import kornia
import cv2 as cv
from timeit import default_timer as timer
from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer
from src.edam.optimization.utils_frame import synthetise_image_and_error
#from edam.utils.file import list_files
from src.edam.optimization.frame import create_frame
from src.edam.optimization.pose_estimation import PoseEstimation
from src.edam.utils.image.convertions import (
    numpy_array_to_pilimage,
    pilimage_to_numpy_array,
)
from src.edam.utils.image.pilimage import (
    pilimage_h_concat,
    pilimage_rgb_to_bgr,
    pilimage_v_concat,
)
class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args
        self.use_viewdirs = cfg['use_viewdirs']
        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']
        self.motion = cfg['motion']
        self.idx = slam.idx
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.logger = slam.logger
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.ckptsdir = slam.ckptsdir
        self.cam_lr = cfg['tracking']['lr']
        #self.cam_lr = args.lr
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.pe = PoseEstimation()
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=0) 
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def optimize_cam_in_batch(self, camera_tensor ,gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.
        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.
        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        
        viewdirs = batch_rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        c=None

        ret = self.renderer.render_batch_ray(c, self.decoders, batch_rays_d,
                                                batch_rays_o, viewdirs,device,stage = 'color',
                                                gt_depth=batch_gt_depth)
        depth, uncertainty, color = ret

        uncertainty = uncertainty.detach()
        if self.handle_dynamic:
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else:
            mask = batch_gt_depth > 0

        loss = (torch.abs(batch_gt_depth-depth) /
                torch.sqrt(uncertainty+1e-10))[mask].sum()
        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            loss += self.w_color_loss*color_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()
     
    def first_tracking_warping(self, gt_depth, gt_color, frame_id):
        '''
        First frame tracking
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        c2w = self.estimate_c2w_list[0]
        gray = cv.cvtColor(
            (((gt_color.cpu()).numpy())*255).astype(np.uint8), cv.COLOR_RGB2GRAY
            ) 
        k_depth = np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]])
        k_depth = kornia.utils.image_to_tensor(k_depth, keepdim=False).squeeze(1)  # Bx3x3
        c2w =  self.cor_convert(c2w.cpu().numpy())
        new_frame_ = create_frame(
                        c_pose_w=np.linalg.inv(c2w),
                        c_pose_w_gt=None,
                        gray_image=gray,
                        rgbimage=None,
                        depth=gt_depth.cpu().numpy(),
                        k=k_depth.numpy().reshape(3, 3),
                        idx=0,
                        ref_camera=True,
                        scales=1,
                        code_size=128,
                        device_name=self.device,
                        uncertainty=None,
                    )
        self.pe.set_ref_keyframe(new_frame_)

    def tracking_warping(self, gt_depth, gt_color, frame_id):

        gray = cv.cvtColor(
            (((gt_color.cpu()).numpy())*255).astype(np.uint8), cv.COLOR_RGB2GRAY
            ) 
        depth_np = gt_depth
        cur_c2w = self.estimate_c2w_list[frame_id - 1].cpu().numpy()
        init_c2w = self.estimate_c2w_list[0].cpu().numpy()
        k_depth = np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]])# type: ignore
        # -- Transform into tensors.
        k_depth = kornia.utils.image_to_tensor(k_depth, keepdim=False).squeeze(1)  # Bx3x3
        cur_c2w = self.cor_convert(cur_c2w)
        new_frame = create_frame(
            c_pose_w=np.linalg.inv(cur_c2w),
            c_pose_w_gt=None,
            gray_image=gray,
            rgbimage=None,
            depth=depth_np.cpu().numpy(),
            k=k_depth.numpy().reshape(3, 3),
            idx=frame_id,
            ref_camera=(frame_id == 0),
            scales=2,
            code_size=128,
            device_name=self.device,
            uncertainty=None,
        )
        new_frame_ = create_frame(
            c_pose_w=new_frame.c_pose_w,
            c_pose_w_gt=None,
            gray_image=gray,
            rgbimage=None,
            depth=depth_np.cpu().numpy(),
            k=k_depth.numpy().reshape(3, 3),
            idx=frame_id,
            ref_camera=True,
            scales=2,
            code_size=128,
            device_name=self.device,
            uncertainty=None,
        )
        self.pe.run(new_frame_, True)
        #self.pe.set_ref_keyframe(new_frame_)
        estimated_new_cam_c2w = np.linalg.inv(new_frame_.c_pose_w)
        estimated_new_cam_c2w = self.cor_convert(estimated_new_cam_c2w)
        self.estimate_c2w_list[frame_id] = torch.tensor(estimated_new_cam_c2w).to(self.device).float()
        return new_frame_

    def cor_convert(self,c2w1):
        c2w = c2w1.copy()
        init_c2w = self.estimate_c2w_list[0].clone().cpu().numpy()
        c2w[2,3] = -(c2w[2,3] - init_c2w[2,3])+init_c2w[2,3]
        c2w[1,3] = -(c2w[1,3] - init_c2w[1,3])+init_c2w[1,3]
        c2w[0,2]= -c2w[0,2]
        c2w[1,2]= -c2w[1,2]
        c2w[2,0]= -c2w[2,0]
        c2w[2,1]= -c2w[2,1] 
        c2w[0,1] *= -1
        c2w[1,0] *= -1
        c2w[1,2] *= -1
        c2w[2,1] *= -1
        
        return c2w

    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose using camera motion model
        '''
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.estimate_c2w_list[frame_id-1].to(self.device)
            self.estimate_c2w_list[frame_id] = c2w_est_prev
            
        else:
            c2w_est_prev_prev = self.estimate_c2w_list[frame_id-2].to(self.device)
            c2w_est_prev = self.estimate_c2w_list[frame_id-1].to(self.device)
            delta = c2w_est_prev@c2w_est_prev_prev.float().inverse()
            self.estimate_c2w_list[frame_id] = delta@c2w_est_prev
        
        return self.estimate_c2w_list[frame_id]

    def tracking_render(self, i,estimated_new_cam_c2w,gt_c2w, gt_depth, gt_color):
        camera_tensor = get_tensor_from_camera(
            torch.tensor(estimated_new_cam_c2w).detach())
        gt_camera_tensor = get_tensor_from_camera(
            torch.tensor(gt_c2w).detach())
        if self.seperate_LR:
            camera_tensor = camera_tensor.to(self.device).detach()
            T = camera_tensor[-3:]
            quad = camera_tensor[:4]
            cam_para_list_quad = [quad]
            quad = Variable(quad, requires_grad=True)
            T = Variable(T, requires_grad=True)
            camera_tensor = torch.cat([quad, T], 0)
            cam_para_list_T = [T]
            cam_para_list_quad = [quad]
            optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                                                    {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])
        else:
            camera_tensor = Variable(
                camera_tensor.to(self.device), requires_grad=True)
            cam_para_list = [camera_tensor]
            optimizer_camera = torch.optim.Adam(
                cam_para_list, lr=self.cam_lr)
        initial_loss_camera_tensor = torch.abs(
            gt_camera_tensor[-3:].to(self.device)-camera_tensor[-3:].to(self.device)).mean().item()
        candidate_cam_tensor = None
        current_min_loss = 10000000000.
        
        for cam_iter in range(self.num_cam_iters):
            if self.seperate_LR:
                camera_tensor = torch.cat([quad, T], 0).to(self.device)
            loss = self.optimize_cam_in_batch(
                    camera_tensor,gt_color, gt_depth, self.tracking_pixels, optimizer_camera)

            if cam_iter == 0:
                initial_loss = loss

            loss_camera_tensor = torch.abs(
                gt_camera_tensor[-3:].to(self.device)-camera_tensor[-3:]).mean().item()
            if cam_iter == self.num_cam_iters-1: 
                print(
                    f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                    f'camera tensor error: {initial_loss_camera_tensor:.8f}->{loss_camera_tensor:.8f}')
            if loss < current_min_loss: 
                current_min_loss = loss
                candidate_cam_tensor = camera_tensor.clone().detach()
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
            [1, 4])).type(torch.float32).to(self.device)
        c2w = get_camera_from_tensor(
            candidate_cam_tensor.clone().detach())
        c2w = torch.cat([c2w, bottom], dim=0)

        return c2w

    def run(self):
        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)
        if self.motion:
            pe = PoseEstimation()
        mm = [0,0,0,0,0,0,0,0,0,0,0,0]
        nn = [0,0,0,0,0,0,0,0,0,0,0,0]
        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            
            idx = idx[0]
            gt_depth = gt_depth[0] 
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]
            
            if self.sync_method == 'strict':
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):

                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[idx-1].to(device)
                    pass
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                # pure parallel, if mesh/vis happens may cause inbalance
                pass

            self.update_para_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0:
                estimated_new_cam_c2w = gt_c2w
                c2w = gt_c2w
                self.estimate_c2w_list[0] = estimated_new_cam_c2w
                if self.motion:
                    self.first_tracking_warping(gt_depth, gt_color, idx)  
            else:
                if self.motion:
                    new_frame_ = self.tracking_warping(gt_depth, gt_color,idx)
                    c2w = self.estimate_c2w_list[idx]
                else:
                    c2w = self.predict_current_pose(idx)
        
            #if idx>5: 
            estimated_pose = self.tracking_render(idx,c2w,gt_c2w, gt_depth, gt_color)
            if idx > 10:
                c2w = estimated_pose
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            self.estimate_c2w_list[idx] = torch.tensor(c2w).clone().cpu()

            if  self.motion and idx % 2 == 0 and idx > 0:
                modified_pose = self.cor_convert(c2w.cpu().numpy())
                new_frame_.modify_pose(c_pose_w=np.linalg.inv(modified_pose))    
                self.pe.set_ref_keyframe(new_frame_)

            save_traj =self.estimate_c2w_list[idx].clone().cpu().numpy()    
            aa = save_traj.reshape(16)[:12]
            
            mm = np.row_stack((mm,aa))
            bb = self.gt_c2w_list[idx].reshape(16)[:12].cpu().numpy()
            nn = np.row_stack((nn,bb))

            self.idx[0] = idx
            """
            if self.low_gpu_mem:
                with torch.cuda.device('cuda:3'):
                    torch.cuda.empty_cache()
            """
            if self.idx == self.n_img - 1 or self.idx %10 == 0:
                np.savetxt(f'{self.output}/mesh/traj.txt',
                    mm[1:],'%.9e',delimiter=' ',newline = '\n')
                np.savetxt(f'{self.output}/mesh/gt.txt',
                    nn[1:],'%.9e',delimiter=' ',newline = '\n')
           