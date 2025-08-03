import os
from pathlib import Path
import json
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
# import torchvision.transforms as transforms
import copy
import matplotlib.pyplot as plt
import h5py
import cv2
from cadlib.visualize import vec2CADsolid, create_CAD
from cadlib.macro import *
from cadlib.extrude import *
# from cadlib.extrude import *
from .bool_op import *
from .three_view import *
from .draw_sketch import *
from .iou import *
from .utils import *
from .three_view import *
from utils.img_utils import normalize_images, denormalize_images

import torchvision.utils as vutils

DEBUG = False

class ViewBuilderDataset(Dataset):
    def __init__(self, data_file, cfg, mode):
        self.filelist = data_file
        self.mode = mode
        self.cfg = cfg

        if self.cfg.norm_img:
            if self.cfg.norm_use_mean_std_file:
                norm_path = Path(self.cfg.data_offload_root) / "mean_std.json"
                with open(norm_path, "r") as f:
                    norm_dict = json.load(f)
                self.img_mean = np.array(norm_dict["mean"]) / 255.0
                self.img_std = np.array(norm_dict["std"]) / 255.0
            else:
                self.img_mean = np.array([0.5, 0.5, 0.5])
                self.img_std = np.array([0.5, 0.5, 0.5])

    def resize_images(self, gt_view=None, shapes_view=None, sketch_img=None):
        """
        Resize images based on the fine-tuned ViT configuration flags.
        
        Args:
            gt_view (np.ndarray, optional): Ground truth view images
            shapes_view (np.ndarray, optional): Shape view images  
            sketch_img (list, optional): Sketch images
            
        Returns:
            tuple: Resized images (gt_view, shapes_view, sketch_img)
        """
        target_height = self.cfg.img_height
        target_width = self.cfg.img_width
        
        resized_gt_view = gt_view
        resized_shapes_view = shapes_view
        resized_sketch_img = sketch_img
        
        # Resize gt_view and shapes_view if use_finetuned_vit_for_gt is enabled
        if hasattr(self.cfg, 'use_finetuned_vit_for_gt') and self.cfg.use_finetuned_vit_for_gt:
            if gt_view is not None:
                resized_gt_view = np.zeros((gt_view.shape[0], target_height, target_width, gt_view.shape[3]), dtype=gt_view.dtype)
                for i in range(gt_view.shape[0]):
                    resized_gt_view[i] = cv2.resize(gt_view[i], (target_width, target_height))
            
            if shapes_view is not None:
                resized_shapes_view = np.zeros((shapes_view.shape[0], shapes_view.shape[1], target_height, target_width, shapes_view.shape[4]), dtype=shapes_view.dtype)
                for j in range(shapes_view.shape[0]):
                    for i in range(shapes_view.shape[1]):
                        resized_shapes_view[j, i] = cv2.resize(shapes_view[j, i], (target_width, target_height))
        
        # Resize sketch_img if use_finetuned_vit_for_sketch is enabled
        if hasattr(self.cfg, 'use_finetuned_vit_for_sketch') and self.cfg.use_finetuned_vit_for_sketch:
            if sketch_img is not None:
                resized_sketch_img = []
                for i in range(len(sketch_img)):
                    if len(sketch_img[i]) > 0:
                        resized_step_imgs = np.zeros((len(sketch_img[i]), target_height, target_width, sketch_img[i].shape[-1]), dtype=sketch_img[i].dtype)
                        for j in range(len(sketch_img[i])):
                            resized_step_imgs[j] = cv2.resize(sketch_img[i][j], (target_width, target_height))
                        resized_sketch_img.append(resized_step_imgs)
                    else:
                        resized_sketch_img.append(sketch_img[i])
        
        return resized_gt_view, resized_shapes_view, resized_sketch_img

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):

        if self.mode != "test":
            # Get the current filename
            filename = self.filelist[idx]
            
            if self.cfg.run_inference_reverse:
                # Legacy h5 file handling for reverse inference
                h5_path = Path(self.cfg.h5_data) / (filename + ".h5")
                with h5py.File(h5_path, "r") as fp:
                    state = fp["state"][:]
                    action = fp["action"][:]
                    reward = fp["reward"][:]
                
                state = torch.tensor(state / 255.0, dtype=torch.float)
                action = torch.tensor(action, dtype=torch.short)
                reward = torch.tensor(reward, dtype=torch.float)
                return {"state": state, "action": action, "reward": reward}
            else:
                if hasattr(self.cfg, 'use_new_data_format') and self.cfg.use_new_data_format:
                    # New data loading from separate h5 files in h5_data
                    data_dirs = {
                        'gt_view': Path(self.cfg.h5_data) / 'gt_view',
                        'vec_raw': Path(self.cfg.h5_data) / 'vec_raw',
                        'shapes_view': Path(self.cfg.h5_data) / 'shapes_view',
                        'sketch_img': Path(self.cfg.h5_data) / 'sketch_img',
                        'iou_seq': Path(self.cfg.h5_data) / 'iou_seq'
                    }
                    
                    # Load ground truth view
                    with h5py.File(data_dirs['gt_view'] / f"{filename}.h5", 'r') as h5f:
                        gt_view = h5f['data'][:]
                    
                    # Load vector data
                    with h5py.File(data_dirs['vec_raw'] / f"{filename}.h5", 'r') as h5f:
                        vec_raw = h5f['data'][:]
                    
                    # Only load visual state and reward data if not using prefix_actions_only mode
                    if not (hasattr(self.cfg, 'use_prefix_actions_only') and self.cfg.use_prefix_actions_only):
                        # Load shapes view
                        with h5py.File(data_dirs['shapes_view'] / f"{filename}.h5", 'r') as h5f:
                            shapes_view = h5f['data'][:]
                        
                        # Load IoU sequence
                        with h5py.File(data_dirs['iou_seq'] / f"{filename}.h5", 'r') as h5f:
                            iou_seq = h5f['data'][:]
                        
                        # Load sketch images
                        sketch_img = []
                        with h5py.File(data_dirs['sketch_img'] / f"{filename}.h5", 'r') as h5f:
                            sketch_group = h5f['steps']
                            for j in range(len(sketch_group.keys())):
                                img = sketch_group[str(j)][:]
                                sketch_img.append(img)
                    else:
                        # Set dummy values when not needed
                        shapes_view = None
                        iou_seq = None
                        sketch_img = None
                else:
                    # Old data loading from a single h5 file
                    h5_path = Path(self.cfg.h5_data) / (filename+".h5")
                    with h5py.File(h5_path, 'r') as h5f:
                        gt_view = h5f['gt_view'][:]
                        vec_raw = h5f['vec_raw'][:]
                        
                        # Only load visual state and reward data if not using prefix_actions_only mode
                        if not (hasattr(self.cfg, 'use_prefix_actions_only') and self.cfg.use_prefix_actions_only):
                            shapes_view = h5f['shapes_view'][:]
                            iou_seq = h5f['iou_seq'][:]
                            
                            # Load sketch images
                            sketch_img = []
                            step_group = h5f['sketch_img']
                            for j in range(len(step_group.keys())):
                                img = step_group[str(j)][:]
                                sketch_img.append(img)
                        else:
                            # Set dummy values when not needed
                            shapes_view = None
                            iou_seq = None
                            sketch_img = None

                # Apply image resizing based on fine-tuned ViT configuration
                gt_view, shapes_view, sketch_img = self.resize_images(gt_view, shapes_view, sketch_img)

                if self.cfg.norm_img:
                    # Normalize images - only normalize non-None data
                    if shapes_view is not None and sketch_img is not None:
                        gt_view, shapes_view, sketch_img = self.normalize_images(gt_view, shapes_view, sketch_img, self.cfg.norm_use_mean_std_file)
                    else:
                        # Only normalize gt_view when other data is None
                        gt_view = self.normalize_gt_view_only(gt_view, self.cfg.norm_use_mean_std_file)

                # Apply gt_view augmentation for training robustness
                gt_view = self.apply_gt_view_augmentation(gt_view)

                return self.collect_data(vec_raw, gt_view, shapes_view, sketch_img, iou_seq)
        else:
            # Test mode handling
            if not DEBUG:
                step_path = str(Path(self.cfg.step_path) / (self.filelist[idx]+".step"))  
                assert os.path.exists(step_path), f"step file {step_path} not found"
                # As we modify the shape in inference, there's no need to load shape from step file
                data = self.collect_data_test()
                data["shape"] = step_path
                data["gt_view"] = torch.zeros((3, self.cfg.img_height, self.cfg.img_width, 3), dtype=torch.float)
            else:
                # for debug
                h5_path = Path(self.cfg.h5_data) / (self.filelist[idx]+".h5")     
                with h5py.File(h5_path, "r") as fp:
                    state = fp["state"][:]
                    action = fp["action"][:]
                    reward = fp["reward"][:]
                state = torch.tensor(state / 255.0, dtype=torch.float)
                action = torch.tensor(action, dtype=torch.short)
                reward = torch.tensor(reward, dtype=torch.float)
                data = {"state": state, "action": action, "reward": reward}
                data["shape"] = self.filelist[idx+4]

            return data

    def normalize_images(self, gt_view, shapes_view, sketch_img, use_extern_norm=False):
        """
        Normalize gt_view, shapes_view, and sketch_img using mean and std from a JSON file
        or default values of [0.5, 0.5, 0.5].

        Args:
            gt_view (np.ndarray): Ground truth view images.
            shapes_view (np.ndarray): Shape view images.
            sketch_img (np.ndarray): Sketch images.
            norm_path (Path): Path to the JSON file containing mean and std values.
            use_default_norm (bool): If True, uses [0.5, 0.5, 0.5] as mean and std for all images.

        Returns:
            tuple: Normalized gt_view, shapes_view, and sketch_img.
        """
        # Helper function to normalize a single image
        def normalize_image(img, mean, std):
            # Convert to channel-first (C, H, W) format and scale to [0, 1]
            img_chw = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            # Normalize
            img_norm = (img_chw - mean[:, None, None]) / std[:, None, None]
            # Convert back to channel-last (H, W, C) format
            return img_norm.transpose(1, 2, 0)

        if not use_extern_norm:
            # Use default normalization values [0.5, 0.5, 0.5] for all images
            default_mean = self.img_mean
            default_std = self.img_std
            
            # Create mean and std with the same structure as the original
            gt_view_mean = [default_mean, default_mean, default_mean]
            gt_view_std = [default_std, default_std, default_std]
            shapes_view_mean = [default_mean, default_mean, default_mean]
            shapes_view_std = [default_std, default_std, default_std]
            sketch_img_mean = default_mean
            sketch_img_std = default_std
        else:
            # Split mean and std for different image types
            gt_view_mean, gt_view_std = self.img_mean[:9], self.img_std[:9]
            shapes_view_mean, shapes_view_std = self.img_mean[9:18], self.img_std[9:18]
            sketch_img_mean, sketch_img_std = self.img_mean[18:], self.img_std[18:]

            # Reshape mean and std for gt_view and shapes_view
            gt_view_mean = [gt_view_mean[:3], gt_view_mean[3:6], gt_view_mean[6:]]
            gt_view_std = [gt_view_std[:3], gt_view_std[3:6], gt_view_std[6:]]
            shapes_view_mean = [shapes_view_mean[:3], shapes_view_mean[3:6], shapes_view_mean[6:]]
            shapes_view_std = [shapes_view_std[:3], shapes_view_std[3:6], shapes_view_std[6:]]

        # Normalize gt_view
        gt_view_norm = np.empty_like(gt_view, dtype=np.float32)
        for i in range(len(gt_view)):
            gt_view_norm[i] = normalize_image(gt_view[i], gt_view_mean[i], gt_view_std[i])
        
        # Normalize shapes_view
        shapes_view_norm = np.empty_like(shapes_view, dtype=np.float32)
        for j in range(shapes_view.shape[0]):
            for i in range(shapes_view.shape[1]):
                shapes_view_norm[j, i] = normalize_image(shapes_view[j, i], shapes_view_mean[i], shapes_view_std[i])

        # Normalize sketch_img (modified in place as in the original code)
        for i in range(len(sketch_img)):
            if len(sketch_img[i]) > 0:
                sketch_img_norm = np.empty_like(sketch_img[i], dtype=np.float32)
                for j in range(len(sketch_img[i])):
                    sketch_img_norm[j] = normalize_image(sketch_img[i][j], sketch_img_mean, sketch_img_std)
                sketch_img[i] = sketch_img_norm
        
        return gt_view_norm, shapes_view_norm, sketch_img

    def normalize_gt_view_only(self, gt_view, use_extern_norm=False):
        """
        Normalize only gt_view images when other data is not needed.
        
        Args:
            gt_view (np.ndarray): Ground truth view images.
            use_extern_norm (bool): Whether to use external normalization parameters.
            
        Returns:
            np.ndarray: Normalized gt_view.
        """
        # Helper function to normalize a single image
        def normalize_image(img, mean, std):
            # Convert to channel-first (C, H, W) format and scale to [0, 1]
            img_chw = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            # Normalize
            img_norm = (img_chw - mean[:, None, None]) / std[:, None, None]
            # Convert back to channel-last (H, W, C) format
            return img_norm.transpose(1, 2, 0)

        if not use_extern_norm:
            # Use default normalization values [0.5, 0.5, 0.5] for all images
            default_mean = self.img_mean
            default_std = self.img_std
            
            # Create mean and std for gt_view
            gt_view_mean = [default_mean, default_mean, default_mean]
            gt_view_std = [default_std, default_std, default_std]
        else:
            # Split mean and std for gt_view only
            gt_view_mean, gt_view_std = self.img_mean[:9], self.img_std[:9]
            # Reshape mean and std for gt_view
            gt_view_mean = [gt_view_mean[:3], gt_view_mean[3:6], gt_view_mean[6:]]
            gt_view_std = [gt_view_std[:3], gt_view_std[3:6], gt_view_std[6:]]

        # Normalize gt_view
        gt_view_norm = np.empty_like(gt_view, dtype=np.float32)
        for i in range(len(gt_view)):
            gt_view_norm[i] = normalize_image(gt_view[i], gt_view_mean[i], gt_view_std[i])
        
        return gt_view_norm

    def apply_gt_view_augmentation(self, gt_view):
        """
        Apply data augmentation to gt_view by randomly setting images to black.
        - 50% probability: mask one image
        - 25% probability: mask two images  
        - 25% probability: no masking
        
        Args:
            gt_view (np.ndarray): Ground truth view images with shape (3, H, W, C)
            
        Returns:
            np.ndarray: Augmented gt_view with same shape
        """
        if not self.cfg.use_gt_view_augmentation or self.mode != "train":
            return gt_view
        
        gt_view_aug = gt_view
        
        # Generate random probability
        rand_prob = random.random()
        
        if rand_prob < self.cfg.gt_view_mask_prob_two:
            # Mask two images (25% probability)
            indices_to_mask = random.sample(range(3), 2)
        elif rand_prob < self.cfg.gt_view_mask_prob_two + self.cfg.gt_view_mask_prob_one:
            # Mask one image (50% probability) 
            indices_to_mask = random.sample(range(3), 1)
        else:
            # No masking (25% probability)
            indices_to_mask = []
        
        # Set selected images to black
        for idx in indices_to_mask:
            gt_view_aug[idx] = np.zeros_like(gt_view_aug[idx])
            
        return gt_view_aug

    def get_vec_shape(self, idx):
        json_path = Path(self.cfg.json_data) / (self.filelist[idx]+".json") 
        with open(json_path, "r", encoding='utf-8') as f:
            json_data = json.load(f)
        cad_seq = CADSequence.from_dict(json_data)
        cad_seq.normalize()                    # normalize
        cad_seq.numericalize()                 # numericalize()
        vec = cad_seq.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=False)
        op_pos_before = np.where(vec[:, 0]>SOL_IDX)[0]
        # remove SOL & EOS
        vec_raw = vec[(vec[:, 0] != SOL_IDX) & (vec[:, 0]!= EOS_IDX)]
        # cut to max_total_len
        vec_raw = vec_raw[:self.cfg.max_total_len]
        # if there's sketch ops left at end, remove them
        op_pos = np.where(vec_raw[:, 0]>SOL_IDX)[0]
        vec_raw = vec_raw[:op_pos[-1]+1]
        # cut cad_seq.seq to the same
        cad_seq.seq = cad_seq.seq[:len(op_pos)]
        # # get shape
        # shape = create_CAD(cad_seq, False)
        # cut vec
        vec = vec[:op_pos_before[len(op_pos)-1]+1]
        vec_raw = np.split(vec_raw, op_pos+1)
        # reverse and concatenate
        vec_raw = np.concatenate(np.flip(vec_raw, axis=0), axis=0)
        return vec_raw,vec,cad_seq
    
    def collect_data(self, vec_raw, gt_view, shapes_view, sketch_img, iou_seq):
        # Check if we're in prefix_actions_only mode
        if hasattr(self.cfg, 'use_prefix_actions_only') and self.cfg.use_prefix_actions_only:
            return self.collect_data_prefix_actions_only(vec_raw, gt_view)
        
        # Full data collection (original logic)
        blank_img = np.zeros((self.cfg.img_height, self.cfg.img_width, 3), dtype=np.uint8)
        
        # Always pre-allocate arrays with max_ep size
        max_ep = self.cfg.max_ep
        
        # Pre-allocate arrays instead of concatenating repeatedly
        state = np.zeros((max_ep, 4, self.cfg.img_height, self.cfg.img_width, 3), dtype=shapes_view.dtype)
        action = np.zeros((max_ep, N_ARGS + 1), dtype=vec_raw.dtype)
        reward = np.zeros(max_ep, dtype=np.float32)
        
        # Initialize first state and action
        state[0, :3] = shapes_view[0]
        state[0, 3] = blank_img
        action[0] = np.array([SOL_IDX] + [-1] * N_ARGS)
        reward[0] = 1. if self.cfg.use_desired_reward else 0.
        
        # Track current position in state/action/reward arrays
        curr_idx = 1
        ep_num = 0
        skt_num = 0
        
        # Fill arrays efficiently
        for i in range(len(vec_raw)):
            if curr_idx >= max_ep:
                break
                
            if vec_raw[i][0] < SOL_IDX:
                # Copy previous state, only update sketch image
                state[curr_idx] = state[curr_idx-1]
                state[curr_idx, 3] = sketch_img[ep_num][skt_num]
                reward[curr_idx] = reward[curr_idx-1]
                skt_num += 1
            else:
                ep_num += 1  # A new episode begins
                state[curr_idx, :3] = shapes_view[ep_num]
                state[curr_idx, 3] = blank_img
                if self.cfg.use_desired_reward:
                    reward[curr_idx] = 1. - iou_seq[ep_num]
                else:
                    reward[curr_idx] = iou_seq[ep_num]
                skt_num = 0
                
            action[curr_idx] = vec_raw[i]
            curr_idx += 1
        
        # If we didn't fill up to max_ep, pad remaining slots
        if curr_idx < max_ep:
            # Pad remaining slots with EOS and zeros
            for i in range(curr_idx, max_ep):
                action[i] = np.array([EOS_IDX] + [-1] * N_ARGS)
                # State remains zeros which is equivalent to blank images
                # Reward remains zeros
        
        # Convert to tensors with proper types
        if self.cfg.norm_img:
            gt_view = torch.tensor(gt_view, dtype=torch.float)
            state_tensor = torch.tensor(state, dtype=torch.float)
        else:
            gt_view = torch.tensor(gt_view / 255.0, dtype=torch.float)
            state_tensor = torch.tensor(state / 255.0, dtype=torch.float)
            
        if self.cfg.use_continuous_params:
            action_tensor = torch.tensor(action, dtype=torch.float)
        else:
            action_tensor = torch.tensor(action, dtype=torch.short)
            
        reward_tensor = torch.tensor(reward, dtype=torch.float)
        
        return {"gt_view": gt_view, "state": state_tensor, "action": action_tensor, "reward": reward_tensor}
    
    def collect_data_prefix_actions_only(self, vec_raw, gt_view):
        """
        Simplified data collection for prefix_actions_only mode.
        Only processes gt_view and action data, skipping visual state and reward processing.
        """
        max_ep = self.cfg.max_ep
        
        # Pre-allocate action array
        action = np.zeros((max_ep, N_ARGS + 1), dtype=vec_raw.dtype)
        
        # Initialize first action
        action[0] = np.array([SOL_IDX] + [-1] * N_ARGS)
        
        # Fill actions efficiently - only process vec_raw
        curr_idx = 1
        for i in range(len(vec_raw)):
            if curr_idx >= max_ep:
                break
            action[curr_idx] = vec_raw[i]
            curr_idx += 1
        
        # Pad remaining slots with EOS if needed
        if curr_idx < max_ep:
            for i in range(curr_idx, max_ep):
                action[i] = np.array([EOS_IDX] + [-1] * N_ARGS)
        
        # Convert to tensors
        if self.cfg.norm_img:
            gt_view = torch.tensor(gt_view, dtype=torch.float)
        else:
            gt_view = torch.tensor(gt_view / 255.0, dtype=torch.float)
            
        if self.cfg.use_continuous_params:
            action_tensor = torch.tensor(action, dtype=torch.float)
        else:
            action_tensor = torch.tensor(action, dtype=torch.short)
        
        # # Create dummy state and reward tensors for compatibility
        # # These won't be used in the model forward pass
        # blank_img = np.zeros((self.cfg.img_height, self.cfg.img_width, 3), dtype=np.uint8)
        # state_dummy = np.zeros((max_ep, 4, self.cfg.img_height, self.cfg.img_width, 3), dtype=np.uint8)
        # reward_dummy = np.zeros(max_ep, dtype=np.float32)
        
        # if self.cfg.norm_img:
        #     state_tensor = torch.tensor(state_dummy, dtype=torch.float)
        # else:
        #     state_tensor = torch.tensor(state_dummy / 255.0, dtype=torch.float)
        # reward_tensor = torch.tensor(reward_dummy, dtype=torch.float)
        
        return {"gt_view": gt_view, "action": action_tensor}
    
    def collect_data_test(self):
        # blank sketch image
        blank_img = np.zeros((self.cfg.img_height, self.cfg.img_width, 3), dtype=np.uint8)
        if self.cfg.run_inference_reverse:
            # get three view shapes
            # views = get_three_view_shapes([shape], self.cfg.img_height, self.cfg.img_width)
            # assemble state, action, reward in new dictionary
            # action is SOL, reward is 0
            state = [[blank_img]*4]     # only paddings
        else:
            tri_view = get_three_view_shapes([None], 384, 384)[0]
            # Apply resizing if fine-tuned ViT for gt is enabled
            if hasattr(self.cfg, 'use_finetuned_vit_for_gt') and self.cfg.use_finetuned_vit_for_gt:
                tri_view_resized = []
                for img in tri_view:
                    tri_view_resized.append(cv2.resize(img, (self.cfg.img_width, self.cfg.img_height)))
                tri_view = tri_view_resized
            if self.cfg.norm_img:
                tri_view = normalize_images(np.array(tri_view) / 255.0, self.img_mean, self.img_std)
            state = [np.concatenate([tri_view, [blank_img]], axis=0)]
        action = [np.array([SOL_IDX] + [-1] * N_ARGS)]
        reward = [0.] if not self.cfg.use_desired_reward else [1.]

        # turn them to tensor
        if self.cfg.norm_img:
            state = torch.tensor(np.array(state), dtype=torch.float)
        else:
            state = torch.tensor(np.array(state) / 255.0, dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.short)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        return {"state": state, "action": action, "reward": reward}

def get_dataloaders(train_file, val_file, cfg):
    train_dataset = ViewBuilderDataset(train_file, cfg, "train")
    val_dataset = ViewBuilderDataset(val_file, cfg, "val")
    # test_dataset = ViewBuilderDataset(test_file, cfg, "test")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_dataloader, val_dataloader
