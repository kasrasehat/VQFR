import os
import cv2
import math
import numpy as np
import torch
import sys
sys.path.append("/home/user1/kasra/pycharm-projects/VQFR")
import random
from vqfr.data import degradations as degradations
from vqfr.data.transforms import augment
from vqfr.utils import img2tensor
import tqdm
import csv
import csv
import cv2
import random
import numpy as np
from scipy.ndimage import convolve

class LowQualityImageGenerator:
    def __init__(self, opt):
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = os.path.join(os.path.dirname(self.gt_folder), 'lq')
        os.makedirs(self.lq_folder, exist_ok=True)

        self.csv_file_path = os.path.join(os.path.dirname(self.gt_folder), 'degradation_params.csv')
        self.create_csv_header()

        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']

        self.crop_components = opt.get('crop_components', False)
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)

        if self.crop_components:
            self.components_list = torch.load(opt.get('component_path'))

        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']
        self.recon_prob = opt.get('recon_prob', 0)

        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        self.gray_prob = opt.get('gray_prob')

        self.color_jitter_shift /= 255.

    def create_csv_header(self):
        header = ['Image_Name', 'Blur_Sigma', 'Downsample_Scale', 'Noise_Std', 'JPEG_Quality', 'Color_Jitter_Shift', 'Grayscale_Applied']
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    def log_degradation_params(self, img_name, blur_sigma, scale, noise_std, jpeg_quality, color_jitter_shift, grayscale_applied):
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([img_name, blur_sigma, scale, noise_std, jpeg_quality, color_jitter_shift, grayscale_applied])

    def generate_lq_images(self):
        image_paths = [os.path.join(self.gt_folder, f) for f in os.listdir(self.gt_folder) if os.path.isfile(os.path.join(self.gt_folder, f))]
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            img_gt = cv2.imread(img_path)
            img_gt = img_gt.astype(np.float32) / 255.0

            # Random horizontal flip
            img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
            h, w, _ = img_gt.shape

            if self.crop_components:
                locations = self.get_component_coordinates(img_name, status)
                loc_left_eye, loc_right_eye, loc_mouth = locations

            if random.random() < self.recon_prob:
                img_lq = img_gt
                blur_sigma, scale, noise_std, jpeg_quality, color_jitter_shift, grayscale_applied = (None, None, None, None, None, False)
            else:
                # Degrade the image
                img_lq, blur_sigma, scale, noise_std, jpeg_quality, color_jitter_shift, grayscale_applied = self.degrade_image(img_gt, w, h)

            # Save the low-quality image
            lq_path = os.path.join(self.lq_folder, img_name)
            cv2.imwrite(lq_path, img_lq * 255.0)

            # Log degradation parameters to CSV
            self.log_degradation_params(img_name, blur_sigma, scale, noise_std, jpeg_quality, color_jitter_shift, grayscale_applied)

    def degrade_image(self, img_gt, w, h):
        
        # Blur
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        blur_sigma = self.blur_sigma

        # Downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)

        # Add noise
        if self.noise_range is not None:
            noise_std = random.uniform(self.noise_range[0], self.noise_range[1])
            img_lq = degradations.random_add_gaussian_noise(img_lq, [noise_std, noise_std])
        else:
            noise_std = None

        # JPEG compression
        if self.jpeg_range is not None:
            jpeg_quality = random.randint(self.jpeg_range[0], self.jpeg_range[1])
            img_lq = degradations.random_add_jpg_compression(img_lq, [jpeg_quality, jpeg_quality])
        else:
            jpeg_quality = None

        # Resize back to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # Random color jitter
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
            color_jitter_shift = self.color_jitter_shift
        else:
            color_jitter_shift = None

        # Random grayscale
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            grayscale_applied = True
        else:
            grayscale_applied = False

        return img_lq, blur_sigma, scale, noise_std, jpeg_quality, color_jitter_shift, grayscale_applied

    @staticmethod
    def color_jitter(img, shift):
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    def get_component_coordinates(self, index, status):
        components_bbox = self.components_list[f'{index:08d}']
        if status[0]:  # hflip
            # Exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # Modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # Get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations
    
 

class LowQualityImageGeneratorV2:
    def __init__(self, opt):
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = os.path.join(os.path.dirname(self.gt_folder), 'lq')
        os.makedirs(self.lq_folder, exist_ok=True)

        self.csv_file_path = os.path.join(os.path.dirname(self.gt_folder), 'degradation_params.csv')
        self.create_csv_header()

        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']

        self.crop_components = opt.get('crop_components', False)
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)

        if self.crop_components:
            self.components_list = torch.load(opt.get('component_path'))

        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.lens_blur_radius = opt['lens_blur_radius']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']
        self.recon_prob = opt.get('recon_prob', 0)

        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        self.gray_prob = opt.get('gray_prob')
        
        # Parameters for motion blur
        self.motion_blur_prob = opt.get('motion_blur_prob', 0.0)
        self.motion_blur_amount = opt.get('motion_blur_amount', [5, 15])
        # parameters  for impulse noise
        self.impulse_noise_prob = opt.get('impulse_noise_prob', 0.0)
        self.impulse_noise_amount = opt.get('impulse_noise_amount', [0.01, 0.05])
        # parameters for denoising
        self.denoise_prob = opt.get('denoise_prob', 0.0)
        self.denoise_strength_range = opt.get('denoise_strength_range', [5, 15])
        # parameters for pixelation
        self.pixelate_prob = opt.get('pixelate_prob', 0.0)
        self.pixelate_factor = opt.get('pixelate_factor', [2, 8])

        self.color_jitter_shift /= 255.

    def create_csv_header(self):
        header = ['Image_Name', 'blur_kernel_size', 'gau_kernel_type', 'gau_sigma_x', 'gau_sigma_y', 'gau_rotation', 'gau_noise', 'Lens_Blur_Radius', 'motion_blur_amount',
                  'Downsample_Scale', 'impulse_noise_amount', 
                  'Noise_Std', 'pixelate_factor', 'JPEG_Quality', 'Denoise_Strength', 'Color_Jitter_Shift', 'Grayscale_Applied']
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    def log_degradation_params(self, img_name, blur_kernel_size, gau_kernel_type, gau_sigma_x, gau_sigma_y, gau_rotation, gau_noise, lens_blur_radius, motion_blur_amount, 
                               scale, impulse_noise_amount, noise_std, pixelate_factor, jpeg_quality, Denoise_Strength, color_jitter_shift, grayscale_applied):
        
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([img_name, blur_kernel_size, gau_kernel_type, gau_sigma_x, gau_sigma_y, gau_rotation, gau_noise, lens_blur_radius, motion_blur_amount, 
                             scale, impulse_noise_amount, noise_std, pixelate_factor, jpeg_quality, Denoise_Strength, color_jitter_shift, grayscale_applied])

    def generate_lq_images(self):
        image_paths = [os.path.join(self.gt_folder, f) for f in os.listdir(self.gt_folder) if os.path.isfile(os.path.join(self.gt_folder, f))]
        for img_path in tqdm.tqdm(image_paths):
            img_name = os.path.basename(img_path)
            img_gt = cv2.imread(img_path)
            img_gt = img_gt.astype(np.float32) / 255.0

            # Random horizontal flip
            img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
            h, w, _ = img_gt.shape

            if self.crop_components:
                locations = self.get_component_coordinates(img_name, status)
                loc_left_eye, loc_right_eye, loc_mouth = locations

            if random.random() < self.recon_prob:
                img_lq = img_gt
                blur_kernel_size, gau_kernel_type, gau_sigma_x, gau_sigma_y, gau_rotation, gau_noise, lens_blur_radius, motion_blur_amount, scale, impulse_noise_amount, noise_std, pixelate_factor, jpeg_quality, Denoise_Strength, color_jitter_shift, grayscale_applied = (None, None, None, None, None, None, None, None, False)
            else:
                # Degrade the image
                img_lq, blur_kernel_size, gau_kernel_type, gau_sigma_x, gau_sigma_y, gau_rotation, gau_noise, lens_blur_radius, motion_blur_amount, scale, impulse_noise_amount, noise_std, pixelate_factor, jpeg_quality, Denoise_Strength, color_jitter_shift, grayscale_applied = self.degrade_image(img_gt, w, h)

            # Save the low-quality image
            lq_path = os.path.join(self.lq_folder, img_name)
            cv2.imwrite(lq_path, img_lq * 255.0)

            # Log degradation parameters to CSV
            self.log_degradation_params(img_name, blur_kernel_size, gau_kernel_type, gau_sigma_x, gau_sigma_y, gau_rotation, gau_noise, lens_blur_radius, motion_blur_amount,
                                         scale, impulse_noise_amount,noise_std, pixelate_factor, jpeg_quality, Denoise_Strength, color_jitter_shift, grayscale_applied)

    def degrade_image(self, img_gt, w, h):
        img_lq = img_gt.copy()  # Start with the original high-quality image

        # Apply Gaussian Blur based on probability
        if random.random() < self.opt.get('gaussian_blur_prob', 0.5):
            # kernel_type = random.choices(self.kernel_list, weights=self.kernel_prob)[0]
            gau_kernel_type, gau_sigma_x, gau_sigma_y, gau_rotation, gau_noise, kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, 
                [-math.pi, math.pi],
                noise_range=None)
            img_lq = cv2.filter2D(img_lq, -1, kernel)

        else:
            gau_kernel_type, gau_sigma_x, gau_sigma_y, gau_rotation, gau_noise = None, None,  None, None, None
            # kernel_type = None



        # Apply Lens Blur based on probability
        if random.random() < self.opt.get('lens_blur_prob', 0.1):
            lens_blur_radius = self.lens_blur_radius  # Set the radius or make it adjustable
            img_lq = degradations.lens_blur(img_lq, radius=lens_blur_radius)
        else:
            lens_blur_radius = None
            
        # Initialize logging parameter
        motion_blur_amount = None
        
        # Apply motion blur
        if random.random() < self.motion_blur_prob:
            amount = random.randint(self.motion_blur_amount[0], self.motion_blur_amount[1])
            img_lq = degradations.imblurmotion(img_lq, amount)
            motion_blur_amount = amount  # For logging    

        # Downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        
        impulse_noise_amount = None  # For logging

        # Add impulse noise
        if random.random() < self.impulse_noise_prob:
            amount = random.uniform(self.impulse_noise_amount[0], self.impulse_noise_amount[1])
            img_lq = degradations.imnoiseimpulse(img_lq, amount)
            impulse_noise_amount = amount  # For logging
        
        # Add Gaussian Noise
        if self.noise_range is not None:
            noise_std = random.uniform(self.noise_range[0], self.noise_range[1])
            img_lq = degradations.random_add_gaussian_noise(img_lq, [noise_std, noise_std])
        else:
            noise_std = None

        pixelate_factor = None  # For logging

        # Apply pixelation
        if random.random() < self.pixelate_prob:
            factor = random.randint(self.pixelate_factor[0], self.pixelate_factor[1])
            img_lq = degradations.impixelate(img_lq, factor)
            pixelate_factor = factor  # For logging

        # JPEG compression
        if self.jpeg_range is not None:
            jpeg_quality = random.randint(self.jpeg_range[0], self.jpeg_range[1])
            img_lq = degradations.random_add_jpg_compression(img_lq, [jpeg_quality, jpeg_quality])
        else:
            jpeg_quality = None
        
        # Resize back to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Random color jitter
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
            color_jitter_shift = self.color_jitter_shift
        else:
            color_jitter_shift = None

        denoise_strength = None
        # Apply denoising
        if random.random() < self.denoise_prob:
            strength = random.uniform(self.denoise_strength_range[0], self.denoise_strength_range[1])
            img_lq = degradations.imdenoise(img_lq, strength)
            denoise_strength = strength  # For logging
        
        # Random grayscale
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            grayscale_applied = True
        else:
            grayscale_applied = False

        return img_lq, self.blur_kernel_size, gau_kernel_type, gau_sigma_x, gau_sigma_y, gau_rotation, gau_noise, lens_blur_radius, motion_blur_amount, scale, impulse_noise_amount, noise_std, pixelate_factor, jpeg_quality, denoise_strength, color_jitter_shift, grayscale_applied

    @staticmethod
    def color_jitter(img, shift):
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    def get_component_coordinates(self, index, status):
        components_bbox = self.components_list[f'{index:08d}']
        if status[0]:  # hflip
            # Exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # Modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # Get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations
   