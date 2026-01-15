import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt
import time
import random
import re

def normalized(image):
    return (image - image.min()) / (image.max() - image.min())

def dowmsampling(image, h_ratio, w_ratio, flag = True):
    h, w = image.shape[:2]
    if flag:
        lr_patch = cv2.resize(image, (w // w_ratio, h // h_ratio),
                              interpolation=cv2.INTER_CUBIC)
    else:
        lr_patch = np.zeros((h, w), dtype=image.dtype)
        lr_patch[::h_ratio, ::w_ratio] = image[::h_ratio, ::w_ratio]
    return lr_patch
def is_high_frequency_patch(img_patch, brightness_threshold=3, variance_threshold1=10, variance_threshold2=200):
    brightness_mean = np.mean(img_patch)
    brightness_var = np.var(img_patch)

    if (brightness_mean < brightness_threshold or brightness_var < variance_threshold1 or
            (brightness_mean< 10 and brightness_var > variance_threshold2)):
        return False, {
            'brightness_mean': brightness_mean,
            'brightness_var': brightness_var,
        }
    return True, {
        'brightness_mean': brightness_mean,
        'brightness_var': brightness_var,
    }

def random_rotate(patch, angle):
    if angle == 90:
        return cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(patch, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return patch

def random_flip(patch, flip_type):
    if flip_type is None:
        return patch
    return cv2.flip(patch, flip_type)

def add_gaussian_noise(patch, sigma):
    gauss = np.random.normal(0, sigma, patch.shape)
    noisy_patch = patch + gauss
    noisy_patch = np.clip(noisy_patch, 0, 1)
    return noisy_patch

class VideoSRDataset(Dataset):
    def __init__(self, __lr_dirs__, __hr_dir__, __hr_index__, __patch_size__=96, __scale_factor__=8):
        self.lr_dirs = __lr_dirs__
        self.hr_dir = __hr_dir__
        self.hr_index = __hr_index__
        self.patch_size = __patch_size__
        self.scale_factor = __scale_factor__
        self.lr_patch_size = __patch_size__//__scale_factor__
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])


    def __getitem__(self, idx):

        hr_frame = cv2.imread(f"{self.hr_dir}/{self.hr_index[idx]}_warped.png", 0)
        h, w = hr_frame.shape
        while True:
            h_start = np.random.randint(0, h - self.patch_size + 1)
            w_start = np.random.randint(0, w - self.patch_size + 1)
            hr_patch = hr_frame[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
            is_high_freq, _ = is_high_frequency_patch(hr_patch)
            if is_high_freq:
                break
        lr_h_start = h_start//self.scale_factor
        hr_h_start = lr_h_start * self.scale_factor
        lr_w_start = w_start//self.scale_factor
        hr_w_start = lr_w_start * self.scale_factor
        hr_patch = hr_frame[hr_h_start:hr_h_start + self.patch_size, hr_w_start:hr_w_start + self.patch_size]

        angle = random.choice([0, 90, 180, 270])
        flip_type = random.choice([-1, 0, 1, None])
        sigma = np.random.uniform(0.001, 0.025)

        # 随机旋转
        hr_patch = random_rotate(hr_patch, angle)
        # 随机翻转
        hr_patch = random_flip(hr_patch, flip_type)
        hr_patch = normalized(hr_patch).astype(np.float32)
        hr_patch = self.transform(hr_patch)
        lr_patches = []
        for t in range(5):
            # Read
            lr_frame = cv2.imread(f"{self.lr_dirs[idx]}/{t+1}_warped.png", 0)
            lr_patch = lr_frame[lr_h_start:lr_h_start + self.lr_patch_size, lr_w_start:lr_w_start + self.lr_patch_size]

            lr_patch = random_rotate(lr_patch, angle)
            lr_patch = random_flip(lr_patch, flip_type)

            lr_patch = add_gaussian_noise(lr_patch,sigma)
            lr_patch = normalized(lr_patch).astype(np.float32)
            lr_patches.append(self.transform(lr_patch))
        __lr_sequence = torch.stack(lr_patches)
        return __lr_sequence, hr_patch

    def __len__(self):
        return  len(self.lr_dirs)

def get_all_folders(path):
    folder_paths = []
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            folder_path = os.path.join(root, directory)
            folder_paths.append(folder_path)
    return folder_paths

if __name__ == "__main__":
    video_dirs  = get_all_folders('dataset6x')
    hr_index = [int(re.search(r'data(\d+)', s).group(1)) for s in video_dirs]
    # print(video_dirs)
    # print(hr_index)
    hr_dir = 'rawdata2'
    train_dataset = VideoSRDataset(__lr_dirs__ = video_dirs, __hr_dir__ = hr_dir, __patch_size__=384, __hr_index__ = hr_index, __scale_factor__=8)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=1)
    # print(train_loader)
    batch_idx, (lr_sequence, hr_target) = list(enumerate(train_loader))[0]
    fig, axes = plt.subplots(2, 2)
    start_epoch = time.time()
    for epoch in range(2):
        for batch_idx,(lr_seq, hr_target) in enumerate(train_loader):
            if batch_idx == 1:
                print(lr_seq.shape)
                print(np.min(lr_seq.squeeze()[0][2].numpy()))
                print(np.min(hr_target.squeeze()[0].numpy()))
                print(hr_target.shape)
                axes[epoch][0].imshow(lr_seq.squeeze()[0][2])
                axes[epoch][1].imshow(hr_target.squeeze()[0])
        else:
            pass
    plt.show()
