from model import VSRRDN
from model_c import FD_UNet, SRResNet
from model_RDN import RDN
import torch
import cv2
import numpy as np
from torchvision import transforms
import statistics
from utilis import PSNRCalculator
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

def upsampling(image, h_ratio, w_ratio, flag = True):
    h, w = image.shape[:2]
    if flag:
        sr_patch = cv2.resize(image, (w * w_ratio, h * h_ratio),
                          interpolation=cv2.INTER_CUBIC)
    else:
        sr_patch = cv2.resize(image, (w * w_ratio, h * h_ratio),
                              interpolation=cv2.INTER_LINEAR)
    return sr_patch

if __name__ == '__main__':
    scale_factor = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_index = 1 #1-Proposed,2-FD_Unet,3-SRRDN,4-SRGAN,5-bicubic,6-bilinear
    Dataset_Path = 'Result/test/data'
    Sample_Flag = True
    if model_index == 1:
        generator = VSRRDN(upscale_factor=scale_factor).to(device)
        generator.eval()
        generator.load_state_dict(torch.load('Result/proposed_{}x.pth'.format(scale_factor), map_location=device))
    elif model_index == 2:
        generator = FD_UNet().to(device)
        generator.eval()
        generator.load_state_dict(torch.load('Result/FD_UNet_{}x.pth'.format(scale_factor), map_location=device))
        Sample_Flag = False
    elif model_index == 3:
        generator = RDN(upscale_factor=scale_factor).to(device)
        generator.eval()
        generator.load_state_dict(torch.load('Result/SRRDN_{}x.pth'.format(scale_factor), map_location=device))
    elif model_index == 4:
        generator = SRResNet(upscale=scale_factor).to(device)
        generator.eval()
        generator.load_state_dict(torch.load('Result/SRGAN_{}x.pth'.format(scale_factor), map_location=device))
    elif model_index == 5:
        interp_method_flag = True
    elif model_index == 6:
        interp_method_flag = False
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    total_ssim = []
    total_psnr = []
    total_lpips = []
    for num in range(10):
        hr_path = 'testdata/HR/{}_warped.png'.format(num+1)
        hr_frame = cv2.imread(hr_path, 0)
        if model_index == 1:
            lr_frames = []
            lr_path = 'testdata/dataset{}x/data{}'.format(scale_factor, num+1)
            for t in range(5):
                # Read
                lr_frame = cv2.imread(f"{lr_path}/{t + 1}_warped.png", 0)
                lr_frame = normalized(lr_frame).astype(np.float32)
                lr_frames.append(transform(lr_frame))
            lr_sequence = torch.stack(lr_frames)
            lr_sequence = lr_sequence.unsqueeze(0).to(device)
            with torch.no_grad():
                hr_pred = generator(lr_sequence)
        elif model_index == 5 or model_index == 6:
            down = dowmsampling(hr_frame, scale_factor, scale_factor, Sample_Flag)
            hr_pred = upsampling(down, scale_factor, scale_factor, interp_method_flag)
            hr_pred = transform(hr_pred).unsqueeze(0).to(device)
        else:
            hr_frame = normalized(hr_frame).astype(np.float32)
            lr_patch = dowmsampling(hr_frame, scale_factor, scale_factor, Sample_Flag)
            lr_patch = normalized(lr_patch)
            lr_patch = transform(lr_patch)
            lr_patch = lr_patch.unsqueeze(0)
            lr_sequence = lr_patch.to(device)
            with torch.no_grad():
                hr_pred = generator(lr_sequence)
        # Count Index
        hr_frame = normalized(hr_frame).astype(np.float32)
        hr_frame = transform(hr_frame)
        lpips_loss_fn = lpips.LPIPS(net='vgg',verbose=False).to(device)
        lpips_value = lpips_loss_fn(hr_pred, hr_frame.unsqueeze(0).to(device)).item()
        psnr_count = PSNRCalculator().to(device)
        ssim_count = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None).to(device)
        psnr = psnr_count(hr_pred, hr_frame.unsqueeze(0).to(device)).item()
        ssim = ssim_count(hr_pred, hr_frame.unsqueeze(0).to(device)).item()
        print(f"{ssim:.4f}, {psnr:.2f}, {lpips_value:.4f}")
        total_ssim.append(ssim)
        total_psnr.append(psnr)
        total_lpips.append(lpips_value)
    maen_ssim = statistics.mean(total_ssim)
    maen_psnr = statistics.mean(total_psnr)
    maen_lpips = statistics.mean(total_lpips)
    var_ssim = statistics.pstdev(total_ssim)
    var_psnr = statistics.pstdev(total_psnr)
    var_lpips = statistics.pstdev(total_lpips)
    print(f"{maen_ssim:.4f}-{var_ssim:.4f}, {maen_psnr:.2f}-{var_psnr:.2f}, {maen_lpips:.4f}-{var_lpips:.4f}")


