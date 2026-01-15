import matplotlib.pyplot as plt
from model import VSRRDN
import torch
import cv2
import numpy as np
from torchvision import transforms
from utilis import PSNRCalculator
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
import warnings
import torch.nn.functional as F
import os
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

def upsampling(image, h_ratio, w_ratio):
    h, w = image.shape[:2]
    sr_patch = cv2.resize(image, (w * w_ratio, h * h_ratio),
                          interpolation=cv2.INTER_CUBIC)
    return sr_patch


if __name__ == '__main__':
    scale_factor = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = VSRRDN(upscale_factor=scale_factor).to(device)
    generator.eval()
    generator.load_state_dict(torch.load('Result/proposed_{}x_2.pth'.format(scale_factor), map_location=device))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    lr_frames = []
    num = 3
    lr_path = 'testdata/dataset{}x/data{}'.format(scale_factor,num)
    hr_path = 'testdata/HR/{}_warped.png'.format(num)
    for t in range(5):
        # Read
        lr_frame = cv2.imread(f"{lr_path}/{t+1}_warped.png", 0)
        lr_frame = normalized(lr_frame).astype(np.float32)
        lr_frames.append(transform(lr_frame))
    lr_sequence = torch.stack(lr_frames)
    hr_frame = cv2.imread(hr_path, 0)
    lr_sequence = lr_sequence.unsqueeze(0).to(device)
    print(hr_frame.shape)
    with torch.no_grad():
        hr_pred = generator(lr_sequence)
    print(hr_pred.shape)
    result = hr_pred.squeeze().detach().cpu().numpy()
    print(np.max(result), np.min(result))
    result = normalized(result).astype(np.float32)
    hr_frame = normalized(hr_frame).astype(np.float32)
    print(np.max(hr_frame), np.min(hr_frame))
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(hr_frame, cmap='hot')
    axes[1].imshow(result, cmap='hot')
    plt.show()
    plt.figure()
    plt.imshow(hr_frame, cmap='afmhot')
    plt.axis('off')
    # plt.savefig('Result/HR.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.figure()
    plt.imshow(result, cmap='afmhot')
    plt.axis('off')
    # plt.savefig('Result/{}x/propose.png'.format(scale_factor), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    #Count Index
    hr_frame = transform(hr_frame).to(device)
    lpips_loss_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)
    lpips_value = lpips_loss_fn(hr_pred, hr_frame.unsqueeze(0).to(device)).item()
    psnr_count = PSNRCalculator().to(device)
    ssim_count = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None).to(device)
    psnr = psnr_count(hr_pred, hr_frame.unsqueeze(0).to(device)).item()
    ssim = ssim_count(hr_pred, hr_frame.unsqueeze(0).to(device)).item()
    print(f"SSIM: {ssim:.4f}, PSNR: {psnr:.2f}, LPIPS: {lpips_value:.4f}")

    # image_uint8 = (result * 255).astype(np.uint8)
    # save_path = 'Result/{}x/raw_propose.png'.format(scale_factor)
    # if os.path.exists(save_path):
    #     try:
    #         os.remove(save_path)
    #     except PermissionError:
    #         pass
    # cv2.imwrite(save_path, image_uint8)


