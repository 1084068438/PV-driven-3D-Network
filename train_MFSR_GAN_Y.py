import argparse
import os
import sys
import re
import torch.optim as optim
from load_dataset import VideoSRDataset
from model import VSRRDN, Discriminator
import torch
from torch.utils.data import DataLoader
from torch.autograd import grad
from utilis import CharbonnierLoss, SSIMLoss, VGGFeatureExtractor
import warnings
import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn as nn
import time
warnings.filterwarnings("ignore", category=UserWarning)

def get_all_folders(path):
    folder_paths = []
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            folder_path = os.path.join(root, directory)
            folder_paths.append(folder_path)
    return folder_paths

def gradient_penalty(gp_discriminator, real_samples, fake_samples, gp_device='cuda'):
    gp_alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(gp_device)
    gp_alpha = gp_alpha.expand_as(real_samples)

    interpolated = gp_alpha * real_samples + ((1 - gp_alpha) * fake_samples)
    interpolated = interpolated.to(gp_device)
    interpolated.requires_grad_(True)

    d_interpolated = gp_discriminator(interpolated)

    gradients = grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(d_interpolated.size()).to(gp_device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale_factor', type=int, default=6, help='Downsampling factor')
    parser.add_argument('--num_frames', type=int, default=5, help='input number of frames')
    parser.add_argument('--PreFlag', type=bool, default=True, help='whether to use pre-trained module')
    parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
    parser.add_argument('--StartEpochs', type=int, default=0, help='number of epochs to train for')
    parser.add_argument('--nPreEpochs', type=int, default=10, help='number of epochs to Pretrain for')
    parser.add_argument('--nEpochs', type=int, default=5000, help='number of epochs to train for')
    parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
    parser.add_argument('--generatorWeights', type=str, default='',
                        help='path to generator weights (to continue training)')
    parser.add_argument('--discriminatorWeights', type=str, default='',
                        help="path to discriminator weights (to continue training)")
    parser.add_argument('--datapath', type=str, default='Yao/dataset6x', help='folder of dataset')
    parser.add_argument('--out', type=str, default='checkpoint/Y_6x', help='folder to output model checkpoints')
    opt = parser.parse_args()
    print(opt)
    try:
        os.makedirs(opt.out)
    except OSError:
        pass
    # TensorBoard
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"exp_{timestamp}"
    log_dir = os.path.join("runs", experiment_name)
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device('cuda')
    video_dirs = get_all_folders(opt.datapath)
    hr_index = [int(re.search(r'data(\d+)', s).group(1)) for s in video_dirs]
    train_dataset = VideoSRDataset(__lr_dirs__=video_dirs, __hr_dir__='Yao/HR', __hr_index__=hr_index, __patch_size__=192, __scale_factor__=opt.scale_factor)
    train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle=True, num_workers=8)
    len_train = len(train_loader)
    generator = VSRRDN(upscale_factor=opt.scale_factor).to(device)
    if opt.generatorWeights != '':
        generator.load_state_dict(torch.load(opt.generatorWeights))
    discriminator = Discriminator(__patch_size__=192).to(device)
    if opt.discriminatorWeights != '':
        discriminator.load_state_dict(torch.load(opt.discriminatorWeights))

    # define Loss
    pixel_criterion = CharbonnierLoss().to(device)
    ssim_criterion = SSIMLoss().to(device)
    # pretrain loss
    optim_generator = optim.AdamW(generator.parameters(), lr=0.0001)
    if opt.PreFlag:
        print('Generator pre-training')
        for epoch in range(opt.nPreEpochs):
            generator.train()
            epoch_loss = 0.0
            for batch_idx, (lr_seq, hr_target) in enumerate(train_loader):
                optim_generator.zero_grad()
                high_res_real = hr_target.to(device)
                high_res_fake = generator(lr_seq.to(device))

                pixel_loss = pixel_criterion(high_res_fake, high_res_real)

                pixel_loss.backward()
                optim_generator.step()
                epoch_loss += pixel_loss.item()
            epoch_loss /= len_train
            sys.stdout.write('\r[%d/%d] Pre_Generator_Loss: %.4f\n' % (epoch + 1, opt.nPreEpochs, epoch_loss))
        # Save checkpoint
        torch.save(generator.state_dict(), '%s/generator_pretrain.pth' % opt.out)

    print('GAN training')
    generator.load_state_dict(torch.load('%s/generator_pretrain.pth' % opt.out))
    optim_generator = optim.AdamW(generator.parameters(), lr=opt.generatorLR, weight_decay=0.0001)
    optim_discriminator = optim.AdamW(discriminator.parameters(), lr=opt.discriminatorLR, weight_decay=0.001)
    feat_criterion = nn.MSELoss().to(device)
    l_fea_w = 0.05
    l_adv_w = 0.01
    ssim_weight = 0.5
    l_fidelity_w = 1.0
    vgg_extractor = VGGFeatureExtractor(layer_index=7).to(device)
    generator_losses = []
    # generator_adv_losses = []
    # generator_fidelity_losses = []
    # generator_feat_losses = []
    # discriminator_losses    len_train = len(train_loader)
    for epoch in range(opt.StartEpochs, opt.nEpochs):
        epoch_start_time = time.time()

        generator.train()
        discriminator.train()
        mean_fidelity_loss = 0.0
        mean_adversarial_loss = 0.0
        mean_feat_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0
        for batch_idx, (lr_seq, hr_target) in enumerate(train_loader):

            for param in discriminator.parameters():
                param.requires_grad = True
            optim_discriminator.zero_grad()
            high_res_real = hr_target.to(device)
            high_res_fake = generator(lr_seq.to(device)).detach()
            pred_d_real = discriminator(high_res_real)
            pred_d_fake = discriminator(high_res_fake)
            discriminator_loss_gp = gradient_penalty(discriminator, high_res_real, high_res_fake)

            d_loss = -torch.mean(pred_d_real) + torch.mean(pred_d_fake) + 10 * discriminator_loss_gp
            d_loss.backward()
            optim_discriminator.step()
            mean_discriminator_loss += d_loss.item()

            for param in discriminator.parameters():
                param.requires_grad = False  # 冻结判别器参数
            optim_generator.zero_grad()
            high_res_fake = generator(lr_seq.to(device))

            fidelity_loss = pixel_criterion(high_res_fake, high_res_real) + ssim_weight * ssim_criterion(high_res_fake, high_res_real)

            pred_g_fake = discriminator(high_res_fake)
            g_adv_loss = -torch.mean(pred_g_fake)

            fake_features = vgg_extractor(high_res_fake)
            real_features = vgg_extractor(high_res_real)
            feat_loss = feat_criterion(fake_features, real_features)

            generator_total_loss = l_fidelity_w * fidelity_loss + l_adv_w * g_adv_loss + l_fea_w * feat_loss
            generator_total_loss.backward()
            optim_generator.step()
            mean_generator_total_loss += generator_total_loss.item()
            mean_fidelity_loss += l_fidelity_w * fidelity_loss.item()
            mean_adversarial_loss += l_adv_w * g_adv_loss.item()
            mean_feat_loss += l_fea_w * feat_loss.item()

        mean_discriminator_loss = mean_discriminator_loss / len_train
        mean_generator_total_loss = mean_generator_total_loss / len_train
        mean_fidelity_loss = mean_fidelity_loss / len_train
        mean_adversarial_loss = mean_adversarial_loss / len_train
        mean_feat_loss = mean_feat_loss / len_train
        generator_losses.append(mean_generator_total_loss)
        epoch_time = time.time() - epoch_start_time
        sys.stdout.write('\r[%d/%d] Generator_Loss: %.4f, Time: %.2f seconds\n' % (epoch+1, opt.nEpochs, mean_generator_total_loss, epoch_time))

        writer.add_scalar('Train/Generator_Loss', mean_generator_total_loss, epoch + 1)
        writer.add_scalar('Train/Mean_fidelity_loss', mean_fidelity_loss, epoch + 1)
        writer.add_scalar('Train/Mean_adversarial_loss', mean_adversarial_loss, epoch + 1)
        writer.add_scalar('Train/Mean_feat_loss', mean_feat_loss, epoch + 1)
        writer.add_scalar('Train/Mean_discriminator_loss', mean_discriminator_loss, epoch + 1)
        if ((epoch+1) % 100 == 0):
            torch.save(generator.state_dict(), f'{opt.out}/generator_epoch_{epoch+1}.pth')
    writer.close()

    plt.figure("Generator_Loss", (18, 6))
    plt.title("Generator_Loss")
    x = [i + 1 for i in range(len(generator_losses))]
    y = [generator_losses[i] for i in range(len(generator_losses))]
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.show()
