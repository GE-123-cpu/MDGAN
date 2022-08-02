import argparse
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision.models.vgg import vgg16
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
from model_our import *
from loss import *
from torch.autograd import Variable
import pytorch_msssim
from torchvision.models.vgg import vgg16

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#path = './data/result'
# LR_image_dir = './noisy'
# HR_image_dir = './clean'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH = 1
EPOCHS = 100

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--step', type=int, default=50,
                    help='Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--gan_type', type=str, default='wgan-gp')
opt = parser.parse_args()

# 图像处理操作，包括随机裁剪，转换张量
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

transform1 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    y = Image.open(filepath).convert('L')
    return y


class DatasetFromFolder(Dataset):
    def __init__(self, LR_image_dir, HR_image_dir, transforms=None):
        super(DatasetFromFolder, self).__init__()
        self.LR_image_filenames = sorted(
            [os.path.join(LR_image_dir, x) for x in os.listdir(LR_image_dir) if is_image_file(x)])
        self.HR_image_filenames = sorted(
            [os.path.join(HR_image_dir, y) for y in os.listdir(HR_image_dir) if is_image_file(y)])
        self.LR_transform = transform
        self.HR_transform = transform1

    def __getitem__(self, index):
        inputs = load_img(self.LR_image_filenames[index])
        labels = load_img(self.HR_image_filenames[index])
        HR_images = self.HR_transform(labels)
        LR_images = self.LR_transform(inputs)
        return LR_images, HR_images

    def __len__(self):
        return len(self.LR_image_filenames)


# 构建数据集
processDataset = DatasetFromFolder(LR_image_dir='./noisy', HR_image_dir='./clean', transforms=transform)
trainData = DataLoader(processDataset, batch_size=BATCH)

# 构造VGG损失中的网络模型
vgg = vgg16(pretrained=True).to(device)
vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1).to(device)
lossNetwork = nn.Sequential(*list(vgg.features)[:31]).eval()
for param in lossNetwork.parameters():
    param.requires_grad = False  # 让VGG停止学习



# 构造模型

netD = PatchGAN(1)
netG = Generator()

netD.to(device)
netG.to(device)
# netG.load_state_dict(torch.load("D:\\chenkun\\amend_GAN\\root\\model_save1\\netG_epoch_4_179.pth"))

# 构造迭代器
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))  # , weight_decay=1e-4)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))  # , weight_decay=1e-4)

# 构造损失函数
cri1 = nn.MSELoss().to(device)
cri2 = nn.L1Loss().to(device)
eloss = edgeV_loss().to(device)


def adjust_learning_rate(epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    if lr < 1e-6:
        lr = 1e-6
    return lr


for epoch in range(EPOCHS):
    lr = adjust_learning_rate(epoch)  # - 1)
    netD.train()
    netG.train()
    processBar = tqdm(enumerate(trainData, 1))

    # Clip weights of discriminator
    # for p in netD.parameters():
    #    p.data.clamp_(-opt.clip_value, opt.clip_value)

    for param_group in optimizerG.param_groups:
        param_group["lr"] = lr
        print("epoch =", epoch, "lr =", optimizerG.param_groups[0]["lr"])

    for i, (LR_images, HR_images) in processBar:
        LR_images = Variable(LR_images.to(device))
        HR_images = Variable(HR_images.to(device))

        fakeImg = netG(LR_images).to(device)

        netD.zero_grad()
        realOut = netD(HR_images).mean()
        fakeOut = netD(fakeImg).mean()

        real = netD(HR_images)
        fake = netD(fakeImg.detach())
        dLoss = lsgan(real, fake, cri1)
        dLoss.backward(retain_graph=True)
        netG.zero_grad()
        fake = netD(fakeImg)
        Identity_loss = cri1(fakeImg, HR_images)
        gLossVGG = 0.006 * cri1(lossNetwork(fakeImg), lossNetwork(HR_images))
        edgloss = 0.0005 * eloss(fakeImg, HR_images)
        gLossGAN = 0.001 * cri1(fake, Variable(torch.ones(fake.size())).cuda())
        gLoss = Identity_loss + gLossGAN #+ gLossVGG# edgloss #+ gLossVGG  # + edgloss# + gLossssim #+ edgloss# + SSIMloss
        gLoss.backward()
        # optimizerD.step()
        optimizerD.step()
        optimizerG.step()

        # 数据可视化
        processBar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, EPOCHS, dLoss.item(), gLoss.item(), realOut.item(), fakeOut.item()))

        if (epoch + 1) % 20 == 0:
            torch.save(netG.state_dict(), 'model_our1/netG_epoch_%d_%d.pth' % (2, epoch))