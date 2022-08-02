import os
import torchvision.transforms as transforms
from PIL import Image
from model_our import *
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([960, 512]),
    transforms.ToTensor()
])

def test():
    time_start = time.time()

    net = Generator()
    net.load_state_dict(torch.load("./model_new/netG_epoch_2_99.pth"))

    noisy = Image.open("./1.jpg").convert("L")
    noisy = transform(noisy)
    noisy = torch.unsqueeze(noisy, dim=0)

    a = net(noisy)
    input = a[0, :, :, :]
    source = input.cpu().detach().numpy()  # 转为numpy
    source = source.transpose((1, 2, 0))  # 切换形状
    source = np.clip(source, 0, 1)  # 修正图片
    source = np.squeeze(source)
    img = Image.fromarray(np.uint8(source * 255))
    img = img.resize((500, 950), Image.BILINEAR)
    img = img.convert('L')


    img.save("./mask_up" + ".png")
    time_end = time.time()
    print('time cost', time_end - time_start, 's')


if __name__ == "__main__":
    test()