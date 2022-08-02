from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from model_our import *
#from model import *
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

time_start = time.time()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cpu")
net = Generator()
BATCH = 1
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    #transforms.Resize([512, 960]),
    transforms.Resize([448, 800]),
    transforms.ToTensor()
])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    y = Image.open(filepath).convert('L')
    return y


class DatasetFromFolder(Dataset):
    def __init__(self, LR_image_dir, transforms=None):
        super(DatasetFromFolder, self).__init__()
        self.LR_image_filenames = sorted(
            [os.path.join(LR_image_dir, x) for x in os.listdir(LR_image_dir) if is_image_file(x)])
        self.LR_transform = transform

    def __getitem__(self, index):
        inputs = load_img(self.LR_image_filenames[index])
        LR_images = self.LR_transform(inputs)
        return LR_images

    def __len__(self):
        return len(self.LR_image_filenames)


net.load_state_dict(torch.load("./model_our3/netG_epoch_2_99.pth"))
processDataset2 = DatasetFromFolder(LR_image_dir='./data_test/1/noisy', transforms=transform)
testData = DataLoader(processDataset2, batch_size=BATCH)


def imshow(save_dir):
    """展示结果"""
    processBar = tqdm(enumerate(testData, 1))
    for i, (HR_images) in processBar:
        HR_images = Variable(HR_images.to(device))
        with torch.no_grad():
            exp_output = net(HR_images)[0, :, :, :]
        source = exp_output.cpu().detach().numpy()  # 转为numpy
        source = source.transpose((1, 2, 0))  # 切换形状
        source = np.clip(source, 0, 1)  # 修正图片
        source = np.squeeze(source)
        img1 = np.uint8(source * 255)
        save_path = os.path.join(save_dir, "{:02d}.jpg".format(i))
        img1 = Image.fromarray(img1)
        img1 = img1.resize((800, 448), Image.BILINEAR)
        #img1 = img1.resize((450, 450), Image.BILINEAR)
        img1 = img1.convert('L')
        img1.save(save_path)

imshow("./result")

time_end = time.time()
print('time cost', time_end - time_start, 's')