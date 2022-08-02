import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

class edgeV_loss(nn.Module):
    def __init__(self):
        super(edgeV_loss, self).__init__()

    def forward(self, X1, X2):
        X1_up = X1[:, :, :-1, :]
        X1_down = X1[:, :, 1:, :]
        X2_up = X2[:, :, :-1, :]
        X2_down = X2[:, :, 1:, :]
        return -np.log(int(torch.sum(torch.abs(X1_up - X1_down))) / int(torch.sum(torch.abs(X2_up - X2_down))))


def lsgan(real, fake, cri):
    real_label = Variable(torch.ones(real.size())).cuda()
    real_loss = cri(real, real_label)

    fake_label = Variable(torch.zeros(fake.size())).cuda()
    fake_loss = cri(fake, fake_label)

    return 0.5 * (fake_loss + real_loss)