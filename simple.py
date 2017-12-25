import argparse
import os
import sys
import csv
from PIL import Image
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

parser = argparse.ArgumentParser(description='PyTorch Neural Style')
parser.add_argument('--content', default='images/content-images/brad_pitt.jpg', type=str, help='content image')
parser.add_argument('--style', default='images/style-images/starry_night.jpg', type=str, help='style image')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='num epochs ')
parser.add_argument('--content-weight', default=1.0, type=float, help='content weight')
parser.add_argument('--style-weight', default=1e5, type=float, help='style weight')

parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--cuda', action='store_true', default=True,help='enables CUDA training')
parser.add_argument('--gpu-id', type=int, default=0, metavar='S',help='gpu id')

args = parser.parse_args()
print (args)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.set_device(args.gpu_id)
    torch.cuda.manual_seed(args.seed)

## load model
from vgg import Vgg16
vgg = Vgg16(requires_grad=False)
if args.cuda:
    vgg.cuda()

mse_loss = nn.MSELoss()
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def content_loss(x, y):
    return mse_loss(x, y)

def style_loss(x, y):
    return mse_loss(gram_matrix(x), gram_matrix(y))

image_transform = transforms.Compose([transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    image = image_transform(image)
    image = Variable(image)
    image = image.unsqueeze(0)
    if args.cuda:
        image = image.cuda()
    return image

kwargs = {'num_workers': 1, 'pin_memory': True}  if args.cuda else {}
content = image_loader(args.content)
style = image_loader(args.style)

input_data = content.data    
input_param = nn.Parameter(input_data)
optimizer = optim.LBFGS([input_param])

Fc = vgg(content)
Fs = vgg(style)

print_flag = [0]
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

for epoch in range(args.start_epoch, args.epochs):
    print_flag[0] = True
    def closure():
        optimizer.zero_grad()

        Fi = vgg(input_param)

        Lc  = content_loss(Fi.relu4_3, Fc.relu4_3) * args.content_weight
        Ls1 = style_loss(Fi.relu1_2, Fs.relu1_2) * args.style_weight
        Ls2 = style_loss(Fi.relu2_2, Fs.relu2_2) * args.style_weight
        Ls3 = style_loss(Fi.relu3_3, Fs.relu3_3) * args.style_weight
        Ls4 = style_loss(Fi.relu4_3, Fs.relu4_3) * args.style_weight
        Ls5 = style_loss(Fi.relu5_3, Fs.relu5_3) * args.style_weight
     
        Lc.backward( retain_graph=True)
        Ls1.backward(retain_graph=True)
        Ls2.backward(retain_graph=True)
        Ls3.backward(retain_graph=True)
        Ls4.backward(retain_graph=True)
        Ls5.backward(retain_graph=True)
        Ls = Ls1 + Ls2 + Ls3 + Ls4 + Ls5
        if print_flag[0]:
            print ('epoch {} Lc:{:.2e}, Ls:{:2e}, Ls1:{:.2e}, Ls2:{:.2e}, Ls3:{:.2e}, Ls4:{:.2e}, Ls5:{:.2e}'.format(
		            epoch, Lc.data[0], Ls.data[0], Ls1.data[0], Ls2.data[0], Ls3.data[0], Ls4.data[0], Ls5.data[0]))
            print_flag[0] = False
        return Ls + Lc
    
    optimizer.step(closure)

    input_param.data.clamp_(0,1)
    save_path = os.path.join(args.out_dir, "epoch_{}.png".format(epoch))
    vutils.save_image(input_param.data, save_path)
    
