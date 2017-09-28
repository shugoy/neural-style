import argparse
import os
import sys
import csv
from PIL import Image
import numpy as np
from visdom import Visdom

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
parser.add_argument('--mask', default='', type=str, help='mask image')
# parser.add_argument('--out-dir', default='out', type=str, help='output image name')
parser.add_argument('--saved-loss', default='loss.npz', type=str, help='save loss array')
parser.add_argument('--image-size', type=int, default=256, metavar='N', help='image size ')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='num epochs ')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='num epochs ')
parser.add_argument('--content-weight', default=1.0, type=float, help='content weight')
parser.add_argument('--style-weight', default=1e5, type=float, help='style weight')

parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--cuda', action='store_true', default=True,help='enables CUDA training')
parser.add_argument('--gpu-id', type=int, default=0, metavar='S',help='gpu id')
parser.add_argument('--init-noize', action='store_true', default=False, help='initial input')
parser.add_argument('--scale', action='store_true', default=False, help='initial input')
parser.add_argument('--center-crop', action='store_true', default=False, help='initial input')
parser.add_argument('--layer-weight', action='store_true', default=False, help='layer weight')

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

def content_loss(x, y, mask):
  #  print(mask.size())
 #   mask = mask.expand(1,x.data.size(1),x.data.size(2),x.data.size(3))
 #   print (mask.size())
#    print (x.size())
  #  x = x.mul(mask)
  #  y = y.mul(mask)
    return mse_loss(x, y)

def style_loss(x, y, mask):
    mask = mask.expand(1, x.data.size(1),x.data.size(2),x.data.size(3))
    x = x.mul(mask)
#    y = y.mul(mask)
    return mse_loss(gram_matrix(x), gram_matrix(y))

image_transform = transforms.Compose([transforms.Scale(args.image_size), transforms.ToTensor()])
if args.scale:
    image_transform = transforms.Compose([transforms.Scale(args.image_size), transforms.ToTensor()])
if args.center_crop:
    image_transform = transforms.Compose([transforms.CenterCrop(args.image_size), transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    image = image_transform(image)
    image = Variable(image)
    image = image.unsqueeze(0)
    if args.cuda:
        image = image.cuda()
    return image
# load images
kwargs = {'num_workers': 1, 'pin_memory': True}  if args.cuda else {}
content = image_loader(args.content)
style = image_loader(args.style)
mask = Variable(torch.ones(1,1,content.data.size(2),content.data.size(3)).cuda())
if args.mask != '':
    mask = Image.open(args.mask)
    mask = mask.convert('L')
    mask = image_transform(mask)
    mask = Variable(mask)
    mask = mask.unsqueeze(0)
    if args.cuda:
        mask = mask.cuda()
mp = nn.MaxPool2d(2, stride=2)
mask1 = mask
mask2 = mp(mask1)
mask3 = mp(mask2)
mask4 = mp(mask3)
mask5 = mp(mask4)

input_data = content.data    
if args.init_noize:
    input_data = torch.randn(content.data.size()).type(torch.cuda.FloatTensor)
input_param = nn.Parameter(input_data)
optimizer = optim.LBFGS([input_param])

Fc = vgg(content)
Fs = vgg(style)


viz = Visdom()
viz.text('style_weight:{:.2e}\ncontent_weight:{:.2e}'.format(args.style_weight, args.content_weight))
# viz.text('content_weight:{}'.format(args.content_weight))
textwindow = viz.text('')
imageWindow = viz.images(input_param.data.cpu().numpy())
styleLossWindow = viz.line(Y=np.array([0]))
contentLossWindow = viz.line(Y=np.array([0]))
loss_style = []
loss_content = []
if args.start_epoch > 0:
    temp = np.load(args.saved_loss)
    loss_style = list(temp['style'])
    loss_content = list(temp['content'])



print_flag = [0]
# if not os.path.exists(args.out_dir):
#     os.mkdir(args.out_dir)
layer_weight = [1,1,1,1,1]
if args.layer_weight:
    layer_weight = [2,1.5,1,0.75,0.5]

for epoch in range(args.start_epoch, args.epochs):
    print_flag[0] = True
    def closure():
        optimizer.zero_grad()

        Fi = vgg(input_param)

        Lc  = content_loss(Fi.relu5_3, Fc.relu5_3, mask5) * args.content_weight
        Ls1 = style_loss(Fi.relu1_2, Fs.relu1_2, mask1) * args.style_weight * layer_weight[0]
        Ls2 = style_loss(Fi.relu2_2, Fs.relu2_2, mask2) * args.style_weight * layer_weight[1]
        Ls3 = style_loss(Fi.relu3_3, Fs.relu3_3, mask3) * args.style_weight * layer_weight[2]
        Ls4 = style_loss(Fi.relu4_3, Fs.relu4_3, mask4) * args.style_weight * layer_weight[3]
        Ls5 = style_loss(Fi.relu5_3, Fs.relu5_3, mask5) * args.style_weight * layer_weight[4]
     
        Lc.backward( retain_graph=True)
        Ls1.backward(retain_graph=True)
        Ls2.backward(retain_graph=True)
        Ls3.backward(retain_graph=True)
        Ls4.backward(retain_graph=True)
        Ls5.backward(retain_graph=True)
        Ls = Ls1 + Ls2 + Ls3 + Ls4 + Ls5
        if print_flag[0]:
            print ('epoch {} Lc:{:.2e}, Ls1:{:.2e}, Ls2:{:.2e}, Ls3:{:.2e}, Ls4:{:.2e}, Ls5:{:.2e}'.format(
		epoch, Lc.data[0], Ls1.data[0], Ls2.data[0], Ls3.data[0], Ls4.data[0], Ls5.data[0]))
            loss_style.append(Ls.data[0])
            loss_content.append(Lc.data[0])
            viz.text('Lc:{:.2e}\n Ls1:{:.2e}\n Ls2:{:.2e}\n Ls3:{:.2e}\n Ls4:{:.2e}\n Ls5:{:.2e}'.format(
		epoch, Lc.data[0], Ls1.data[0], Ls2.data[0], Ls3.data[0], Ls4.data[0], Ls5.data[0]),
            win=textwindow, opts=dict(title='epoch_{}'.format(epoch))
            )
            
            print_flag[0] = False
        return Ls + Lc
    
    optimizer.step(closure)

    input_param.data.clamp_(0,1)
    
    viz.images(input_param.data.cpu().numpy(), win=imageWindow, opts=dict(title='epoch_{}'.format(epoch)))
    viz.line(Y=np.array(loss_style), X=np.array(range(epoch+1)),win=styleLossWindow, opts=dict(title='style loss epoch:{}'.format(epoch)))
    viz.line(Y=np.array(loss_content), X=np.array(range(epoch+1)),win=contentLossWindow, opts=dict(title='content loss epoch_{}'.format(epoch)))
    np.savez(args.saved_loss, style=np.array(loss_style), content=np.array(loss_content))
   
print ('finish')
