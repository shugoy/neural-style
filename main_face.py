import argparse
import os
import sys
from PIL import Image

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
parser.add_argument('--content', default='content.png', type=str, help='content image')
#parser.add_argument('--style', default='style.png', type=str, help='style image')
parser.add_argument('--data-dir', default='data', type=str, help='database dir name')
#parser.add_argument('--input-image', default='image', type=str, help='input image name')
parser.add_argument('--out-dir', default='out', type=str, help='output image name')
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='batch size ')
parser.add_argument('--image-size', type=int, default=512, metavar='N', help='image size ')
parser.add_argument('--epochs', type=int, default=20, metavar='N', help='num epochs ')
parser.add_argument('--content-weight', default=1.0, type=float, help='content weight')
parser.add_argument('--style-weight', default=1000, type=float, help='style weight')

parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--cuda', action='store_true', default=True,help='enables CUDA training')
parser.add_argument('--gpu-id', type=int, default=0, metavar='S',help='gpu id')
parser.add_argument('--init', type=str, default='content', metavar='S',help='initial input')

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

def content_loss(x, y, w):
    return mse_loss(x*w, y*w)

def style_loss(x, y, w):
    return mse_loss(gram_matrix(x)*w, gram_matrix(y)*w)

image_transform = transforms.Compose([transforms.CenterCrop(args.image_size), transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    image = image_transform(image)
    image = Variable(image)
    image = image.unsqueeze(0)
    if args.cuda:
        image = image.cuda()
    return image

## load images
kwargs = {'num_workers': 1, 'pin_memory': True}  if args.cuda else {}

# datasets.ImageFolder
#styledir = os.path.join(args.data_dir, 'test')

style_dataset = datasets.ImageFolder(args.data_dir,
        transforms.Compose([transforms.CenterCrop(args.image_size), transforms.ToTensor()])
        )

style_loader = torch.utils.data.DataLoader(
        style_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

content = image_loader(args.content)
#style = image_loader(args.style)

#if args.init == 'content':
input_data = content.data    
#else:
#    input_data = torch.randn(content.data.size()).type(torch.cuda.FloatTensor)
input_param = nn.Parameter(input_data)
optimizer = optim.LBFGS([input_param])

Fc = vgg(content)
#Fs = vgg(style)

Gc1 = gram_matrix(Fc.relu1_2)
Gc2 = gram_matrix(Fc.relu2_2)
Gc3 = gram_matrix(Fc.relu3_3)
Gc4 = gram_matrix(Fc.relu4_3)
Gs1 = Variable(torch.cuda.FloatTensor(Gc1.size()).zero_())
Gs2 = Variable(torch.cuda.FloatTensor(Gc2.size()).zero_())
Gs3 = Variable(torch.cuda.FloatTensor(Gc3.size()).zero_())
Gs4 = Variable(torch.cuda.FloatTensor(Gc4.size()).zero_())
for batch_idx, (images, _) in enumerate(style_loader):
    images = Variable(images)
    if args.cuda:
        images = images.cuda()
    Fs = vgg(images)
    Gs1 += gram_matrix(Fs.relu1_2)
    Gs2 += gram_matrix(Fs.relu2_2)
    Gs3 += gram_matrix(Fs.relu3_3)
    Gs4 += gram_matrix(Fs.relu4_3)
    print ('load {}th image'.format(batch_idx), "\r", end="")
print("")
Gs1 /= len(style_loader.dataset)
Gs2 /= len(style_loader.dataset)
Gs3 /= len(style_loader.dataset)
Gs4 /= len(style_loader.dataset)
    
loss_ = []
print_flag = [0]
if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

for epoch in range(args.epochs):
    print_flag[0] = True
    def closure():
        optimizer.zero_grad()

        Fi = vgg(input_param)
        Gi1 = gram_matrix(Fi.relu1_2)
        Gi2 = gram_matrix(Fi.relu2_2)
        Gi3 = gram_matrix(Fi.relu3_3)
        Gi4 = gram_matrix(Fi.relu4_3)
        
        Lc  = mse_loss(Fi.relu4_3, Fc.relu4_3) * args.content_weight
        Ls1 = mse_loss(Gi1, Gs1) * args.style_weight
        Ls2 = mse_loss(Gi2, Gs2) * args.style_weight
        Ls3 = mse_loss(Gi3, Gs3) * args.style_weight
        Ls4 = mse_loss(Gi4, Gs4) * args.style_weight

        Lc.backward( retain_graph=True)
        Ls1.backward(retain_graph=True)
        Ls2.backward(retain_graph=True)
        Ls3.backward(retain_graph=True)
        Ls4.backward(retain_graph=True)
        loss = Lc + Ls1 + Ls2 + Ls3 + Ls4
        if print_flag[0]:
            print ('epoch {} Lc:{:.2e}, Ls1:{:.2e}, Ls2:{:.2e}, Ls3:{:.2e}, Ls4:{:.2e}'.format(epoch, Lc.data[0], Ls1.data[0], Ls2.data[0], Ls3.data[0], Ls4.data[0]))
            loss_.append(loss.data[0])
            print_flag[0] = False
        return loss
    
    optimizer.step(closure)

    input_param.data.clamp_(0,1)
    vutils.save_image(input_param.data, os.path.join(args.out_dir, 'epoch_{}.png'.format(epoch)))
    
    fig, axL = plt.subplots()
    axL.plot(range(len(loss_)), loss_, label='Train', linewidth = 2.0)
    axL.set_title('train')
    fig.savefig(os.path.join(args.out_dir, 'loss.png'))


print ('finish')
