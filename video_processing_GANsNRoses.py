import traceback
import cv2
import numpy as np
import sys
import argparse
from datetime import datetime
import os

# GANsNRoses
# https://github.com/mchong6/GANsNRoses.git

# Install steps:
# git clone https://github.com/mchong6/GANsNRoses.git
# pip install tqdm gdown kornia scipy opencv-python dlib moviepy lpips aubio ninja

# Status: working

pathToProject='../GANsNRoses'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
import copy
from util import *
from PIL import Image

from model import *
import moviepy.video.io.ImageSequenceClip
import scipy
import cv2
import dlib
import kornia.augmentation as K
from aubio import tempo, source

from IPython.display import HTML
from base64 import b64encode


def get_image(image, size=None, mode='nearest', unnorm=False, title=''):
    # image is [3,h,w] or [1,3,h,w] tensor [0,1]
    if image.is_cuda:
        image = image.cpu()
    if size is not None and image.size(-1) != size:
        image = F.interpolate(image, size=(size,size), mode=mode)
    if image.dim() == 4:
        image = image[0]
    image = image.permute(1, 2, 0).detach().numpy()
    return image
    # plt.figure()
    # plt.title(title)
    # plt.axis('off')
    # plt.imshow(image)

def init_model(transform):
    device = 'cuda'
    latent_dim = 8
    n_mlp = 5
    num_down = 3

    G_A2B = Generator(256, 4, latent_dim, n_mlp, channel_multiplier=1, lr_mlp=.01,n_res=1).to(device).eval()

    ensure_checkpoint_exists('GNR_checkpoint.pt')
    ckpt = torch.load('GNR_checkpoint.pt', map_location=device)

    G_A2B.load_state_dict(ckpt['G_A2B_ema'])

    # mean latent
    truncation = 1
    with torch.no_grad():
        mean_style = G_A2B.mapping(torch.randn([1000, latent_dim]).to(device)).mean(0, keepdim=True)

    test_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
    ])

    torch.manual_seed(84986)

    reverse_preprocess = transforms.Compose([
        transforms.ToPILImage(),
        np.array,
    ])

    num_styles = 1
    style = torch.randn([num_styles, latent_dim]).to(device)

    return (device,G_A2B,style,num_styles,test_transform,reverse_preprocess),None


def process_image(transform,processing_model,img):
    tracks = []
    try:
        (device,G_A2B,style,num_styles,test_transform,reverse_preprocess) = processing_model
        real_A = Image.fromarray(img)

        real_A = test_transform(real_A).unsqueeze(0).to(device)

        with torch.no_grad():
            A2B_content, _ = G_A2B.encode(real_A)
            # print('num_styles',num_styles)
            fake_A2B = G_A2B.decode(A2B_content.repeat(num_styles,1,1,1), style)
            # print(fake_A2B)
            A2B = fake_A2B #torch.cat([real_A, fake_A2B], 0)

            image = reverse_preprocess(utils.make_grid(A2B.cpu(), normalize=True, range=(-1, 1), nrow=10))
            img = image 
            # print(image.shape)

            # plt.figure()
            # plt.title('img')
            # plt.axis('off')
            # plt.imshow(image)          
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("GANsNRoses Exception",e)
        pass
                
    return tracks,img

