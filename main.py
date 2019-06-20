# Package
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy

from tqdm import tqdm
import numpy as np
import os
import cv2

from utils import *
from sematic_neural_style_transfer import *

import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    # Image Path
    content_img_path = "images/content_img/people2.jpg"
    style_img_1_path = "images/style_img/bright.jpeg"
    style_img_2_path = "images/style_img/dark.jpg"
    num_steps = 300
    
    # Read Image (to Tensor)
    content_img = image_loader(content_img_path)
    style_img_1 = image_loader(style_img_1_path)
    style_img_2 = image_loader(style_img_2_path) 
    
    # Get Sematic Map (to Tensor)
    sematic_map = get_image_sematic_map(content_img).cpu()
    map_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ])

    # Get the sematic map corresponding to human
    PERSON = label_to_index("person")
    sematic_map_1 = sematic_map.clone()
    sematic_map_1[sematic_map_1 != PERSON] = 0
    sematic_map_1[sematic_map_1 == PERSON] = 1
    sematic_map_1 = sematic_map_1.repeat(3,1,1).type(torch.FloatTensor)
    sematic_map_1 = map_transform(sematic_map_1).to(device)

    # Get the sematic map corresponding to non-human
    sematic_map_2 = sematic_map.clone()
    sematic_map_2[sematic_map_2 != PERSON] = 1
    sematic_map_2[sematic_map_2 == PERSON] = 0
    sematic_map_2 = sematic_map_2.repeat(3,1,1).type(torch.FloatTensor)
    sematic_map_2 = map_transform(sematic_map_2).to(device)

    # Create CNN 
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # --- Vallina Neural Style Transfer --- #
    print("# --- Vallina Neural Style Transfer --- #")
    # Get the output of style 1
    print("(Style 1)")
    vanilla_style_img_1 = run_style_transfer(cnn, content_img, style_img_1, num_steps=num_steps, style_weight=1e6, content_weight=1)
    # Get the output of style 1
    print("(Style 2)")
    vanilla_style_img_2 = run_style_transfer(cnn, content_img, style_img_2, num_steps=num_steps, style_weight=1e6, content_weight=1)
    # Synthesis two output style image accroding to sematic map 
    vanilla_synthesis_img = vanilla_style_img_1 * sematic_map_1 + vanilla_style_img_2 * sematic_map_2

    # --- Sematic Neural Style Transfer --- #
    print("# --- Sematic Neural Style Transfer --- #")
    style_weight=1e6
    tv_weight=0.000085
    input_img_1_list = []
    input_img_2_list = []
    synthesis_img_list = []
    ratio_list = []

    for i in range(0, 101, 20):
        ratio = i / 100
        ratio_list.append(ratio)
        style_blend_weight = style_weight * ratio
        print("--Blend Ratio:", ratio)
        input_img_1, input_img_2, synthesis_img = run_sematic_style_transfer(
            cnn, style_img_1, style_img_2, content_img, sematic_map_1.clone(), sematic_map_2,
            style_weight=style_weight, style_blend_weight=style_blend_weight, content_weight=1, tv_weight=tv_weight, num_steps=num_steps
        )
        input_img_1_list.append(input_img_1)
        input_img_2_list.append(input_img_2)
        synthesis_img_list.append(synthesis_img)

    # Save Result
    dir_name = "results/{}_{}_{}".format(
        os.path.basename(content_img_path).split('.')[0],
        os.path.basename(style_img_1_path).split('.')[0],
        os.path.basename(style_img_2_path).split('.')[0],
    )
    print("Save Result to", dir_name)
    save_result(dir_name, content_img, style_img_1, style_img_2, sematic_map_1, sematic_map_2,
        vanilla_style_img_1, vanilla_style_img_2, vanilla_synthesis_img,
        input_img_1_list, input_img_2_list, synthesis_img_list, ratio_list
    )
    
