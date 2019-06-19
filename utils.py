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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image_name):
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # scale imported image
        transforms.ToTensor()
    ])  # transform it into a torch tensor
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def to_numpy_image(img_tensor):
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    if len(img_tensor.shape) == 3:
        img_numpy = img_tensor.permute(1, 2, 0).data.cpu().numpy()
    else:
        img_numpy = img_tensor.data.cpu().numpy()
    return img_numpy

def get_image_sematic_map(content_img, toTensor=True):
    from PIL import Image
    import torchvision
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(content_img[0]).to(device)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    # set it to evaluation mode, as the model behaves differently
    # during training and during evaluation
    model.eval().to(device)

    output = model(input_batch)
    import matplotlib.pyplot as plt
    _, sematic_map = output['out'].max(dim=1)
    return sematic_map

def save_result(dir_name, content_img, style_img_1, style_img_2, sematic_map_1, sematic_map_2,
    vanilla_style_img_1, vanilla_style_img_2, vanilla_synthesis_img,
    input_img_1_list, input_img_2_list, synthesis_img_list, ratio_list):

    os.makedirs(dir_name, exist_ok=True)

    # Plot Content, Style Image and Sematic Map 
    plt.figure(figsize=(20,15))
    plt.subplot(231), plt.imshow(to_numpy_image(style_img_1)), plt.axis("off"), plt.title('Style Image 1')
    plt.subplot(232), plt.imshow(to_numpy_image(style_img_2)), plt.axis("off"), plt.title('Style Image 2')
    plt.subplot(233), plt.imshow(to_numpy_image(content_img)), plt.axis("off"), plt.title('Content Image')
    plt.subplot(234), plt.imshow(to_numpy_image(sematic_map_1)), plt.axis("off"), plt.title('Sematic Map 1')
    plt.subplot(235), plt.imshow(to_numpy_image(sematic_map_2)), plt.axis("off"), plt.title('Sematic Map 2')
    plt.savefig(os.path.join(dir_name, "Content_Style_Sematic.jpg"))

    # Plot Vanilla vs Sematic
    plt.figure(figsize=(20,10))
    plt.subplot(121), plt.imshow(to_numpy_image(vanilla_synthesis_img)), plt.axis("off"), plt.title("Vanilla Synthesis Image")
    plt.subplot(122), plt.imshow(to_numpy_image(synthesis_img_list[0])), plt.axis("off"), plt.title("Sematic Synthesis Image")
    plt.savefig(os.path.join(dir_name, "Vanilla_Sematic.jpg"))

    # Save Ratio
    N = len(synthesis_img_list)
    plt.figure(figsize=(10*N,30))
    for i in range(N):
        plt.subplot(3, N, 0*N+i+1), plt.imshow(to_numpy_image(input_img_1_list[i])), plt.axis("off"), plt.title("Ratio %.2f"%(ratio_list[i]))
        plt.subplot(3, N, 1*N+i+1), plt.imshow(to_numpy_image(input_img_2_list[i])), plt.axis("off")
        plt.subplot(3, N, 2*N+i+1), plt.imshow(to_numpy_image(synthesis_img_list[i])), plt.axis("off")
        img_RGB = cv2.cvtColor(to_numpy_image(synthesis_img_list[i]), cv2.COLOR_BGR2RGB) *255.0
        cv2.imwrite(os.path.join(dir_name, "Blend_Ratio_%.2f.jpg"%(ratio_list[i])), img_RGB.astype(np.uint8))
    plt.savefig(os.path.join(dir_name, "Blend_Ratio.jpg"))



def get_rest_map(map_list):
    rest_map = torch.ones(map_list[0].shape).to(device)
    for sematic_map in map_list:
        rest_map = rest_map - sematic_map
    return rest_map

def contain_one(tensor):
    return len(torch.nonzero(tensor)) > 0

def contain_zero(tensor):
    return len(torch.nonzero(1 - tensor)) > 0

def get_boundary(sematic_map, length=2):
    boundary = torch.zeros(sematic_map[0].shape)
    for i in range(length, boundary.shape[0]-length):
        for j in range(length, boundary.shape[1] -length):
            if contain_zero(sematic_map[0, i-length:i+length, j-length:j+length]) and contain_one(sematic_map[0, i-length:i+length, j-length:j+length]):
                boundary[i, j] = 1
    return boundary.repeat(3,1,1).type(torch.FloatTensor).to(device)


LABELS  = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
"diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
INDICES = [i for i in range(len(LABELS))]
L2I = dict(zip(LABELS, INDICES))
I2L = dict(zip(INDICES, LABELS))

def label_to_index(label):
    return L2I[label.lower()]

def index_to_label(index):
    return I2L[index]