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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Gram Matrix
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# Content Loss
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# Style Loss
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input



# #######################
# # --- Vallina NSF --- # 
# #######################
def get_style_model_and_losses(cnn, style_img, content_img):
    cnn = copy.deepcopy(cnn)

    # normalization module
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

     # desired depth layers to compute style/content losses :
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, content_img, style_img,
    num_steps=300, style_weight=1e6, content_weight=1):

    cnn = copy.deepcopy(cnn)

    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

    input_img = content_img.clone()
    optimizer = get_input_optimizer(input_img)
    run = [0]
    pbar = tqdm(total=num_steps+20, ascii=True)
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            pbar.set_postfix({
                'Style Loss' : '%.4f'%style_score.item(),
                'Content Loss': '%.4f'%content_score.item(),
            })
            pbar.update()

            return style_score + content_score

        optimizer.step(closure)

    pbar.close()
    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


#######################
# --- Sematic NSF --- # 
#######################
class TVLoss(nn.Module):
    
    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength
    
    def forward(self, input):
        self.x_diff = input[:,:,1:,:] - input[:,:,:-1,:]
        self.y_diff = input[:,:,:,1:] - input[:,:,:,:-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input


def get_sematic_style_model_and_losses(cnn, style_img_1, style_img_2, content_img, tv_weight=0.000085):
    
    cnn = copy.deepcopy(cnn)
    
    # normalization module
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
    
    # desired depth layers to compute style/content losses :
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # content_layers = ['relu_10', 'relu_14']
    # style_layers = ['relu_1', 'relu_3', 'relu_5', 'relu_9', 'relu_13']

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses_1 = []
    style_losses_2 = []
    tv_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    content_model = nn.Sequential(normalization)
    style_model_1 = nn.Sequential(normalization)
    style_model_2 = nn.Sequential(normalization)


    tv_mod = TVLoss(tv_weight).to(device)
    content_model.add_module(str(len(content_model)), tv_mod)
    tv_losses.append(tv_mod)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        content_model.add_module(name, layer)
        style_model_1.add_module(name, layer)
        style_model_2.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = content_model(content_img).detach()
            content_loss = ContentLoss(target)
            content_model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss 1:
            target_feature_1 = style_model_1(style_img_1).detach()
            style_loss_1 = StyleLoss(target_feature_1)
            style_model_1.add_module("style_loss_{}".format(i), style_loss_1)
            style_losses_1.append(style_loss_1)
            
            # add style loss 2:
            target_feature_2 = style_model_2(style_img_2).detach()
            style_loss_2 = StyleLoss(target_feature_2)
            style_model_2.add_module("style_loss_{}".format(i), style_loss_2)
            style_losses_2.append(style_loss_2)

    # now we trim off the layers after the last content and style losses
    for i in range(len(content_model) - 1, -1, -1):
        if isinstance(content_model[i], ContentLoss) or isinstance(style_model_1[i], StyleLoss):
            break

    content_model = content_model[:(i + 1)]
    style_model_1 = style_model_1[:(i + 1)]
    style_model_2 = style_model_2[:(i + 1)]

    return (content_model, style_model_1, style_model_2), (content_losses, style_losses_1, style_losses_2), tv_losses


def get_inputs_optimizer(input_img1, input_img2, sematic_map):
    # this line to show that input is a parameter that requires a gradient
    #optimizer = optim.LBFGS([input_img1.requires_grad_(), input_img2.requires_grad_(), sematic_map.requires_grad_()])
    optimizer = optim.LBFGS([input_img1.requires_grad_(), input_img2.requires_grad_()])
    return optimizer

def run_sematic_style_transfer(cnn, style_img_1, style_img_2, content_img,
                       sematic_map_1, sematic_map_2,
                       style_weight=1e6, style_blend_weight=1e3, content_weight=1, tv_weight=0.000085,
                       num_steps=300):
    
    sematic_map_1_origin = sematic_map_1.clone()

    input_img_1 = content_img.clone()
    input_img_2 = content_img.clone()

    # rest_map = torch.ones(sematic_map_1.shape).to(device)
    # rest_map = rest_map - sematic_map_1 - sematic_map_2 
    # rest_map.data.clamp_(0, 1)
    synthesis_img = input_img_1 * sematic_map_1 + input_img_2 * sematic_map_2 
    
    models, losses, tv_losses  = get_sematic_style_model_and_losses(cnn, style_img_1, style_img_2, content_img, tv_weight=tv_weight)
    content_model, style_model_1, style_model_2 = models
    content_losses, style_losses_1, style_losses_2 = losses
    
    optimizer = get_inputs_optimizer(input_img_1, input_img_2, sematic_map_1)
    # optimizer_2 = get_input_optimizer(input_img_2)
    run = [0]
    
    pbar = tqdm(total=num_steps+20, ascii=True)
    
    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            input_img_1.data.clamp_(0, 1)
            input_img_2.data.clamp_(0, 1)
            optimizer.zero_grad()
            synthesis_img = input_img_1 * sematic_map_1 + input_img_2 * sematic_map_2
            
            style_score = 0
            style_blend_score = 0
            content_score = 0
            
            # --- Style Loss --- #
            style_model_1(input_img_1)
            style_model_2(input_img_2)
            for sl in style_losses_1:
                style_score += sl.loss
            for sl in style_losses_2:
                style_score += sl.loss
            style_score *= style_weight
            
            # --- Content Loss --- # (Add TV loss)
            content_model(synthesis_img)
            for cl in content_losses:
                content_score += cl.loss
            for mod in tv_losses:
                content_score += mod.loss
            content_score *= content_weight
            
            # --- Style Blending Loss --- #
            style_model_1(input_img_2)
            style_model_2(input_img_1)
            for sl in style_losses_1:
                style_blend_score += sl.loss
            for sl in style_losses_2:
                style_blend_score += sl.loss
            style_blend_score *= style_blend_weight
            
            # Backward
            loss = style_score + content_score + style_blend_score
            loss.backward()

            run[0] += 1
            pbar.set_postfix({
                'Style Loss' : '%.2f'%style_score.item(),
                'Content Loss': '%.2f'%content_score.item(),
                'Style Blend Loss':'%.2f'%style_blend_score.item()
            })
            pbar.update()
            return style_score + content_score + style_blend_score
        optimizer.step(closure)
        # optimizer_2.step(closure)
    pbar.close()
    
    # a last correction...
    input_img_1.data.clamp_(0, 1)
    input_img_2.data.clamp_(0, 1)
    synthesis_img = input_img_1 * sematic_map_1 + input_img_2 * sematic_map_2
    return input_img_1, input_img_2, synthesis_img