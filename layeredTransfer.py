import argparse
import os
from pathlib import Path
import glob

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from imageio import imwrite
import cv2
from PIL import Image
import shutil

from st_helper import *
import utils
from utils import *

color_dict = {
    "sky": [255, 0, 0],
    "mountain": [0, 255, 0],
    "sea": [0, 0, 255]
}

exeption_dict = {
    "other": [255, 255, 255],
}

def run_st(content_path, style_path, content_weight, max_scl, coords, use_guidance,regions, output_path='./output.png'):
    smll_sz = 64
    content_im_big = utils.to_device(Variable(load_path_for_pytorch(content_path,512,force_scale=True).unsqueeze(0)))
    for scl in range(1,max_scl):
        long_side = smll_sz*(2**(scl-1))
        lr = 2e-3

        ### Load Style and Content Image ###
        content_im = utils.to_device(Variable(load_path_for_pytorch(content_path,long_side,force_scale=True).unsqueeze(0)))
        content_im_mean = utils.to_device(Variable(load_path_for_pytorch(style_path,long_side,force_scale=True).unsqueeze(0))).mean(2,keepdim=True).mean(3,keepdim=True)
        
        ### Compute bottom level of laplaccian pyramid for content image at current scale ###
        lap = content_im.clone()-F.interpolate(F.interpolate(content_im,(content_im.size(2)//2,content_im.size(3)//2),mode='bilinear',align_corners=True),(content_im.size(2),content_im.size(3)),mode='bilinear')
        nz = torch.normal(lap*0.,0.1)
        canvas = F.interpolate( lap, (content_im_big.size(2),content_im_big.size(3)), mode='bilinear',align_corners=True)[0].data.cpu().numpy().transpose(1,2,0)
        if scl == 1:
            canvas = F.interpolate(content_im,(content_im.size(2)//2,content_im.size(3)//2),mode='bilinear',align_corners=True)[0].data.cpu().numpy().transpose(1,2,0)

        ### Initialize by zeroing out all but highest and lowest levels of Laplaccian Pyramid ###
        if scl == 1:
            if 1:
                stylized_im = Variable(content_im_mean+lap)
            else:
                stylized_im = Variable(content_im.data)

        ### Otherwise bilinearly interpolate previous scales output and add back bottom level of Laplaccian pyramid for current scale of content image ###
        if scl > 1 and scl < max_scl-1:
            stylized_im = F.interpolate(stylized_im.clone(),(content_im.size(2),content_im.size(3)),mode='bilinear',align_corners=True)+lap
        if scl == max_scl-1:
            stylized_im = F.interpolate(stylized_im.clone(),(content_im.size(2),content_im.size(3)),mode='bilinear',align_corners=True)
            lr = 1e-3

        ### Style Transfer at this scale ###
        stylized_im, final_loss = style_transfer(stylized_im, content_im, style_path, output_path, scl, long_side, 0., use_guidance=use_guidance, coords=coords, content_weight=content_weight, lr=lr, regions=regions)
        canvas = F.interpolate(stylized_im,(content_im.size(2),content_im.size(3)),mode='bilinear',align_corners=True)[0].data.cpu().numpy().transpose(1,2,0)
        
        ### Decrease Content Weight for next scale ###
        content_weight = content_weight/2.0

    canvas = torch.clamp( stylized_im[0], 0., 1.).data.cpu().numpy().transpose(1,2,0)
    imwrite(output_path,canvas)
    return final_loss, canvas

def create_layer(im_path, map_path, save_dir, color_dict, paint_color=None, mean_color=False):
    im_layer_path = {}
    map_layer_path = {}

    # check existing file
    layer_dir = os.path.join(save_dir, Path(im_path).stem+'_layers')
    if os.path.exists(layer_dir):
        for k, v in color_dict.items():
            im_layer_path[k] = os.path.join(layer_dir, k+".png")
            map_layer_path[k] = os.path.join(layer_dir, k+"_map.png")
        return im_layer_path, map_layer_path

    os.makedirs(layer_dir, exist_ok=True)
    im = np.array(Image.open(im_path))
    im_map = np.array(Image.open(map_path))
    h, w = im_map.shape[0:2]
    im = cv2.resize(im, (w, h))
    kernel = np.ones((5,5),np.uint8)

    for k, v in color_dict.items():
        # set the save path
        save_path_im = os.path.join(layer_dir, k+".png")
        save_path_map = os.path.join(layer_dir, k+"_map.png")

        # masking process
        im_mask = cv2.inRange(im_map, np.array(v), np.array(v))
        im_mask = im_mask//255
        im_mask = im_mask.reshape(h,w,1)
        masked_im = im * im_mask

        # fill the outer area
        after_color = [255, 255, 255]
        if mean_color:
            after_color = np.array([masked_im[:, :, 0].sum(), masked_im[:, :, 1].sum(), masked_im[:, :, 2].sum()])/im_mask.sum()
            after_color = after_color.astype(np.uint8).tolist()
        masked_im[np.where((masked_im == [0, 0, 0]).all(axis=2))] = after_color

        # save rayered image
        maskim = Image.fromarray(masked_im)
        maskim.save(save_path_im)
        im_layer_path[k] = save_path_im

        # save rayered map image
        color = color_dict[k]
        if paint_color:
            color = paint_color
        mask_layer = np.array([[color]*w]*h, dtype=np.uint8) * im_mask
        mask_layer[np.where((mask_layer == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        mask_layer = Image.fromarray(mask_layer)
        mask_layer.save(save_path_map)
        map_layer_path[k] = save_path_map

    return im_layer_path, map_layer_path

def create_ex_layer(im_path, save_dir, color_dict, paint_color):
    im_layer_path = {}
    map_layer_path = {}

    # for the content or stye image
    im = np.array(Image.open(im_path))
    layer_dir = os.path.join(save_dir, Path(im_path).stem+'_layers')
    os.makedirs(layer_dir, exist_ok=True)
    save_path = os.path.join(layer_dir, "other.png")
    im_layer_path[list(color_dict.keys())[0]] = save_path
    Image.fromarray(im).save(save_path)
    h, w = im.shape[:2]

    # for the region map
    result = np.array([[paint_color]*w]*h, dtype=np.uint8)
    result = Image.fromarray(result)
    save_path = os.path.join(layer_dir, "other_map.png")
    result.save(save_path)
    map_layer_path[list(color_dict.keys())[0]] = save_path
    return im_layer_path, map_layer_path

def run_strotss(content_path, style_path, content_map_path, style_map_path, content_weight, max_scale, output_path):
    content_weight = content_weight*16.0
    max_scl = max_scale
    use_guidance_points = False

    ### Preprocess User Guidance if Required ###
    coords=0.
    regions = utils.extract_regions(content_map_path, style_map_path)

    ### Style Transfer and save output ###
    loss,canvas = run_st(content_path,style_path,content_weight,max_scl,coords,use_guidance_points,regions,output_path=output_path)


def run_layered_transfer(full_content_path, full_content_map_path, full_style_path, full_style_map_path, content_weight, max_scale, save_dir, use_cpu):
    # layer creation
    content_im_layer_path, content_map_layer_path = create_layer(full_content_path, full_content_map_path, save_dir, color_dict)
    style_im_layer_path, style_map_layer_path = create_layer(full_style_path, full_style_map_path, os.path.dirname(save_dir), color_dict, mean_color=True)
    
    # exeption layer process
    paint_color = [255, 255, 0]
    ex_content_im_path, ex_content_map_path = create_ex_layer(full_content_path, save_dir, exeption_dict, paint_color)
    ex_style_im_path, ex_style_map_path = create_ex_layer(full_style_path, save_dir, exeption_dict, paint_color)
    
    # concat dictionaries
    color_dict.update(exeption_dict)
    content_im_layer_path.update(ex_content_im_path)
    content_map_layer_path.update(ex_content_map_path)
    style_im_layer_path.update(ex_style_im_path)
    style_map_layer_path.update(ex_style_map_path)
    use_gpu = not use_cpu
    utils.use_gpu = use_gpu
    for k in color_dict.keys():
        content_path = content_im_layer_path[k]
        style_path = style_im_layer_path[k]
        content_map_path = content_map_layer_path[k]
        style_map_path = style_map_layer_path[k]
        
        # print logs
        output_path = os.path.join(save_dir, 'output_' + k + '.png')
        print('Generating {} layer...'.format(k), end='')
        run_strotss(content_path, style_path, content_map_path, style_map_path, content_weight, max_scale, output_path)
        print(' Done.')

    # concat layer
    mask = np.array(Image.open(full_content_map_path))
    im_other_path = list(Path(save_dir).glob('*%s.png'%'other'))[0]
    result_alpha = Image.open(im_other_path)
    w, h = result_alpha.size
    mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    kernel_size = int(min(h, w) * 0.015) * 2 + 1
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in color_dict.items():
        im_file = list(Path(save_dir).glob('*%s.png'%k))[0]
        im = np.array(Image.open(im_file))
        im_mask = cv2.inRange(mask, np.array(v), np.array(v)).reshape(h,w,1)
        ch_a = im_mask
        ch_a = cv2.GaussianBlur(ch_a, (kernel_size, kernel_size), 0)
        im_mask = im_mask//255
        masked_im = im * im_mask
        result += masked_im
        if not k == 'other':
            result_alpha.paste(Image.fromarray(im), (0, 0), Image.fromarray(ch_a))

    # save final image
    result = Image.fromarray(result)
    path = Path(save_dir) / "result.png"
    result.save(path)
    result_alpha.save(os.path.join(save_dir, 'result_alpha.png'))


if __name__=='__main__':
    ### Parse Command Line Arguments ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', required=True)
    parser.add_argument('--style', required=True)
    parser.add_argument('--content_map_dir', required=True)
    parser.add_argument('--style_map', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--max_scale', type=int, default=5)
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()

    content_paths = glob(os.path.join(args.content_dir,'*'))
    content_map_paths = glob(os.path.join(args.content_map_dir,'*'))
    num = len(content_paths)
    count = 0
    for content_path, content_map_path in zip(content_paths, content_map_paths):
        name = os.path.splitext(os.path.basename(content_path))[0]
        save_dir = os.path.join(args.save_dir, name)
        run_layered_transfer(content_path, content_map_path, args.style, args.style_map, args.content_weight, args.max_scale, save_dir, args.use_cpu)
        count +=1
        print('[{} / {} is finished.]'.format(count, num))

    # move result images to the final_images folder
    parent_dir = args.save_dir
    save_dir = os.path.join(parent_dir, 'final_images')
    os.makedirs(save_dir, exist_ok=True)
    dirs = glob(os.path.join(parent_dir, '*'))
    for dir in dirs:
        file = os.path.join(dir, 'result_alpha.png')
        name = os.path.basename(dir)
        path = os.path.join(save_dir, name + '.png')
        if os.path.exists(file):
            shutil.copyfile(file, path)