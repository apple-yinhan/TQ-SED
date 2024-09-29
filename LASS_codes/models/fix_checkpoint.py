#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:34:52 2024

@author: bfzystudent
"""
import torch
from resunet import HResUNet30

if 1: # check
    system = 'fsd_clo_caps_HResUNet_V1_3'
    model_name = 'model-epoch=01-val_loss=0.0153.ckpt'
    checkpoint_path = f'/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes/workspace/checkpoints/{system}/{model_name}'
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
    model_param = checkpoint_data['state_dict']
    alpha = model_param['ss_model.base.alpha']
    print(alpha)
if 0: # fix
    checkpoint_path = '/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes/pretrained_models/Audiosep/model-epoch=190-val_loss=0.0150.ckpt'
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
    
    model_param = checkpoint_data['state_dict']
    param_key = list(model_param.keys())
    
    new_model = HResUNet30(input_channels=1, output_channels=1, condition_size=512)
    new_model.eval()
    
    name_list = []
    for name, paramer in new_model.named_parameters():
        name = f'ss_model.{name}'
        name_list.append(name)
        if name not in param_key:
            model_param[name] = paramer
            if 'bn' in name and 'bias' in name:
                model_param[name.replace('bias', 'running_mean')] = paramer
                model_param[name.replace('bias', 'running_var')] = paramer
            if 'bn' in name and 'weight' in name:
                model_param[name.replace('weight', 'running_mean')] = paramer
                model_param[name.replace('weight', 'running_var')] = paramer
    
    checkpoint_data['state_dict'] = model_param
    checkpoint_path = '/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes/pretrained_models/Audiosep/model-epoch=190-val_loss=0.0150_new.ckpt'
    torch.save(checkpoint_data, checkpoint_path)    
