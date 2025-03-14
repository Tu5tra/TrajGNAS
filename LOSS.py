import os
import math
import sys
from thop import profile 
import time 
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from zc_metrics import *
import torch.distributions.multivariate_normal as torchdist


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def bivariate_loss(V_pred,V_trgt):      
    normx = V_trgt[:,:,0]- V_pred[:,:,0]   
    normy = V_trgt[:,:,1]- V_pred[:,:,1]

    sx = torch.exp(V_pred[:,:,2]) 
    sy = torch.exp(V_pred[:,:,3])
    corr = torch.tanh(V_pred[:,:,4]) 

    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2
    for i in range(negRho.shape[0]):
        for j in range(negRho.shape[1]):
            if negRho[i,j]==0.0000:
                negRho[i,j]=1e-10
    result = torch.exp(-z/(2*negRho))
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    for i in range(denom.shape[0]):
        for j in range(denom.shape[1]):
            if denom[i,j]==0.0000:
                denom[i,j]=1e-10
    result = result / denom
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    return result



    


def region_loss(V_pred, obs_len, obs_traj, start, end, sa_out):
    mean = V_pred[:, :, 0:2]
    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr


    cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2, device=V_pred.device)
    cov[:, :, 0, 0] = sx * sx + 1e-3
    cov[:, :, 1, 1] = sy * sy + 1e-3
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = cov[:, :, 0, 1]  # Symmetry

    pred_traj_generated = []
    aa = obs_traj[start:end, :, -1].clone()

    for t in range(obs_len):
        try:
            mvnormal = torchdist.MultivariateNormal(mean[t, :, :], cov[t, :, :, :])
            sampled_traj = mvnormal.sample()
        except ValueError as e:
            print(f"Skipping invalid covariance matrix at time step {t}: {e}")
            sampled_traj = torch.zeros_like(mean[t, :, :], device=V_pred.device)

        sampled_traj_abs = nodes_rel_to_nodes_abs_frame(sampled_traj, aa)
        pred_traj_generated.append(sampled_traj_abs)
        aa = sampled_traj_abs

    pred_traj_generated = torch.stack(pred_traj_generated, dim=-1)

    total_loss = 0
    for t in range(pred_traj_generated.shape[2]):
        for obj_idx in range(pred_traj_generated.shape[0]):
            x, y = pred_traj_generated[obj_idx, :, t]
            cx, cy = sa_out[t, obj_idx, -2], sa_out[t, obj_idx, -1]
            width, height = sa_out[t, obj_idx, -4], sa_out[t, obj_idx, -3]

            left_dist = torch.relu((cx - width / 2) - x)
            right_dist = torch.relu(x - (cx + width / 2))
            bottom_dist = torch.relu((cy - height / 2) - y)
            top_dist = torch.relu(y - (cy + height / 2))

            distance = torch.min(torch.stack([left_dist, right_dist, bottom_dist, top_dist]))
            total_loss += 0.0001 * (distance ** 2)

    # Normalize total loss
    num_objs = pred_traj_generated.shape[0]
    num_steps = pred_traj_generated.shape[2]
    if num_objs > 0 and num_steps > 0:
        total_loss = total_loss / (num_objs * num_steps)

    return total_loss



def compute_cost_loss(model, flops_weight=1e-6, sparsity_weight=1e-5):
    cost_loss = 0.0

    # FLOPs 估算损失
    flops_loss = 0.0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):  
            flops_loss += (
                module.weight.numel() * 
                module.kernel_size[0] * module.kernel_size[1] *  
                module.out_channels  
            )
        elif isinstance(module, torch.nn.Linear):  
            flops_loss += module.weight.numel()  
    flops_loss *= flops_weight


    sparsity_loss = 0.0
    for param in model.parameters():
        sparsity_loss += torch.sum(torch.abs(param)) 
    sparsity_loss *= sparsity_weight

 
    cost_loss =0.1*(flops_loss + sparsity_loss) 
    return cost_loss