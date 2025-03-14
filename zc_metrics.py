import os
import math
import sys

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


def ade(predAll, targetAll, count_):
    All = len(predAll) 
    sum_all = 0 
    for s in range(All):  
        pred = predAll[s][:, :count_[s], :].permute(1, 0, 2)  
        target = targetAll[s][:, :count_[s], :].permute(1, 0, 2)

        dist = torch.sqrt(torch.sum((pred - target)**2, dim=-1))  
        sum_all += dist.mean().item()  
    return sum_all / All

def fde(predAll, targetAll, count_):
    All = len(predAll)  
    sum_all = 0 
    for s in range(All):  
        pred = predAll[s][:, :count_[s], :].permute(1, 0, 2)  
        target = targetAll[s][:, :count_[s], :].permute(1, 0, 2)
      
        pred_last = pred[:, -1, :]  
        target_last = target[:, -1, :]  
        
        dist = torch.sqrt(torch.sum((pred_last - target_last)**2, dim=-1))  
        sum_all += dist.mean().item()  
    return sum_all / All



def fde_frame(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = predAll[s][:count_[s],:]
        target = targetAll[s][:count_[s],:]
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            sum_+=math.sqrt((pred[i,0] - target[i,0])**2+(pred[i,1] - target[i,1])**2)
        sum_all += sum_/(N)
    return sum_all/All

def seq_to_nodes(seq_,max_nodes = 88):
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = torch.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
    return V.squeeze()


def nodes_rel_to_nodes_abs_frame(nodes,init_node):

    nodes_ = torch.zeros_like(nodes)
    for ped in range(nodes.shape[0]):
        nodes_[ped,:] = nodes[ped][:] + init_node[ped][:]
    
    return nodes_.squeeze()

def seq_to_nodes(seq_,max_nodes = 88):
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = torch.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()
#
def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = torch.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = torch.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
        
def bivariate_loss(V_pred,V_trgt):   
    normx = V_trgt[:,:,0]- V_pred[:,:,0]
    normy = V_trgt[:,:,1]- V_pred[:,:,1]

    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr

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
   