import torch 
import torch.nn.functional as F
from utils.utils import * 
from torch import nn
def loss_Fn(predictions,labels,device):
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()
    gxy,gwh,gt_class,indices=[l.to(device) for l in labels]
    batch_id,grid_j,grid_i,a=indices.T


    tconf=torch.zeros_like(predictions[...,0]).to(device).float()
    tconf[batch_id,grid_j,grid_i,a]=1.0

    pi=predictions[batch_id,grid_j,grid_i,a]



    lxy=8*MSE(pi[...,0:2].sigmoid(),gxy)
    lwh=4*MSE(pi[...,2:4],gwh)
    lclass=CE(pi[:,5:],gt_class)
    lconf=64*BCE(predictions[...,4],tconf)
    loss=lxy+lwh+lclass+lconf
    return lxy,lwh,lclass,lconf,loss 