import torch 
import os 
import sys
from torch import nn

# from utils.utils import *
from backbone.build_backbone import build_backbone
from backbone.basic import * 
class yolov2(nn.Module):
    def __init__(self,
                 input_size=416,
                 num_classes=20,
                 anchors=[[1.19,1.98],[2.79,4.59],[4.53,8.92],[8.06,5.29],[10.32,10.65]]):

        super().__init__()
        self.input_size=input_size
        self.num_classes=num_classes
        self.anchors=torch.tensor(anchors)
        self.num_anchors=len(anchors)
        self.stride=32
        self.fmp=self.input_size//self.stride



        self.backbone=build_backbone(model_name='darknet19',pretrained=True)

        #head_p5

        self.head_p5=nn.Sequential(
            
            cnnblock(1024,1024,3),
            cnnblock(1024,1024,3),
            )
        #head_p4
        self.route=cnnblock(512,64,1) #p4
        self.reorg=reorg()

        self.head_cat=cnnblock(1280,1024,1) #p4+p5

        self.pred=nn.Conv2d(1024,self.num_anchors*(self.num_classes+1+4),1)
    

    def _generate_grid(self,fmp):

        grid_y,grid_x=torch.meshgrid(torch.arange(fmp),torch.arange(fmp),indexing='ij')
        grid=torch.stack([grid_y,grid_x],-1).unsqueeze(-2) #hw 1 2 
        return grid

    def forward(self,x):
        device=x.device
        p3,p4,p5=self.backbone(x).values()

        p4=self.reorg(self.route(p4))
        p5=self.head_p5(p5)
        p5=torch.cat([p4,p5],1)

        p5=self.head_cat(p5)
        pred=self.pred(p5)

        batch_size,channels,h,w=pred.size()

        pred=pred.permute(0,2,3,1).view(batch_size,h,w,self.num_anchors,self.num_classes+5).contiguous().float()

        if not self.training:
            
            pred[...,0:2]=(torch.sigmoid(pred[...,0:2])+self._generate_grid(self.fmp).to(device))*self.stride
            pred[...,2:4]=(torch.exp(pred[...,2:4])*self.anchors.to(device))*self.stride

            
        return pred 



