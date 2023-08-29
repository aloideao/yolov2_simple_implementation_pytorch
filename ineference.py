import torch 
torch.set_printoptions(precision=4,sci_mode=False)
import numpy as np
np.set_printoptions(precision=4,suppress=True)

from data.voc import * 
from data.augment import * 
from utils.utils import *
import matplotlib.pyplot as plt 
import torchvision 


import numpy as np 
import torch
import cv2
from model.yolov2 import * 
import random
import argparse
from utils.utils import * 

def parse_args():
    parser=argparse.ArgumentParser(description='YOLO Inference')

    parser.add_argument('--cuda',action='store_true',default=True,
                        help='device is cuda')
    
    parser.add_argument('--weights',default="model.pth",
                        type=str,help='model weights')
    

    parser.add_argument('--source',default=0
                        ,help='0 for camera,or path to image')
    
    parser.add_argument('--imgsz',default=416
                        ,help='image size')
    
    parser.add_argument('--conf_threshold',default=.15,
                       type=float,help='conf_threshold')
    
    parser.add_argument('--iou',default=.1,
                        type=float,help='nms threshold')
    
    return parser.parse_args()











args = parse_args()

def test(args=args,transform=albumentations()):
    conf_threshold=args.conf_threshold
    iou_threshold=args.iou
    source_path=args.source
    model_path=args.weights
    size=args.imgsz
    device='cuda' if args.cuda else 'cpu'
    

    model=yolov2(input_size=size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    


    camera = cv2.VideoCapture(0)
    while True:
        if source_path == 0:
             return_value, image = camera.read()
             source=image.astype(np.uint8)
        else:
            source=cv2.imread(source_path).astype(np.uint8)
            cv2.namedWindow("predict", cv2.WINDOW_NORMAL)        
            cv2.resizeWindow("predict", 640, 640) 

        height,width,_=source.shape
        img=letterbox(source,new_shape=[size,size])[0] #letterbox to perserve scales 

        input=transform(img).to(device)
        result=model(input.unsqueeze(0))

        scores=result[...,4:5].sigmoid()*result[...,5:].softmax(-1)
        scores,labels=scores.view(1,-1,20).max(-1)
        xywh=result[...,:4].view(1,-1,4)


        keep=scores>args.conf_threshold



        filtered_result=xw2xy(xywh[keep])/416
        scores=scores[keep]
        labels=labels[keep]

        f=rescale_bbx(filtered_result[:,:4],height,width)

        # #apply nms

        ids=torch.ops.torchvision.nms(torch.tensor(f),
                                        torch.tensor(scores),
                                        iou_threshold=iou_threshold)

        bbx_rescaled=f[ids][None] if len(f[ids].shape)==1 else f[ids]
        labels=torch.vstack([scores,labels]).T[ids]
        bbx=torch.hstack([bbx_rescaled,labels])
        # #plotting
        img=plot_bbox_labels(source,bbx)

        cv2.imshow('predict',img)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

                 

if __name__=='__main__':
    test()
