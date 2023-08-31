import torch 
from evaluator.eval import VOCAPIEvaluator
torch.set_printoptions(precision=4,sci_mode=False)
import numpy as np
np.set_printoptions(precision=4,suppress=True)

from data.voc import * 
from data.augment import * 
from utils.utils import *
import matplotlib.pyplot as plt 
import torchvision
from utils.loss import * 
from data.grid_creator import matcher
from model.yolov2 import yolov2 
from test import evaluate 
#the most basic trainer ever 

input_size=416
num_of_classes=20
device='cuda' if torch.cuda.is_available() else 'cpu'
root=r"E:\b\yolov1_final\data\VOCdevkit"

epochs=50
model=yolov2(416)


model.train().to(device)

optimizer=torch.optim.SGD(model.parameters(),.001)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.1,patience=10)

dataset=VOCDetection(
                        data_dir=r'data\VOCdevkit',
                        img_size=416,transform=TrainTransforms())

dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=16, 
                    shuffle=True, 
                    collate_fn=detection_collate_fn,
                    pin_memory=True
                    )

validation_dataset = VOCDetection(data_dir=r'data/VOCdevkit',image_sets=[('2007', 'test')], img_size=416, transform=ValTransforms())
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=detection_collate_fn,
    pin_memory=True
)


for epoch in range(epochs):
    model.train()
    for j,(x,y) in enumerate(dataloader):
        
        x=x.to(device)
        gt=matcher(y,13,13,model.anchors)
        pred=model(x)
        lxy,lwh,lclass,lconf,loss=loss_Fn(pred,gt,device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if j % 50==0:
            print(lxy.item(),lwh.item(),lclass.item(),lconf.item(),loss.item())

    scheduler.step(loss)
    print('epoch',epoch)
    if (epoch)%10==0:
        torch.save(model.state_dict(),f'epoch_{epoch+1}')
        val=evaluate(validation_dataloader,model,device=device)
