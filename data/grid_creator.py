import torch
import numpy as np 
from utils.utils import * 

def matcher(labels,hs,ws,anchors):
   batch_size=len(labels)
   gxy,gwh,gt_class,indices=[],[],[],[]
   # gt_bboxes=np.zeros([batch_size,hs,ws,5,5])
   for batch_id in range(batch_size):
                  for label in labels[batch_id]:
                     xyxy=torch.tensor(label[:-1]).float()*hs
                     xywh=xy2wh(xyxy)

                     iou=[wh_iou(xywh[2:4],anch) for anch in anchors]
                     iou,a=torch.stack(iou).max(0)
                     txy=xywh[:2]
                     grid_i,grid_j=txy.long()
                     twh=torch.log(xywh[2:4]/anchors[a])
                     txy=txy-txy.long()

                     gxy.append(txy)
                     gwh.append(twh)
                     gt_class.append(int(label[-1]))
                     indices.append([batch_id,grid_j,grid_i,a])
   
   return torch.stack(gxy),torch.stack(gwh),torch.tensor(gt_class),torch.tensor(indices)
                








