import numpy as np
import torch 
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import cv2 
def xywh2xyxy(bbx):
   bbx=np.array(bbx)
   bbx_new=np.zeros_like(bbx)
   bbx_new[...,0]=bbx[...,0]-bbx[...,2]/2
   bbx_new[...,1]=bbx[...,1]-bbx[...,3]/2
   bbx_new[...,2]=bbx[...,0]+bbx[...,2]/2
   bbx_new[...,3]=bbx[...,1]+bbx[...,3]/2
   return bbx_new 

def set_anchors(anchors):
    #input list anchors [[w,h]]
    #output will be array [[0,0,w,h]] 
    padding=np.zeros_like(anchors)
    anchors=np.array(anchors)
    return np.c_[padding,anchors]



def iou_achors(gt_box,anchor_size,mode='xywh'):

    gt_box=np.repeat(gt_box,len(anchor_size)).reshape(4,-1).T
    iou=compute_iou(gt_box,anchor_size,mode)
    return iou



def compute_iou(b1,b2,mode=None):

    if mode =='xywh':
        b1=xywh2xyxy(b1)
        b2=xywh2xyxy(b2)
    
    box1_x1=b1[...,0]
    box1_y1=b1[...,1]
    box1_x2=b1[...,2]
    box1_y2=b1[...,3]

    box2_x1=b2[...,0]
    box2_y1=b2[...,1]
    box2_x2=b2[...,2]
    box2_y2=b2[...,3]

    x1=np.maximum(box1_x1,box2_x1)
    y1=np.maximum(box1_y1,box2_y1)
    x2=np.minimum(box1_x2,box2_x2)
    y2=np.minimum(box1_y2,box2_y2)


    area_1=np.abs((box1_x2-box1_x1)*(box1_y2-box1_y1))

    area_2=np.abs((box2_x2-box2_x1)*(box2_y2-box2_y1))
    intersection=(x2-x1).clip(0)*(y2-y1).clip(0) #clip to 0 in case it doesnt intersect 
    union=area_1+area_2-intersection+1e-20
    iou=intersection/union
    return iou


def compute_iou_t(b1,b2,mode=None):

    if mode =='xywh':
        b1=xywh2xyxy(b1)
        b2=xywh2xyxy(b2)
    
    box1_x1=b1[...,0:1]
    box1_y1=b1[...,1:2]
    box1_x2=b1[...,2:3]
    box1_y2=b1[...,3:4]

    box2_x1=b2[...,0:1]
    box2_y1=b2[...,1:2]
    box2_x2=b2[...,2:3]
    box2_y2=b2[...,3:4]

    x1=torch.maximum(box1_x1,box2_x1)
    y1=torch.maximum(box1_y1,box2_y1)
    x2=torch.minimum(box1_x2,box2_x2)
    y2=torch.minimum(box1_y2,box2_y2)


    area_1=torch.abs((box1_x2-box1_x1)*(box1_y2-box1_y1))

    area_2=torch.abs((box2_x2-box2_x1)*(box2_y2-box2_y1))
    intersection=(x2-x1).clamp(0)*(y2-y1).clamp(0) #clip to 0 in case it doesnt intersect 
    union=area_1+area_2-intersection+1e-20
    iou=intersection/union
    return iou

def detection_collate_fn(batch):
    imgs,labels=list(zip(*batch))
    imgs=[img.float() for img in imgs]
    return torch.stack(imgs,0),labels
def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.T

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou

def xy2wh(xyxy):
    xywh=torch.zeros_like(xyxy)
    xywh[...,:2]=(xyxy[...,:2]+xyxy[...,2:4])/2
    xywh[...,2:4]=xyxy[...,2:4]-xyxy[...,:2]
    return xywh


def xw2xy(xywh):    
    xyxy=torch.zeros_like(xywh)
    xyxy[...,:2]=xywh[...,:2]-(xywh[...,2:4]/2)
    xyxy[...,2:4]=xywh[...,:2]+(xywh[...,2:4]/2)
    return xyxy


VOC_CLASSES = [  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']


class albumentations:
    def __init__(self):
        T=[
                  A.Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      max_pixel_value=255.0,
                  ),
                  ToTensorV2(),
              ]
        
        self.transform=A.Compose(T)
    def __call__(self,img):
        return self.transform(image=img)['image']
    





#plotting copied from repo 
def plot_bbox_labels(img, bboxes, cls_color=(0,255,255), text_scale=0.4):
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        label=VOC_CLASSES[int(bbox[-1])]+f': {bbox[-2]}'
        cls_color=class_colors[int(bbox[-1])]

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        t_size = cv2.getTextSize(label, 0, fontScale=3, thickness=1)[0]
        # plot bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
        
        if label is not None:
            # plot title bbox
            cv2.rectangle(img, (x1, y1-t_size[1]+5), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
            # put the test on the title bbox
            cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img  

#letterbox copied from yolov5

def letterbox(im, new_shape=None, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)  

#function to rescale infered bbx to the original image size 
def rescale_bbx(bbx,original_height,original_width,input_size=None):
    bbx[:,0]=bbx[:,0]*original_width
    bbx[:,2]=bbx[:,2]*original_width
    bbx[:,1]=bbx[:,1]*original_height
    bbx[:,3]=bbx[:,3]*original_height
    return bbx
def plot_bbox_labels(img, bboxes, cls_color=(0,255,255), text_scale=3.5):
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        label=VOC_CLASSES[int(bbox[-1])]+f': {bbox[-2]}'
        cls_color=class_colors[int(bbox[-1])]

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        t_size = cv2.getTextSize(label, 0, fontScale=6, thickness=5)[0]
        # plot bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 5)
        
        if label is not None:
            # plot title bbox
            cv2.rectangle(img, (x1, y1-t_size[1]+5), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
            # put the test on the title bbox
            cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)


    return img  