import numpy as np
import torch 
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import cv2 
from terminaltables import AsciiTable
import tqdm
import torchvision
import time 

#copied from SSD repo

class_names = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')



def process_targets(targets,input_size=416.0):
    z=[np.c_[np.zeros(len(j)),j] for j in targets]
    for i,j in enumerate(z):
        j[:,0]=i
    z=[torch.tensor(j) for j in z]
    z=torch.cat(z,0)
    z[:,1:5]=z[:,1:5]*input_size
    return z 
def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xw2xy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, -1] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:,:-1]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics
def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

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
# def plot_bbox_labels(img, bboxes, cls_color=(0,255,255), text_scale=0.4):
#     np.random.seed(0)
#     class_colors = [(np.random.randint(255),
#                      np.random.randint(255),
#                      np.random.randint(255)) for _ in range(20)]

#     for bbox in bboxes:
#         x1, y1, x2, y2 = bbox[:4]
#         label=VOC_CLASSES[int(bbox[-1])]+f': {bbox[-2]}'
#         cls_color=class_colors[int(bbox[-1])]

#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         t_size = cv2.getTextSize(label, 0, fontScale=3, thickness=1)[0]
#         # plot bbox
#         cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
        
#         if label is not None:
#             # plot title bbox
#             cv2.rectangle(img, (x1, y1-t_size[1]+5), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
#             # put the test on the title bbox
#             cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

#     return img  

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
def plot_bbox_labels(img, bboxes, cls_color=(0,255,255), text_scale=2.5):
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
        cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 7)
        
        if label is not None:
            # plot title bbox
            cv2.rectangle(img, (x1, y1-t_size[1]+5), (int(x1 + t_size[0]), y1), cls_color, -1)
            # put the test on the title bbox
            cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)


    return img  
