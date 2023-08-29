import torch 
from .darknet19 import build_darknet19

def build_backbone(model_name='darknet19',pretrained=True):
    if model_name=="darknet19":
        backbone=build_darknet19(pretrained)
    return backbone
if __name__=='__name__':
    net=build_backbone()
    print(net)