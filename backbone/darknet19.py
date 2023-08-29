import torch 
from torch import nn 
from .basic import cnnblock


class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()
        
        #p1/2
        self.cnn_1=nn.Sequential(
                    cnnblock(3,32,3),
                    nn.MaxPool2d(2)
                         )  
        #p2/4
        self.cnn_2=nn.Sequential(
                    cnnblock(32,64,3),
                    nn.MaxPool2d(2)
                         )   
        #p3/8
        self.cnn_3=nn.Sequential(
                    cnnblock(64,128,3),
                    cnnblock(128,64,1),
                    cnnblock(64,128,3),
                    nn.MaxPool2d(2)
                         )   
        
        #p4/16
        self.cnn_4=nn.Sequential(
                    cnnblock(128,256,3),
                    cnnblock(256,128,1),
                    cnnblock(128,256,3),
                         )  
        #p5/32
        self.max_pool_4=nn.MaxPool2d(2)

        self.cnn_5=nn.Sequential(
                    cnnblock(256,512,3),
                    cnnblock(512,256,1),
                    cnnblock(256,512,3),
                    cnnblock(512,256,1),
                    cnnblock(256,512,3),
                         ) 

        self.max_pool_5=nn.MaxPool2d(2)

        self.cnn_6=nn.Sequential(
                    cnnblock(512,1024,3),
                    cnnblock(1024,512,1),
                    cnnblock(512,1024,3),
                    cnnblock(1024,512,1),
                    cnnblock(512,1024,3),
                         )   
        

    def forward(self,x):
        c1=self.cnn_1(x)
        c2=self.cnn_2(c1)
        c3=self.cnn_3(c2)
        c3=self.cnn_4(c3)
        c4=self.cnn_5(self.max_pool_4(c3))
        c5=self.cnn_6(self.max_pool_5(c4))

        output={
            "c3":c3,
            "c4":c4,
            "c5":c5
        }

        return output

    
def load_model_names(my_model,pretrained_model):

    ''''
    my_model : model 
    pretrained_model : state_dict
    '''

    model_params=my_model.state_dict()
    new=list(pretrained_model.items())
    count=0
    for key,vals in model_params.items():
         layer_name,weights=new[count]
         model_params[key]=weights
         count+=1
    

    
    my_model.load_state_dict(model_params)
    print('='*50,'\nweights have been loaded\n'+'='*50)

def build_darknet19(pretrained=True,weights_path=r'pretrained_models\darknet19.pth'):
    backbone=Darknet19()
    if pretrained:
       pretrained_model= torch.load(weights_path) 
       load_model_names(backbone,pretrained_model)
    return backbone








     

