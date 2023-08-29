import torch 
from torch import nn 
import torch.nn.functional as F

class cnnblock(nn.Module):
    def __init__(self,in_,
                 out_,
                 kernel,
                 padding='same',
                 activation=True,
                 **kwargs):
        
        super().__init__()
        self.cnn=nn.Conv2d(in_channels=in_,
                           out_channels=out_,
                           kernel_size=kernel,
                           bias=True,
                           padding=kernel//2 if padding=='same' else 0,            
                           **kwargs)
        
        self.batch=nn.BatchNorm2d(out_)
        self.activation=nn.LeakyReLU(0.1,inplace=True) if activation else nn.Identity()


    def forward(self,x):
        return self.activation(self.batch(self.cnn(x)))

class reorg(nn.Module):
    #add mid level features to high level features 
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
    def forward(self,x):
        batch_size,channels,h,w=x.size()
        assert h % self.stride == 0 and w % self.stride == 0, "Input dimensions must be divisible by stride."
        _h,_w=h//self.stride,w//self.stride
        x=x.view(batch_size,channels,_h,self.stride,_w,self.stride)
        x=x.permute(0,1,2,4,3,5).contiguous()
        x=x.view(batch_size,-1,_h,_w)
        return x
#neck 
class SPP(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x1=F.max_pool2d(x,5,stride=1,padding=2)
        x2=F.max_pool2d(x,9,stride=1,padding=4)
        x3=F.max_pool2d(x,13,stride=1,padding=6)
        return torch.cat([x1,x2,x3,x],1)