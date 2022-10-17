import jittor as jt
from jittor import nn

class NeRF_Net(nn.Module):
    def __init__(self,D = 8,W = 256,skips = [4]):
        super(NGP_Net,self).__init__()
        
        self.D = D
        self.W = W
        self.skips = skips
        