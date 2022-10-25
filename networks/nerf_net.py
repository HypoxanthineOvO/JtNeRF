import jittor as jt
from jittor import nn

class NeRF_Net(nn.Module):
    def __init__(self,D = 8,W = 256,skips = [4],input_size = 39,output_size = 4):

        
        self.D = D
        self.W = W
        self.skips = skips
        self.I = input_size
        self.O = output_size
        
        self.linears = nn.ModuleList(
            [nn.Linear(self.I,self.W)]+
            [nn.Linear(self.W,self.W) if i not in self.skips
            else nn.Linear(self.W+self.I,self.W) for i in range(self.D-1)
            ]
        )
        self.output_linear = nn.Linear(self.W,self.O)
        
    def execute(self, x_b) -> None:
        x = x_b
        for i in range(0,self.D-1):
            x = self.linears[i](x)
            x = nn.relu(x)
            if i in self.skips:
                x = jt.concat([x_b,x],dim = -1)
            #print(x.shape)
        return self.output_linear(x)
        