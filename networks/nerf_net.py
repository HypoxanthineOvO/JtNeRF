import jittor as jt
from jittor import nn

class NeRF_Net(nn.Module):
    def __init__(self,D = 8,W = 256,skips = [4],input_size = 39,output_size = 3):

        
        self.D = D
        self.W = W
        self.skips = skips
        self.I = input_size
        self.O = output_size
        
        self.input_linear = nn.Linear(self.I,self.W)
        
        self.linears = nn.ModuleList(
            [nn.Linear(self.W,self.W)] if i not in self.skips
            else [nn.Linear(self,W+self.I,self.W)] for i in range(1,self.D-1)
        )
        self.output_linear = nn.Linear(self.W,self.O)
        
    def execute(self, x_b) -> None:
        x = x_b
        x = self.input_linear(x)
        for i in range(self.D-2):
            x = self.linears[i](x)
            x = nn.relu(x)
            if i in self.skips:
                x = jt.concat([x,x_b],dim = -1)
        return self.output_linear(x)
        