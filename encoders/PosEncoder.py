import jittor as jt
import numpy as np
from jittor import nn

class PositionalEncoder(nn.Module):
    def __init__(self,L_enc = 6):
        self.L_enc = L_enc
        size = 3+6*L_enc
        self.input_size = 3
        self.output_size = size 
        
    def positional_encoding(self,x):
        '''
        x.size = (3,)
        '''
        rets = [x]
        for i in range(self.L_enc):
            for functions in [jt.sin,jt.cos]:
                rets.append(functions(2. ** i * x))
        return jt.concat(rets,dim=-1)
    
    def execute(self,x):
        return self.positional_encoding(x)
