import numpy as np
import jittor as jt
from jittor import nn 

class Losser(nn.Module):
    '''
    We assume that the input images are all normalized in [0,1]
    '''
    def __init__(self):
        pass 
    def mse(self,x,target):
        return jt.mean((x/1.0-target/1.0) ** 2)
    
    def psnr(self,x,target):
        x *= 255.
        target *= 255.
        msevalue = self.mse(x,target)
        if msevalue < 1e-10:
            return 100
        
        return 10. *(2 * np.log10(255.) - np.log10(msevalue))
    def mse2psnr(self,mse):
        mse *= 255.**2
        if mse < 1e-10:
            return 100
        
        return 10. *(2 * np.log10(255.) - np.log10(mse))