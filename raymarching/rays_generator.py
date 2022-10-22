import numpy as np
import jittor as jt

class NeRF_RayGen:
    def __init__(self,H,W,focal):
        self.H = H
        self.W = W
        self.focallength = focal
    def get_rays(self,c2w):
        i,j = jt.meshgrid(
            jt.arange(0,self.W,dtype= jt.float32),
            jt.arange(0,self.H,dtype= jt.float32),
            )
        dirs = jt.stack([(i-self.W/2)/self.focallength,-(j-self.H/2)/self.focallength,-jt.ones_like(i)],dim = -1)
        rays_d = jt.sum(dirs[...,np.newaxis,:] * c2w[...,:3,:3],dim=-1)
        rays_o = c2w[...,:3,-1].expand(rays_d.shape)
        return rays_o,rays_d