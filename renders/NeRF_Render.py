import numpy as np
import jittor as jt


class NeRF_Render:
    def __init__(self,net,rays_o,rays_d,z,N_samples,encoding):
        self.net = net
        self.rays_o = jt.array(rays_o)
        self.rays_d = jt.array(rays_d)
        self.zmin = z[0]
        self.zmax = z[1]
        self.N_samples = N_samples
        
        # For positional encoding, it could be a function
        # But for hash encoding, we may need a class.
        # So we need to create a class for them.
        self.enc = encoding
        
    def cumprod_exclusice(self,input):
        dim = -1
        cumprod = np.cumprod(input,dim)
        cumprod = np.roll(cumprod,1,dim)
        cumprod[...,0] = 1.0
        return cumprod
        
    def render(self):
        z_values = jt.linspace(self.zmin,self.zmax,self.N_samples)
        
        pts = self.rays_o[...,np.newaxis,:]+ self.rays_d[...,np.newaxis,:]*z_values[...,:,np.newaxis]
        pts_flat = jt.reshape(pts,[-1,3])
        pts_flat = self.enc(pts_flat)
        raw = self.net(pts_flat)
        