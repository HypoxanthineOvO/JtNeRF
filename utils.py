from pickletools import optimize
import jittor as jt
from jittor import nn
import numpy as np
import pandas as pd
import tqdm 

from dataset import NeRFDataset
from raymarching import NeRF_RayGen
from encoder import PositionalEncoder
from networks import NeRF_Net
from renders import NeRF_Render


class Trainer():
    def __init__(self,
                 name,
                 data,
                 encoder,
                 lr = 1e-5,
                 iters = 1000,
                 model = None,
                 lossfn = None,
                 optim = None,
                 ):
        '''
        name: string
        data: dict
            type, root_dir,batch_size
        encoder: dict
            type:{'pos','hash'}
            para:   For pos: an int
                    For hash: None
        '''
        self.name = name
        
        self.lr = lr
        self.N_iters = iters
        self.batchsize = data['batch_size']
        
        # Dataloader
        self.dataloader = NeRFDataset(
            data_type = data['type'],
            root_dir=data['root_dir'],
            batch_size= data['batch_size']
            )
        
        # Get Rays
        H,W,focal = self.dataloader.get_para()
        self.rays_gen = NeRF_RayGen(H,W,focal)
        
        # Encoder
        if encoder['type'] == 'pos':
            self.encoder = PositionalEncoder(L_enc=encoder['para'])
        elif encoder['type'] == 'hash':
            print("还没写")
            self.encoder = PositionalEncoder()
            
        # Network
        if model is None:
            self.model = NeRF_Net()
        else:
            print("还没写")
            self.model = NeRF_Net()
            
        # Render
        '''Need to change'''
        self.render = NeRF_Render(self.model,
                                  (2,6),
                                  self.dataloader.batch_size,
                                  100,
                                  self.encoder)
            
        # Optimizer
        if optim is None:
            self.optimizer = nn.Adam(self.model.parameters(),self.lr)
        else:
            print("还没写")
            self.optimizer = nn.Adam(self.model.parameters(),self.lr)
        
        # Training parameters
        self.train_index = 0
        
    def train_one_step(self):
        '''
        A train loop
        '''
        idxs,imgs,poses = next(self.dataloader)
        rays_o,rays_d = self.rays_gen.get_rays(poses)
        rgbs = []
        rays_o,rays_d = np.split(rays_o,self.batchsize),np.split(rays_d,self.batchsize)
        for i in range(len(rays_o)):
            #ro,rd = jt.array(rays_o[i]),jt.array(rays_d[i])
            ro,rd = rays_o[i],rays_d[i]
            rgb = self.render.rendering(ro,rd)
            
            rgbs.append(rgb)

        loss = jt.mean(jt.sqr(rgb-imgs))
        self.optimizer.step(loss)
        return loss
        
    def train(self):
        for i in tqdm.tqdm(range(self.N_iters)):
            loss = self.train_one_step()
            if i % 10 == 0:
                print(loss)
            
