from pickletools import optimize
import jittor as jt
from jittor import nn, save
import numpy as np
import pandas as pd
import tqdm 
from PIL import Image
import cv2 as cv
import os

from dataloaders import NeRFDataset
from raymarching import NeRF_RayGen
from encoders import PositionalEncoder
from networks import NeRF_Net
from renders import NeRF_Render
from utils import Losser


class Trainer():
    def __init__(self,
                 name,
                 data,
                 encoder,
                 para,
                 model = None,
                 optim = None,
                 save_checkpoint = True,
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
        
        self.lr = para['lr']
        self.N_iters = para["iters"]
        self.batchsize = data['batch_size']
        self.show_iter = self.N_iters/10
        self.save_checkpoint = save_checkpoint
        self.out = data['out']
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
        
        if os.path.exists(model):
            self.model = jt.load(model)
            print(f"\nLOAD model from {model}")
        else:
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

        # Compute loss
        self.losser = Losser()
        
        

    def train_one_step(self):
        '''
        A train loop
        '''
        idxs,imgs,poses = next(self.dataloader)

        rays_o,rays_d = self.rays_gen.get_rays(poses)
        """rays_o,rays_d = np.split(rays_o,self.batchsize),np.split(rays_d,self.batchsize)
        rgbs = np.array([self.render.rendering(rays_o[i],rays_d[i]).numpy() for i in range(len(rays_o))])
        rgbs = np.concatenate(rgbs,0)"""
        rays_o,rays_d = jt.split(rays_o,1),jt.split(rays_d,1)
        rgbs = jt.concat([self.render.rendering(rays_o[i],rays_d[i]) for i in range(self.batchsize)])
        loss = jt.mean(jt.sqr(rgbs-imgs))
        self.optimizer.step(loss)
        return loss
    
    def val(self):
        idxs,imgs,poses = next(self.dataloader)
        rays_o,rays_d = self.rays_gen.get_rays(poses)
        rays_o,rays_d = jt.split(rays_o,1),jt.split(rays_d,1)
        rgbs = jt.concat([self.render.rendering(rays_o[i],rays_d[i]) for i in range(self.batchsize)])
        return rgbs,imgs

    def train(self):
        for i in tqdm.tqdm(range(self.N_iters)):
            loss = self.train_one_step()
            if i % self.show_iter == 0:
                print("")
                print(f"Loss = {round(float(loss.numpy()),4)}, PSNR = {round(float(self.losser.mse2psnr(loss)),2)}")

        # Save the model
        if self.save_checkpoint:
            jt.save(self.model,self.out+'test_model.pkl')
        
        # Finally show the answer
        rgbs,imgs = self.val()
        psnr = np.mean([self.losser.psnr(rgbs[j],imgs[j]) for j in range(self.batchsize)])
        print(f"Finally, PSNR = {psnr}")
        for i,rgb in enumerate(rgbs):
            cv.imwrite(self.out+f"gt{i}.jpg",imgs[i].numpy()*255)
            cv.imwrite(self.out+f"op{i}.jpg",rgb.numpy()*255)
            
    def test(self):
        
        psnrs = []
        for i in tqdm.tqdm(range(self.dataloader.image_cnt/self.batchsize)):
            rgbs,imgs = self.val()
            psnr = np.mean([self.losser.psnr(rgbs[j],imgs[j]) for j in range(self.batchsize)])
            psnrs.append(psnr)
        finalpsnr = np.mean(np.array(psnrs))
        print(f"PSNR = {finalpsnr}")
        