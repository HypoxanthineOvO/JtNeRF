import jittor as jt
import numpy as np
from jittor import nn
from tqdm import tqdm

from dataset.NeRF_dataset import NeRFDataset
from raymarching.rays_generator import NeRF_RayGen
from encoder.PosEncoder import PositionalEncoder
from networks.nerf_net import NeRF_Net
from renders.NeRFRender import NeRF_Render

if jt.has_cuda:
    print("Yes!")
    jt.flags.use_cuda = 1
batch_size = 10

data = NeRFDataset(data_type='npz',root_dir="./data/tiny_nerf_data.npz",batch_size= batch_size)
rays_generator = NeRF_RayGen(data.H,data.W,data.focal_length)
encoder = PositionalEncoder(6)
network = NeRF_Net()
render = NeRF_Render(network,(2.,6.),batch_size,100,encoder)
optimizer = nn.Adam(network.parameters(),0.0001)

print(network)

def train():
    idxs,imgs,poses = next(data)
    rays_o,rays_d = rays_generator.get_rays(poses)
    rgb = render.rendering(rays_o,rays_d)
    loss = jt.mean(jt.sqr(rgb-imgs))
    optimizer.step(loss)
    
for i in tqdm(range(20)):
    train()