import random
import jittor as jt
from jittor.dataset import Dataset
import numpy as np
import os,json,imageio,cv2

from math import pi,tan
from tqdm import tqdm

from .data_utils import *

class NeRFDataset():
    def __init__(self,
                 root_dir,
                 batch_size,
                 mode = "train",
                 H = 0, W = 0,
                 correct_pose = [1,-1,-1]):
        self.root_dir = root_dir
        self.batch_size = batch_size
        
        self.imgsize = [H,W]
        self.correct_pose = correct_pose
        
        # Datas
        self.img_count = 0
        self.image_data = [] # Images of the dataset
        self.focal_lengths = [] # focal length of each images
        
        # Modes
        modes = ["train","test"]
        assert mode in modes 
        self.mode = mode 
        
    def __next__(self):
        '''
        The NeRF-Dataset a iterator
        '''
        
        
        
        
    