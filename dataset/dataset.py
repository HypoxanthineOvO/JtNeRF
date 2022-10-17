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
        
        # Iterator parameters
        self.idx = 0
        
    def __next__(self):
        '''
        The NeRF-Dataset a iterator
        '''
        
    def load_data(self,root_dir = None):
        '''
        Load data from file
        '''
        if root_dir == None:
            root_dir = self.root_dir
        
        jsonfile = []
        for root,dirs,files in os.walk(root_dir):
            '''
            root: 当前文件夹的绝对路径
            dirs: 当前文件夹下的子文件夹路径(一层)
            files: 当前文件夹下所有文件的名称
            '''
            # Get .json path
            json_paths = []
            for file in files:
                if os.path.splitext(file)[1] == ".json":
                    if(self.mode in os.path.splitext(file)[0]):
                        json_paths.append(os.path.join(root,file))
            
            # Get frames
            json_data = None        
            for json_path in json_paths:
                with open(json_path,"r") as f:
                    data = json.load(f)
                if json_data is None:
                    json_data = data
                else:
                    '''
                    There only cone_angle and frame parameters of data.
                    The 'frames' means all the data
                    '''
                    json_data['frames'] += data['frames']                   
            
            frames = json_data['frames']
            for frame in tqdm(frames):
                img_path = os.path.join(self.root_dir,frame['file_path'])
                if not os.path.exists(img_path):
                    img_path = img_path + ".png"
                    if not os.path.exists(img_path):
                        print("俺的图图呢?")
                        continue
                img = read_image(img_path)
                    
                    
        
        
        
        
    