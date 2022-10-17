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
                 correct_pose = [1,-1,-1],
                 scale = None,offset = None):
        self.root_dir = root_dir
        self.batch_size = batch_size
        
        self.H = H
        self.W = W
        self.imgsize = [H,W]
        self.correct_pose = correct_pose
        
        # Datas
        self.img_count = 0
        self.image_data = [] # Images of the dataset
        self.focal_lengths = [] # focal length of each images
        self.transform_matrixs = [] # Transform matrixs
        
        # To correct the scale of NeRF and NGP
        if scale is None:
            self.scale = NERF_SCALE
        else:
            self.scale = scale 
        if offset is None:
            self.offset = [0.5,0.5,0.5]
        else:
            self.offset = offset
        
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
                '''
                Load the images
                '''
                img_path = os.path.join(self.root_dir,frame['file_path'])
                if not os.path.exists(img_path):
                    img_path = img_path + ".png"
                    if not os.path.exists(img_path):
                        print("俺的图图呢?")
                        continue
                img = read_image(img_path)
                if self.H == 0 or self.W == 0:
                    self.H = int(img.shape[0])
                    self.W = int(img.shape[1])
                self.image_data.append(img)
                
                self.img_count += 1
                matrix = np.array(frame['transform_matrix'],np.float32)[:-1,:] # Remove the last column
                self.transform_matrixs.append(
                    self.matrix_NeRF2NGP(matrix,self.scale,self.offset)
                )
    # 原JNeRF P116
                
    
    def matrix_NeRF2NGP(self,matrix,scale,offset):
        '''
        Copy from JNeRF
        '''
        matrix[:, 0] *= self.correct_pose[0]
        matrix[:, 1] *= self.correct_pose[1]
        matrix[:, 2] *= self.correct_pose[2]
        matrix[:, 3] = matrix[:, 3] * scale + offset
        matrix=matrix[[1,2,0]]
        return matrix
        
    def matrix_NGP2NeRF(self, matrix, scale, offset):
        '''
        Copy from JNeRF
        '''
        matrix=matrix[[2,0,1]]
        matrix[:, 0] *= self.correct_pose[0]
        matrix[:, 1] *= self.correct_pose[1]
        matrix[:, 2] *= self.correct_pose[2]  
        matrix[:, 3] = (matrix[:, 3] - offset) / scale
        return matrix
                    
                    
        
        
        
        
    