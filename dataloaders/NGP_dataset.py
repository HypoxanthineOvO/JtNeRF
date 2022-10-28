import random
import jittor as jt
from jittor.dataset import Dataset
import numpy as np
import os,json

from math import pi,tan
from tqdm import tqdm

from .data_utils import *

class NGPDataset():
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
        self.resolution = [H,W]
        self.correct_pose = correct_pose
        
        # aavv
        self.aabb_scale = None
        
        # Datas
        self.img_count = 0
        self.image_data = [] # Images of the dataset
        self.metadata = []
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
        self.resolution = [self.H,self.W]
        self.resolution_GPU = jt.array(self.resolution)
        
        metadata = np.empty([11],np.float32)
        # Dinctionary.get(key,default): Return the value of key.
        metadata[0] = json_data.get('k1',0)
        metadata[1] = json_data.get('k1',0)
        metadata[2]= json_data.get('p1',0)
        metadata[3]=json_data.get('p2',0)
        metadata[4]=json_data.get('cx',self.W/2)/self.W
        metadata[5]=json_data.get('cy',self.H/2)/self.H
        
        # Read the parameter of focal length.
        def read_focal_length(resolution,axis):
            if 'fl_'+axis in json_data:
                return json_data['fl_'+axis]
            elif 'camera_angle_'+axis in json_data:
                return fovangle_to_focallength(resolution,json_data['camera_angle_'+axis]*180/pi)
            else:
                return 0
            
        # Resolution means the size of image.
        x_fl = read_focal_length(self.resolution[0],'x')
        y_fl = read_focal_length(self.resolution[1],'y')
        focal_length = []
        if x_fl == 0 and y_fl == 0:
            raise RuntimeError("Fuck You!!! Where are the fov???")
        elif x_fl != 0:
            focal_length = [x_fl,y_fl]
            if y_fl != 0:
                focal_length[1] = y_fl
        else:
            focal_length = [y_fl,y_fl]
            
        self.focal_lengths.append(focal_length)
        metadata[6] = focal_length[0]
        metadata[7] = focal_length[1]
        
        light_direction = np.array([0,0,0])
        metadata[8:] = light_direction
        
        self.metadata = np.expand_dims(metadata,0).repeat(self.img_count,axis = 0)
        if self.aabb_scale is None:
            self.aabb_scale = json_data.get('aabb_scale',1)
        aabb_range = (0.5,0.5)
        self.aabb_range = (aabb_range[0]-self.aabb_scale/2,aabb_range[1]+self.aabb_scale/2)
        
        self.H = int(self.H)
        self.W = int(self.W)
        
        self.image_data = jt.array(self.image_data)
        self.transform_matrixs = jt.array(self.transform_matrixs)
        self.focal_lengths = jt.array(self.focal_lengths).repeat(self.img_count,1)
        
        # Transpose to adapt Eigen::Matrix memory
        self.transform_matrixs = self.transform_matrixs.transpose(0,2,1)
        self.metadata = jt.array(self.metadata)
        
        self.shuffle_index = jt.randperm(self.H*self.W*self.img_count).detach()
        jt.gc()
        
        
        # 直接跑的话肯定会出问题，因为不是复制粘贴的内容。
        # 后面看情况，如果限定了训练集，可以尝试自己重新写一个 dataset 类。
        # 暂时就先到这里吧
        
        
                
    
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
                    
                    
        
        
        
        
    