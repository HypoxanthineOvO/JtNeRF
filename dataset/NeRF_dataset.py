import jittor as jt
import numpy as np

class NeRFDataset:
    def __init__(self,
                 data_type,
                 root_dir,
                 mode = 'train',
                 batch_size = 10,
                 ):
        types = ['npz','nerf_synthetic','nerf_llff']
        assert data_type in types
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.mode = mode 
        modes = ['train','test']
        assert self.mode in modes 
        self.images = None
        self.poses = None
        self.focal_length = 0
        self.H = 0
        self.W = 0
        
        self.idx = 0
        self.image_cnt = 0
        
        if (data_type == 'npz'):
            self.load_npz_data()
            
        self.images = jt.array(self.images)
        self.poses = jt.array(self.images)
        self.focal_length = jt.array(self.focal_length)
        
            
    # Use next(object) to get the new data.
    def __next__(self):
        assert self.batch_size < self.image_cnt
        if self.idx+self.batch_size >=self.image_cnt:
            self.idx = 0
        indexs = np.arange(self.idx,self.idx+self.batch_size)
        
        imgs = self.images[indexs]
        poss = self.poses[indexs]
        self.idx += self.batch_size
        return indexs,imgs,poss
        
        
        
        
        
    def load_npz_data(self):
        # Check the type of data
        data = np.load(self.root_dir)
        images = data['images']
        poses = data['poses']
        self.focal_length = data['focal']
        
        self.H,self.W = images.shape[1],images.shape[2]
        
        if self.mode == 'train':
            self.images = images[:100,...,:3]
            self.poses = poses[:100]
            self.image_cnt = 100
        elif self.mode == 'test':
            self.images = images[101]
            self.poses = poses[101]
            self.image_cnt = 1
            
    def reset(self):
        self.idx = 0
        
        
        
        