import cv2,imageio,os
from math import pi,tan
import numpy as np
import jittor as jt

# Const value
NERF_SCALE = 0.33

def read_image_imageio(image_file):
    img = imageio.imread(image_file)
    img = np.asarray(img).astype(np.float32)
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
    img /= 255.
    return img

def read_image(file):
    '''
    Will support the '.bin' image
    '''
    return read_image_imageio(file)

def fovangle_to_focallength(resolution:int,degrees:float):
    return 0.5*resolution/tan(0.5*degrees*(pi/180))