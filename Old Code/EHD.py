"""
File:   EHD.py
Author: Anthony Khoury and Keegan Bess
Date:   04/18/2022
Desc:    
"""

import torch.nn.functional as F
from scipy import signal, ndimage
import numpy as np
import torch



def Edge_Histogram_Descriptor(X, parameters, device=None):
    
    #Generate masks based on parameters
    masks = Generate_masks(mask_size=parameters['mask_size'],
                           angle_res=parameters['angle_res'],
                           normalize=parameters['normalize_kernel'])
  
    #Convolve input with filters, expand masks to match input channels
    #TBD works for grayscale images (single channel input)
    #Check for multiimage input
    in_channels = X.shape[1]
    masks = torch.tensor(masks).float()
    masks = masks.unsqueeze(1)
    
    #Replicate masks along channel dimension
    masks = masks.repeat(1,in_channels,1,1)
    
    if device is not None:
        masks = masks.to(device)
    
    edge_responses = F.conv2d(X, masks, dilation=parameters['dilation'])
    
    #Find max response
    [value,index] = torch.max(edge_responses,dim=1)
    
    #Set edge responses to "no edge" if not larger than threshold
    index[value<parameters['threshold']] = masks.shape[0] 
    
    feat_vect = []
    window_scale = np.prod(np.asarray(parameters['window_size']))
    
    for edge in range(0,masks.shape[0]+1):
        # #Sum count
        if parameters['normalize_count']:
           #Average count
            feat_vect.append((F.avg_pool2d((index==edge).unsqueeze(1).float(),
                              parameters['window_size'],stride=parameters['stride'],
                              count_include_pad=False).squeeze(1)))
        else:
            feat_vect.append(window_scale*F.avg_pool2d((index==edge).unsqueeze(1).float(),
                              parameters['window_size'],stride=parameters['stride'],
                              count_include_pad=False).squeeze(1))
        
    
    #Return vector
    feat_vect = torch.stack(feat_vect,dim=1)
        
    return feat_vect

def Generate_masks(mask_size=3,angle_res=45,normalize=False,rotate=False):
    
    #Make sure masks are appropiate size. Should not be less than 3x3 and needs
    #to be odd size
    if type(mask_size) is list:
        mask_size = mask_size[0]
    if mask_size < 3:
        mask_size = 3
    elif ((mask_size % 2) == 0):
        mask_size = mask_size + 1
    else:
        pass
    
    if mask_size == 3:
        if rotate:
            Gy = np.outer(np.array([1,2,1]).T,np.array([1,0,-1]))
        else:
            Gy = np.outer(np.array([1,0,-1]).T,np.array([1,2,1]))
    else:
        if rotate:
            Gy = np.outer(np.array([1,2,1]).T,np.array([1,0,-1]))
        else:
            Gy = np.outer(np.array([1,0,-1]).T,np.array([1,2,1]))
        dim = np.arange(5,mask_size+1,2)
        expand_mask =  np.outer(np.array([1,2,1]).T,np.array([1,2,1]))
        for size in dim:
            # Gx = signal.convolve2d(expand_mask,Gx)
            Gy = signal.convolve2d(expand_mask,Gy)
    
    #Generate horizontal masks
    angles = np.arange(0,360,angle_res)
    masks = np.zeros((len(angles),mask_size,mask_size))
    
    #TBD: improve for masks sizes larger than 
    for rot_angle in range(0,len(angles)):
        masks[rot_angle,:,:] = ndimage.rotate(Gy,angles[rot_angle],reshape=False,
                                              mode='nearest')
        
    
    #Normalize masks if desired
    if normalize:
        if mask_size == 3:
            masks = (1/8) * masks
        else:
            masks = (1/8) * (1/16)**len(dim) * masks 
    return masks



# EHD parameters
mask_size = [3,3]           # Convolution kernel size for edge responses
window_size = [5,5]         # Threshold for no edge orientation
angle_res = 45              # Angle resolution for masks rotations
normalize_count = True      # Set whether to use sum (unnormalized count) or average pooling (normalized count)
normalize_kernel = True     # Need to be normalized for histogram layer (maybe b/c of hist initialization)
threshold = 1/int(360/angle_res) 
stride = 1
dilation = 1

EHD_Parameters = {
    "mask_size"         : mask_size,
    "window_size"       : window_size,
    "angle_res"         : angle_res,
    "normalize_count"   : normalize_count,
    "normalize_kernel"  : normalize_kernel,
    "threshold"         : threshold,
    "stride"            : stride,
    "dilation"          : dilation
}