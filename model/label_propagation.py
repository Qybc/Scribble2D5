import numpy as np
from skimage.segmentation import felzenszwalb, slic
import torch
import torch.nn.functional as F
from skimage.segmentation import watershed
from scipy.ndimage import median_filter
import cv2

def label_propagation(image, scribble, dataset='ACDC', method='felzenszwalb'):
    """
    image: BCHWD 0~1 torch.tensor
    scribble: BCHWD torch.tensor
    """
    
    # img 归一化
    image = (image - image.min())/(image.max() - image.min())*255
    # array
    image = image.cpu().numpy().astype(np.uint8)
    scribble = np.array(scribble)
    if dataset=='ACDC':
        scribble[scribble==4] = 0
    elif dataset=='CHAOS':
        scribble[scribble==5] = 0
    elif dataset=='VS':
        scribble[scribble==2] = 0
    elif dataset=='RUIJIN':
        scribble[scribble==2] = 0
    
    B,C,H,W,D = image.shape
    pseudo_mask = np.zeros(image.shape)
    su_mask = np.zeros(image.shape)
    for b in range(B):
        # 找前景区域，只在前景区域寻找pseudo label
        x,y,z = np.where(scribble[b,0,:,:,:]!=0)
        if x.size == 0:
            continue
        x_min, x_max, y_min, y_max, z_min, z_max = max((x.min()-10),0), min((x.max()+10), scribble.shape[2]), \
                                                   max((y.min()-10),0), min((y.max()+10), scribble.shape[3]), \
                                                   z.min(), z.max()+1
        img_fg = image[b,0,x_min:x_max, y_min:y_max, z_min:z_max]
        scr_fg = scribble[b,0,x_min:x_max, y_min:y_max, z_min:z_max]
        H_fg,W_fg,D_fg = img_fg.shape
        pseudo_fg = np.zeros(img_fg.shape)
        su_fg = np.zeros(img_fg.shape)
        

        for d in range(D_fg):
            img = img_fg[:,:,d]
            scr = scr_fg[:,:,d]
            su = felzenszwalb(img, scale=50, sigma=0.5, min_size=30)

            
            su_fg[:,:,d] = su
            scribble_value_list = np.unique(scr)
            scribble_value_ignore = 0
            for scribble_value in scribble_value_list:
                if scribble_value != scribble_value_ignore:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
                    tmp = scr.copy()
                    tmp[scr==scribble_value] = 1
                    tmp[scr!=scribble_value] = 0
                    if dataset=='ACDC':
                        valid_mask = cv2.dilate(tmp,kernel,iterations=1)
                    if 'CHAOS' in dataset:
                        valid_mask = cv2.dilate(tmp,kernel,iterations=5)
                    if dataset=='VS':
                        valid_mask = cv2.dilate(tmp,kernel,iterations=1)
                    if dataset=='RUIJIN':
                        valid_mask = cv2.dilate(tmp,kernel,iterations=1)
                    supervoxel_under_scribble_marking = np.unique(su[scr == scribble_value])
                    tmp_mask = np.zeros(img.shape)
                    for i in supervoxel_under_scribble_marking:
                        tmp_mask[su==i] = scribble_value
                    if dataset != 'VS':
                        tmp_mask*=valid_mask
                    for h in range(H_fg):
                        for w in range(W_fg):
                            if tmp_mask[h,w]!=0:
                                pseudo_fg[h,w,d] = tmp_mask[h,w]
        pseudo_mask[b,0,x_min:x_max, y_min:y_max, z_min:z_max] = pseudo_fg
        su_mask[b,0,x_min:x_max, y_min:y_max, z_min:z_max] = su_fg
           

    return torch.Tensor(pseudo_mask), torch.Tensor(su_mask)