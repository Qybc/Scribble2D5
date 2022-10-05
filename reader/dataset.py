import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib
import cv2

class trainDataset(Dataset):
    def __init__(self, path_list, opt):
        self.resizesize = opt.spatial_shape
        self.cropsize = opt.crop_shape
        H,W,D = self.resizesize
        h,w,d = self.cropsize
        self.H_min = int((H - h)/2)
        self.H_max = int((H + h)/2)
        self.W_min = int((W - w)/2)
        self.W_max = int((W + w)/2)
        self.D_min = int((D - d)/2)
        self.D_max = int((D + d)/2)

        self.images = [item["img"] for item in path_list]
        self.scribbles = [item["scribble"] for item in path_list]
        self.gts = [item["gt"] for item in path_list]
        self.edges = [item["edge"] for item in path_list]
        self.supervoxels = [item["supervoxel"] for item in path_list]
        
        self.images = sorted(self.images)
        self.scribbles = sorted(self.scribbles)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        self.supervoxels = sorted(self.supervoxels)
        
        self.filter_files()
        self.size = len(self.images)
        # 限制对比度自适应直方图均衡化
        self.clahe = cv2.createCLAHE(3,(8,8))

    def __getitem__(self, index):

        image = self.nii_loader(self.images[index])
        scribble = self.nii_loader(self.scribbles[index])
        gt = self.nii_loader(self.gts[index])
        edge = self.nii_loader(self.edges[index])
        supervoxel = self.nii_loader(self.supervoxels[index])
        # import pdb;pdb.set_trace()
        # fuse gt(scribble gt) and superpixel to pseudo gt
        pseudo_gt = self.gen_pgt(scribble, supervoxel)

        # image/edge normalization
        dst=np.zeros(image.shape)
        image = (image - np.min(image))/(np.max(image) - np.min(image)) * 255
        image = image.astype(np.uint8)
        for d in range(image.shape[2]):
            dst[:,:,d] = self.clahe.apply(image[:,:,d]) / 255.0
        
        edge = (edge - np.min(edge))/(np.max(edge) - np.min(edge))

        # numpy array to torch tensor
        image = torch.Tensor(dst)
        scribble = torch.Tensor(scribble)
        gt = torch.Tensor(gt)
        edge = torch.Tensor(edge)
        supervoxel = torch.Tensor(supervoxel)
        pseudo_gt = torch.Tensor(pseudo_gt)


        # resize
        image = F.interpolate(image.unsqueeze(0).unsqueeze(0), size=self.resizesize, mode='trilinear', align_corners=True).squeeze(0)
        scribble = F.interpolate(scribble.unsqueeze(0).unsqueeze(0), size=self.resizesize, mode='nearest').squeeze(0)
        gt = F.interpolate(gt.unsqueeze(0).unsqueeze(0), size=self.resizesize, mode='nearest').squeeze(0)
        edge = F.interpolate(edge.unsqueeze(0).unsqueeze(0), size=self.resizesize, mode='trilinear', align_corners=True).squeeze(0)
        supervoxel = F.interpolate(supervoxel.unsqueeze(0).unsqueeze(0), size=self.resizesize, mode='nearest').squeeze(0)
        pseudo_gt = F.interpolate(pseudo_gt.unsqueeze(0).unsqueeze(0), size=self.resizesize, mode='nearest').squeeze(0)

        # crop
        image = image[:,self.H_min:self.H_max, self.W_min:self.W_max, self.D_min:self.D_max]
        scribble = scribble[:,self.H_min:self.H_max, self.W_min:self.W_max, self.D_min:self.D_max]
        gt = gt[:,self.H_min:self.H_max, self.W_min:self.W_max, self.D_min:self.D_max]
        edge = edge[:,self.H_min:self.H_max, self.W_min:self.W_max, self.D_min:self.D_max]
        supervoxel = supervoxel[:,self.H_min:self.H_max, self.W_min:self.W_max, self.D_min:self.D_max]
        pseudo_gt = pseudo_gt[:,self.H_min:self.H_max, self.W_min:self.W_max, self.D_min:self.D_max]

        return image, scribble, gt, edge, supervoxel, pseudo_gt

    def filter_files(self):
        images = []
        scribbles = []
        gts= []
        edges = []
        supervoxels = []
        for img_path, scribble_path, gt_path, edge_path, supervoxel_path in zip(self.images, self.scribbles, self.gts, self.edges, self.supervoxels):
            images.append(img_path)
            scribbles.append(scribble_path)
            gts.append(gt_path)
            edges.append(edge_path)
            supervoxels.append(supervoxel_path)
        self.images = images
        self.scribbles = scribbles
        self.gts = gts
        self.edges = edges
        self.supervoxels = supervoxels

    def nii_loader(self, path):
       return nib.load(path).get_fdata().transpose((1, 0, 2))

    def gen_pgt(self, scribble, supervoxel):
        H,W,D = scribble.shape
        pseudo_gt = np.zeros((H,W,D))

        scribble_value_list = np.unique(scribble)
        scribble_value_ignore = 0
        for scribble_value in scribble_value_list:
            if scribble_value != scribble_value_ignore:
                supervoxel_under_scribble_marking = np.unique(supervoxel[scribble[:, :, :] == scribble_value])
                for i in supervoxel_under_scribble_marking:
                    pseudo_gt[supervoxel==i] = scribble_value
        return pseudo_gt

    def __len__(self):
        return self.size


class valDataset(Dataset):
    def __init__(self, path_list, opt):
        self.resizesize = opt.spatial_shape
        self.cropsize = opt.crop_shape
        H,W,D = self.resizesize
        h,w,d = self.cropsize
        self.H_min = int((H - h)/2)
        self.H_max = int((H + h)/2)
        self.W_min = int((W - w)/2)
        self.W_max = int((W + w)/2)
        self.D_min = int((D - d)/2)
        self.D_max = int((D + d)/2)
        self.images = [item["img"] for item in path_list]
        self.gts = [item["gt"] for item in path_list]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.nii_loader(self.images[index])
        gt = self.nii_loader(self.gts[index])

        # normalization
        image = (image - np.min(image))/(np.max(image) - np.min(image))

        # numpy array to torch tensor
        image = torch.Tensor(image)
        gt = torch.Tensor(gt)

        # resize
        image = F.interpolate(image.unsqueeze(0).unsqueeze(0), size=self.resizesize, mode='trilinear', align_corners=True).squeeze(0)
        gt = F.interpolate(gt.unsqueeze(0).unsqueeze(0), size=self.resizesize, mode='nearest').squeeze(0)

        # crop
        image = image[:,self.H_min:self.H_max, self.W_min:self.W_max, self.D_min:self.D_max]
        gt = gt[:,self.H_min:self.H_max, self.W_min:self.W_max, self.D_min:self.D_max]

        return image, gt

    def nii_loader(self, path):
       return nib.load(path).get_fdata().transpose((1, 0, 2))

    def __len__(self):
        return self.size
