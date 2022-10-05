#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from calendar import c
import os
import time
import numpy as np
import pandas as pd
# from tqdm import tqdm

import torch
from torch import nn 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import tensorboardX

from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    SpatialPadd,
    NormalizeIntensityd,
    RandFlipd,
    RandSpatialCropd,
    Orientationd,
    ToTensord,
)


from model.label_propagation import label_propagation
from model.unet2d5_spvPA_scribble import UNet2d5_spvPA_scribble

from loss.scribble2d5_loss import Scribble2D5Loss

from medpy.metric.binary import dc, precision, hd95
import SimpleITK as sitk

import utils.vis as vis_utils

import logging as logger
logger.basicConfig(
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logger.INFO)

# Create logger and tensorboard writer.
summary_writer = tensorboardX.SummaryWriter()
color_map = vis_utils.load_color_map('misc/colormapvoc.mat')

# GPU CHECKING
if torch.cuda.is_available():
    logger.info("[INFO] GPU available.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    raise logger.error(
        "[INFO] No GPU found")

def parsing_data():
    parser = argparse.ArgumentParser(
        description="Script to train the models using scribbles as supervision")

    parser.add_argument("--epochs",
                    type=int,
                    default=3000,
                    help="epochs (default: 30)")

    parser.add_argument("--batch_size",
                    type=int,
                    default=6,
                    help="Size of the batch size (default: 4)")

    parser.add_argument("--model_dir",
                    type=str,
                    default="ckpts/",
                    help="Path to the model dir")

    parser.add_argument("--learning_rate",
                    type=float,
                    default=1e-4,
                    help="Initial learning rate")

    parser.add_argument("--spatial_shape",
                    type=int,
                    nargs="+",
                    default=(224,224,32),
                    help="Size of the resize spatial shape")

    parser.add_argument("--method",
                    type=str,
                    default='felzenszwalb',
                    help="Type of the generate supervoxel")

    parser.add_argument("--resume",
                    type=bool,
                    default=True,
                    help="wether resume from previous training")

    parser.add_argument("--data",
                    type=str,
                    default="CHAOS") 

    opt = parser.parse_args()

    opt.save_path = os.path.join(opt.model_dir, 'UNet2d5_spvPA', opt.data, "./CP_{}.pth")

    if opt.data == 'ACDC':
        opt.num_classes = 4
        opt.dataset_split = 'splits/split_ACDC.csv'
        opt.path_data = 'data/ACDC'
        opt.spatial_shape = (224,224,8)
        opt.batch_size = 4

    if opt.data == 'CHAOS':
        opt.num_classes = 5
        opt.dataset_split = 'splits/split_CHAOS.csv'
        opt.path_data = 'data/CHAOS-MR-T1-inphase'
        opt.spatial_shape = (192,192,32)
        opt.batch_size = 2

    if opt.data == 'VS':
        opt.num_classes = 2
        opt.dataset_split = 'splits/split_VS.csv'
        opt.path_data = 'data/VS'
        opt.spatial_shape = (192,128,48)
        opt.batch_size = 2

    return opt


def main():

    # parse args
    opt = parsing_data()

    # split train/val/infer dataset
    logger.info("Spliting data")
    assert os.path.isfile(opt.dataset_split), logger.error("[ERROR] Invalid split")
    df_split = pd.read_csv(opt.dataset_split,header = None)
    list_file = dict()
    phases = ["training", "validation"]
    for split in phases:
        list_file[split] = df_split[df_split[1].isin([split])][0].tolist()
    
    paths_dict = {split:[] for split in phases}
    for split in phases:
        for subject in list_file[split]:
            subject_data = dict()

            subject_data["img"] = os.path.join(opt.path_data,'images', str(subject)+'.nii.gz')
            subject_data["gt"] = os.path.join(opt.path_data,'full_annos', str(subject)+'.nii.gz')
            subject_data["scribble"] = os.path.join(opt.path_data,'scribbles', str(subject)+'.nii.gz')
            subject_data["edge"] = os.path.join(opt.path_data,'edges', str(subject)+'.nii.gz')
            paths_dict[split].append(subject_data)
        logger.info(f"patients in {split} data: {len(list_file[split])}")

    # reader
    # PREPROCESSING
    transforms = dict()
    all_keys = ["img", "gt", "scribble", "edge"]

    transforms_training = (
        LoadNiftid(keys=all_keys),
        AddChanneld(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        NormalizeIntensityd(keys=["img"]),
        SpatialPadd(keys=all_keys, spatial_size=opt.spatial_shape), # 填充后输出数据的空间大小，如果输入数据大小的维度大于填充大小，则不会填充该维度
        RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
        RandSpatialCropd(keys=all_keys, roi_size=opt.spatial_shape, random_center=True, random_size=False), # 从中心crop spatial_shape (224,224,8)大小
        ToTensord(keys=all_keys),
        )   
    transforms["training"] = Compose(transforms_training)   

    transforms_validation = (
        LoadNiftid(keys=all_keys),
        AddChanneld(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        NormalizeIntensityd(keys=["img"]),
        SpatialPadd(keys=all_keys, spatial_size=opt.spatial_shape),  # 填充后输出数据的空间大小，如果输入数据大小的维度大于填充大小，则不会填充该维度
        ToTensord(keys=all_keys)
        )   

    if opt.data == 'VS':
            transforms_training = (
                LoadNiftid(keys=all_keys),
                AddChanneld(keys=all_keys),
                Orientationd(keys=all_keys, axcodes="RAS"),
                NormalizeIntensityd(keys=["img"]),
                SpatialPadd(keys=all_keys, spatial_size=opt.spatial_shape), # 填充后输出数据的空间大小，如果输入数据大小的维度大于填充大小，则不会填充该维度
                RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
                RandSpatialCropd(keys=all_keys, roi_size=opt.spatial_shape, random_center=False, random_size=False), # 从中心crop spatial_shape (224,224,8)大小
                ToTensord(keys=all_keys),
                )   
            transforms["training"] = Compose(transforms_training)   

            transforms_validation = (
                LoadNiftid(keys=all_keys),
                AddChanneld(keys=all_keys),
                Orientationd(keys=all_keys, axcodes="RAS"),
                NormalizeIntensityd(keys=["img"]),
                SpatialPadd(keys=all_keys, spatial_size=opt.spatial_shape),  # 填充后输出数据的空间大小，如果输入数据大小的维度大于填充大小，则不会填充该维度
                RandSpatialCropd(keys=all_keys, roi_size=opt.spatial_shape, random_center=False, random_size=False), # 从中心crop spatial_shape (224,224,8)大小
                ToTensord(keys=all_keys)
                ) 
    transforms["validation"] = Compose(transforms_validation)

 
    # model
    logger.info("Building model")
    model = UNet2d5_spvPA_scribble(
            dimensions=3, # number of spatial dimensions.
            in_channels=1,
            out_channels=opt.num_classes,
            channels=(16, 32, 48, 64, 80, 96),
            strides=(
                (2, 2, 1),
                (2, 2, 1),
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
            ),
            kernel_sizes=(
                (3, 3, 1),
                (3, 3, 1),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
            sample_kernel_sizes=(
                (3, 3, 1),
                (3, 3, 1),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.1,
            attention_module=True,
        ).to(device)
    
    # loss function
    logger.info("Building loss function")
    loss_function = Scribble2D5Loss(to_onehot_y=True, softmax=True, supervised_attention=True, hardness_weighting=True)
    
    logger.info("[INFO] Training")
    train(paths_dict, 
        model, 
        transforms, 
        loss_function, # loss
        opt)
    

def train(paths_dict, model, transformation, loss_function, opt):
    
    since = time.time()
    PHASES = ["training", "validation"] #

    # Define transforms for data normalization and augmentation
    subjects_train = Dataset(
        paths_dict["training"], 
        transform=transformation["training"])

    subjects_val = Dataset(
        paths_dict["validation"], 
        transform=transformation["validation"])
    
    # Dataloaders
    def infinite_iterable(i):
        while True:
            yield from i
    dataloaders = dict()
    dataloaders["training"] = infinite_iterable(
        DataLoader(subjects_train, batch_size=opt.batch_size, num_workers=2, shuffle=True)
        )
    dataloaders["validation"] = infinite_iterable(
        DataLoader(subjects_val, batch_size=1, num_workers=2)
        )

    nb_batches = {
        "training": 30, # One image patch per epoch for the full dataset
        "validation": len(paths_dict["validation"])
        }

    # load model
    # Training parameters are saved 
    df_path = os.path.join(opt.model_dir,'UNet2d5_spvPA', opt.data, "log.csv")
    if os.path.isfile(df_path) and opt.resume: # If the training already started
        logger.info("Resume...")
        df = pd.read_csv(df_path, index_col=False)
        epoch = df.iloc[-1]["epoch"]
        best_epoch = df.iloc[-1]["best_epoch"]
        best_val = df.iloc[-1]["best_val"]
        initial_lr = df.iloc[-1]["lr"]
        model.load_state_dict(torch.load(opt.save_path.format("best")))

    else: # If training from scratch
        columns=["epoch","best_epoch", "MA", "best_MA", "lr", "timeit"]
        df = pd.DataFrame(columns=columns)
        best_val = None
        best_epoch = 0
        epoch = 0
        initial_lr = opt.learning_rate


    # Optimisation policy mimicking nnUnet training policy
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-7)

    # Training loop
    continue_training = True
    while continue_training:
        epoch+=1
        logger.info("-" * 10)
        logger.info("Epoch {}/".format(epoch))

        for param_group in optimizer.param_groups:
            logger.info("Current learning rate is: {}".format(param_group["lr"]))
            
        # Each epoch has a training and validation phase

        for phase in PHASES:
            if phase == "training":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode 

            # Initializing the statistics
            running_loss = 0.0
            running_dice = 0.0
            epoch_samples = 0
            running_time = 0.0

            list_dice = []
            list_hd95 = []
            list_precision = []

            # Iterate over data
            for index in range(nb_batches[phase]):
                batch = next(dataloaders[phase])
                inputs = batch["img"].to(device) # T2 images
                labels = batch["gt"].to(device)
                scribbles = batch["scribble"].to(device)
                edges = batch["edge"].to(device)
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "training"):
                    if phase=="training": # Random patch predictions
                        # import pdb;pdb.set_trace()
                        outputs = model(inputs)
                        
                        
                    else:  # if validation, Inference on the full image
                        outputs = sliding_window_inference(
                            inputs=inputs,
                            roi_size=opt.spatial_shape,
                            sw_batch_size=1,
                            predictor=model,
                            index=4, # modify MONAI
                            mode="gaussian",
                        )
                        

                    
                    

                    # Segmentation loss
                    if phase == 'training':
                        init_time_pesudo_labels = time.time()
                        pseudo_labels, su_mask = label_propagation(inputs.cpu(), scribbles.cpu(), opt.data)
                        time_pesudo_labels = time.time() - init_time_pesudo_labels
                        loss = loss_function(outputs, pseudo_labels.to(device), inputs, edges)
                        logger.info("the {}-st batch {} loss: {}".format(index, phase, loss))

                        dice_score = dc(np.array(outputs[4].cpu().argmax(dim=1,keepdim=True)), 
                                        np.array(labels.cpu()))

                        dice_pseudo_label = dc(np.array(pseudo_labels.cpu()), 
                                               np.array(labels.cpu()))
                        dice_init = dc(np.array(outputs[2].cpu().argmax(dim=1,keepdim=True)), 
                                       np.array(labels.cpu()))
                        dice_ref = dc(np.array(outputs[4].cpu().argmax(dim=1,keepdim=True)), 
                                      np.array(labels.cpu()))
                    elif phase == 'validation':
                        pred = np.array(outputs.cpu().argmax(dim=1,keepdim=True))
                        gt = np.array(labels.cpu())
                        dice_score = dc(pred, gt)
                        prec = precision(pred, gt)
                        hd_score = 0.0
                        if np.sum(pred)>0:
                            hd_score = hd95(pred, gt)
                        print('dice: {}, hd95: {}, precision: {}'.format(dice_score, hd_score, prec))
                        list_dice.append(dice_score*100)
                        list_hd95.append(hd_score)
                        list_precision.append(prec*100)
                                        
                    logger.info("the {}-st batch {} dice: {}".format(index, phase, dice_score))
                    if phase == 'training':
                        logger.info("the {}-st batch {} pseudo label dice: {}".format(index, phase, dice_pseudo_label))
                        logger.info("the {}-st batch {} init dice: {}".format(index, phase, dice_init))
                        logger.info("the {}-st batch {} ref dice: {}".format(index, phase, dice_ref))

                    # visualize loss
                    summary_val = {'DiceLoss': {}, 'DiceScore': {}}
                    if phase == 'training':
                        summary_val['DiceLoss'][phase] = loss.to('cpu')

                    summary_val['DiceScore'][phase] = dice_score
                    if phase == 'training':
                        summary_val['DiceScore']['train_pseudo_label'] = dice_pseudo_label
                        summary_val['DiceScore']['train_init'] = dice_init
                        summary_val['DiceScore']['train_ref'] = dice_ref
                    vis_utils.write_scalars_to_tensorboard(summary_writer, summary_val, (epoch-1) * 30 + index)

                    # visualize tensors
                    summary_vis = []
                    # image
                    summary_vis.append(vis_utils.convert_image(inputs.cpu().permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1))) # 把batchsize和depth flatten成一个维度展示
                    # gt
                    summary_vis.append(vis_utils.convert_label_to_color(labels.cpu().permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1), color_map))

                    if phase == 'training': 
                        # scribble
                        summary_vis.append(vis_utils.convert_label_to_color(scribbles.cpu().permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1), color_map))
                        # edge
                        summary_vis.append(vis_utils.convert_image(edges.cpu().permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1)))
                        # pseudo_label
                        summary_vis.append(vis_utils.convert_label_to_color(pseudo_labels.cpu().permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1), color_map))
                        # supervoxel
                        summary_vis.append(vis_utils.convert_label_to_color(su_mask.cpu().permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1), color_map))
                        summary_vis.append(vis_utils.convert_label_to_color(outputs[0].cpu().argmax(dim=1,keepdim=True).permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1), color_map))
                        # visualize attps
                        attps = outputs[1]
                        for item in attps:
                            item = F.interpolate(item, attps[-1].size()[2:], mode='trilinear', align_corners=True)
                            summary_vis.append(vis_utils.convert_image(item.cpu().permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1)))

                        init, edge_map, ref, init_1c, ref_1c = outputs[2:]
                        summary_vis.append(vis_utils.convert_label_to_color(init.cpu().argmax(dim=1,keepdim=True).permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1), color_map))
                        summary_vis.append(vis_utils.convert_image(torch.sigmoid(edge_map).cpu().permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1)))
                        summary_vis.append(vis_utils.convert_label_to_color(ref.cpu().argmax(dim=1,keepdim=True).permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1), color_map))
                    else:
                        # pred
                        summary_vis.append(vis_utils.convert_label_to_color(outputs.cpu().argmax(dim=1,keepdim=True).permute(0,4,1,2,3).contiguous().flatten(start_dim=0, end_dim=1), color_map))
                    vis_utils.write_image_to_tensorboard(summary_writer, summary_vis, summary_vis[-1].shape[-2:], (epoch-1) * 30 + index, name=phase)

                    if phase == "training":
                        loss.backward()
                        optimizer.step()

                
                
                # Iteration statistics
                epoch_samples += 1
                running_dice += dice_score
                if phase == "training":
                    running_loss += loss.item()
                    running_time += time_pesudo_labels

            # 指标
            if phase == 'validation':
                mean_dice = np.round(np.mean(list_dice),1) 
                std_dice = np.round(np.std(list_dice),1)
                mean_hd = np.round(np.mean(list_hd95),1) 
                std_hd = np.round(np.std(list_hd95),1) 
                mean_precision = np.round(np.mean(list_precision),1) 
                std_precision = np.round(np.std(list_precision),1) 
                # import pdb;pdb.set_trace()
                print('dice mean/avg {},{}, hd95 mean/avg {},{}, precision mean/avg {},{}'.format(mean_dice, std_dice, mean_hd, std_hd, mean_precision, std_precision))

            # Epoch statistcs
            epoch_dice = running_dice / epoch_samples
            logger.info("{}  Dice: {:.4f}".format(phase, epoch_dice))
            if phase == "training":
                epoch_loss = running_loss / epoch_samples
                epoch_time = running_time / epoch_samples
                logger.info("{}  Loss: {:.4f}".format(phase, epoch_loss))
                logger.info("{}  Time Pesudo Labels: {:.4f}".format(phase, epoch_time))
                
            # Saving best model on the validation set
            if phase == "validation":
                if best_val is None: # first iteration
                    best_val = epoch_dice
                    torch.save(model.state_dict(), opt.save_path.format("best"))

                if epoch_dice >= best_val:
                    best_val = epoch_dice
                    best_epoch = epoch
                    torch.save(model.state_dict(), opt.save_path.format("best"))

                df = df.append(
                    {"epoch":epoch,
                    "best_epoch":best_epoch,
                    "best_val":best_val,  
                    "lr":param_group["lr"],
                    "timeit":epoch_time}, 
                    ignore_index=True)
                df.to_csv(df_path, index=False)

                def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
                    """Learning rate policy used in nnUNet."""
                    return initial_lr * (1 - epoch / max_epochs)**exponent
                optimizer.param_groups[0]["lr"] = poly_lr(epoch, opt.epochs, opt.learning_rate, 0.9)
        
            if epoch == opt.epochs:
                torch.save(model.state_dict(), opt.save_path.format("final"))
                continue_training=False
    
    time_elapsed = time.time() - since
    logger.info("[INFO] Training completed in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info(f"[INFO] Best validation epoch is {best_epoch}")


if __name__ == "__main__":
    main()