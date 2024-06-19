import os
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import DataLoaderGenerator
from utils.losses import CustomLoss,SegmentationLoss

from networks.localization_net import NetworkAngles, NetworkTranslation
from networks.spatial_transform import spatial_transform
from networks.segmentation_net import UNet3D
from model_train_val import train_model_reorientation, train_model_reorientation_segmentation

os.environ["CUDA_VISIBLE_DEVICES"]= '1'
print('GPU Device Number is %d '%(torch.cuda.current_device()))

def parse_args():
    parser = argparse.ArgumentParser(description="Train NetworkAngles, NetworkTranslation and Segmentation")
    parser.add_argument('--root_path', type=str, default="path/to/train_val/dataset", help='Root path for the data')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('--aug', type=bool, default=False, help='Use augmentation (True/False)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train each network')
    parser.add_argument('--save_path', type=str, default="./models/", help='Path to save the model weights')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    main_args = parse_args()

    if not os.path.exists(main_args.save_path):
        os.makedirs(main_args.save_path)

    # Create dataloaders
    data_loader_generator = DataLoaderGenerator(main_args)
    train_loader, val_loader = data_loader_generator.generate()

    # Define the loss function and the optimizer
    criterion = CustomLoss().cuda()

    # Initialize networks
    full_network_trans = NetworkTranslation().cuda()
    full_network_rotate = NetworkAngles().cuda()
    segmentation_network = UNet3D(in_channels=2, out_channels=4).cuda()

    #optimizers
    optimizer_trans = optim.Adam(full_network_trans.parameters(), lr=1e-5) 
    scheduler_trans = optim.lr_scheduler.StepLR(optimizer_trans, step_size=30, gamma=0.6)

    optimizer_rotate = optim.Adam(full_network_rotate.parameters(), lr=1e-4) 
    scheduler_rotate = optim.lr_scheduler.StepLR(optimizer_rotate, step_size=30, gamma=0.6)


    optimizer_trans_joint = optim.Adam(full_network_trans.parameters(), lr=1e-7)
    optimizer_rotate_joint = optim.Adam(full_network_rotate.parameters(), lr=1e-6)
    optimizer_seg_joint = optim.Adam(segmentation_network.parameters(), lr=1e-3)
    scheduler_seg = optim.lr_scheduler.StepLR(optimizer_seg_joint, step_size=30, gamma=0.6)

    print("Training FullNetworkTrans...")
    train_model_reorientation(main_args,full_network_trans, optimizer_trans, scheduler_trans, train_loader, val_loader, criterion, epochs=main_args.epochs, target_key='Tdash')
    torch.save(full_network_trans.state_dict(), f"{main_args.save_path}/network_translation.pth")

    print("Training FullNetworkRotate...")
    train_model_reorientation(main_args,full_network_rotate, optimizer_rotate, scheduler_rotate, train_loader, val_loader, criterion, epochs=main_args.epochs, target_key='angles')
    torch.save(full_network_rotate.state_dict(), f"{main_args.save_path}/network_angles.pth")

    print("Training Fullnetwork...")
    train_model_reorientation_segmentation(main_args,full_network_trans, full_network_rotate, segmentation_network, optimizer_trans_joint,optimizer_rotate_joint,optimizer_seg_joint, scheduler_seg, train_loader, val_loader, criterion, epochs=main_args.epochs, target_key='reorientation')
    torch.save(full_network_trans.state_dict(), f"{main_args.save_path}/network_seg_translation.pth")
    torch.save(full_network_rotate.state_dict(), f"{main_args.save_path}/network_seg_angles.pth")
    torch.save(segmentation_network.state_dict(), f"{main_args.save_path}/network_seg.pth")
