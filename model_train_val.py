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

def train_model_reorientation(main_args,model, optimizer, scheduler, train_loader, val_loader, criterion, epochs, target_key):

    trigger = 0
    best_loss = 65535
    temp_loss = 65535

    for epoch in range(epochs):
        # Training
        model.train()
        running_train_loss = 0.0
        for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
            PET_inputs = sample['PET_in'].cuda()  # Adjust according to the actual keys in your dataset
            target_parameters = sample[target_key].cuda()  # Adjust according to the actual keys in your dataset
            optimizer.zero_grad()
            predict_parameters = model(PET_inputs)
            predict_PET, predict_MRI = spatial_transform(predict_parameters, sample, target_key)
            loss, l1 = criterion(predict_parameters, target_parameters, predict_PET, predict_MRI, sample, target_key)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        scheduler.step()
        trigger += 1
        mean_train_loss = running_train_loss/len(train_loader)
        if mean_train_loss < best_loss:
            if target_key == 'Tdash':
                torch.save(model.state_dict(), f"{main_args.save_path}/network_train_best_trans.pth")
            if target_key == 'angles':
                torch.save(model.state_dict(), f"{main_args.save_path}/network_train_best_angles.pth")
            best_loss = mean_train_loss
            print("=> saved best train model")
            trigger = 0

        # Validation
        running_val_loss = 0.0
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for i, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
                    PET_inputs = sample['PET_in'].cuda()  # Adjust according to the actual keys in your dataset
                    target_parameters = sample[target_key].cuda()  # Adjust according to the actual keys in your dataset
                    predict_parameters = model(PET_inputs)
                    predict_PET, predict_MRI = spatial_transform(predict_parameters, sample, target_key)
                    loss, l1 = criterion(predict_parameters, target_parameters, predict_PET, predict_MRI, sample, target_key)                    
                    running_val_loss += loss.item()

                print(f"val loss L1 of {target_key} in {epoch+1} is : [{l1}]")
                print(predict_parameters[0:4])
                print(target_parameters[0:4])

                mean_val_loss = running_val_loss/len(val_loader)
                if mean_val_loss < temp_loss:
                    if target_key == 'Tdash':
                        torch.save(model.state_dict(), f"{main_args.save_path}/network_val_best_trans.pth")
                    if target_key == 'angles':
                        torch.save(model.state_dict(), f"{main_args.save_path}/network_val_best_angles.pth")
                    temp_loss = mean_val_loss
                    print("=> saved best val model")
                    trigger = 0

        early_stop = True
        if early_stop:
            if trigger >= 60:
                print("=> early stopping")
                break

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {running_train_loss/len(train_loader)}, Val Loss: {running_val_loss/len(val_loader)}")

def train_model_reorientation_segmentation(main_args,full_network_trans, full_network_rotate, seg_model, optimizer_trans_joint,optimizer_rotate_joint, optimizer_seg_joint, scheduler, train_loader, val_loader, criterion, epochs, target_key='reorientation'):
    
    trigger = 0
    best_loss = 65535
    temp_loss = 65535

    full_network_trans.load_state_dict(torch.load(f"{main_args.save_path}/network_val_best_trans.pth"))
    full_network_rotate.load_state_dict(torch.load(f"{main_args.save_path}/network_val_best_angles.pth"))

    for epoch in range(epochs):
        full_network_trans.train()
        full_network_rotate.train()
        seg_model.train()

        running_train_loss = 0.0
        running_val_loss = 0.0

        for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = sample['PET_in'].cuda()  # Adjust according to the actual keys in your dataset
            targets_trans = sample['Tdash'].cuda()  # Adjust according to the actual keys in your dataset
            targets_rotate = sample['angles'].cuda()  # Adjust according to the actual keys in your dataset

            optimizer_trans_joint.zero_grad()
            optimizer_rotate_joint.zero_grad()
            optimizer_seg_joint.zero_grad()

            predict_parameters_trans = full_network_trans(inputs)
            predict_parameters_rotate = full_network_rotate(inputs)
            predict_parameters = torch.cat((predict_parameters_trans, predict_parameters_rotate), dim=1)
            target_parameters = torch.cat((targets_trans, targets_rotate), dim=1)

            predict_PET, predict_MRI, affine_matrix = spatial_transform(predict_parameters, sample, target_key)
            
            combined_input = torch.cat((predict_PET, predict_MRI), dim=1)

            segmentation_output = seg_model(combined_input)
            predicted_labels = torch.argmax(segmentation_output, dim=1)
            loss_rotation, l1 = criterion(predict_parameters, target_parameters, predict_PET, predict_MRI, sample, target_key)
            # print(segmentation_output.shape, sample['label'].shape)
            loss_seg, dice_seg = SegmentationLoss()(segmentation_output, sample['label'].cuda())
            loss = loss_rotation + loss_seg

            loss.backward()
            optimizer_trans_joint.step()
            optimizer_rotate_joint.step()
            optimizer_seg_joint.step()

            running_train_loss += loss.item()
            
        scheduler.step()

        trigger += 1
        mean_train_loss = running_train_loss/len(train_loader)
        if mean_train_loss < best_loss:
            torch.save(seg_model.state_dict(), f"{main_args.save_path}/network_train_best_seg.pth")
            torch.save(full_network_trans.state_dict(), f"{main_args.save_path}/network_train_best_seg_trans.pth")
            torch.save(full_network_rotate.state_dict(), f"{main_args.save_path}/network_train_best_seg_angles.pth")
            best_loss = mean_train_loss
            print("=> saved best train seg model")
            trigger = 0

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                for i, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
                    inputs = sample['PET_in'].cuda()   # Adjust according to the actual keys in your dataset
                    targets_trans = sample['Tdash'].cuda()  # Adjust according to the actual keys in your dataset
                    targets_rotate = sample['angles'].cuda()  # Adjust according to the actual keys in your dataset

                    predict_parameters_trans = full_network_trans(inputs)
                    predict_parameters_rotate = full_network_rotate(inputs)
                    predict_parameters = torch.cat((predict_parameters_trans, predict_parameters_rotate), dim=1)
                    target_parameters = torch.cat((targets_trans, targets_rotate), dim=1)

                    predict_PET, predict_MRI, affine_matrix = spatial_transform(predict_parameters, sample, target_key)
                    
                    combined_input = torch.cat((predict_PET, predict_MRI), dim=1)

                    segmentation_output = seg_model(combined_input)
                    predicted_labels = torch.argmax(segmentation_output, dim=1)

                    loss_rotation, l1 = criterion(predict_parameters, target_parameters, predict_PET, predict_MRI, sample, target_key)
                    loss_seg, dice_seg = SegmentationLoss().cuda()(segmentation_output, sample['label'].cuda())
                    loss_val = loss_rotation + loss_seg

                    running_val_loss += loss_val.item()

                print(f"val loss dice in {epoch+1} is : [{dice_seg}]")
                print(predict_parameters[0:4])
                print(target_parameters[0:4])


                mean_val_loss = running_val_loss/len(val_loader)
                if mean_val_loss < temp_loss:
                    torch.save(seg_model.state_dict(), f"{main_args.save_path}/network_val_best_seg.pth")
                    torch.save(full_network_trans.state_dict(), f"{main_args.save_path}/network_val_best_seg_trans.pth")
                    torch.save(full_network_rotate.state_dict(), f"{main_args.save_path}/network_val_best_seg_angles.pth")
                    temp_loss = mean_val_loss
                    print("=> saved best val seg model")
                    trigger = 0

        early_stop = True
        if early_stop:
            if trigger >= 60:
                print("=> early stopping")
                break            
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {running_train_loss/len(train_loader)}, Val Loss: {running_val_loss/len(val_loader)}")

    print("Training complete.")