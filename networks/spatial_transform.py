import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('.')
from utils.module import rotation_matrix

def matrix_generator(parameters, sample, target_key):

    if target_key == 'angles':
        angles = parameters.view(-1, 1, 3)[:,0,:]
        Smat = sample['Smat'].cuda()
        Rdash = rotation_matrix(angles[:,[0]],angles[:,[1]],angles[:,[2]],'zyx').view(-1,3,3)

        ZS = torch.bmm(sample['Zdash'].cuda() , Smat)
        RZS = torch.bmm(Rdash, ZS) #10 3 3

        A44 = torch.eye(4).cuda() 
        A44 = A44.reshape((1, 4, 4))
        A44 = A44.repeat(sample['PET_in'].shape[0], 1, 1)

        A44[:,:3,:3] = RZS
        A44[:,:-1,-1] = sample['Tdash'].cuda() #theta[:,0,:] #x['Tdash']  #

        matrix_PET = A44

    if target_key == 'Tdash':
        trans = parameters.view(-1, 1, 3)[:,0,:]
        Smat = sample['Smat'].cuda()
        Rdash = rotation_matrix(sample['angles'][:,[0]],sample['angles'][:,[1]],sample['angles'][:,[2]],'zyx').view(-1,3,3)
        ZS = torch.bmm(sample['Zdash'].cuda(), Smat)
        RZS = torch.bmm(Rdash.cuda(), ZS) #10 3 3

        A44 = torch.eye(4).cuda() 
        A44 = A44.reshape((1, 4, 4))
        A44 = A44.repeat(sample['PET_in'].shape[0], 1, 1)

        A44[:,:3,:3] = RZS
        A44[:,:-1,-1] = trans

        matrix_PET = A44
    
    if target_key == 'reorientation':

        trans = parameters[:,:3].view(-1, 1, 3)[:,0,:]
        angles = parameters[:,3:].view(-1, 1, 3)[:,0,:]

        Rdash = rotation_matrix(angles[:,[0]],angles[:,[1]],angles[:,[2]],'zyx').view(-1,3,3)

        Smat = sample['Smat'].cuda()
        ZS = torch.bmm(sample['Zdash'].cuda(), Smat)
        RZS = torch.bmm(Rdash, ZS) #10 3 3

        A44 = torch.eye(4).cuda() 
        A44 = A44.reshape((1, 4, 4))
        A44 = A44.repeat(sample['PET_in'].shape[0], 1, 1)

        A44[:,:3,:3] = RZS
        A44[:,:-1,-1] = trans

        matrix_PET = A44

    return matrix_PET

def spatial_transform(parameters, sample, target_key):

    affine_matrix = matrix_generator(parameters, sample, target_key)
    # affine_matrix = sample['PET_matrix']
    # print(sample['PET_in'].shape)
    PET_in = sample['PET_in'].permute(0, 1, 4, 3, 2).cuda()
    MRI_in = sample['MRI_in'].permute(0, 1, 4, 3, 2).cuda()
    PET_out = sample['PET_out'].permute(0, 1, 4, 3, 2).cuda()
    MRI_out = sample['MRI_out'].permute(0, 1, 4, 3, 2).cuda()

    arr_trans = PET_in
    arr_SAX = PET_out

    _,_, W1,H1,D1 = sample['PET_in'].shape
    _,_, W2,H2,D2 = sample['PET_out'].shape

    t=1
    M1_np =np.array([[2/(W1-t),0,0,-1],
      [0,2/(H1-t),0,-1],
      [0,0,2/(D1-t),-1],
      [0,0,0,1]])
    M2 = affine_matrix
    M3_temp =np.array([[2/(W2-t),0,0,-1],
      [0,2/(H2-t),0,-1],
      [0,0,2/(D2-t),-1],
      [0,0,0,1]])
    M3_np = np.linalg.inv(M3_temp)

    M1 = torch.tensor(M1_np, dtype=torch.float32).cuda()
    M3 = torch.tensor(M3_np, dtype=torch.float32).cuda()

    theta_matrix = torch.matmul(torch.matmul(M1, affine_matrix), M3)
    # print(theta_matrix)
    # print(sample['theta_matrix'])
    theta = torch.tensor(theta_matrix[:,:3,:], dtype=torch.float32)
    grid = F.affine_grid(theta, arr_SAX.size())

    predict_PET_SAX = F.grid_sample(arr_trans, grid, mode='bilinear')
    predict_MRI_SAX = F.grid_sample(MRI_in, grid, mode='bilinear')

    if target_key == 'reorientation':
        PETtoMRI_matrix = torch.matmul(torch.tensor(sample['PET_trans_affine'], dtype=torch.float32).cuda(), affine_matrix)
        return predict_PET_SAX.permute(0, 1, 4, 3, 2), predict_MRI_SAX.permute(0, 1, 4, 3, 2), PETtoMRI_matrix

    return predict_PET_SAX.permute(0, 1, 4, 3, 2), predict_MRI_SAX.permute(0, 1, 4, 3, 2)
    # x_predict_SAX = x_predict_SAX.squeeze()


