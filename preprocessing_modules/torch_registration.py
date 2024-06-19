import os
import numpy as np
import SimpleITK as sitk
import skimage.io as io
import nibabel as nib
from nibabel.affines import to_matvec, from_matvec
import scipy
import nilearn
import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchTransferAlign():
    def __init__(self, img, reformat_img, affine):

        self.trans_img = nib.load(img)
        self.SAX_img = reformat_img
        self.affine_matrix = affine

    def torch_affine_transfer(self):

        arr_trans = self.trans_img.get_data()

        SAX_header = self.SAX_img.header
        arr_SAX = self.SAX_img.get_data()
        arr_SAX_affine = SAX_header.get_best_affine()

        W1,H1,D1 = arr_trans.shape
        W2,H2,D2 = arr_SAX.shape

        t=1
        M1 =np.array([[2/(W1-t),0,0,-1],
        [0,2/(H1-t),0,-1],
        [0,0,2/(D1-t),-1],
        [0,0,0,1]])
        M2 = self.affine_matrix
        M3_temp =np.array([[2/(W2-t),0,0,-1],
        [0,2/(H2-t),0,-1],
        [0,0,2/(D2-t),-1],
        [0,0,0,1]])
        M3 = np.linalg.inv(M3_temp)

        theta_matrix = M1.dot(M2).dot(M3)

        theta = torch.tensor(theta_matrix[:3,:].copy(), dtype=torch.float32).unsqueeze(axis=0)

        x_trans = torch.tensor(arr_trans.copy().transpose(2,1,0), dtype=torch.float32).unsqueeze(axis=0).unsqueeze(axis=0)
        x_SAX = torch.tensor(arr_SAX.transpose(2,1,0), dtype=torch.float32).unsqueeze(axis=0).unsqueeze(axis=0)


        grid = F.affine_grid(theta, x_SAX.size())
        x_predict_SAX = F.grid_sample(x_trans, grid, mode='bilinear')
        x_predict_SAX = x_predict_SAX.squeeze().numpy()

        predict_SAX = nib.Nifti1Image(x_predict_SAX.transpose(2,1,0), arr_SAX_affine)
       
        return predict_SAX, theta_matrix, M1, M3

    def torch_SAX_align(self):

        predict_SAX, theta_matrix, M1, M3 = self.torch_affine_transfer()
        return predict_SAX, theta_matrix, M1, M3