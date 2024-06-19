import os
import numpy as np
import SimpleITK as sitk
import skimage.io as io
import nibabel as nib
from nibabel.affines import to_matvec, from_matvec
import scipy
import nilearn
from transforms3d.affines import decompose44, compose
from transforms3d.axangles import aff2axangle, mat2axangle
from scipy.spatial.transform import Rotation 
from preprocessing_modules.rotation_transfer_matrix import rotation_matrix, rotation_angles 

class MatrixTransform():

    def __init__(self, M1, theta_matrix, M3, theta_matrix_MRI, patient_origin_path):

        self.M1 = M1
        self.theta_matrix = theta_matrix #for grid genetation
        self.M3 = M3
        self.matrix_PET = np.linalg.inv(M1).dot(theta_matrix).dot(np.linalg.inv(M3)) #SAX_affine

        self.patient_origin_path = patient_origin_path
        self.theta_matrix_MRI = theta_matrix_MRI

    def decompose_save(self):

        Tdash, Rdash, Zdash, Sdash = decompose44(self.matrix_PET)
        # print(self.matrix_PET)
        Smat = np.array([[1, Sdash[0], Sdash[1]],
                        [0,    1, Sdash[2]],
                        [0,    0,    1]])

        RZS = np.dot(Rdash, np.dot(np.diag(Zdash), Smat))
        A44 = np.eye(4)
        A44[:3,:3] = RZS
        A44[:-1,-1] = Tdash

        angles = rotation_angles(Rdash, 'zyx')
        matrix = rotation_matrix(angles[0],angles[1],angles[2],'zyx')
        # print(Zdash)
        # print(angles)
        # print(Rdash)
        # print(Tdash)
        # print(Smat)
        # print(self.theta_matrix)
        np.save(os.path.join(self.patient_origin_path, 'data_train_SS', 'Smat.npy'),Smat)
        np.save(os.path.join(self.patient_origin_path, 'data_train_SS', 'Zdash.npy'),Zdash)
        np.save(os.path.join(self.patient_origin_path, 'data_train_SS', 'Angles.npy'),angles)
        np.save(os.path.join(self.patient_origin_path, 'data_train_SS', 'Rdash.npy'),Rdash)
        np.save(os.path.join(self.patient_origin_path, 'data_train_SS', 'Tdash.npy'),Tdash)
        np.save(os.path.join(self.patient_origin_path, 'data_train_SS', 'matrix_PET.npy'),self.matrix_PET)
        np.save(os.path.join(self.patient_origin_path, 'data_train_SS', 'M1.npy'),self.M1)
        np.save(os.path.join(self.patient_origin_path, 'data_train_SS', 'M3.npy'),self.M3)
        np.save(os.path.join(self.patient_origin_path, 'data_train_SS', 'theta_matrix.npy'),self.theta_matrix)

        return angles, Tdash