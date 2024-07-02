#convert2nii.py
import os
import numpy as np
import SimpleITK as sitk
import skimage.io as io
from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize
from collections import OrderedDict
import nibabel as nib
from nibabel.processing import resample_from_to,resample_to_output,adapt_affine
from nibabel.spaces import vox2out_vox
from nibabel.affines import to_matvec, from_matvec
import scipy
import nilearn
import joblib
from nilearn.image import new_img_like,resample_to_img
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dicom_sequence_reader import dicoms_reader, dicoms_multi_reader, dicoms_reader_transfer, apply_dicom_info_to_nii
from MRI_sequence_reader import MRI_Multi_save, MRI_Single_save
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="The way to save PET_suv")
    parser.add_argument('--method', type=str, default="one_step", help='one_step/two_step')
    args = parser.parse_args()
    return args

def patient_save_PET(root_path):

    patient_dirs = os.listdir(root_path)
    
    for patient_name in patient_dirs:
        if patient_name[0]=='.' or patient_name=='folders.cache':
            continue
        patient_origin_path = os.path.join(root_path, patient_name)
        dicom_PET_path = os.path.join(patient_origin_path, 'raw_dicom','PET_dicom')
        PET_dir = os.listdir(dicom_PET_path)

        if len(PET_dir)!=1:
            print(patient_name)
            # continue

        PET_dicom_path = os.path.join(dicom_PET_path, PET_dir[0])

        PET_dicom_path = dicom_PET_path

        img_PET = dicoms_reader(PET_dicom_path)

        PET_nii_save_path = os.path.join(patient_origin_path, 'data_nii', 'PET_origin.nii')

        sitk.WriteImage(img_PET, PET_nii_save_path)

def patient_save_MRI(root_path):

    patient_dirs = os.listdir(root_path)
    
    for patient_name in patient_dirs:
        if patient_name[0]=='.' or patient_name=='folders.cache':
            continue
        patient_origin_path = os.path.join(root_path, patient_name)
        dicom_MRI_path = os.path.join(patient_origin_path, 'raw_dicom','MRI_dicom')
        MRI_dir = os.listdir(dicom_MRI_path)
        save_MRI_path = os.path.join(patient_origin_path, 'data_nii','MRI_origin.nii')

        if len(MRI_dir)!=1:
            # print("Multi", patient_name)
            MRI_4D_img = MRI_Multi_save(dicom_MRI_path)
            sitk.WriteImage(MRI_4D_img, save_MRI_path)
            t = t+1
        if len(MRI_dir)==1:
            # print("Single", patient_name)
            MRI_4D_img = MRI_Single_save(dicom_MRI_path)

            sitk.WriteImage(MRI_4D_img, save_MRI_path)

def patient_save_PET_transfer(root_path):

    patient_dirs = os.listdir(root_path)
    
    for patient_name in patient_dirs:
        if patient_name[0]=='.' or patient_name=='folders.cache':
            continue
        patient_origin_path = os.path.join(root_path, patient_name)
        dicom_PET_path = os.path.join(patient_origin_path, 'raw_dicom','PET_dicom')
        PET_dir = os.listdir(dicom_PET_path)

        if len(PET_dir)!=1:
            print(patient_name)
            # continue

        PET_dicom_path = os.path.join(dicom_PET_path, PET_dir[0])
        PET_dicom_path = dicom_PET_path
        img_PET = dicoms_reader_transfer(PET_dicom_path)
        PET_nii_save_path = os.path.join(patient_origin_path, 'data_nii', 'PET_origin.nii')

        sitk.WriteImage(img_PET, PET_nii_save_path)

if __name__ == "__main__":

    root_path = r"path/to/dataset"

    main_args = parse_args()

    patient_save_MRI(root_path)

    if main_args.method =='one_step':
        #read and save dicoms to PET_suv at the same time
        patient_save_PET(root_path)

    if main_args.method =='two_step':
        #save dicoms to nii first, then apply metadata to nii to generate PET_suv
        patient_save_PET_transfer(root_path)
        apply_dicom_info_to_nii(root_path)

