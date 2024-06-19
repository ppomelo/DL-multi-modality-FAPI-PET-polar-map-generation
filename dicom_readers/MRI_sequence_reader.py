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
from dicom_sequence_reader import dicoms_reader, dicoms_multi_reader

def MRI_Multi_save(dicom_Cine_path):
    
    Cine_dirs = os.listdir(dicom_Cine_path)
    slice_num = len(Cine_dirs)
    print(dicom_Cine_path)
    print(slice_num)
    origin_z_collect = []

    holder = 0
    for i, Cine_dir in enumerate(Cine_dirs):

        if Cine_dir[0]=='.':
            holder = holder + 2
            continue

        Cine_dicom_path = os.path.join(dicom_Cine_path, Cine_dir)
        img_single_Cine = dicoms_reader(Cine_dicom_path)
        array_single_Cine = sitk.GetArrayFromImage(img_single_Cine).transpose(1,2,0)

        if holder%2 == 0:
            h,w,s = array_single_Cine.shape
            # print(h,w,s)
            array_4D_cine = np.zeros((slice_num, h, w, s))
            holder = holder + 1
        array_4D_cine[i,:,:,:] = array_single_Cine
        img_single_origin = img_single_Cine.GetOrigin()

        origin_z_collect.append([i, img_single_origin[2]])
        # print(img_single_origin)

    sorted_origin = np.array(sorted(origin_z_collect, key=lambda x: x[1]),dtype=int)
    z_order = sorted_origin[:,0]
    array_cine = array_4D_cine[z_order,:,:,:]


    first_slice_num = z_order[0]
    first_dicom_path = os.path.join(dicom_Cine_path, Cine_dirs[first_slice_num])
    dicom_path = os.path.join(first_dicom_path, os.listdir(first_dicom_path)[0])
    # first_cine_img = dicoms_reader(first_dicom_path)
    first_cine_img = sitk.ReadImage(dicom_path)
    direction=first_cine_img.GetDirection()
    spacing_temp=first_cine_img.GetSpacing()
    # print(first_cine_img.GetMetaDataKeys())
    spacing_z = first_cine_img.GetMetaData('0018|0088')
    print(spacing_z)
    spacing = (spacing_temp[0],spacing_temp[1],float(spacing_z))
    origin=first_cine_img.GetOrigin()
    print(direction,spacing,origin)
    new_image = sitk.GetImageFromArray(array_cine)
    new_image.SetSpacing(spacing)
    new_image.SetOrigin(origin)
    new_image.SetDirection(direction)

    return new_image

def MRI_Single_save(dicom_Cine_path):

    Cine_dir = os.listdir(dicom_Cine_path)
    Cine_dicom_path = os.path.join(dicom_Cine_path, Cine_dir[0])

    img_multi_Cine, single_dicom_slice = dicoms_multi_reader(Cine_dicom_path)

    slice_num = len(img_multi_Cine)
    print(slice_num)
    origin_z_collect = []

    for i, img_single_Cine in enumerate(img_multi_Cine):
        array_single_Cine = sitk.GetArrayFromImage(img_single_Cine).transpose(1,2,0)

        if i == 0:
            h,w,s = array_single_Cine.shape
            print(h,w,s)
            array_4D_cine = np.zeros((slice_num, h, w, s))

        array_4D_cine[i,:,:,:] = array_single_Cine
        img_single_origin = img_single_Cine.GetOrigin()
        # print(img_single_origin)
        origin_z_collect.append([i, img_single_origin[2]])

    sorted_origin = np.array(sorted(origin_z_collect, key=lambda x: x[1]),dtype=int)
    z_order = sorted_origin[:,0]
    array_cine = array_4D_cine[z_order,:,:,:]

    first_slice_num = z_order[0]
    first_cine_img = img_multi_Cine[first_slice_num]

    direction=first_cine_img.GetDirection()
    spacing_temp=first_cine_img.GetSpacing()
    # print(first_cine_img.GetMetaDataKeys())

    single_dicom = sitk.ReadImage(single_dicom_slice)
    spacing_z = single_dicom.GetMetaData('0018|0088')
    print(spacing_z)
    spacing = (spacing_temp[0],spacing_temp[1],float(spacing_z))
    origin=first_cine_img.GetOrigin()
    print(direction,spacing,origin)
    new_image = sitk.GetImageFromArray(array_cine)
    new_image.SetSpacing(spacing)
    new_image.SetOrigin(origin)
    new_image.SetDirection(direction)

    return new_image