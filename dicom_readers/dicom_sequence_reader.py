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
import pydicom

def SUV_with_decay(pet_file,PET_img):

    PET_array = sitk.GetArrayFromImage(PET_img)

    slope = pet_file[0x0028, 0x1053].value
    intercept = pet_file[0x0028, 0x1052].value
    activity_concentration = PET_array * slope + intercept

    #check corrected or not
    if slope == 1 and intercept == 0:
        print("already corrected")
        print(slope, intercept)
        activity_concentration = PET_array

    injected_dose = pet_file[0x0054,0x0016][0][0x0018,0x1074].value
    half_life = pet_file[0x0054,0x0016][0][0x0018, 0x1075].value
    
    # don't consider this situation: the series date and the start data are not the same day
    # check bellow
    series_date = pet_file[0x0008, 0x0021].value
    acquisition_date = pet_file[0x0008, 0x0022].value

    if series_date != acquisition_date:
        print(series_date, acquisition_date)
        print('Warning: the PET series date is not the same as the acquisition date, Pay Attention!!!')
    
    scan_time = pet_file[0x0008, 0x0031].value
    start_time = pet_file[0x0054,0x0016][0][0x0018, 0x1072].value

    if scan_time < start_time:
        print(scan_time, start_time)
        print('Warning: the PET series time is not the same as the start time, Pay Attention!!!')

    scan_time = pet_file[0x0008, 0x0031].value
    time_hour = float(scan_time[0:2])
    time_minute = float(scan_time[2:4])
    time_second = float(scan_time[4:6])
    
    start_time = pet_file[0x0054,0x0016][0][0x0018, 0x1072].value
    start_hour = float(start_time[0:2])
    start_minute = float(start_time[2:4])
    start_second = float(start_time[4:6])
    
    delta_time = (time_hour - start_hour) * 3600 + (time_minute - start_minute) * 60 + (time_second - start_second)
    decay_correction = pow(2, (- delta_time / half_life))


    weight = pet_file[0x0010, 0x1030].value * 1000
    
    pet_file_data = activity_concentration * weight / (injected_dose * decay_correction)

    print('patient weight : ', weight )
    print('patient injected dose :', injected_dose)

    # dose_type = pet_file[0x0054,0x0016][0][0x0018,0x0031].value
    # print('dose type:', dose_type)

    suv_image = sitk.GetImageFromArray(pet_file_data)

    return suv_image

def tansfer_SUV(dcm_dir, img):

    all_dcms = os.listdir(dcm_dir)
    dicom_file = pydicom.read_file(os.path.join(dcm_dir, all_dcms[0]))
    full_pet = SUV_with_decay(dicom_file, img)
    img_origin = img
    full_pet.SetDirection(img_origin.GetDirection())
    full_pet.SetOrigin(img_origin.GetOrigin())
    full_pet.SetSpacing(img_origin.GetSpacing())

    return full_pet

def dicoms_reader(dcm_dir):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(dcm_dir)
    dcm_series = reader.GetGDCMSeriesFileNames(dcm_dir, seriesIDs[0])
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    img = tansfer_SUV(dcm_dir,img)
    return img

def dicoms_multi_reader(dcm_dir):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(dcm_dir)
    multi_img = []
    for seriesID in seriesIDs:
        dcm_series = reader.GetGDCMSeriesFileNames(dcm_dir, seriesID)
        reader.SetFileNames(dcm_series)
        img = reader.Execute()
        multi_img.append(img)
    return multi_img, dcm_series[0]

def dicoms_reader_transfer(dcm_dir):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(dcm_dir)
    dcm_series = reader.GetGDCMSeriesFileNames(dcm_dir, seriesIDs[0])
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    return img

def apply_dicom_info_to_nii(root_path):

    patient_dirs = os.listdir(root_path)
    
    for patient_name in patient_dirs:
        if patient_name[0]=='.' or patient_name=='folders.cache':
            continue
        
        dicom_path = os.path.join(root_path,patient_name,'raw_dicom/PET_dicom')
        pet_file_path = os.path.join(root_path,patient_name,'data_nii')

        pet_file_full_path = os.path.join(pet_file_path,'PET_origin.nii')

        all_dcms = os.listdir(dicom_path)
        dicom_file = pydicom.read_file(os.path.join(dicom_path, all_dcms[0]))
        
        pet_img = sitk.ReadImage(pet_file_full_path)

        new_full_pet = SUV_with_decay(dicom_file, pet_img)
        new_path = os.path.join(pet_file_path,'PET_origin.nii')
        img_origin = sitk.ReadImage(pet_file_full_path)
        new_full_pet.SetDirection(img_origin.GetDirection())
        new_full_pet.SetOrigin(img_origin.GetOrigin())
        new_full_pet.SetSpacing(img_origin.GetSpacing())

        sitk.WriteImage(new_full_pet, new_path)
