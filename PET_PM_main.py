import os
import csv
import argparse

import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage.measurements import center_of_mass

from matplotlib import colors

from PM_modules.PolarMap_PET import PolarMap_PET
from PM_modules.PlotBullEye import plot_bullseye
from PM_modules.DefineColor import cdict

def parse_args():
    parser = argparse.ArgumentParser(description='Generate PET Polar Maps.')
    parser.add_argument('--prediction_folder', type=str, default='./predictions', help='Directory containing patient data.')
    parser.add_argument('--save_PolarMap_path', type=str, default='./PET_PolarMaps', help='Directory to save the results.')
    return parser.parse_args()

def arr_load_crop(data_path, mask_path, start_temp=2, end_temp=2):

    data_img = nib.load(data_path)
    mask_img = nib.load(mask_path)

    data_arr=data_img.get_data()
    mask_arr=mask_img.get_data()

    h, w = data_arr.shape[1], data_arr.shape[0] 
    crop_start = h // 2 - w // 2
    crop_end = h // 2 + w // 2

    data_arr = data_arr[:, crop_start:crop_end, :].transpose((1,0,2))
    mask_arr = mask_arr[:, crop_start:crop_end, :].transpose((1,0,2))

    slices_nonzero = np.nonzero(mask_arr)[2]
    start_slice = slices_nonzero.min() + start_temp 
    end_slice = slices_nonzero.max() - end_temp

    return data_arr, mask_arr, start_slice, end_slice

def polarmap_generation(data_arr, mask_arr, start_slice, end_slice, save_results_path, patient_num):

    polarmap_pet = PolarMap_PET(data_arr, mask_arr, start_slice, end_slice)
    results = polarmap_pet.project_to_aha_polar_map()
    E, mu_r = polarmap_pet.construct_polar_map(results['V_proj'])

    new_cmap = colors.LinearSegmentedColormap('new_cmap',segmentdata=cdict)

    fig_name = patient_num +'_polar_PET'  + '.png'
    fig_path = os.path.join(save_results_path,fig_name)
    plot_bullseye(E, mu_r, savepath = fig_path, cmap = new_cmap)

    print(f"{patient_num}_PolarMap saved.")

if __name__ == "__main__":

    args = parse_args()
    root_path = args.prediction_folder
    save_results_path = args.save_PolarMap_path

    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    patient_dirs = os.listdir(root_path)

    for patient_name in patient_dirs:

        if patient_name[0]=='.' or patient_name=='folders.cache':
            continue

        patient_origin_path = os.path.join(root_path, patient_name)
        patient_num = patient_name[:10]

        PET_path = os.path.join(patient_origin_path, patient_num+'_predict_PET.nii')
        mask_path = os.path.join(patient_origin_path,  patient_num+'_predicted_labels.nii')

        PET_arr, mask_arr, start_slice, end_slice = arr_load_crop(PET_path, mask_path)
        
        polarmap_generation(PET_arr, mask_arr, start_slice, end_slice, save_results_path, patient_num)



