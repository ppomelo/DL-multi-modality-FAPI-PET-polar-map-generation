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

##save PET

def dicoms_reader(dcm_dir):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(dcm_dir)
    dcm_series = reader.GetGDCMSeriesFileNames(dcm_dir, seriesIDs[0])
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
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