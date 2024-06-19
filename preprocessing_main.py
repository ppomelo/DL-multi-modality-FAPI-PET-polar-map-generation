import os
import numpy as np
import SimpleITK as sitk
import skimage.io as io
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from preprocessing_modules.registration_spacing import AlignSpacingTransform
from preprocessing_modules.torch_registration import TorchTransferAlign
from preprocessing_modules.torch_matrix_generation import MatrixTransform

def preprocessing(PET_nii_path, MRI_nii_path):

    Align_SAX_SS = AlignSpacingTransform(PET_nii_path, MRI_nii_path)
    PET_reformat_SAX_img, MRI_reformat_SAX_img, SAX_SS_affine, PETtoMRI_affine, MRItoSS_affine = Align_SAX_SS.PET_MRI_align()

    if not os.path.exists(os.path.join(patient_origin_path, 'data_nii_SS')):
        os.makedirs(os.path.join(patient_origin_path, 'data_nii_SS'))
        
    if not os.path.exists(os.path.join(patient_origin_path, 'data_train_SS')):
        os.makedirs(os.path.join(patient_origin_path, 'data_train_SS'))

    nib.save(PET_reformat_SAX_img, os.path.join(patient_origin_path, 'data_nii_SS', 'PET_SAX_SS.nii'))  
    nib.save(MRI_reformat_SAX_img, os.path.join(patient_origin_path, 'data_nii_SS', 'MRI_SAX_SS.nii'))

    '''preparing training dataset'''
    PET_trans, MRI_trans = Align_SAX_SS.MRI_SAXtoTAX_reformat(MRI_reformat_SAX_img)

    nib.save(PET_trans, os.path.join(patient_origin_path, 'data_train_SS', 'PET_trans.nii'))  
    nib.save(MRI_trans, os.path.join(patient_origin_path, 'data_train_SS', 'MRI_trans.nii'))

    MRI_trans_path = os.path.join(patient_origin_path, 'data_train_SS', 'MRI_trans.nii')

    TorchTrans_PET = TorchTransferAlign(PET_nii_path, PET_reformat_SAX_img, PETtoMRI_affine)
    reformat_PET_torch, theta_matrix, M1, M3 = TorchTrans_PET.torch_SAX_align()
    
    TorchTrans_MRI = TorchTransferAlign(MRI_trans_path, MRI_reformat_SAX_img, PETtoMRI_affine)
    reformat_MRI_torch, theta_matrix_MRI, M1_MRI, M3_MRI = TorchTrans_MRI.torch_SAX_align()

    if not os.path.exists(os.path.join(patient_origin_path, 'data_train_SS')):
        os.makedirs(os.path.join(patient_origin_path, 'data_train_SS'))
    nib.save(reformat_PET_torch, os.path.join(patient_origin_path, 'data_train_SS', 'PET_SAX_train_SS.nii'))
    nib.save(reformat_MRI_torch, os.path.join(patient_origin_path, 'data_train_SS', 'MRI_SAX_train_SS.nii'))

    Matrix_transform = MatrixTransform(M1, theta_matrix, M3, theta_matrix_MRI, patient_origin_path)
    # print(theta_matrix)
    angles, Tdash = Matrix_transform.decompose_save()
    # print(angles, Tdash)

if __name__ == "__main__":

    patient_root_path = r"path/to/dataset"
    patient_dirs = os.listdir(patient_root_path)

    for patient_name in patient_dirs:
        if patient_name[0]=='.' or patient_name=='folders.cache':
            continue
        print(patient_name)

        patient_origin_path = os.path.join(patient_root_path, patient_name)
        PET_nii_path = os.path.join(patient_origin_path, 'data_nii', 'PET_origin.nii')
        MRI_nii_path = os.path.join(patient_origin_path, 'data_nii', 'MRI_origin.nii')

        preprocessing(PET_nii_path, MRI_nii_path)


