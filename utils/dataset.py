import os
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
import random
from utils.module import normalize_image
import matplotlib.pyplot as plt

def show_slice(image, slice_index=50):
    """
    Show a specific slice of a 3D image.
    
    Parameters:
    image (numpy.ndarray): The input 3D image.
    slice_index (int): The index of the slice to show.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image[:, :, slice_index], cmap='gray')
    plt.title(f'Slice {slice_index}')
    plt.axis('off')
    plt.show()
    # pass

class PairedDataset(Dataset):
    def __init__(self, patient_dirs, root_path, augment=False):
        self.root_path = root_path
        self.patient_dirs = patient_dirs
        self.num = len(self.patient_dirs)
        self.augment = augment

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        out = {}
        patient_name = self.patient_dirs[idx % self.num]
        patient_origin_path = os.path.join(self.root_path, patient_name)

        PET_in_path = os.path.join(patient_origin_path, 'data_train_SS', 'PET_trans.nii')
        MRI_in_path = os.path.join(patient_origin_path, 'data_train_SS', 'MRI_trans.nii')

        PET_out_path = os.path.join(patient_origin_path, 'data_train_SS', 'PET_SAX_train_SS.nii')
        MRI_out_path = os.path.join(patient_origin_path, 'data_train_SS', 'MRI_SAX_train_SS.nii')

        seg_path = os.path.join(patient_origin_path, 'data_train_SS', 'SS_SAX_'+ patient_name[-3:] +'.nii')

        PET_in_array = nib.load(PET_in_path).get_data() 
        MRI_in_array = nib.load(MRI_in_path).get_data()

        PET_out_array = nib.load(PET_out_path).get_data()
        MRI_out_array = nib.load(MRI_out_path).get_data()

        seg_array = nib.load(seg_path).get_data()

        out['name'] = patient_name
        # out['theta_matrix'] = np.array(np.load(os.path.join(patient_origin_path, 'data_train_SS', 'theta_matrix.npy')), dtype='float32')

        out['PET_matrix'] = np.array(np.load(os.path.join(patient_origin_path, 'data_train_SS', 'matrix_PET.npy')), dtype='float32')
        out['Rdash'] = np.array(np.load(os.path.join(patient_origin_path, 'data_train_SS', 'Rdash.npy')), dtype='float32')
        #just for monitoring, useless.
        out['Smat'] = np.array(np.load(os.path.join(patient_origin_path, 'data_train_SS', 'Smat.npy')), dtype='float32')
        out['Zddash'] = np.array(np.load(os.path.join(patient_origin_path, 'data_train_SS', 'Zdash.npy')), dtype='float32')
        out['Zdash'] = np.diag(out['Zddash'])
        out['angles'] = np.array(np.load(os.path.join(patient_origin_path, 'data_train_SS', 'Angles.npy')), dtype='float32')
        out['Tdash'] = np.array(np.load(os.path.join(patient_origin_path, 'data_train_SS', 'Tdash.npy')), dtype='float32')

        # # show_slice(PET_in_array,slice_index=50)
        # PET_in_array = normalize_image(PET_in_array)
        # # show_slice(PET_in_array,slice_index=60)

        # # show_slice(MRI_in_array,slice_index=50)
        # MRI_in_array = normalize_image(MRI_in_array)
        # # show_slice(MRI_in_array,slice_index=60)

        # # show_slice(PET_out_array,slice_index=11)
        # PET_out_array = normalize_image(PET_out_array)
        # # show_slice(PET_out_array,slice_index=11)

        # # show_slice(MRI_out_array,slice_index=10)
        # MRI_out_array = normalize_image(MRI_out_array)
        # # show_slice(MRI_out_array,slice_index=9)

        out['PET_in'] = np.array(np.expand_dims(PET_in_array, axis=0), dtype='float32')
        out['MRI_in'] = np.array(np.expand_dims(MRI_in_array, axis=0), dtype='float32')
        out['PET_out'] = np.array(np.expand_dims(PET_out_array, axis=0), dtype='float32')
        out['MRI_out'] = np.array(np.expand_dims(MRI_out_array, axis=0), dtype='float32')

        out['label'] = np.array(np.expand_dims(seg_array, axis=0), dtype='float32')
        # print(out['label'].shape, out['PET_in'].shape)

        SAX_header = nib.load(MRI_out_path).header
        out['SAX_affine'] = SAX_header.get_best_affine()

        Trans_header = nib.load(PET_in_path).header
        out['PET_trans_affine'] = Trans_header.get_best_affine()

        if self.augment:

            PET_in_array, MRI_in_array, angles, Tdash = self.random_transform(PET_in_array, MRI_in_array, out['angles'], out['Tdash'])

            out['angles'] = np.array(angles,dtype='float32')
            out['Tdash'] = np.array(Tdash,dtype='float32')
            # show_slice(PET_in_array,slice_index=60)
            # show_slice(MRI_in_array,slice_index=60)
            out['PET_in'] = np.array(np.expand_dims(PET_in_array, axis=0),dtype='float32')
            out['MRI_in'] = np.array(np.expand_dims(MRI_in_array, axis=0), dtype='float32')

        return out

    def random_transform(self, PET_image, MRI_image, angles, Tdash):
        # Random rotation
        angle = random.uniform(-0.5, 0.5)  # random rotation between -5 and 5 degrees
        rotated_PET = self.rotate(PET_image, angle)
        rotated_MRI = self.rotate(MRI_image, angle)
        
        # Random translation
        translation = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))  # random translation
        translated_PET = self.translate(rotated_PET, translation)
        translated_MRI = self.translate(rotated_MRI, translation)

        # Apply the same transformation to angles and Tdash
        angles = angles - angle
        Tdash = Tdash - np.array(translation)

        return translated_PET, translated_MRI, angles, Tdash

    def rotate(self, image, angle):
        # Apply rotation to the image
        from scipy.ndimage import rotate
        return rotate(image, angle, reshape=False)

    def translate(self, image, translation):
        # Apply translation to the image
        from scipy.ndimage import shift
        return shift(image, translation)
