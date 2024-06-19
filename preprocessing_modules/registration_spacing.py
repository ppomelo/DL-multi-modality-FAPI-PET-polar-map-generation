import os
import numpy as np
import SimpleITK as sitk
import skimage.io as io
import nibabel as nib
from nibabel.processing import resample_from_to,resample_to_output,adapt_affine
from nibabel.spaces import vox2out_vox
from nibabel.affines import to_matvec, from_matvec
import scipy
from transforms3d.affines import decompose44, compose


class AlignSpacingTransform():
    def __init__(self, PET_path, MRI_path):
    # signature unrelated magic nums
        # variables
        self.PET_path = PET_path
        self.MRI_path = MRI_path

    def data_reader(self):
        PET_image = nib.load(self.PET_path)
        PET_header = PET_image.header
        arr_PET = PET_image.get_data()

        MRI_image = nib.load(self.MRI_path)
        MRI_header = MRI_image.header
        arr_MRI = MRI_image.get_data()

        PET_affine = PET_header.get_best_affine()
        MRI_affine = MRI_header.get_best_affine()

        return arr_PET, arr_MRI, PET_affine, MRI_affine

    def affine_transform_SAX_SS(self):
        
        arr_PET, arr_MRI, PET_affine, MRI_affine = self.data_reader()

        "transfer for PET"
        to_affine = MRI_affine
        Tdash, Rdash, Zdash, Sdash = decompose44(to_affine)
        Zdash = [Zdash[0],Zdash[1], 6]
        to_affine = compose(Tdash, Rdash, Zdash, Sdash)

        from_affine = PET_affine
        a_from_affine = adapt_affine(from_affine, len(arr_PET.shape))
        a_to_affine = adapt_affine(to_affine, len(arr_MRI.shape))

        matrix = np.linalg.inv(a_from_affine) .dot(a_to_affine)
        rzs, trans = to_matvec(matrix)
        Tdash, Rdash, Zdash, Sdash = decompose44(matrix)
        out_shape = [arr_MRI.shape[0],arr_MRI.shape[1],20]

        PET_reformat_SAX = scipy.ndimage.affine_transform(arr_PET,
                                    rzs,
                                    trans,
                                    output_shape=out_shape,
                                    order=3,
                                    mode='constant',
                                    cval=0)
        SAX_SS_affine = from_affine.dot(from_matvec(rzs,trans))
        PET_reformat_SAX_img = nib.Nifti1Image(PET_reformat_SAX, SAX_SS_affine)

        "affine from PET to MRI - SAX"
        PETtoMRI_affine = from_matvec(rzs,trans)

        "transfer for MRI - interpolation"
        from_affine_MRI = MRI_affine
        to_affine_MRI = a_to_affine
        matrix_MRI = np.linalg.inv(from_affine_MRI) .dot(to_affine_MRI)
        rzs_MRI, trans_MRI = to_matvec(matrix_MRI)
        data_MRI = scipy.ndimage.affine_transform(arr_MRI,
                                    rzs_MRI,
                                    trans_MRI,
                                    output_shape=out_shape,
                                    order=3,
                                    mode='constant',
                                    cval=0)
        MRI_reformat_SAX_img = nib.Nifti1Image(data_MRI, SAX_SS_affine)
        
        "affine from MRI to MRI - Small Spacing"
        MRItoSS_affine = from_matvec(rzs_MRI,trans_MRI)

        return PET_reformat_SAX_img, MRI_reformat_SAX_img, SAX_SS_affine, PETtoMRI_affine, MRItoSS_affine 

    def MRI_SAXtoTAX_reformat(self, MRI_reformat_SAX_img):

        PET_image = nib.load(self.PET_path)
        PET_trans = PET_image
        PET_header = PET_image.header
        arr_PET = PET_image.get_data()

        MRI_header = MRI_reformat_SAX_img.header
        arr_MRI = MRI_reformat_SAX_img.get_data()

        PET_affine = PET_header.get_best_affine()
        MRI_affine = MRI_header.get_best_affine()

        to_affine = PET_affine
        from_affine = MRI_affine

        a_to_affine = adapt_affine(to_affine, len(arr_PET.shape))
        a_from_affine = adapt_affine(from_affine, len(arr_MRI.shape))
        matrix = np.linalg.inv(a_from_affine) .dot(a_to_affine)
        rzs, trans = to_matvec(matrix)

        out_shape = arr_PET.shape
        data = scipy.ndimage.affine_transform(arr_MRI,
                                    rzs,
                                    trans,
                                    output_shape=out_shape,
                                    order=3,
                                    mode='constant',
                                    cval=0)

        data_affine = from_affine.dot(from_matvec(rzs,trans))
        data_affine = PET_affine
        MRI_trans = nib.Nifti1Image(data, data_affine)

        return PET_trans, MRI_trans

    def PET_MRI_align(self):

        PET_reformat_SAX_img, MRI_reformat_SAX_img, SAX_SS_affine, PETtoMRI_affine, MRItoSS_affine = self.affine_transform_SAX_SS()
        
        return PET_reformat_SAX_img, MRI_reformat_SAX_img, SAX_SS_affine, PETtoMRI_affine, MRItoSS_affine