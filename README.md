# Multi-modality deep learning-based 68Ga-DOTA-FAPI-04 PET polar map generation

## Overview

This repository provided a deep-learning-based method that fuses multi-modality images to compensate for the cardiac structural information lost in 68Ga-DOTA-FAPI-04 PET images and accurately generated Polar Maps.

There are four main modules in the repo. In the data preprocessing module (./preprocessing_modules), we conducted the PET/MRI image alignment and registration, then generated dataset for train/val/test purposes. In the cardiac reorientation module (./networks), we estimated the cardiac trans-axial to short-axis view transformation to localize and reorient the heart to the short-axis view. In the multi-modal segmentation module(./networks), the cardiac structure is segmented from short-axis view PET images with the reference of MRI image. In the 17-segment Polar Map generation module (./PM_modules), we project the left ventricular myocardium volume to a standard 2D surface to create the Polar Map according to the American Heart Association's standard 17-segment model.

**Below are detailed descriptions of each major step.**

## Install/Check dependencies

Ensure you have the necessary dependencies installed:

```
pip3 install -r requirements.txt
```

## Data Preprocessing

##### Convert DICOM to NIfTI Format

```bash
python dicom_readers/convert_dicom2nii_main.py
```

Convert DICOM files to NIfTI format and save in the `data_nii` folder as `PET_origin.nii` and `MRI_origin.nii`.

##### PET/MRI registration with pytorch and train/val/test dataset generation

```
python preprocessing_main.py
```

The input is the trans-axial view `PET_origin.nii` and `MRI_origin.nii` from the `data_nii` folder. 

The outputs are the well-organized dataset for train/val/test saved in `data_train_SS`, including short-axis view and trans-axial view PET and MRI images, and transformation parameters (referring to the Dataset Structure section). 

## Network Training

```
python train_main.py --root_path <data_path> --batch_size <batch_size> --epochs <epochs> --save_path <./models>
```

Train the whole network, including localization, spatial transformation (reorientation), and segmentation models. 

The well-trained models will be saved in the ./models folder.

## Prediction

```
python predict_main.py --test_path <data_path> --model_path <./models> --prediction_folder <./predictions>
```

Reorient transaxial cardiac PET images into short-axis images and segment cardiac structures with the well-trained models. The prediction results are saved in the ./predictions folder as follows, including predicted segmentation, short-axis view PET, and MRI images.

```
./
├── predictions/
│   ├── patient001_prediction/
│   │   ├── patient001_predict_PET.nii
│   │   ├── patient001_predict_MRI.nii
│   │   ├── patient001_predicted_labels.nii
│   ├── patient002_prediction/
│   │   ├── ...
```

## Post Processing

```
python postprocessing.py --prediction_folder <./predictions>
```

Post-process segmentation results, including 3D largest connected component filtering and 2D morphological closing operations. The processed label files will be saved in the ./predictions folder.

## Dataset Structure

The train/val/test dataset folder structure is as follows:

```
dataset/
├── patient001/
│   ├── data_nii/
│   │   ├── MRI_origin.nii
│   │   └── PET_origin.nii
│   ├── data_nii_SS/
│   │   ├── MRI_SAX_SS.nii
│   │   └── PET_SAX_SS.nii
│   ├── data_train_SS/
│   │   ├── MRI_SAX_train_SS.nii #short-axis view
│   │   ├── PET_SAX_train_SS.nii #short-axis view
│   │   ├── MRI_trans.nii #trans-axial view
│   │   ├── PET_trans.nii #trans-axial view
│   │   ├── SS_SAX_001.nii #segmentation
│   │   ├── Angles.npy #angle adjust according to your own datasets
│   │   ├── Tdash.npy #translation adjust according to your own datasets
│   │   ├── ...
│   └── raw_dicom/
│       ├── PET_dicom/
│       └── MRI_dicom/
├── patient002/
│   ├── ...
```
## Trained Weights

Our trained model weights have been uploaded to OneDrive:
https://1drv.ms/f/s!Av73VuhJIDgopG0n5FCXYvu466pC?e=NpJEI5

Segmentation model weights: network_seg.pth

Reorientation model weights: network_seg_angles.pth for rotation parameters prediction; network_seg_translation.pth for translation parameters prediction

## Citations
To be continued...
