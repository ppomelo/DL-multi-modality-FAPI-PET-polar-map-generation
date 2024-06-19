import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import DataLoaderGenerator, DataLoaderGeneratorTest
from networks.localization_net import NetworkAngles, NetworkTranslation
from networks.spatial_transform import spatial_transform
from networks.segmentation_net import UNet3D
import nibabel as nib
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Predict using trained models")
    parser.add_argument('--test_path', type=str, default="path/to/test/dataset", help='Root path for the data')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for prediction')
    parser.add_argument('--model_path', type=str, default="./models/", help='Path to the saved model weights')
    parser.add_argument('--prediction_folder', type=str, default="./predictions/", help='Path to save the predictions')
    args = parser.parse_args()
    return args

def load_models(model_path):

    network_trans = NetworkTranslation().cuda()
    network_trans.load_state_dict(torch.load(os.path.join(model_path, 'network_val_best_seg_trans.pth')))
    
    network_angles = NetworkAngles().cuda()
    network_angles.load_state_dict(torch.load(os.path.join(model_path, 'network_val_best_seg_angles.pth')))
    
    segmentation_network = UNet3D(in_channels=2, out_channels=4).cuda()
    segmentation_network.load_state_dict(torch.load(os.path.join(model_path, 'network_val_best_seg.pth')))
    
    return network_trans, network_angles, segmentation_network

def predict(models, dataloader, output_path):

    os.makedirs(output_path, exist_ok=True)
    
    network_trans, network_angles, segmentation_network = models
    network_trans.eval()
    network_angles.eval()
    segmentation_network.eval()
    
    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = sample['PET_in'].cuda()
            # print(sample['PET_out'].shape)
            predict_parameters_trans = network_trans(inputs)
            predict_parameters_angles = network_angles(inputs)
            predict_parameters = torch.cat((predict_parameters_trans, predict_parameters_angles), dim=1)

            predict_PET, predict_MRI, affine_matrix = spatial_transform(predict_parameters, sample, 'reorientation')

            combined_input = torch.cat((predict_PET, predict_MRI), dim=1)
            segmentation_output = segmentation_network(combined_input)
            predicted_labels = torch.argmax(segmentation_output, dim=1).cpu().numpy()
            # print(predicted_labels.shape)
            # Save predictions
            for j in range(predicted_labels.shape[0]):
                patient_id = sample['name'][j]
                patient_folder = os.path.join(output_path, f"{patient_id}_prediction")
                os.makedirs(patient_folder, exist_ok=True)

                affine = affine_matrix[j].squeeze().cpu().numpy()

                output_file = os.path.join(patient_folder, f"{patient_id}_predicted_labels.nii")
                nib.save(nib.Nifti1Image(predicted_labels[j], affine, dtype=np.float32), output_file)
                print(f"Saved prediction for {patient_id} at {output_file}")

                # Save predict_PET                                
                output_file_pet = os.path.join(patient_folder, f"{patient_id}_predict_PET.nii")
                nib.save(nib.Nifti1Image(predict_PET[j].squeeze().cpu().numpy(), affine, dtype=np.float32), output_file_pet)
                print(predict_PET[j].cpu().numpy().shape)
                print(f"Saved predict_PET for {patient_id} at {output_file_pet}")
                
                # Save predict_MRI
                output_file_mri = os.path.join(patient_folder, f"{patient_id}_predict_MRI.nii")
                nib.save(nib.Nifti1Image(predict_MRI[j].squeeze().cpu().numpy(), affine, dtype=np.float32), output_file_mri)
                print(f"Saved predict_MRI for {patient_id} at {output_file_mri}")

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.prediction_folder):
        os.makedirs(args.prediction_folder)
    
    data_loader_generator = DataLoaderGeneratorTest(args)
    test_loader = data_loader_generator.generate()
    
    # Load models
    models = load_models(args.model_path)
    
    # Predict
    predict(models, test_loader, args.prediction_folder)
