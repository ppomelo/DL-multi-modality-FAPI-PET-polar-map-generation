import os
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import label, binary_closing
from skimage.morphology import ball, square

def parse_args():
    parser = argparse.ArgumentParser(description="Post-process predicted labels")
    parser.add_argument('--prediction_folder', type=str,default="./predictions/",help='Directory containing the predicted labels')
    args = parser.parse_args()
    return args

def largest_connected_component_3d(image, num_labels):
    largest_components = np.zeros_like(image)
    for label_num in range(1, num_labels):  # Skip background label 0
        labeled_array, num_features = label(image == label_num)
        if num_features == 0:
            continue
        largest_label = 1 + np.argmax(np.bincount(labeled_array.flat)[1:])
        largest_components[labeled_array == largest_label] = label_num
    return largest_components

def morphological_closing_2d(image, num_labels, kernel_size=(3, 3)):
    closed_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for label_num in range(1, num_labels):  # Skip background label 0
            closed_image[i] += binary_closing(image[i] == label_num, structure=square(kernel_size[0])) * label_num
    return closed_image

def post_process(predicted_label_path, output_path):
    predicted_img = nib.load(predicted_label_path)
    predicted_labels = predicted_img.get_fdata()
    num_labels = int(predicted_labels.max()) + 1
    # Apply 3D largest connected component filtering
    largest_component = largest_connected_component_3d(predicted_labels, num_labels)

    # Apply 2D morphological closing
    closed_labels = morphological_closing_2d(largest_component, num_labels)
    closed_labels = np.clip(closed_labels, 0, num_labels - 1).astype(np.int16)

    # Save the post-processed labels
    nib.save(nib.Nifti1Image(closed_labels.astype(np.float32), predicted_img.affine), output_path)
    print(f"Saved post-processed labels at {output_path}")

def main():
    args = parse_args()
    patient_dirs = os.listdir(args.prediction_folder)

    for patient_name in patient_dirs:
        patient_path = os.path.join(args.prediction_folder, patient_name)

        for filename in os.listdir(patient_path):
            if filename.endswith("_predicted_labels.nii"):
                input_path = os.path.join(patient_path, filename)
                output_path = os.path.join(patient_path, filename)
                post_process(input_path, output_path)

if __name__ == "__main__":
    main()
