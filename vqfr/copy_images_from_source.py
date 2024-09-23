import os
import shutil
import random

def copy_images_to_hq_folder(source_path, destination_path, num_images):
    # Ensure the source and destination paths exist
    if not os.path.exists(source_path):
        raise ValueError(f"Source path {source_path} does not exist.")
    
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Create the 'hq' folder in the destination path
    hq_folder = os.path.join(destination_path, 'hq')
    if not os.path.exists(hq_folder):
        os.makedirs(hq_folder)
    
    # List all image files in the source directory
    image_files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
    
    # Check if the number of requested images is not more than available images
    if num_images > len(image_files):
        raise ValueError(f"Requested {num_images} images, but only {len(image_files)} are available in the source path.")
    
    # Randomly select the specified number of images
    selected_images = random.sample(image_files, num_images)
    
    # Copy selected images to the 'hq' folder
    for image in selected_images:
        src = os.path.join(source_path, image)
        dst = os.path.join(hq_folder, image)
        shutil.copyfile(src, dst)
    
    print(f"Copied {num_images} images to {hq_folder}")

# Example usage:
copy_images_to_hq_folder("/home/user1/kasra/pycharm-projects/face-super-resolution/dataset/validation/hq", "/home/user1/kasra/pycharm-projects/VQFR/datasets/FFHQ/validation/hq", 1000)
