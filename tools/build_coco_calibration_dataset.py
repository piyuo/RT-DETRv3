# tools/build_coco_calibration_dataset.py

import os
import json
import random
import shutil
from pycocotools.coco import COCO
from tqdm import tqdm

def create_calibration_dataset(
    source_image_dir,
    source_annotation_path,
    target_dir,
    num_samples=1000
):
    """
    Creates a new COCO-format calibration dataset by sampling a subset of images.

    Args:
        source_image_dir (str): Path to the source images directory (e.g., 'dataset/coco/val2017').
        source_annotation_path (str): Path to the source annotations file (e.g., 'dataset/coco/annotations/instances_val2017.json').
        target_dir (str): Path to the new directory for the calibration dataset.
        num_samples (int): The number of images to sample.
    """
    print("Loading COCO annotations...")
    coco = COCO(source_annotation_path)

    # Get all image IDs from the source dataset
    all_img_ids = coco.getImgIds()
    print(f"Total images available: {len(all_img_ids)}")

    # Randomly sample the desired number of image IDs
    if num_samples > len(all_img_ids):
        print(f"Warning: Number of samples ({num_samples}) is greater than available images. Using all available images.")
        sampled_img_ids = all_img_ids
    else:
        sampled_img_ids = random.sample(all_img_ids, num_samples)

    print(f"Selected {len(sampled_img_ids)} images for calibration.")

    # Create the target directories
    target_images_dir = os.path.join(target_dir, 'images')
    target_annotations_dir = os.path.join(target_dir, 'annotations')
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_annotations_dir, exist_ok=True)

    # Initialize a new COCO-format dictionary
    new_annotations = {
        "info": coco.dataset["info"],
        "licenses": coco.dataset["licenses"],
        "categories": coco.dataset["categories"],
        "images": [],
        "annotations": []
    }

    # Copy images and their annotations
    print("Copying images and collecting annotations...")
    for img_id in tqdm(sampled_img_ids):
        # Get image info
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']

        # Copy the image file
        source_image_path = os.path.join(source_image_dir, file_name)
        target_image_path = os.path.join(target_images_dir, file_name)
        shutil.copy(source_image_path, target_image_path)

        # Add image info to the new annotations
        new_annotations['images'].append(img_info)

        # Get annotations for the current image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Add annotations to the new annotations
        new_annotations['annotations'].extend(anns)

    # Save the new annotation file
    target_annotation_path = os.path.join(target_annotations_dir, 'instances_calibration.json')
    with open(target_annotation_path, 'w') as f:
        json.dump(new_annotations, f)

    print(f"Calibration dataset created successfully in '{target_dir}'")
    print(f"New annotations saved to: {target_annotation_path}")


if __name__ == '__main__':
    # Define your paths and desired number of samples
    SOURCE_IMAGES_DIR = 'dataset/coco/val2017'
    SOURCE_ANNOTATIONS_PATH = 'dataset/coco/annotations/instances_val2017.json'
    TARGET_DIR = 'dataset/coco/calibration_dataset'
    NUM_SAMPLES = 1000  # You can adjust this number

    create_calibration_dataset(
        source_image_dir=SOURCE_IMAGES_DIR,
        source_annotation_path=SOURCE_ANNOTATIONS_PATH,
        target_dir=TARGET_DIR,
        num_samples=NUM_SAMPLES
    )