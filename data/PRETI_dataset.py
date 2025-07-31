import os
import os.path as osp
import random

# import albumentations as A
import cv2
# import drnoon_image_transform as dit
import numpy as np
import pandas as pd
import torch
from PIL import Image
import json
import itertools
import torchvision.transforms.v2 as T2
from torchvision.transforms.v2 import functional as F
from .util.image_utils import RRC

class FundusDataset_w_meta(torch.utils.data.Dataset):
    def __init__(self, csv_file, args, cache_file="/home/leeyeonkyung/PRETI/data/csv/PROCESSED_TRAIN_2_automorph_with_masks.csv", transform=None, force_rebuild=False):
        """
        Args:
            csv_file (str): Path to the CSV file
            args: Object containing augmentation settings and other configs
            cache_file (str): Path to the cached paired_data file
            transform (callable, optional): Optional additional image transforms (default uses DualViewAugmentation)
            force_rebuild (bool): Whether to force rebuild the cached file
        """
        interpolation = F.InterpolationMode.BILINEAR  # Set default interpolation
        interpolation_mask = F.InterpolationMode.NEAREST # Use nearest for masks
        
        self.transform_left = T2.Compose([
            T2.RandomResizedCrop(
                size=(args.input_size, args.input_size),
                scale=(args.random_area_min_1, args.random_area_max_1),  # Crop between 20% ~ 80% of the image
                ratio=(args.random_aspect_ratio_min_1, args.random_aspect_ratio_max_1),  # Aspect ratio of the crop
                interpolation=interpolation,
                antialias=True
            ),
            T2.RandomHorizontalFlip(p=args.horizontal_flip_p_1),
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_right = T2.Compose([
            T2.RandomResizedCrop(
                size=(args.input_size, args.input_size),
                scale=(args.random_area_min_2, args.random_area_max_2),
                ratio=(args.random_aspect_ratio_min_2, args.random_aspect_ratio_max_2),
                interpolation=interpolation,
                antialias=True
            ),
            T2.RandomHorizontalFlip(p=args.horizontal_flip_p_2),
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_left_mask = T2.Compose([
            T2.RandomResizedCrop(
                size=(args.input_size, args.input_size),
                scale=(args.random_area_min_1, args.random_area_max_1),
                ratio=(args.random_aspect_ratio_min_1, args.random_aspect_ratio_max_1),
                interpolation=interpolation_mask,  # Use nearest interpolation for masks
                antialias=False
            ),
            T2.RandomHorizontalFlip(p=args.horizontal_flip_p_1),
            T2.ToImage(),
            T2.ToDtype(torch.uint8),  # Keep masks as integers
        ])
        self.transform_right_mask = T2.Compose([
            T2.RandomResizedCrop(
                size=(args.input_size, args.input_size),
                scale=(args.random_area_min_2, args.random_area_max_2),
                ratio=(args.random_aspect_ratio_min_2, args.random_aspect_ratio_max_2),
                interpolation=interpolation_mask,
                antialias=False
            ),
            T2.RandomHorizontalFlip(p=args.horizontal_flip_p_2),
            T2.ToImage(),
            T2.ToDtype(torch.uint8),
        ])
        
        if not force_rebuild and os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.paired_data = json.load(f)
            print(f"Loaded cached paired_data from {cache_file}")
        else:
            print(f"Generating new paired_data from {csv_file}...")
            self.paired_data = self._process_csv(csv_file)
            with open(cache_file, "w") as f:
                json.dump(self.paired_data, f)
            print(f"Cached paired_data saved to {cache_file}")

    def _process_csv(self, csv_file):
        """Read the CSV file and build the paired_data list"""
        data = pd.read_csv(csv_file)
        
        # Convert laterality (>= 0.5 → 'R', < 0.5 → 'L')
        data["laterality"] = data["laterality"].apply(lambda x: "R" if x >= 0.5 else "L")
        
        # Convert gender ('Male' → 1, 'Female' → 0)
        data["gender"] = data["gender"].apply(lambda x: 1 if x == "Male" else 0)

        paired_data = []
        for (pid, eid), group in data.groupby(["patient_id", "exam_id"]):
            eye_list = group.to_dict("records")  # All eye records for the same patient

            # Generate all possible pairs (including self-pairs)
            for eye1, eye2 in itertools.combinations(eye_list, 2):
                paired_data.append({
                    "left_image": eye1["jpg_h512_path_automorph"],
                    "right_image": eye2["jpg_h512_path_automorph"],
                    "left_mask": eye1["mask_path"],
                    "right_mask": eye2["mask_path"],
                    "age": eye1["age"],
                    "gender": eye1["gender"]
                })
                # Add the reversed pair
                paired_data.append({
                    "left_image": eye2["jpg_h512_path_automorph"],
                    "right_image": eye1["jpg_h512_path_automorph"],
                    "left_mask": eye2["mask_path"],
                    "right_mask": eye1["mask_path"],
                    "age": eye2["age"],
                    "gender": eye2["gender"]
                })

            # If only one eye exists, create a self-pair
            if len(eye_list) == 1:
                eye = eye_list[0]
                paired_data.append({
                    "left_image": eye["jpg_h512_path_automorph"],
                    "right_image": eye["jpg_h512_path_automorph"],
                    "left_mask": eye["mask_path"],
                    "right_mask": eye["mask_path"],
                    "age": eye["age"],
                    "gender": eye["gender"]
                })

        return paired_data
    
    def _load_mask(self, mask_path, transform = None):
        """Load a mask image (no normalization, keep 0–1 range as uint8 Tensor)"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if mask is None:
            raise ValueError(f"Mask file not found: {mask_path}")
        
        mask = Image.fromarray(mask)  # Convert to PIL image

        if transform:
            mask = transform(mask)  # Apply transforms to the mask

        mask = torch.tensor(np.array(mask), dtype=torch.uint8)  # Keep 0–255 range, no normalization
        mask = mask.unsqueeze(0)  # Convert to shape (1, H, W)
        return mask

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        item = self.paired_data[idx]
        img_left = self._load_image(item["left_image"], transform=self.transform_left)
        img_right = self._load_image(item["right_image"], transform=self.transform_right)
        imgs = torch.stack([img_left, img_right], dim=0)  # Stack both images into a single tensor

        # Load masks
        mask_left = self._load_mask(item["left_mask"], transform=self.transform_left_mask)
        mask_right = self._load_mask(item["right_mask"], transform=self.transform_right_mask)
        masks = torch.stack([mask_left, mask_right], dim=0)  # Stack both masks

        meta_data = {
            "age": torch.tensor(item["age"], dtype=torch.float32),
            "gender": torch.tensor(item["gender"], dtype=torch.float32)
        }

        return imgs, meta_data, masks

    def _load_image(self, image_path, transform=None):
        """Image loading function (using PIL)"""
        image = Image.open(image_path).convert("RGB")
        if transform:
            image = transform(image)
        return image