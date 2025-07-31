import torch
import torchvision.transforms.v2 as T2
from torchvision.transforms.v2 import functional as F
from .image_utils import RRC


# Multi Images (YK)
class DualViewAugmentation:
    """
    Applies different augmentations to two input images separately.
    """
    def __init__(self, args):
        super().__init__()
        
        interpolation = getattr(F.InterpolationMode, args.interpolation_method.upper())
        
        self.crop_1 = RRC(
            size=(args.input_size, args.input_size),
            scale=(args.random_area_min_1, args.random_area_max_1),
            ratio=(args.random_aspect_ratio_min_1, args.random_aspect_ratio_max_1),
            interpolation=interpolation,
            antialias=True
        )

        self.crop_2 = RRC(
            size=(args.input_size, args.input_size),
            scale=(args.random_area_min_2, args.random_area_max_2),
            ratio=(args.random_aspect_ratio_min_2, args.random_aspect_ratio_max_2),
            interpolation=interpolation,
            antialias=True
        )

        self.transforms_1 = T2.Compose([
            T2.RandomHorizontalFlip(p=args.horizontal_flip_p_1),
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.transforms_2 = T2.Compose([
            T2.RandomHorizontalFlip(p=args.horizontal_flip_p_2),
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        add_trans_1 = []
        if args.use_color_jitter_1:
            add_trans_1.append(T2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
        if args.use_gaussian_blur_1:
            add_trans_1.append(T2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)))
        if args.use_elastic_transform_1:
            add_trans_1.append(T2.ElasticTransform())
        
        self.add_trans_1 = T2.Compose(add_trans_1) if add_trans_1 else None
        
        add_trans_2 = []
        if args.use_color_jitter_2:
            add_trans_2.append(T2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
        if args.use_gaussian_blur_2:
            add_trans_2.append(T2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)))
        if args.use_elastic_transform_2:
            add_trans_2.append(T2.ElasticTransform())
        
        self.add_trans_2 = T2.Compose(add_trans_2) if add_trans_2 else None

    def _create_views(self, img1, img2):
        img1_cropped = self.crop_1(img1)
        img2_cropped = self.crop_2(img2)
        return img1_cropped, img2_cropped

    def _process_views(self, img1, img2):
        img1 = self.transforms_1(img1)
        img2 = self.transforms_2(img2)
        return img1, img2

    def _apply_additional_transforms(self, img1, img2):
        if self.add_trans_1:
            img1 = self.add_trans_1(img1)
        if self.add_trans_2:
            img2 = self.add_trans_2(img2)
        return img1, img2

    def __call__(self, img1, img2):
        img1_cropped, img2_cropped = self._create_views(img1, img2)
        img1_aug, img2_aug = self._process_views(img1_cropped, img2_cropped)
        img1_aug, img2_aug = self._apply_additional_transforms(img1_aug, img2_aug)
        return torch.cat([img1_aug.unsqueeze(0), img2_aug.unsqueeze(0)], dim=0)