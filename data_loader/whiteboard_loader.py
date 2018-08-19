import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from constants import CATAL_MEAN, CATAL_STD
from data_loader.image_folder_with_paths import ImageFolderWithPaths
from PIL import ImageFile


class WhiteboardLoader(data.DataLoader):
    """DataLoader for whiteboard images."""
    def __init__(self, data_dir, phase, batch_size, shuffle, do_augment, num_workers):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.data_dir = os.path.join(data_dir, phase)
        self.phase = phase
        to_tensor_fn = transforms.ToTensor()
        normalize_fn = transforms.Normalize(mean=CATAL_MEAN, std=CATAL_STD)
        if do_augment:
            transforms_list = [transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               transforms.RandomAffine(20),
                               transforms.RandomRotation(20),
                               to_tensor_fn,
                               normalize_fn]
        else:
            transforms_list = [transforms.Resize(256),
                               transforms.TenCrop(224),
                               transforms.Lambda(lambda crops: torch.stack([normalize_fn(to_tensor_fn(crop))
                                                                            for crop in crops]))]

        dataset = ImageFolderWithPaths(self.data_dir, transforms.Compose(transforms_list))
        super(WhiteboardLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
