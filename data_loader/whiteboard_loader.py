import os
import torch.utils.data as data
import torchvision.transforms as transforms

from constants import CATAL_MEAN, CATAL_STD
from data_loader.image_folder_with_paths import ImageFolderWithPaths


class WhiteboardLoader(data.DataLoader):
    """DataLoader for whiteboard images."""
    def __init__(self, data_dir, phase, batch_size, shuffle, do_augment, num_workers):
        self.data_dir = os.path.join(data_dir, phase)
        self.phase = phase
        if do_augment:
            transforms_list = [transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               transforms.RandomAffine(20),
                               transforms.RandomRotation(20)]
        else:
            transforms_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        transforms_list += [transforms.ToTensor(),
                            transforms.Normalize(mean=CATAL_MEAN, std=CATAL_STD)]

        dataset = ImageFolderWithPaths(self.data_dir, transforms.Compose(transforms_list))
        super(WhiteboardLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
