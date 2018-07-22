import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import os


class WhiteboardLoader(data.DataLoader):
    """DataLoader for whiteboard images."""
    def __init__(self, data_dir, phase, batch_size, shuffle, do_augment, num_workers):
        self.data_dir = os.path.join(data_dir, phase)
        self.phase = phase
        if do_augment:
            transforms_list = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        else:
            transforms_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        transforms_list += [transforms.ToTensor(),
                            transforms.Normalize(mean=[0.520, 0.443, 0.374], std=[0.136, 0.135, 0.134])]

        dataset = torchvision.datasets.ImageFolder(self.data_dir, transforms.Compose(transforms_list))
        super(WhiteboardLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
