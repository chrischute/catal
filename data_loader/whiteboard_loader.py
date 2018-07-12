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
                            # TODO: Find mean and std dev. of training set
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        dataset = torchvision.datasets.ImageFolder(self.data_dir, transforms.Compose(transforms_list))
        super(WhiteboardLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
