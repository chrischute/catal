import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class FakeDataLoader(data.DataLoader):
    """DataLoader for whiteboard images."""
    def __init__(self, size, num_classes, phase, batch_size, shuffle, do_augment, num_workers):
        self.phase = phase
        if do_augment:
            transforms_list = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        else:
            transforms_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        transforms_list += [transforms.ToTensor(),
                            # TODO: Find mean and std dev. of training set
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        dataset = FakeData(size, num_classes=num_classes,
                           transform=transforms.Compose(transforms_list))
        super(FakeDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class FakeData(data.Dataset):
    """A fake dataset that returns randomly generated images and returns them as PIL images

    Args:
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): Number of classes in the datset. Default: 10
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        random_offset (int): Offsets the index-based random seed used to
            generate each image. Default: 0

    """

    def __init__(self, size=1000, image_size=(3, 224, 224), num_classes=10,
                 transform=None, target_transform=None, random_offset=0):
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform
        self.random_offset = random_offset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        target = torch.tensor(1, dtype=torch.int64).random_(0, self.num_classes)
        torch.set_rng_state(rng_state)

        # convert to PIL Image
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.size

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
