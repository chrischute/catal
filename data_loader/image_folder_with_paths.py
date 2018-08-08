from torchvision import datasets


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder that also returns paths.

    Adapted from:
        https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __getitem__(self, index):
        input_and_label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        input_and_label_and_path = (input_and_label + (path,))

        return input_and_label_and_path
