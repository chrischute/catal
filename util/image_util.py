class UnNormalize(object):
    """Module to reverse normalization of input images."""
    def __init__(self, mean, std):
        if len(mean) != 3 or len(std) != 3:
            raise ValueError('Mean and standard deviations must have length 3 (one per channel).')
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        return tensor
