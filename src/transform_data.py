import torch
import numpy as np
import random
from skimage import transform


class Resize_img(object):
    def __init__(self, size):
        # assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resize_image = transform.resize(img, self._size, preserve_range=True)
        # the resize will return a float64 array
        return resize_image

class DRR(object):
    def __init__(self, b1=20, b2=50):
        self.b = random.randint(b1, b2)

    def __call__(self, data):
        return data * self.b / ((self.b - 1) * data + 1)

class Numpy2Tensor(object):
    def __call__(self, data):
        data = np.transpose(data, (2,0,1))
        return torch.Tensor(data)

class Normalize_img(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (img - self.mean) / self.std

        return img

class Numpy2Tensor_img(object):
    """Convert a 1-channel ``numpy.ndarray`` to 1-c or 3-c tensor,
    depending on the arg parameter of "channels"
    """
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, img):
        """
        for SAR images (.npy), shape H * W, we should transform into C * H * W
        :param img:
        :return:
        """
        channels = self.channels

        img_copy = np.zeros([channels, img.shape[0], img.shape[1]])

        for i in range(channels):
            img_copy[i, :, :] = np.reshape(img, [1, img.shape[0], img.shape[1]]).copy()

        if not isinstance(img_copy, np.ndarray) and (img_copy.ndim in {2, 3}):
            raise TypeError('img should be ndarray. Got {}'.format(type(img_copy)))

        if isinstance(img_copy, np.ndarray):
            # handle numpy array
            img_copy = torch.Tensor(img_copy)
            # backward compatibility
            return img_copy.float()