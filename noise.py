import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
from skimage.util import random_noise

import torch
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None

#from ..utils import _log_api_usage_once
#from . import functional as F
#from .functional import _interpolation_modes_from_int, InterpolationMode

from types import FunctionType
from typing import Any
def _log_api_usage_once(obj: Any) -> None:
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")
    
__all__ = [
    "RandomSpeckleNoise",
    "RandomGaussianNoise",
    "RandomSaltPepperNoise"
]


# add speckle noise p: probability var: noise intensity
class RandomSpeckleNoise(torch.nn.Module):
    def __init__(self, p=0.5, var=0.05):
        super().__init__()
        _log_api_usage_once(self)
        self.p = p
        self.var = var

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return torch.Tensor(random_noise(img, mode='speckle', mean=0, var=self.var, clip=True))
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

# add gaussian noise p: probability var: noise intensity
class RandomGaussianNoise(torch.nn.Module):
    def __init__(self, p=0.5, var=0.05):
        super().__init__()
        _log_api_usage_once(self)
        self.p = p
        self.var = var

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return torch.Tensor(random_noise(img, mode='gaussian', mean=0, var=self.var, clip=True))
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

# add sal&paper noise p: probability var: noise intensity
class RandomSaltPepperNoise(torch.nn.Module):
    def __init__(self, p=0.5, amount=0.1, salt_vs_pepper=0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.p = p
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return torch.tensor(random_noise(img, mode='s&p', amount=self.amount, salt_vs_pepper=self.salt_vs_pepper, clip=True))
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
    