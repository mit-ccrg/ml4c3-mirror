# Imports: standard library
from abc import ABC, abstractmethod

# Imports: third party
import numpy as np

# Imports: first party
from definitions.globals import EPS

# pylint: disable=unnecessary-pass, no-self-use, redefined-builtin


class Normalizer(ABC):
    """
    Normalizer is an abstract class that defines transformations to normalize a
    tensor.
    """

    @abstractmethod
    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        """Shape preserving transformation"""
        pass

    def un_normalize(self, tensor: np.ndarray) -> np.ndarray:
        """The inverse of normalize if possible. Otherwise identity."""
        return tensor


class Standardize(Normalizer):
    """
    Normalize a tensor by subtracting a constant mean and dividing by a
    constant standard deviation.
    """

    def __init__(self, mean: float, std: float):
        self.mean, self.std = mean, std

    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        return (tensor - self.mean) / (self.std + EPS)

    def un_normalize(self, tensor: np.ndarray) -> np.ndarray:
        return tensor * (self.std + EPS) + self.mean


class RobustScaler(Normalizer):
    """
    Normalize a tensor by subtracting a constant median and dividing by a
    constant interquartile range.
    """

    def __init__(self, median: float, iqr: float):
        self.median, self.iqr = median, iqr

    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        return (tensor - self.median) / (self.iqr + EPS)

    def un_normalize(self, tensor: np.ndarray) -> np.ndarray:
        return tensor * (self.iqr + EPS) + self.median


class ZeroMeanStd1(Normalizer):
    """
    Normalize a tensor by subtracting the mean of the tensor and dividing by
    the standard deviation of the tensor.
    """

    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        tensor -= np.mean(tensor)
        tensor /= np.std(tensor) + EPS
        return tensor


class MinMax(Normalizer):
    """
    Normalize a tensor by subtracting the minimum of the tensor and dividing by
    the difference between the maximum and the minimum of the tensor.
    """

    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        return (tensor - self.min) / (self.max - self.min)

    def un_normalize(self, tensor: np.ndarray) -> np.ndarray:
        return tensor * (self.max - self.min) + self.min
