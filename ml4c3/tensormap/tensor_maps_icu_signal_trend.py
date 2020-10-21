# Imports: standard library
from typing import Any, List
from itertools import chain, groupby

# Imports: third party
import numpy as np
from matplotlib import pyplot as plt

# pylint: disable=too-many-branches


class TrendComputing:
    """
    Implementation of a class to compute dominant trends of the given
    timeseries signal.
    """

    def __init__(self, signal: np.ndarray, time: np.ndarray):
        """
        Init TrendComputing instance.

        :param signal: <np.ndarray> Numpy array containing the values of the signal.
        :param time: <np.ndarray> Numpy array containing the timestamp corresponding
                     to each value.
        """
        self.signal = signal
        self.time = time
        self.groups = [0] * len(self.signal)
        self.change_p_trend = 0
        self.change_n_trend = 0
        self.slope_p_trend = 0
        self.slope_n_trend = 0

    def compute_trend(self, max_size: int = -np.inf, min_size: int = np.inf):
        """
        Function to compute dominant trends features.

        Given a time window, compute the change in dominant positive and
        negative trends, That is the difference between the maximal and
        minimal point of the largest positive and negative trends. The
        largest trend is defined by means of the number of consecutive
        data points.

        If a consecutive set of points with the same trend is shorter than
        min_size is bounded by a segments with same slope is merged with the
        boundary segments.

        :param max_size: <int> Maximum size of a segment to consider merging it
                         with the surrounding segments.
        :param min_size: <int> Minimum size of each surrounding segment to consider
                         merging with the segment inbetween.
        """
        grad = np.gradient(self.signal)
        grad = (grad > 0).astype(int) - (grad < 0).astype(int)

        result: List[List[Any]] = []
        indices: List[int] = []
        acum_indices: List[int] = []
        acum_groups = []
        len_acum_groups = 0
        previous_index: int = -1
        previous = None
        current_index = 0
        for k, _group in groupby(grad):
            group = list(_group)
            if len(group) >= min_size:
                if not previous:
                    previous_index = current_index
                    previous = group
                else:
                    if k == previous[0]:
                        previous = [k] * (len(group) + len(previous))
                        for middle_group in acum_groups:
                            previous += [k] * len(middle_group)
                    else:
                        result.append(previous)
                        previous_index = current_index
                        previous = group
                        for i, middle_group in enumerate(acum_groups):
                            indices.append(acum_indices[i])
                            result.extend(middle_group)
                    acum_indices = []
                    acum_groups = []
                    len_acum_groups = 0
            elif len(group) + len_acum_groups <= max_size and previous:
                len_acum_groups += len(group)
                acum_indices.append(current_index)
                acum_groups.append(group)
            else:
                if previous:
                    indices.append(previous_index)
                    result.append(previous)
                    previous_index = -1
                    previous = None
                    for i, middle_group in enumerate(acum_groups):
                        indices.append(acum_indices[i])
                        result.append(middle_group)
                    acum_indices = []
                    acum_groups = []
                    len_acum_groups = 0
                indices.append(current_index)
                result.append(group)
            current_index += len(group)
        if previous:
            indices.append(previous_index)
            result.append(previous)

        p_max = 0
        n_max = 0
        p_index = 0
        n_index = 0
        for index, group in enumerate(result):
            k = group[0]
            length = len(group)
            if k == 1 and length > p_max:
                p_max = length
                p_index = indices[index]
            elif k == -1 and length > n_max:
                n_max = length
                n_index = indices[index]

        if p_max != 0:
            self.change_p_trend = (
                self.signal[p_index + p_max - 1] - self.signal[p_index]
            )
            self.slope_p_trend = (
                self.signal[p_index + p_max - 1] - self.signal[p_index]
            ) / (self.time[p_index + p_max - 1] - self.time[p_index])
        if n_max != 0:
            self.change_n_trend = (
                self.signal[n_index + n_max - 1] - self.signal[n_index]
            )
            self.slope_n_trend = (
                self.signal[n_index + n_max] - self.signal[n_index]
            ) / (self.time[n_index + n_max - 1] - self.time[n_index])
        self.groups = list(chain.from_iterable(result))

    def plot(self):
        """
        Function to plot the signal colored with a different color if it has a
        positive, negative or zero slope.
        """
        plt.figure()
        plt.scatter(self.time, self.signal, c=self.groups)
        plt.show()
