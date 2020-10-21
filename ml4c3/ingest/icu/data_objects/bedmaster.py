# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.definitions.icu import ALARMS_FILES

from .data_object import ICUDiscreteData, ICUContinuousData


class BMSignal(ICUContinuousData):
    """
    Implementation of the parent ICU Continuous Data Object class for Bedmaster
    data.
    """

    def __init__(
        self,
        name: str,
        source: str,
        channel: str,
        value: np.ndarray,
        time: np.ndarray,
        units: str,
        sample_freq: np.ndarray,
        scale_factor: float,
        time_corr_arr: np.ndarray,
        samples_per_ts: np.ndarray,
    ):
        """
        Init a BMSignal object.

        :param name: <str> Signal name.
        :param source: <str> Database from which the signal is extracted.
        :param channel: <str> Channel of the signal.
        :param value: <np.ndarray> Array of floats containing the signal
                      value in each timestamp.
        :param time: <np.ndarray> Array of timestamps corresponding to
                     each signal value.
        :param units: <str> Unit associated to the signal value.
        :param sample_freq: <np.ndarray> Signal's sampling frequency.
        :param scale_factor: <float> Signal's scale factor.
        :param time_corr_arr: <np.ndarray> Compressed array defining if the
                             corresponding Timestamp has been corrected or not.
                             Compressed using np.packbits (np.unpackbits to
                             decompress)
        :param samples_per_ts: <np.ndarray> Array with the number of values
                              recorded in the period between each timestamp.
        """
        super().__init__(name, source, value, time, units)
        self.channel = channel
        self.sample_freq = sample_freq
        self.scale_factor = scale_factor
        self.time_corr_arr = time_corr_arr
        self.samples_per_ts = samples_per_ts

    @property
    def _source_type(self):
        return self.source.replace("BM_", "")


class BMAlarm(ICUDiscreteData):
    """
    Implementation of the parent ICU Discrete Data Object class for Bedmaster
    alarms data.
    """

    def __init__(
        self,
        name: str,
        start_date: np.ndarray,
        duration: np.ndarray,
        level: int,
    ):
        """
        Init a BMAlarm object.

        :param name: <str> Alarm name.
        :param start_date: <np.ndarray> List of unix timestamps when the alarm is
                           triggered.
        :param duration: <np.ndarray> List of alarm triggered durations in seconds.
        :param level: <int> Alarm Level
        """
        super().__init__(name, ALARMS_FILES["source"], start_date)
        self.duration = duration
        self.level = level

    @property
    def _source_type(self):
        return self.source.replace("BM_", "")
