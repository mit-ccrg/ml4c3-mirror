# Imports: standard library
from abc import ABC, abstractmethod

# Imports: third party
import numpy as np

# pylint: disable=abstract-method, unnecessary-pass


class ICUDataObject(ABC):
    """
    Abstract class that is the parent to all the Data Objects.

    It may be left empty.
    """

    def __init__(self, source: str):
        """
        Init an ICUDataObject.

        :param source: <str> Database from which the signal is extracted.
        """
        self.source = source

    @property
    @abstractmethod
    def _source_type(self):
        """
        Parse out the source type from the source.
        """
        pass


class ICUContinuousData(ICUDataObject):
    """
    Parent class for the wrappers of continuous data.

    Continuous data objects are BedmasterObject and MeasurementObject. It
    shouldn't be instanced directly, instance through its children
    instead.
    """

    def __init__(
        self,
        name: str,
        source: str,
        value: np.ndarray,
        time: np.ndarray,
        units: str,
    ):
        """
        Init an ICUContinuousData object.

        :param name: <str> Signal name.
        :param source: <str> Database from which the signal is extracted.
        :param value: <np.ndarray> Array of floats containing the signal
                     value in each timestamp.
        :param time: <np.ndarray> Array of timestamps corresponding
                    to each signal value.
        :param units: <str> Unit associated to the signal value.
        """
        self.name = name
        self.value = value
        self.time = time
        self.units = units
        super().__init__(source)

    def __str__(self):
        return f"Signal name: {self.name!r}, Source: {self.source!r}"


class ICUDiscreteData(ICUDataObject):
    """
    Parent class for our wrappers of punctual data.

    Punctual data are MedicationObject, EventObject and
    TransfusionObject.

    It shouldn't be instanced directly, use its children instead.
    """

    def __init__(
        self,
        name: str,
        source: str,
        start_date: np.ndarray,
    ):
        """
        Init an ICUDiscreteData object.

        :param name: <str> Signal name.
        :param source: <str> Database from which the signal is extracted.
        :param start_date: <np.ndarray> Array of timestamps corresponding
                            to the start of each captured discrete event.
        """
        self.name = name
        self.start_date = start_date
        super().__init__(source)

    def __str__(self):
        return f"Signal name: {self.name!r}, Source: {self.source!r}"


class BedmasterSignal(ICUContinuousData):
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
        Init a BedmasterSignal object.

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
        return self.source.replace("Bedmaster_", "")


BedmasterType = BedmasterSignal
