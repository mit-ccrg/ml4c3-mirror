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

    Continuous data objects are BMObject and MeasurementObject. It
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
