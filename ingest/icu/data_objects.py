# Imports: standard library
from abc import ABC, abstractmethod
from typing import Dict

# Imports: third party
import numpy as np

# Imports: first party
from definitions.icu import EDW_FILES, ALARMS_FILES

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


class BedmasterAlarm(ICUDiscreteData):
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
        Init a BedmasterAlarm object.

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
        return self.source.replace("Bedmaster_", "")


class StaticData(ICUDataObject):
    """
    Independent class to wrap Static objects.

    Static objects are EDW demographic and admission data.
    """

    def __init__(
        self,
        department_id: np.ndarray,
        department_nm: np.ndarray,
        room_bed: np.ndarray,
        move_time: np.ndarray,
        weight: float,
        height: float,
        admin_type: str,
        admin_diag: str,
        admin_date: str,
        birth_date: str,
        race: str,
        sex: str,
        end_date: str,
        end_stay_type: str,
        local_time: np.ndarray,
        medical_hist: np.ndarray,
        surgical_hist: np.ndarray,
        tobacco_hist: str,
        alcohol_hist: str,
        source=EDW_FILES["adm_file"]["source"],
    ):
        """
        Init a StaticData object.

        :param department_id: <np.ndarray> Patient's location (department id).
        :param department_nm: <np.ndarray> Patient's location (department name).
        :param room_bed: <np.ndarray> Patient's location (room and bed)
        :param move_time: <np.ndarray> Timestamp array indicating when the patient is
                                    transferred from one location to another. This array
                                    is directly linked with department_id, department_nm
                                    and room_bed arrays. Each element corresponds to the
                                    timestamp information of the location with same
                                    array index.
        :param weight: <float> Patient's admission weight in pounds.
        :param height: <float> Patient's height in meters.
        :param admin_type: <str> Patient's admission type.
        :param admin_diag: <str> Patient's admission diagnosis.
        :param admin_date: <str> Patient's admission date (Timestamp).
        :param birth_date: <str> Patient's date of birth (Timestamp).
        :param race: <str> Patient's ethnicity.
        :param sex: <str> Patient's sex.
        :param end_date: <str> Timestamp indicating when the patient leaves the ICU.
        :param end_stay_type: <str> Reason, 'Alive' or 'Deceased'.
        :param local_time: <np.ndarray> Array of local times. In general only contains
                                        one value, it may contain two if there is a time
                                        change (e.g. summer to winter time) while the
                                        patient is in the hospital.
        :param medical_hist: <np.ndarray> Array of previous diagnoses. Each element
                                        contains past information about the diagnosis,
                                        physician comments and the date if known.
        :param surgical_hist: <np.ndarray> Array of previous surgical procedures. Each
                                        element contains past information about the
                                        surgery, physician comments and the date if
                                        known.
        :tobacco_hist: <str> String of tobacco consumption information.
        :param alcohol_hist: <str> String of alcohol consumption information.
        :param source: <str> Database from which the signal is extracted.
                      For this class is always 'EDW_static'.
        """
        self.department_id = department_id
        self.department_nm = department_nm
        self.room_bed = room_bed
        self.move_time = move_time
        self.weight = weight
        self.height = height
        self.admin_type = admin_type
        self.admin_diag = admin_diag
        self.admin_date = admin_date
        self.birth_date = birth_date
        self.race = race
        self.sex = sex
        self.end_date = end_date
        self.end_stay_type = end_stay_type
        self.local_time = local_time
        self.medical_hist = medical_hist
        self.surgical_hist = surgical_hist
        self.tobacco_hist = tobacco_hist
        self.alcohol_hist = alcohol_hist
        super().__init__(source)

    def __str__(self):
        return f"Source: {self.source!r}"

    @property
    def _source_type(self):
        return self.source.replace("EDW_", "")


class Measurement(ICUContinuousData):
    """
    Implementation of ICUContinuousData for EDW measurement data.

    Measurement data are `labs` or `vitals`.
    """

    def __init__(
        self,
        name: str,
        source: str,
        value: np.ndarray,
        time: np.ndarray,
        units: str,
        data_type: str,
        metadata: Dict[str, np.ndarray] = None,
    ):
        """
        Init a Measurement object.

        :param name: <str> Signal name.
        :param source: <str> Database from which the signal is extracted.
        :param value: <np.ndarray> Array of floats containing the signal
                     value in each timestamp.
        :param time: <np.ndarray> Array of Unix timestamps corresponding to
                    each signal value.
        :param units: <str> Unit associated to the signal value.
        :param data_type: <str> "categorical" or "numerical".
        :param metadata: Dictionary of additional measurement metadata
        """
        super().__init__(name, source, value, time, units)
        self.data_type = data_type
        if metadata:
            self.metadata = metadata

    @property
    def _source_type(self):
        return self.source.replace("EDW_", "")


class Medication(ICUDiscreteData):
    """
    Implementation of ICUContinuousData for EDW medication data.

    One instance holds a list of doses of the same type of medication.
    """

    def __init__(
        self,
        name: str,
        dose: np.ndarray,
        units: str,
        start_date: np.ndarray,
        action: np.ndarray,
        route: str,
        wt_based_dose: bool,
        source=EDW_FILES["med_file"]["source"],
    ):
        """
        Init a Medication object.

        :param name: <str> Medication name.
        :param dose: <np.ndarray> Array of floats containing the dose
                    administered in each timestamp.
        :param units: <str> Unit associated to the dosage value.
        :param start_date: <np.ndarray> Array of Unix timestamps corresponding
                         the med administration date of the specified dose...
        :param action: <np.ndarray> Array of strings describing the medication
                        action. Given, New Bag, Rate Change, Stopped...
        :param route: <str> The way the patient has taken the medication.
        :param wt_based_dose: <str> Indicates whether the dose for this
                             medication order is based on the patient's weight.
        :param source: <str> Database from which the signal is extracted.
                      For this class is always 'EDW_med'.
        """
        super().__init__(name, source, start_date)
        self.dose = dose
        self.units = units
        self.action = action
        self.route = route
        self.wt_based_dose = wt_based_dose

    @property
    def _source_type(self):
        return self.source.replace("EDW_", "")


class Event(ICUDiscreteData):
    """
    Implementation of ICUDiscreteData for EDW events.

    One instance holds a list of event of the same type.
    """

    def __init__(
        self,
        name: str,
        start_date: np.ndarray,
    ):
        """
        Init an Event object.

        :param name: <str> Event name.
        :param start_date: <np.ndarray> Array of Unix timestamps corresponding
                          to the start of each captured discrete event.
        :param source: <str> Database from which the signal is extracted.
        """
        super().__init__(name, EDW_FILES["events_file"]["source"], start_date)

    @property
    def _source_type(self):
        return self.source.replace("EDW_", "")


class Procedure(ICUDiscreteData):
    """
    Implementation of ICUDiscreteData for EDW procedures.

    One instance holds a list of procedures of the same type (Surgery,
    dialysis or transfusion procedures)
    """

    def __init__(
        self,
        name: str,
        source: str,
        start_date: np.ndarray,
        end_date: np.ndarray,
    ):
        """
        Init an Event object.

        :param name: <str> Procedure name.
        :param start_date: <np.ndarray> Array of Unix timestamps corresponding
                          to the start of each captured discrete event.
        :param end_date: <np.ndarray> Array of timestamps corresponding
                        to the end of each captured discrete event.
        :param source: <str> Database from which the signal is extracted.
        """
        super().__init__(name, source, start_date)
        self.end_date = end_date

    @property
    def _source_type(self):
        return self.source.replace("EDW_", "")


EDWType = (StaticData, Measurement, Procedure, Event, Medication)
BedmasterType = (BedmasterSignal, BedmasterAlarm)
