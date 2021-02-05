# Imports: standard library
import os
import re
import logging
from abc import ABC
from typing import Any, Set, Dict, List, Tuple, Optional
from datetime import datetime

# Imports: third party
import h5py
import numpy as np
import pandas as pd
import unidecode

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from definitions.edw import EDW_FILES, MED_ACTIONS
from definitions.icu import ALARMS_FILES, ICU_SCALE_UNITS
from definitions.globals import TIMEZONE
from tensorize.edw.data_objects import (
    Event,
    Procedure,
    Medication,
    StaticData,
    Measurement,
)
from tensorize.bedmaster.data_objects import BedmasterAlarm, BedmasterSignal
from tensorize.bedmaster.bedmaster_stats import BedmasterStats
from tensorize.bedmaster.match_patient_bedmaster import PatientBedmasterMatcher

# pylint: disable=too-many-branches, dangerous-default-value


class Reader(ABC):
    """
    Parent class for our Readers class.

    As an abstract class, it can't be directly instanced. Its children
    should be used instead.
    """

    @staticmethod
    def _ensure_contiguous(data: np.ndarray) -> np.ndarray:
        if len(data) > 0:
            dtype = Any
            try:
                data = data.astype(float)
                if all(x.is_integer() for x in data):
                    dtype = int
                else:
                    dtype = float
            except ValueError:
                dtype = "S"
            try:
                data = np.ascontiguousarray(data, dtype=dtype)
            except (UnicodeEncodeError, SystemError):
                logging.info("Unknown character. Not ensuring contiguous array.")
                new_data = []
                for element in data:
                    new_data.append(unidecode.unidecode(str(element)))
                data = np.ascontiguousarray(new_data, dtype="S")
            except ValueError:
                logging.exception(
                    f"Unknown method to convert np.ndarray of "
                    f"{dtype} objects to numpy contiguous type.",
                )
                raise
        return data


class EDWReader(Reader):
    """
    Implementation of the Reader for EDW data.

    Usage:
    >>> reader = EDWReader('MRN')
    >>> hr = reader.get_measurement('HR')
    """

    def __init__(
        self,
        path: str,
        mrn: str,
        csn: str,
        med_file: str = EDW_FILES["med_file"]["name"],
        move_file: str = EDW_FILES["move_file"]["name"],
        adm_file: str = EDW_FILES["adm_file"]["name"],
        demo_file: str = EDW_FILES["demo_file"]["name"],
        vitals_file: str = EDW_FILES["vitals_file"]["name"],
        lab_file: str = EDW_FILES["lab_file"]["name"],
        surgery_file: str = EDW_FILES["surgery_file"]["name"],
        other_procedures_file: str = EDW_FILES["other_procedures_file"]["name"],
        transfusions_file: str = EDW_FILES["transfusions_file"]["name"],
        events_file: str = EDW_FILES["events_file"]["name"],
        medhist_file: str = EDW_FILES["medhist_file"]["name"],
        surghist_file: str = EDW_FILES["surghist_file"]["name"],
        socialhist_file: str = EDW_FILES["socialhist_file"]["name"],
    ):
        """
        Init EDW Reader.

        :param path: absolute path of files.
        :param mrn: MRN of the patient.
        :param csn: CSN of the patient visit.
        :param med_file: file containing the medicines data from the patient.
                        Can be inferred if None.
        :param move_file: file containing the movements of the patient
                        (admission, transfer and discharge) from the patient.
                        Can be inferred if None.
        :param demo_file: file containing the demographic data from
                        the patient. Can be inferred if None.
        :param vitals_file: file containing the vital signals from
                        the patient. Can be inferred if None.
        :param lab_file: file containing the laboratory signals from
                        the patient. Can be inferred if None.
        :param adm_file: file containing the admission data from
                        the patient. Can be inferred if None.
        :param surgery_file: file containing the surgeries performed to
                        the patient. Can be inferred if None.
        :param other_procedures_file: file containing procedures performed to
                        the patient. Can be inferred if None.
        :param transfusions_file: file containing the transfusions performed to
                        the patient. Can be inferred if None.
        :param eventss_file: file containing the events during
                        the patient stay. Can be inferred if None.
        :param medhist_file: file containing the medical history information of the
                        patient. Can be inferred if None.
        :param surghist_file: file containing the surgical history information of the
                        patient. Can be inferred if None.
        :param socialhist_file: file containing the social history information of the
                        patient. Can be inferred if None.
        """
        self.path = path

        self.mrn = mrn
        self.csn = csn

        self.move_file = self.infer_full_path(move_file)
        self.demo_file = self.infer_full_path(demo_file)
        self.vitals_file = self.infer_full_path(vitals_file)
        self.lab_file = self.infer_full_path(lab_file)
        self.med_file = self.infer_full_path(med_file)
        self.adm_file = self.infer_full_path(adm_file)
        self.surgery_file = self.infer_full_path(surgery_file)
        self.other_procedures_file = self.infer_full_path(other_procedures_file)
        self.transfusions_file = self.infer_full_path(transfusions_file)
        self.events_file = self.infer_full_path(events_file)
        self.medhist_file = self.infer_full_path(medhist_file)
        self.surghist_file = self.infer_full_path(surghist_file)
        self.socialhist_file = self.infer_full_path(socialhist_file)

        self.timezone = TIMEZONE

    def infer_full_path(self, file_name: str) -> str:
        """
        Infer a file name from MRN and type of data.

        Used if a file is not specified on the input.

        :param file_name: <str> 8 possible options:
                           'medications.csv', 'demographics.csv', 'labs.csv',
                           'flowsheet.scv', 'admission-vitals.csv',
                           'surgery.csv','procedures.csv', 'transfusions.csv'
        :return: <str> the inferred path
        """
        if not file_name.endswith(".csv"):
            file_name = f"{file_name}.csv"

        full_path = os.path.join(self.path, self.mrn, self.csn, file_name)
        return full_path

    def list_vitals(self) -> List[str]:
        """
        List all the vital signs taken from the patient.

        :return: <List[str]>  List with all the available vital signals
                from the patient
        """
        signal_column = EDW_FILES["vitals_file"]["columns"][0]
        vitals_df = pd.read_csv(self.vitals_file)

        # Remove measurements out of dates
        time_column = EDW_FILES["vitals_file"]["columns"][3]
        admit_column = EDW_FILES["adm_file"]["columns"][3]
        discharge_column = EDW_FILES["adm_file"]["columns"][4]
        admission_df = pd.read_csv(self.adm_file)
        init_date = admission_df[admit_column].values[0]
        end_date = admission_df[discharge_column].values[0]

        vitals_df = vitals_df[vitals_df[time_column] >= init_date]
        if str(end_date) != "nan":
            vitals_df = vitals_df[vitals_df[time_column] <= end_date]

        return list(vitals_df[signal_column].astype("str").str.upper().unique())

    def list_labs(self) -> List[str]:
        """
        List all the lab measurements taken from the patient.

        :return: <List[str]>  List with all the available lab measurements
                from the patient.
        """
        signal_column = EDW_FILES["lab_file"]["columns"][0]
        labs_df = pd.read_csv(self.lab_file)
        return list(labs_df[signal_column].astype("str").str.upper().unique())

    def list_medications(self) -> List[str]:
        """
        List all the medications given to the patient.

        :return: <List[str]>  List with all the medications on
                the patients record
        """
        signal_column = EDW_FILES["med_file"]["columns"][0]
        status_column = EDW_FILES["med_file"]["columns"][1]
        med_df = pd.read_csv(self.med_file)
        med_df = med_df[med_df[status_column].isin(MED_ACTIONS)]
        return list(med_df[signal_column].astype("str").str.upper().unique())

    def list_surgery(self) -> List[str]:
        """
        List all the types of surgery performed to the patient.

        :return: <List[str]>  List with all the event types associated
                with the patient
        """
        return self._list_procedures(self.surgery_file, "surgery_file")

    def list_other_procedures(self) -> List[str]:
        """
        List all the types of procedures performed to the patient.

        :return: <List[str]>  List with all the event types associated
                with the patient
        """
        return self._list_procedures(
            self.other_procedures_file,
            "other_procedures_file",
        )

    def list_transfusions(self) -> List[str]:
        """
        List all the transfusions types that have been done on the patient.

        :return: <List[str]>  List with all the transfusions type of
                the patient
        """
        return self._list_procedures(self.transfusions_file, "transfusions_file")

    @staticmethod
    def _list_procedures(file_name, file_key) -> List[str]:
        """
        Filter and list all the procedures in the given file.
        """
        signal_column, status_column, start_column, end_column = EDW_FILES[file_key][
            "columns"
        ]
        data = pd.read_csv(file_name)
        data = data[data[status_column].isin(["Complete", "Completed"])]
        data = data.dropna(subset=[start_column, end_column])
        return list(data[signal_column].astype("str").str.upper().unique())

    def list_events(self) -> List[str]:
        """
        List all the event types during the patient stay.

        :return: <List[str]>  List with all the events type.
        """
        signal_column, _ = EDW_FILES["events_file"]["columns"]
        data = pd.read_csv(self.events_file)
        return list(data[signal_column].astype("str").str.upper().unique())

    def get_static_data(self) -> StaticData:
        """
        Get the static data from the EDW csv file (admission + demographics).

        :return: <StaticData> wrapped information
        """
        movement_df = pd.read_csv(self.move_file)
        admission_df = pd.read_csv(self.adm_file)
        demographics_df = pd.read_csv(self.demo_file)

        # Obtain patient's movement (location and when they move)
        department_id = np.array(movement_df["DepartmentID"], dtype=int)
        department_nm = np.array(movement_df["DepartmentDSC"], dtype="S")
        room_bed = np.array(movement_df["BedLabelNM"], dtype="S")
        move_time = np.array(movement_df["TransferInDTS"], dtype="S")

        # Convert weight from ounces to pounds
        weight = float(admission_df["WeightPoundNBR"].values[0]) / 16

        # Convert height from feet & inches to meters
        height = self._convert_height(admission_df["HeightTXT"].values[0])

        admin_type = admission_df["HospitalAdmitTypeDSC"].values[0]

        # Find possible diagnosis at admission
        diag_info = admission_df["AdmitDiagnosisTXT"].dropna().drop_duplicates()
        if list(diag_info):
            diag_info = diag_info.astype("str")
            admin_diag = diag_info.str.cat(sep="; ")
        else:
            admin_diag = "UNKNOWN"

        admin_date = admission_df["HospitalAdmitDTS"].values[0]
        birth_date = demographics_df["BirthDTS"].values[0]
        race = demographics_df["PatientRaceDSC"].values[0]
        sex = demographics_df["SexDSC"].values[0]
        end_date = admission_df["HospitalDischargeDTS"].values[0]

        # Check whether it exists a deceased date or not
        end_stay_type = (
            "Alive"
            if str(demographics_df["DeathDTS"].values[0]) == "nan"
            else "Deceased"
        )

        # Find local time, if patient is still in hospital, take today's date
        if str(end_date) != "nan":
            offsets = self._get_local_time(admin_date[:-1], end_date[:-1])
        else:
            today_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S.%f")
            offsets = self._get_local_time(admin_date[:-1], today_date)

        offsets = list(set(offsets))  # Take unique local times
        local_time = np.empty(0)
        for offset in offsets:
            local_time = np.append(local_time, f"UTC{int(offset/3600)}:00")
        local_time = local_time.astype("S")

        # Find medical, surgical and social history of patient
        medical_hist = self._get_med_surg_hist("medhist_file")
        surgical_hist = self._get_med_surg_hist("surghist_file")
        tobacco_hist, alcohol_hist = self._get_social_hist()

        return StaticData(
            department_id,
            department_nm,
            room_bed,
            move_time,
            weight,
            height,
            admin_type,
            admin_diag,
            admin_date,
            birth_date,
            race,
            sex,
            end_date,
            end_stay_type,
            local_time,
            medical_hist,
            surgical_hist,
            tobacco_hist,
            alcohol_hist,
        )

    def get_med_doses(self, med_name: str) -> Medication:
        """
        Get all the doses of the input medication given to the patient.

        :param medication_name: <string> name of the medicine
        :return: <Medication> wrapped list of medications doses
        """
        (
            signal_column,
            status_column,
            time_column,
            route_column,
            weight_column,
            dose_column,
            dose_unit_column,
            infusion_column,
            infusion_unit_column,
            duration_column,
            duration_unit_column,
        ) = EDW_FILES["med_file"]["columns"]
        source = EDW_FILES["med_file"]["source"]

        med_df = pd.read_csv(self.med_file)
        med_df = med_df[med_df[status_column].isin(MED_ACTIONS)]
        med_df = med_df.sort_values(time_column)

        if med_name not in med_df[signal_column].astype("str").str.upper().unique():
            raise ValueError(f"{med_name} was not found in {self.med_file}.")

        idx = np.where(med_df[signal_column].astype("str").str.upper() == med_name)[0]
        route = np.array(med_df[route_column])[idx[0]]
        wt_base_dose = (
            bool(1) if np.array(med_df[weight_column])[idx[0]] == "Y" else bool(0)
        )

        if med_df[duration_column].isnull().values[idx[0]]:
            start_date = self._get_unix_timestamps(np.array(med_df[time_column])[idx])
            action = np.array(med_df[status_column], dtype="S")[idx]
            if (
                np.array(med_df[status_column])[idx[0]] in [MED_ACTIONS[0]]
                or med_df[infusion_column].isnull().values[idx[0]]
            ):
                dose = np.array(med_df[dose_column], dtype="S")[idx]
                units = np.array(med_df[dose_unit_column])[idx[0]]
            else:
                dose = np.array(med_df[infusion_column])[idx]
                units = np.array(med_df[infusion_unit_column])[idx[0]]
        else:
            dose = np.array([])
            units = np.array(med_df[infusion_unit_column])[idx[0]]
            start_date = np.array([])
            action = np.array([])
            for _, row in med_df.iloc[idx, :].iterrows():
                dose = np.append(dose, [row[infusion_column], 0])
                time = self._get_unix_timestamps(np.array([row[time_column]]))[0]
                conversion = 1
                if row[duration_unit_column] == "Seconds":
                    conversion = 1
                elif row[duration_unit_column] == "Minutes":
                    conversion = 60
                elif row[duration_unit_column] == "Hours":
                    conversion = 3600
                start_date = np.append(
                    start_date,
                    [time, time + float(row[duration_column]) * conversion],
                )
                action = np.append(action, [row[status_column], "Stopped"])

        dose = self._ensure_contiguous(dose)
        start_date = self._ensure_contiguous(start_date)
        action = self._ensure_contiguous(action)

        return Medication(
            med_name,
            dose,
            units,
            start_date,
            action,
            route,
            wt_base_dose,
            source,
        )

    def get_vitals(self, vital_name: str) -> Measurement:
        """
        Get the vital signals from the EDW csv file 'flowsheet'.

        :param vital_name: <string> name of the signal
        :return: <Measurement> wrapped measurement signal
        """
        vitals_df = pd.read_csv(self.vitals_file)

        # Remove measurements out of dates
        time_column = EDW_FILES["vitals_file"]["columns"][3]
        admit_column = EDW_FILES["adm_file"]["columns"][3]
        discharge_column = EDW_FILES["adm_file"]["columns"][4]
        admission_df = pd.read_csv(self.adm_file)
        init_date = admission_df[admit_column].values[0]
        end_date = admission_df[discharge_column].values[0]
        vitals_df = vitals_df[vitals_df[time_column] >= init_date]
        if str(end_date) != "nan":
            vitals_df = vitals_df[vitals_df[time_column] <= end_date]

        return self._get_measurements(
            "vitals_file",
            vitals_df,
            vital_name,
            self.vitals_file,
        )

    def get_labs(self, lab_name: str) -> Measurement:
        """
        Get the lab measurement from the EDW csv file 'labs'.

        :param lab_name: <string> name of the signal
        :return: <Measurement> wrapped measurement signal
        """
        labs_df = pd.read_csv(self.lab_file)
        return self._get_measurements("lab_file", labs_df, lab_name, self.lab_file)

    def get_surgery(self, surgery_type: str) -> Procedure:
        """
        Get all the surgery information of the input type performed to the
        patient.

        :param surgery_type: <string> type of surgery
        :return: <Procedure> wrapped list surgeries of the input type
        """
        return self._get_procedures("surgery_file", self.surgery_file, surgery_type)

    def get_other_procedures(self, procedure_type: str) -> Procedure:
        """
        Get all the procedures of the input type performed to the patient.

        :param procedure: <string> type of procedure
        :return: <Procedure> wrapped list procedures of the input type
        """
        return self._get_procedures(
            "other_procedures_file",
            self.other_procedures_file,
            procedure_type,
        )

    def get_transfusions(self, transfusion_type: str) -> Procedure:
        """
        Get all the input transfusions type that were done to the patient.

        :param transfusion_type: <string> Type of transfusion.
        :return: <Procedure> Wrapped list of transfusions of the input type.
        """
        return self._get_procedures(
            "transfusions_file",
            self.transfusions_file,
            transfusion_type,
        )

    def get_events(self, event_type: str) -> Event:
        """
        Get all the input event type during the patient stay.

        :param event_type: <string> Type of event.
        :return: <Event> Wrapped list of events of the input type.
        """
        signal_column, time_column = EDW_FILES["events_file"]["columns"]

        data = pd.read_csv(self.events_file)
        data = data.dropna(subset=[time_column])
        data = data.sort_values([time_column])
        if event_type not in data[signal_column].astype("str").str.upper().unique():
            raise ValueError(f"{event_type} was not found in {self.events_file}.")

        idx = np.where(data[signal_column].astype("str").str.upper() == event_type)[0]
        time = self._get_unix_timestamps(np.array(data[time_column])[idx])
        time = self._ensure_contiguous(time)

        return Event(event_type, time)

    def _get_local_time(self, init_date: str, end_date: str) -> np.ndarray:
        """
        Obtain local time from init and end dates.

        :param init_date: <str> String with initial date.
        :param end_date: <str> String with end date.
        :return: <np.ndarray> List of offsets from UTC (it may be two in
                case the time shift between summer/winter occurs while the
                patient is in the hospital).
        """
        init_dt = datetime.strptime(init_date, "%Y-%m-%d %H:%M:%S.%f")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S.%f")
        offset_init = self.timezone.utcoffset(  # type: ignore
            init_dt,
            is_dst=True,
        ).total_seconds()
        offset_end = self.timezone.utcoffset(  # type: ignore
            end_dt,
            is_dst=True,
        ).total_seconds()
        return np.array([offset_init, offset_end], dtype=float)

    def _get_unix_timestamps(self, time_stamps: np.ndarray) -> np.ndarray:
        """
        Convert readable time stamps to unix time stamps.

        :param time_stamps: <np.ndarray> Array with all readable time stamps.
        :return: <np.ndarray> Array with Unix time stamps.
        """
        try:
            arr_timestamps = pd.to_datetime(time_stamps)
        except pd.errors.ParserError as error:
            raise ValueError("Array contains non datetime values.") from error

        # Convert readable local timestamps in local seconds timestamps
        local_timestamps = (
            np.array(arr_timestamps, dtype=np.datetime64)
            - np.datetime64("1970-01-01T00:00:00")
        ) / np.timedelta64(1, "s")

        # Find local time shift to UTC
        if not (pd.isnull(local_timestamps[0]) or pd.isnull(local_timestamps[-1])):
            offsets = self._get_local_time(time_stamps[0][:-1], time_stamps[-1][:-1])
        else:
            offsets = np.random.random(2)  # pylint: disable=no-member

        # Compute unix timestamp (2 different methods: 1st ~5000 times faster)
        if offsets[0] == offsets[1]:
            unix_timestamps = local_timestamps - offsets[0]
        else:
            unix_timestamps = np.empty(np.size(local_timestamps))
            for idx, val in enumerate(local_timestamps):
                if not pd.isnull(val):
                    ntarray = datetime.utcfromtimestamp(val)
                    offset = self.timezone.utcoffset(  # type: ignore
                        ntarray,
                        is_dst=True,
                    )
                    unix_timestamps[idx] = val - offset.total_seconds()  # type: ignore
                else:
                    unix_timestamps[idx] = val

        return unix_timestamps

    def _get_med_surg_hist(self, file_key: str) -> np.ndarray:
        """
        Read medical or surgical history table and its information as arrays.

        :param file_key: <str> Key name indicating the desired file name.
        :return: <Tuple> Tuple with tobacco and alcohol information.
        """
        if file_key == "medhist_file":
            hist_df = pd.read_csv(self.medhist_file)
        else:
            hist_df = pd.read_csv(self.surghist_file)

        hist_df = (
            hist_df[EDW_FILES[file_key]["columns"]].fillna("UNKNOWN").drop_duplicates()
        )

        info_hist = []
        for _, row in hist_df.iterrows():
            id_num, name, comment, date = row
            info_hist.append(
                f"ID: {id_num}; DESCRIPTION: {name}; "
                f"COMMENTS: {comment}; DATE: {date}",
            )

        return self._ensure_contiguous(np.array(info_hist))

    def _get_social_hist(self) -> Tuple:
        """
        Read social history table and return tobacco and alcohol patient
        status.

        :return: <Tuple> Tuple with tobacco and alcohol information.
        """
        hist_df = pd.read_csv(self.socialhist_file)
        hist_df = hist_df[EDW_FILES["socialhist_file"]["columns"]].drop_duplicates()

        concat = []
        for col in hist_df:
            information = hist_df[col].drop_duplicates().dropna()
            if list(information):
                information = information.astype(str)
                concat.append(information.str.cat(sep=" - "))
            else:
                concat.append("NONE")

        tobacco_hist = f"STATUS: {concat[0]}; COMMENTS: {concat[1]}"
        alcohol_hist = f"STATUS: {concat[2]}; COMMENTS: {concat[3]}"

        return tobacco_hist, alcohol_hist

    def _get_measurements(self, file_key: str, data, measure_name: str, file_name: str):
        (
            signal_column,
            result_column,
            units_column,
            time_column,
            additional_columns,
        ) = EDW_FILES[file_key]["columns"]
        source = EDW_FILES[file_key]["source"]

        # Drop repeated values and sort
        data = data[
            [signal_column, result_column, units_column, time_column]
            + additional_columns
        ].drop_duplicates()
        data = data.sort_values(time_column)

        if measure_name not in data[signal_column].astype("str").str.upper().unique():
            raise ValueError(f"{measure_name} was not found in {file_name}.")

        idx = np.where(data[signal_column].astype("str").str.upper() == measure_name)[0]
        value = np.array(data[result_column])[idx]
        time = self._get_unix_timestamps(np.array(data[time_column])[idx])
        units = np.array(data[units_column])[idx[0]]
        value = self._ensure_contiguous(value)
        time = self._ensure_contiguous(time)
        data_type = "Numerical"

        additional_data = {}
        for col in additional_columns:
            col_data = np.array(data[col])[idx]
            if "DTS" in col:
                col_data = self._get_unix_timestamps(col_data)
            col_data = self._ensure_contiguous(col_data)
            additional_data[col] = col_data

        return Measurement(
            measure_name,
            source,
            value,
            time,
            units,
            data_type,
            additional_data,
        )

    def _get_procedures(
        self,
        file_key: str,
        file_name: str,
        procedure_type: str,
    ) -> Procedure:

        signal_column, status_column, start_column, end_column = EDW_FILES[file_key][
            "columns"
        ]
        source = EDW_FILES[file_key]["source"]

        data = pd.read_csv(file_name)
        data = data[data[status_column].isin(["Complete", "Completed"])]

        data = data.dropna(subset=[start_column, end_column])
        data = data.sort_values([start_column, end_column])

        if procedure_type not in data[signal_column].astype("str").str.upper().unique():
            raise ValueError(f"{procedure_type} was not found in {file_name}.")

        idx = np.where(data[signal_column].astype("str").str.upper() == procedure_type)[
            0
        ]
        start_date = self._get_unix_timestamps(np.array(data[start_column])[idx])
        end_date = self._get_unix_timestamps(np.array(data[end_column])[idx])
        start_date = self._ensure_contiguous(start_date)
        end_date = self._ensure_contiguous(end_date)

        return Procedure(procedure_type, source, start_date, end_date)

    @staticmethod
    def _convert_height(height_i):
        if str(height_i) != "nan":
            height_i = height_i[:-1].split("' ")
            height_f = float(height_i[0]) * 0.3048 + float(height_i[1]) * 0.0254
        else:
            height_f = np.nan
        return height_f


class BedmasterReader(h5py.File, Reader):
    """
    Implementation of the Reader for Bedmaster data.

    Usage:
    >>> reader = BedmasterReader('file.mat')
    >>> hr = reader.get_vs('HR')
    """

    def __init__(
        self,
        file: str,
        scaling_and_units: Dict[str, Dict[str, Any]] = ICU_SCALE_UNITS,
        summary_stats: BedmasterStats = None,
    ):
        super().__init__(file, "r")
        self.max_segment = {
            "vs": {"segmentNo": 0, "maxTime": -1, "signalName": ""},
            "wv": {"segmentNo": 0, "maxTime": -1, "signalName": ""},
        }
        self.interbundle_corr: Dict[str, Optional[Dict]] = {
            "vs": None,
            "wv": None,
        }
        self.scaling_and_units: Dict[str, Dict[str, Any]] = scaling_and_units
        self.summary_stats = summary_stats
        if self.summary_stats:
            self.summary_stats.add_file_stats("total_files")

    def _update_max_segment(self, sig_name, sig_type, max_time):
        """
        Update the signal that holds the segment with the last timespan.

        Needed for inter-bundle correction.

        :param sig_name: <str> name of the new candidate signal
        :param sig_type: <str> wv or vs
        :param max_time: <int> latest timespan for that signal
        """
        packet = self["vs_packet"] if sig_type == "vs" else self["wv_time_original"]
        max_seg = self.max_segment[sig_type]
        max_seg["maxTime"] = max_time
        max_seg["segmentNo"] = packet[sig_name]["SegmentNo"][-1][0]
        max_seg["signalName"] = sig_name

    def get_interbundle_correction(self, previous_max):
        """
        Calculate interbundle correction parameters from previous bundle maxs.

        Based on the signal with maximum time from the previous bundle,
        it calculates the 'maxTime': the last timespan that is overlapped
        with the previous bundle, and 'timeCorr': the time shifting to be
        applied on this bundle.

        Parameters are stored on attribute 'interbundle_corr'.

        :param previous_max: <Dict> dict with the max timepans info from
            the previous bundle. Same format than 'max_sement' attribute.
        """

        def _ib_corr(previous_max, segments, time):
            ib_corr = None
            overlap_idx = np.where(segments[()] == previous_max["segmentNo"])[0]
            if overlap_idx.size > 0:  # Bundles overlap
                last_overlap_idx = overlap_idx[-1]

                if last_overlap_idx >= len(time):
                    last_overlap_idx = len(time) - 1

                last_overlap_time = time[last_overlap_idx][0]
                time_corr = previous_max["maxTime"] - last_overlap_time

                ib_corr = {"maxTime": last_overlap_time, "timeCorr": time_corr}
            return ib_corr

        vs_corr = None
        last_max_vs = previous_max["vs"]["signalName"]
        if self.contains_group("vs"):
            if last_max_vs in self["vs"].keys():
                vs_corr = _ib_corr(
                    previous_max=previous_max["vs"],
                    segments=self["vs_packet"][last_max_vs]["SegmentNo"],
                    time=self["vs_time_corrected"][last_max_vs]["res_vs"],
                )

        wv_corr = None
        last_max_wv = previous_max["wv"]["signalName"]
        if self.contains_group("wv"):
            if last_max_wv in self["wv"].keys():
                wv_corr = _ib_corr(
                    previous_max=previous_max["wv"],
                    segments=self["wv_time_original"][last_max_wv]["SegmentNo"],
                    time=self["wv_time_corrected"][last_max_wv]["res_wv"],
                )
        self.max_segment = previous_max
        self.interbundle_corr["vs"] = vs_corr
        self.interbundle_corr["wv"] = wv_corr

    def apply_ibcorr(self, signal: BedmasterSignal):
        """
        Apply inter-bundle correction on a given signal.

        The correction will be applied based on the 'interbundle_corr'
        attribute, which needs is updated using the method:
        'get_interbundle_correction'

        The correction will cut the overlapping values between this bundle
        and the previous one. In addition, it will shift the timespans so that
        the first timespan on this bundle is the continuation of the last
        timespan of the previouz value.

        Note that this shifting will occur until a dataevent 1 or 5 is found.

        :param signal: <BedmasterSignal> a Bedmaster signal.
        """
        source = "vs" if signal._source_type == "vitals" else "wv"
        if not self.interbundle_corr[source]:
            return

        overlap_idx = np.where(
            signal.time <= self.interbundle_corr[source]["maxTime"],  # type: ignore
        )[0]
        if overlap_idx.size > 0:
            first_non_ol_idx = overlap_idx[-1] + 1
            signal.time = signal.time[first_non_ol_idx:]
            signal.time_corr_arr = signal.time_corr_arr[first_non_ol_idx:]
            value_cut_idx = (
                first_non_ol_idx
                if source == "vs"
                else np.sum(signal.samples_per_ts[:first_non_ol_idx])
            )
            signal.value = signal.value[value_cut_idx:]
            signal.samples_per_ts = signal.samples_per_ts[first_non_ol_idx:]
            if signal.source == "waveform":
                signal.sample_freq = self.get_sample_freq_from_channel(
                    channel=signal.channel,
                    first_idx=first_non_ol_idx,
                )

        corr_to_apply = self.interbundle_corr[source]["timeCorr"]  # type: ignore
        if corr_to_apply:
            de_idx = np.where(signal.time_corr_arr == 1)[0]
            if de_idx.size > 0:  # Contains data events
                first_event = de_idx[0]
                signal.time[:first_event] = signal.time[:first_event] + corr_to_apply
            else:
                signal.time = signal.time + corr_to_apply

        if self.summary_stats and overlap_idx.size > 0:
            if signal.value.size > 0:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "overlapped_points",
                    first_non_ol_idx,
                    source=signal.source,
                )

    def contains_group(self, group_name: str) -> bool:
        """
        Check if the .mat file contains the given group.
        """
        has_group = False
        if group_name in self.keys():
            if isinstance(self[group_name], h5py.Group):
                has_group = True
        return has_group

    def list_vs(self) -> List[str]:
        """
        Get the JUST the names of vital signals contained on the .mat file.

        It doesn't return the value of the vital signs.
        :return: <list[str]> A list with the vital signals' names contained
                on the .mat file
        """
        if not self.contains_group("vs"):
            logging.warning(f"No BM vitalsign found on file {self.filename}.")
            if self.summary_stats:
                self.summary_stats.add_file_stats("missing_vs")
            return []
        return list(self["vs"].keys())

    def list_wv(self) -> Dict[str, str]:
        """
        Get the the names of waveform signals contained on the .mat file.

        The format is : {wv_name: channel}, where `channel` is the input
        channel where the the signal enters. If a channel contains
        no waveform or contains multiple waveforms, it will be ignored.

        :return: <Dict[str:str]> A dict with the wave form signals
                contained on the .mat file, along with their input channel.
        """
        wv_signals: Dict[str, str] = {}

        if not self.contains_group("wv"):
            logging.warning(f"No BM waveform found on file {self.filename}.")
            if self.summary_stats:
                self.summary_stats.add_file_stats("missing_wv")
            return wv_signals

        for ch_name in self["wv"].keys():
            signal_name = self.get_wv_from_channel(ch_name)
            if signal_name:
                wv_signals[signal_name] = ch_name

        return wv_signals

    def format_data(self, data) -> np.ndarray:
        """
        Format multidimensional data into 1D arrays.

        :param data: <np.array> Data to be formatted
        :return: <np.array> formatted data
        """
        # Pseudo 1D data to 1D data
        if data.shape[1] == 1:  # Case [[0],[1]]
            data = np.transpose(data)
        if data.shape[0] == 1:  # Case [[0, 1]]
            data = data[0]

        # 2D data unicode encoded to 1D decoded
        if data.ndim == 2:
            if data.shape[0] < data.shape[1]:
                data = np.transpose(data)
            data = self.decode_data(data)

        return data

    @staticmethod
    def decode_data(data: np.ndarray) -> np.ndarray:
        """
        Decodes data stored as unicode identifiers and returns a 1D array.

        Example:
        >>> data  # 3D array with unicode codes for '0','.','2'
        array([[48, 46, 50],
               [48, 46, 50],
               [48, 46, 50],
               [48, 46, 50]])

        >>> BedmasterReader.decode_data(data)
        array([0.2, 0.2, 0.2, 0.2])

        :param data: <np.ndarray> Data to decode
        :return: <np.ndarray> decoded data
        """

        def _decode(row):
            row = "".join([chr(code) for code in row]).strip()
            if row in ("X", "None"):
                return np.nan
            return row

        data = np.apply_along_axis(_decode, 1, data)
        try:
            data = data.astype(float)
            if all(x.is_integer() for x in data):
                dtype = int  # type: ignore
            else:
                dtype = float  # type: ignore
        except ValueError:
            dtype = "S"  # type: ignore

        data = data.astype(dtype)
        return data

    def get_vs(self, signal_name: str) -> Optional[BedmasterSignal]:
        """
        Get the corrected vs signal from the.mat file.

        2. Applies corrections on the signal
        3. Wraps the corrected signal and its metadata on a BedmasterDataObject

        :param signal_name: <string> name of the signal
        :return: <BedmasterSignal> wrapped corrected signal
        """
        if signal_name not in self["vs"].keys():
            raise ValueError(
                f"In bedmaster_file {self.filename}, the signal {signal_name} "
                "was not found.",
            )

        # Get values and time
        values = self["vs"][signal_name][()]

        if values.ndim == 2:
            values = self.format_data(values)

        if values.dtype.char == "S":
            logging.warning(
                f"{signal_name} on .mat file  {self.filename}, has unexpected "
                "string values.",
            )
            return None

        if values.ndim >= 2:
            raise ValueError(
                f"Signal {signal_name} on file: {self.filename}. The values"
                f"of the signal have higher dimension than expected (>1) after"
                f"being formatted. The signal is probably in a bad format so it "
                f"won't be written.",
            )

        time = np.transpose(self["vs_time_corrected"][signal_name]["res_vs"][:])[0]

        # Get the occurrence of event 1 and 5
        de_1 = self["vs_time_corrected"][signal_name]["data_event_1"]
        de_5 = self["vs_time_corrected"][signal_name]["data_event_5"]
        events = (de_1[:] | de_5[:]).astype(np.bool)

        # Get scaling factor and units
        if signal_name in self.scaling_and_units:
            scaling_factor = self.scaling_and_units[signal_name]["scaling_factor"]
            units = self.scaling_and_units[signal_name]["units"]
        else:
            scaling_factor = 1
            units = "UNKNOWN"

        # Samples per timespan
        samples_per_ts = np.array([1] * len(time))

        signal = BedmasterSignal(
            name=signal_name,
            source="vitals",
            channel=signal_name,
            value=self._ensure_contiguous(values),
            time=self._ensure_contiguous(time),
            units=units,
            sample_freq=np.array([(0.5, 0)], dtype="float,int"),
            scale_factor=scaling_factor,
            time_corr_arr=events,
            samples_per_ts=self._ensure_contiguous(samples_per_ts),
        )

        # Apply inter-bundle correction
        if self.interbundle_corr["vs"]:
            self.apply_ibcorr(signal)

        if signal.time.size == 0:
            logging.info(
                f"Signal {signal} on .mat file {self.filename} doesn't contain new "
                f"information (only contains overlapped values from previous bundles). "
                f"It won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "total_overlap_bundles",
                    source=signal.source,
                )
            return None

        # Compress time_corr_arr
        signal.time_corr_arr = np.packbits(np.transpose(signal.time_corr_arr)[0])

        # Update the max segment time (for inter-bundle correction)
        max_time = time[-1]
        if max_time > self.max_segment["vs"]["maxTime"]:
            self._update_max_segment(signal_name, "vs", max_time)

        # Quality check on data
        if not signal.time.shape[0] == signal.value.shape[0]:
            logging.warning(
                f"Something went wrong with signal {signal.name} on file: "
                f"{self.filename}. Time vector doesn't have the same length than "
                f"values vector. The signal won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "defective_signal",
                    source=signal.source,
                )
            return None

        if not signal.samples_per_ts.shape[0] == signal.time.shape[0]:
            logging.warning(
                f"Something went wrong with signal {signal.name} on file: "
                f"{self.filename}. Time vector doesn't have the same length than "
                f"values vector. The signal won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "defective_signal",
                    source=signal.source,
                )
            return None

        if self.summary_stats:
            self.summary_stats.add_from_signal(signal)

        return signal

    def get_wv(
        self,
        channel_n: str,
        signal_name: str = None,
    ) -> Optional[BedmasterSignal]:
        """
        Get the corrected wv signal from the.mat file.

        1. Gets the signal and its metadata from the .mat file
        2. Applies corrections on the signal
        3. Wraps the corrected signal and its metadata on a BedmasterDataObject

        :param channel_n: <string> channel where the signal is
        :param signal_name: <string> name of the signal
        :return: <BedmasterSignal> wrapped corrected signal
        """
        if channel_n not in self["wv"].keys():
            raise ValueError(
                f"In bedmaster_file {self.filename}, the signal {channel_n} was "
                "not found.",
            )

        if not signal_name:
            signal_name = self.get_wv_from_channel(channel_n)
            if not signal_name:
                signal_name = "?"

        values = np.array(np.transpose(self["wv"][channel_n][:])[0])
        if values.ndim == 2:
            values = self.format_data(values)

        if values.ndim >= 2:
            raise ValueError(
                f"Something went wrong with signal {signal_name} "
                f"on file: {self.filename}. Dimension of values "
                f"formatted values is higher than expected (>1).",
            )

        time = np.transpose(self["wv_time_corrected"][channel_n]["res_wv"][:])[0]

        # Get scaling factor and units
        scaling_factor, units = self.get_scaling_and_units(channel_n, signal_name)

        # Get the occurrence of event 1 and 5
        de_1 = self["wv_time_corrected"][channel_n]["data_event_1"]
        de_5 = self["wv_time_corrected"][channel_n]["data_event_5"]
        time_reset_events = de_1[:] | de_5[:].astype(np.bool)

        # Get sample frequency
        sample_freq = self.get_sample_freq_from_channel(channel_n)

        # Get samples per timespan
        samples_per_ts = self["wv_time_original"][channel_n]["Samples"][()]
        if samples_per_ts.ndim == 2:
            samples_per_ts = self.format_data(samples_per_ts)

        signal = BedmasterSignal(
            name=signal_name,
            source="waveform",
            channel=channel_n,
            value=values[:],
            time=time[:],
            units=units,
            sample_freq=sample_freq,
            scale_factor=scaling_factor,
            time_corr_arr=time_reset_events,
            samples_per_ts=samples_per_ts,
        )

        # Apply inter-bundle correction
        if self.interbundle_corr["wv"]:
            self.apply_ibcorr(signal)

        if signal.time.size == 0:
            logging.info(
                f"In bedmaster_file {self.filename}, {signal} is completely "
                "overlapped, it won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "total_overlap_bundles",
                    source=signal.source,
                )
            return None

        # Add the rest of events and compress the array
        tc_len = len(signal.time_corr_arr)
        de_2 = self["wv_time_corrected"][channel_n]["data_event_2"]
        de_3 = self["wv_time_corrected"][channel_n]["data_event_3"]
        de_4 = self["wv_time_corrected"][channel_n]["data_event_4"]

        events = signal.time_corr_arr | de_2[-tc_len:] | de_3[-tc_len:] | de_4[-tc_len:]
        events = np.packbits(np.transpose(events)[0])
        signal.time_corr_arr = events

        # Update the max segment time (for inter-bundle correction)
        max_time = time[-1]
        if max_time > self.max_segment["wv"]["maxTime"]:
            self._update_max_segment(channel_n, "wv", max_time)

        # Quality check on data
        if not signal.time.shape[0] == signal.samples_per_ts.shape[0]:
            logging.warning(
                f"Something went wrong with signal: "
                f"{signal.name} on file: {self.filename}. "
                f"Time vector doesn't have the same length than "
                f"'samples_per_ts' vector. The signal won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "defective_signal",
                    source=signal.source,
                )
            return None

        if not signal.value.shape[0] == np.sum(signal.samples_per_ts):
            logging.warning(
                f"Something went wrong with signal: "
                f"{signal.name} on file: {self.filename} "
                f"'samples_per_ts' vector's sum isn't equal to "
                f"values vector's length. This seems an error on the primitive "
                f".stp file. The signal won't be written.",
            )
            if self.summary_stats:
                self.summary_stats.add_signal_stats(
                    signal.name,
                    "defective_signal",
                    source=signal.source,
                )
            return None

        if self.summary_stats:
            self.summary_stats.add_from_signal(signal)

        return signal

    def get_wv_from_channel(self, channel: str) -> Optional[str]:
        path = f"wv_time_original/{channel}/Label"
        length = self[path].shape[-1]
        if length < 10:
            signals = self[path][:]
        else:
            signals = self[path][..., range(0, length, length // 10)]
        signals = np.unique(signals.T, axis=0)
        signals = signals[(signals != 32) & (signals != 0)]

        if signals.ndim > 1:
            logging.warning(
                f"Channel {channel} on file {self.filename} "
                f"is a mix of different signals: {signals}. "
                f"This situation is not supported. "
                f"The channel will be ignored.",
            )
            if self.summary_stats:
                self.summary_stats.add_file_stats("multiple_label_signal")
            return None

        if signals.size == 0:
            logging.warning(
                f"The signal on channel {channel} on file {self.filename} "
                f"has no name. It is probably an empty signal or a badly "
                f"recorded one. It won't be written to the tensorized file.",
            )
            if self.summary_stats:
                self.summary_stats.add_file_stats("no_label_signal")
            return None

        name = "".join([chr(letter) for letter in signals])
        return name

    def get_sample_freq_from_channel(self, channel: str, first_idx=0):
        sf_arr = self["wv_time_original"][channel]["SampleRate"][first_idx:].T[0]
        if sf_arr.shape[0] <= 0:
            logging.info(
                f"The signal on channel {channel} on file {self.filename} has an "
                f"incorrect sample frequency format. Either it doesn't have sample "
                f"frequency or it has an incongruent one. Sample frequency will be set "
                f"to Nan for this signal.",
            )
            return np.array([(np.nan, 0)], dtype="float,int")
        changes = np.concatenate([[-1], np.where(sf_arr[:-1] != sf_arr[1:])[0]])
        return np.fromiter(
            ((sf_arr[index + 1], index + 1) for index in changes),
            dtype="float,int",
        )

    def get_scaling_and_units(self, channel_n, signal_name):
        if signal_name in self.scaling_and_units:
            scaling_factor = self.scaling_and_units[signal_name]["scaling_factor"]
            units = self.scaling_and_units[signal_name]["units"]
        else:
            try:
                calibration = self["wv_time_original"][channel_n]["Cal"][()]
                calibration = self.decode_data([calibration.T[0]])[0].decode("utf-8")
                calibration = [
                    part for part in re.split(r"(\d*\.?\d+)", calibration) if part
                ]
                if len(calibration) == 2:
                    scaling_factor, units = calibration
                else:
                    raise ValueError
            except (KeyError, ValueError):
                logging.warning(
                    f"Scaling factor or units not found "
                    f"for signal {signal_name} on file {self.filename}. They will "
                    f"be set to units: UNKNOWN, scaling_factor: 0.",
                )
                scaling_factor = 0
                units = "UNKNOWN"

        return float(scaling_factor), units


class BedmasterAlarmsReader(Reader):
    """
    Implementation of the Reader for Bedmaster Alarms data.
    """

    def __init__(
        self,
        alarms_path: str,
        edw_path: str,
        mrn: str,
        csn: str,
        adt: str,
        move_file: str = EDW_FILES["move_file"]["name"],
    ):
        """
        Iinit Bedmaster Alarms Reader.

        :param alarms_path: Absolute path of Bedmaster alarms directory.
        :param edw_path: Absolute path of edw directory.
        :param mrn: MRN of the patient.
        :param csn: CSN of the patient visit.
        :param adt: Path to adt table.
        :param move_file: File containing the movements of the patient
                          (admission, transfer and discharge) from the patient.
                          Can be inferred if None.
        """
        self.alarms_path = alarms_path
        self.edw_path = edw_path
        self.mrn = mrn
        self.csn = csn
        if not move_file.endswith(".csv"):
            move_file += ".csv"
        self.move_file = os.path.join(self.edw_path, self.mrn, self.csn, move_file)
        self.adt = adt
        self.alarms_dfs = self._get_alarms_dfs()

    def list_alarms(self) -> List[str]:
        """
        List all the Bedmaster alarms registered from the patient.

        :return: <List[str]>  List with all the registered Bedmaster alarms
                 from the patient
        """
        alarms: Set[str] = set()
        alarm_name_column = ALARMS_FILES["columns"][3]
        for alarms_df in self.alarms_dfs:
            alarms = alarms.union(
                set(alarms_df[alarm_name_column].astype("str").str.upper()),
            )
        return list(alarms)

    def get_alarm(self, alarm_name: str) -> BedmasterAlarm:
        """
        Get the Bedmaster alarms data from the Bedmaster Alarms .csv files.

        :return: <BedmasterAlarm> wrapped information.
        """
        dates = np.array([])
        durations = np.array([])
        date_column, level_column, alarm_name_column, duration_column = ALARMS_FILES[
            "columns"
        ][1:5]
        first = True
        for alarm_df in self.alarms_dfs:
            idx = np.where(
                alarm_df[alarm_name_column].astype("str").str.upper() == alarm_name,
            )[0]
            dates = np.append(dates, np.array(alarm_df[date_column])[idx])
            durations = np.append(durations, np.array(alarm_df[duration_column])[idx])
            if len(idx) > 0 and first:
                first = False
                level = np.array(alarm_df[level_column])[idx[0]]
        dates = self._ensure_contiguous(dates)
        durations = self._ensure_contiguous(durations)
        return BedmasterAlarm(
            name=alarm_name,
            start_date=dates,
            duration=durations,
            level=level,
        )

    def _get_alarms_dfs(self) -> List[pd.core.frame.DataFrame]:
        """
        List all the Bedmaster alarms data frames containing data for the given
        patient.

        :return: <List[pd.core.frame.DataFrame]> List with all the Bedmaster alarms
                 data frames containing data for the given patient.
        """
        if os.path.isfile(self.move_file):
            movement_df = pd.read_csv(self.move_file)
        else:
            adt_df = pd.read_csv(self.adt)
            movement_df = adt_df[adt_df["MRN"].astype("str") == self.mrn]
            movement_df = movement_df[
                movement_df["PatientEncounterID"].astype("str") == self.csn
            ]

        department_nm = np.array(movement_df["DepartmentDSC"], dtype=str)
        room_bed = np.array(movement_df["BedLabelNM"], dtype=str)
        transfer_in = np.array(movement_df["TransferInDTS"], dtype=str)
        transfer_out = np.array(movement_df["TransferOutDTS"], dtype=str)
        alarms_dfs = []
        for i, dept in enumerate(department_nm):
            move_in = self._get_unix_timestamp(transfer_in[i])
            move_out = self._get_unix_timestamp(transfer_out[i])
            if dept in ALARMS_FILES["names"]:
                names = ALARMS_FILES["names"][dept]
            else:
                logging.warning(
                    f"Department {dept} is not found in ALARMS_FILES['names'] "
                    "in ml4c3/definitions.py. No alarms data will be searched for this "
                    "department. Please, add this information to "
                    "ALARMS_FILES['names'].",
                )
                continue
            bed = room_bed[i][-3:]
            if any(s.isalpha() for s in bed):
                bed = room_bed[i][-5:-2] + room_bed[i][-1]
            for csv_name in names:
                if not os.path.isfile(
                    os.path.join(self.alarms_path, f"bedmaster_alarms_{csv_name}.csv"),
                ):
                    continue
                alarms_df = pd.read_csv(
                    os.path.join(self.alarms_path, f"bedmaster_alarms_{csv_name}.csv"),
                    low_memory=False,
                )
                alarms_df = alarms_df[alarms_df["Bed"].astype(str) == bed]
                alarms_df = alarms_df[alarms_df["AlarmStartTime"] >= move_in]
                alarms_df = alarms_df[alarms_df["AlarmStartTime"] <= move_out]
                if len(alarms_df.index) > 0:
                    alarms_dfs.append(alarms_df)
        return alarms_dfs

    @staticmethod
    def _get_unix_timestamp(time_stamp_str: str) -> int:
        """
        Convert readable time stamps to unix time stamps.

        :param time_stamps: <str> String with readable time stamps.
        :return: <int> Integer Unix time stamp.
        """
        try:
            time_stamp = pd.to_datetime(time_stamp_str)
        except pd.errors.ParserError as error:
            raise ValueError("Array contains non datetime values.") from error
        # Convert readable local timestamps in local seconds timestamps
        local_timestamp = (
            np.array(time_stamp, dtype=np.datetime64)
            - np.datetime64("1970-01-01T00:00:00")
        ) / np.timedelta64(1, "s")
        # Find local time shift to UTC
        init_dt = datetime.strptime(time_stamp_str[:-1], "%Y-%m-%d %H:%M:%S.%f")
        offset = TIMEZONE.utcoffset(  # type: ignore
            init_dt,
            is_dst=True,
        ).total_seconds()
        unix_timestamp = int(local_timestamp) - int(offset)
        return unix_timestamp


class CrossReferencer:
    """
    Class that cross-references Bedmaster and EDW data.

    Used to ensure correspondence between the data.
    """

    def __init__(
        self,
        bedmaster_dir: str,
        edw_dir: str,
        xref_file: str,
        adt: str,
        bedmaster_index: str = None,
    ):
        self.bedmaster_dir = bedmaster_dir
        self.edw_dir = edw_dir
        self.xref_file = xref_file
        self.adt = adt
        self.bedmaster_index = bedmaster_index
        self.crossref: Dict[str, Dict[str, List[str]]] = {}

    def get_xref_files(
        self,
        mrns: List[str] = None,
        starting_time: int = None,
        ending_time: int = None,
        overwrite_hd5: bool = True,
        n_patients: int = None,
        tensors: str = None,
        allow_one_source: bool = False,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Get the cross-referenced Bedmaster files and EDW files.

        The output dictionary will have the format:

        {"MRN1": {"visitID":[bedmaster_files],
                  "visitID2": ...,
                  ...
                  }
         "MRN2: ...}

        :param mrns: <List[str]> list with the MRNs.
                    If None, it take all the existing ones
        :param starting_time: <int> starting time in Unix format.
                             If None, timestamps will be taken
                             from the first one.
        :param ending_time: <int> ending time in Unix format.
                            If None, timestamps will be taken
                            until the last one.
        :param overwrite_hd5: <bool> indicates if the mrns of the existing
                              hd5 files are excluded from the output dict.
        :param n_patients: <int> max number of patients to tensorize.
        :param tensors: <str> directory to check for existing hd5 files.
        :param allow_one_source: <bool> bool indicating whether a patient with
                                just one type of data will be tensorized or not.
        :return: <dict> a dictionary with the MRNs, visit ID and Bedmaster files.
        """
        self.crossref = {}
        if not os.path.exists(self.xref_file):
            if self.bedmaster_index is None or not os.path.exists(self.bedmaster_index):
                raise ValueError(
                    "No method to get xref table.  Specify a valid path to an existing "
                    "xref table or a bedmaster index table.",
                )
            bedmaster_matcher = PatientBedmasterMatcher(
                bedmaster=self.bedmaster_dir,
                adt=self.adt,
            )
            bedmaster_matcher.match_files(
                bedmaster_index=self.bedmaster_index,
                xref=self.xref_file,
            )

        adt_df = pd.read_csv(self.adt)
        adt_columns = EDW_FILES["adt_file"]["columns"]
        adt_df = adt_df[adt_columns].drop_duplicates()

        xref = pd.read_csv(self.xref_file)
        xref = xref.drop_duplicates(subset=["MRN", "PatientEncounterID", "Path"])
        xref["MRN"] = xref["MRN"].astype(str)

        edw_mrns = [
            folder
            for folder in os.listdir(self.edw_dir)
            if os.path.isdir(os.path.join(self.edw_dir, folder))
        ]

        if mrns:
            xref = xref[xref["MRN"].isin(mrns)]
            edw_mrns = [ele for ele in edw_mrns if ele in mrns]
        if starting_time:
            xref = xref[xref["TransferInDTS"] > starting_time]
            adt_df = adt_df[adt_df[adt_columns[0]].isin(edw_mrns)]
            adt_df[adt_columns[4]] = get_unix_timestamps(adt_df[adt_columns[4]].values)
            adt_df = adt_df[adt_df[adt_columns[4]] > starting_time]
            edw_mrns = list(adt_df[adt_columns[0]].drop_duplicates().astype(str))
        if ending_time:
            xref = xref[xref["TransferOutDTS"] < ending_time]
            adt_df = adt_df[adt_df[adt_columns[0]].isin(edw_mrns)]
            adt_df[adt_columns[3]] = get_unix_timestamps(adt_df[adt_columns[3]].values)
            adt_df = adt_df[adt_df[adt_columns[3]] < ending_time]
            edw_mrns = list(adt_df[adt_columns[0]].drop_duplicates().astype(str))

        if not overwrite_hd5 and tensors and os.path.isdir(tensors):
            existing_mrns = [
                hd5file[:-4]
                for hd5file in os.listdir(tensors)
                if hd5file.endswith(".hd5")
            ]
            xref = xref[~xref["MRN"].isin(existing_mrns)]
            edw_mrns = [ele for ele in edw_mrns if ele not in existing_mrns]
        elif not overwrite_hd5 and not tensors:
            logging.warning(
                "overwrite_hd5 is set to False, but output_dir option is "
                "not set, ignoring overwrite_hd5 option. HD5 files are "
                "going to be overwritten.",
            )
        self.add_bedmaster_elements(
            xref=xref,
            edw_mrns=edw_mrns,
            allow_one_source=allow_one_source,
        )
        if allow_one_source:
            self.add_edw_elements(edw_mrns)
        self.assess_coverage()
        self.stats()

        # Get only the first n patients
        if (n_patients or 0) > len(self.crossref):
            logging.warning(
                f"Number of patients set to tensorize "
                f"exceeds the amount of patients stored. "
                f"Number of patients to tensorize will be changed to "
                f"{len(self.crossref)}.",
            )
            return self.crossref
        return dict(list(self.crossref.items())[:n_patients])

    def add_bedmaster_elements(self, xref, edw_mrns, allow_one_source):
        # Add elements from xref.csv
        for _, row in xref.iterrows():
            mrn = str(row["MRN"])
            if not allow_one_source and mrn not in edw_mrns:
                continue
            try:
                csn = str(int(row["PatientEncounterID"]))
            except ValueError:
                csn = str(row["PatientEncounterID"])
            bedmaster_path = os.path.join(self.bedmaster_dir, row["Path"])
            if mrn not in self.crossref:
                self.crossref[mrn] = {csn: [bedmaster_path]}
            elif csn not in self.crossref[mrn]:
                self.crossref[mrn][csn] = [bedmaster_path]
            else:
                self.crossref[mrn][csn].append(bedmaster_path)

        for _mrn, visits in self.crossref.items():
            for _csn, bedmaster_files in visits.items():
                bedmaster_files.sort(
                    key=lambda x: (int(re.split("[_-]", x)[-3]), int(x.split("_")[-2])),
                )

    def add_edw_elements(self, edw_mrns):
        # Add elements from EDW folders
        for mrn in edw_mrns:
            csns = [
                csn
                for csn in os.listdir(os.path.join(self.edw_dir, mrn))
                if os.path.isdir(os.path.join(self.edw_dir, mrn, csn))
            ]
            if mrn not in self.crossref:
                self.crossref[mrn] = {}
                for csn in csns:
                    self.crossref[mrn][csn] = []
            else:
                for csn in csns:
                    if csn not in self.crossref[mrn]:
                        self.crossref[mrn][csn] = []

    def assess_coverage(self):
        for mrn in self.crossref:
            for csn in self.crossref[mrn]:
                # Check if there exist Bedmaster data for this mrn-csn pair
                if not self.crossref[mrn][csn]:
                    logging.warning(f"No Bedmaster data for MRN: {mrn}, CSN: {csn}.")
                # Check if there exist EDW data for this mrn-csn pair
                edw_file_path = os.path.join(self.edw_dir, mrn, csn)
                if (
                    not os.path.isdir(edw_file_path)
                    or len(os.listdir(edw_file_path)) == 0
                ):
                    logging.warning(f"No EDW data for MRN: {mrn}, CSN: {csn}.")

    def stats(self):
        """
        :param crossref: <dict> a dictionary with the MRNs, visit ID and Bedmaster
                         files.
        """
        edw_mrns_set = {
            mrn
            for mrn in os.listdir(self.edw_dir)
            if os.path.isdir(os.path.join(self.edw_dir, mrn))
        }
        edw_csns_set = {
            csn
            for mrn in edw_mrns_set
            for csn in os.listdir(os.path.join(self.edw_dir, mrn))
            if os.path.isdir(os.path.join(self.edw_dir, mrn, csn))
        }

        xref = pd.read_csv(self.xref_file, dtype=str)
        xref_mrns_set = set(xref["MRN"].unique())
        try:
            _xref_csns_set = np.array(
                list(xref["PatientEncounterID"].unique()),
                dtype=float,
            )
            xref_csns_set = set(_xref_csns_set.astype(int).astype(str))
        except ValueError:
            xref_csns_set = set(xref["PatientEncounterID"].unique())
        xref_bedmaster_files_set = set(xref["Path"].unique())

        crossref_mrns_set = set(self.crossref.keys())
        crossref_csns_set = set()
        crossref_bedmaster_files_set = set()
        for _, visits in self.crossref.items():
            for visit_id, bedmaster_files in visits.items():
                crossref_csns_set.add(visit_id)
                for bedmaster_file in bedmaster_files:
                    crossref_bedmaster_files_set.add(bedmaster_file)
        logging.info(
            f"MRNs in {self.edw_dir}: {len(edw_mrns_set)}\n"
            f"MRNs in {self.xref_file}: {len(xref_mrns_set)}\n"
            f"Union MRNs: {len(edw_mrns_set.intersection(xref_mrns_set))}\n"
            f"Intersect MRNs: {len(crossref_mrns_set)}\n"
            f"CSNs in {self.edw_dir}: {len(edw_csns_set)}\n"
            f"CSNs in {self.xref_file}: {len(xref_csns_set)}\n"
            f"Union CSNs: {len(edw_csns_set.intersection(xref_csns_set))}\n"
            f"Intersect CSNs: {len(crossref_csns_set)}\n"
            f"Bedmaster files IDs in {self.xref_file}: "
            f"{len(xref_bedmaster_files_set)}\n"
            f"Intersect Bedmaster files: {len(crossref_bedmaster_files_set)}\n",
        )
