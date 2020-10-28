# Imports: standard library
import os
from typing import List, Tuple
from datetime import datetime

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.definitions.icu import EDW_FILES, MED_ACTIONS
from ml4c3.definitions.globals import TIMEZONE
from ml4c3.ingest.icu.data_objects import (
    Event,
    Procedure,
    Medication,
    StaticData,
    Measurement,
)

from .reader import Reader


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

        return list(vitals_df[signal_column].str.upper().unique())

    def list_labs(self) -> List[str]:
        """
        List all the lab measurements taken from the patient.

        :return: <List[str]>  List with all the available lab measurements
                from the patient
        """
        signal_column = EDW_FILES["lab_file"]["columns"][0]
        labs_df = pd.read_csv(self.lab_file)
        return list(labs_df[signal_column].str.upper().unique())

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
        return list(med_df[signal_column].str.upper().unique())

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
        return list(data[signal_column].str.upper().unique())

    def list_events(self) -> List[str]:
        """
        List all the event types during the patient stay.

        :return: <List[str]>  List with all the events type.
        """
        signal_column, _ = EDW_FILES["events_file"]["columns"]
        data = pd.read_csv(self.events_file)
        return list(data[signal_column].str.upper().unique())

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

        if med_name not in med_df[signal_column].str.upper().unique():
            raise ValueError(f"{med_name} was not found in {self.med_file}.")

        idx = np.where(med_df[signal_column].str.upper() == med_name)[0]
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
        if event_type not in data[signal_column].str.upper().unique():
            raise ValueError(f"{event_type} was not found in {self.events_file}.")

        idx = np.where(data[signal_column].str.upper() == event_type)[0]
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
        signal_column, result_column, units_column, time_column = EDW_FILES[file_key][
            "columns"
        ]
        source = EDW_FILES[file_key]["source"]

        # Drop repeated values and sort
        data = data[
            [signal_column, result_column, units_column, time_column]
        ].drop_duplicates()
        data = data.sort_values(time_column)

        if measure_name not in data[signal_column].str.upper().unique():
            raise ValueError(f"{measure_name} was not found in {file_name}.")

        idx = np.where(data[signal_column].str.upper() == measure_name)[0]
        value = np.array(data[result_column])[idx]
        time = self._get_unix_timestamps(np.array(data[time_column])[idx])
        units = np.array(data[units_column])[idx[0]]
        value = self._ensure_contiguous(value)
        time = self._ensure_contiguous(time)
        data_type = "Numerical"

        return Measurement(measure_name, source, value, time, units, data_type)

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

        if procedure_type not in data[signal_column].str.upper().unique():
            raise ValueError(f"{procedure_type} was not found in {file_name}.")

        idx = np.where(data[signal_column].str.upper() == procedure_type)[0]
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
