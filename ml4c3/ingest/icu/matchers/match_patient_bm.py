# Imports: standard library
import os
import re
import logging
from typing import Any, Dict, List, Tuple

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from ml4c3.definitions.icu import EDW_FILES, MAPPING_DEPARTMENTS

# pylint: disable=too-many-branches


class PatientBMMatcher:
    """
    Implementation of Patient Bedmaster matching algorithm.
    """

    def __init__(
        self,
        flag_lm4: bool,
        bm_dir: str,
        edw_dir: str,
        des_depts: List[str] = None,
        adt_file: str = EDW_FILES["adt_file"]["name"],
    ):
        """
        Init Patient BM matcher.

        :param flag_lm4: <bool> If True, the matching is performed with all subfolders
                         in bm_dir. If False, matching is performed with all .mat files
                         in bm_dir.
        :param bm_dir: <str> Directory containing all the BM .mat data.
        :param edw_dir: <str> Directory containing the adt_file.
        :param des_depts: <List[str]> List of all desired departments.
        :param adt_file: <str> File containing the admission, transfer and
                         discharge from patients (.csv).
        """
        self.flag_lm4 = flag_lm4
        self.bm_dir = bm_dir
        self.edw_dir = edw_dir
        self.adt_file = adt_file
        if not des_depts:
            self.des_depts = list(MAPPING_DEPARTMENTS.keys())
        else:
            self.des_depts = des_depts

        self.dept_to_folder = MAPPING_DEPARTMENTS
        self.folder_to_dept = {}
        for key, values in MAPPING_DEPARTMENTS.items():
            for value in values:
                self.folder_to_dept.update({value: key})

        self.table_dic: Dict[str, List[Any]] = self._new_table_dic()

    @staticmethod
    def _new_table_dic():
        """
        Clear "self.table_dic" when initializing class and running match_files
        method.
        """
        return {
            "MRN": [],
            "PatientEncounterID": [],
            "transferIn": [],
            "transferOut": [],
            "fileID": [],
            "unixFileStartTime": [],
            "unixFileEndTime": [],
            "Department": [],
        }

    @staticmethod
    def _take_bmfile_info(bm_file: str) -> Tuple[str, ...]:
        """
        Take information from BM files name.

        :param bm_file: <str> bedmaster file name.
        :return: <Tuple[str]> array with all the info from the name. Element 0
                              is department, element 1 is room, element 2 is
                              starting time.
        """
        # Split BM file name into fields:
        # <department>_<roomBed>-<start_t>-#_v4.mat --> department, roomBed, start_time.
        split_info = re.split("[^a-zA-Z0-9]", bm_file)
        # If department name contains a hyphen, rejoin the name again:
        # ELL-7 --> ELL, 7 --> ELL-7
        if split_info[0].isalpha() and len(split_info[1]) == 1:
            split_info[0] = split_info[0] + "-" + split_info[1]
            del split_info[1]
        k = 2
        # If BM file name contains a "+" or "~" sign, take next field as it will contain
        # an empty one.
        if split_info[k] == "":
            k = 3
        return split_info[0], split_info[1], split_info[k]

    def match_files(self, xref_path: str = None, store_unmatched: bool = False):
        """
        Match BM files with patient's MRN and EncounterID.

        :param xref_path: <str> directory of the output xref table.
        :param store_unmatched: <bool> Bool indicating if the unmatched Bedmaster files
                                are saved in the xref table.
        """
        # Clear dictionary if it has any information
        if any(self.table_dic[key] for key in self.table_dic.keys()):
            self.table_dic = self._new_table_dic()

        # Read ADT table and take desired columns, remove duplicates and sort
        adt_df = pd.read_csv(os.path.join(self.edw_dir, self.adt_file))
        adt_df = (
            adt_df[EDW_FILES["adt_file"]["columns"]]
            .sort_values("TransferInDTS")
            .drop_duplicates()
        )
        # Convert dates to Unix Time Stamps
        adt_df["TransferInDTS"] = get_unix_timestamps(adt_df["TransferInDTS"].values)
        adt_df["TransferOutDTS"] = get_unix_timestamps(adt_df["TransferOutDTS"].values)
        # Get departments from ADT:
        adt_departments = np.array(adt_df["DepartmentDSC"].drop_duplicates(), dtype=str)
        if len(set(adt_departments) - set(self.des_depts)) > 0:
            logging.info(
                "There are more departments in ADT table than the desired ones "
                "introduced in the mode argument. Be sure of not losing "
                "patients. ADT departments you did not consider: "
                f"{str(list(set(adt_departments)-set(self.des_depts)))[1:-1]}",
            )
        if len(set(self.des_depts) - set(adt_departments)) > 0:
            logging.info(
                "There are less departments in ADT table than the desired ones "
                "introduced in the mode argument. Be sure of not losing "
                "patients. Departments that not appear in ADT: "
                f"{str(list(set(self.des_depts)-set(adt_departments)))[1:-1]}",
            )

        # Check if the matching is either performed in arbitrary directory or in LM4
        if self.flag_lm4 and (
            not any(
                folder in self.folder_to_dept.keys()
                for folder in os.listdir(self.bm_dir)
            )
        ):
            raise OSError(
                "No BM subfolders in this LM4 directory. If you wanted to "
                "use a personal folder, do not set the --lm4 argument.",
            )
        if not self.flag_lm4 and (
            not any(file.endswith(".mat") for file in os.listdir(self.bm_dir))
        ):
            raise OSError(
                "No .mat files encountered in this directory. If you wanted "
                "to match patients with BM files in LM4, set the --lm4 "
                "argument.",
            )

        # Get names of all the files
        if self.flag_lm4:
            bm_files = []
            for dept in sorted(set(self.des_depts) & set(adt_departments)):
                try:
                    folders = self.dept_to_folder[dept]
                    for folder in folders:
                        if folder is None:
                            continue
                        try:
                            bm_files.extend(
                                sorted(os.listdir(os.path.join(self.bm_dir, folder))),
                            )
                        except FileNotFoundError:
                            logging.info(
                                f"Folder {folder} is not found in the current LM4 "
                                "directory.",
                            )
                except KeyError:
                    logging.warning(
                        f"Department {dept} is not found in MAPPING_DEPARTMENTS "
                        "(ml4c3/definitions.py). No matching will be performed with "
                        "patients from this department. Please, add this information "
                        "in MAPPING_DEPARTMENTS (ml4c3/definitions.py).",
                    )
        else:
            bm_files = sorted(os.listdir(self.bm_dir))
        if not all(file.endswith(".mat") for file in bm_files):
            logging.warning(
                "Not all listed files end with .mat extension. They are "
                "going to be removed from the list.",
            )
            bm_files = [file for file in bm_files if file.endswith(".mat")]

        # Iterate over bm_files and match them to the corresponding
        # patient in the ADT table
        unmatched_files = []
        for _, bm_file in enumerate(bm_files):
            dept, room_bed, start_time = self._take_bmfile_info(bm_file)
            if self.folder_to_dept[dept] in self.des_depts:
                # Set room and bed string properly, if it has no room already, add
                # letter A (default in ADT)
                if any(s.isalpha() for s in room_bed):
                    room_bed = room_bed[:-1] + " " + room_bed[-1]
                else:
                    room_bed = room_bed + " A"

                while len(room_bed) < 6:
                    room_bed = "0" + room_bed
                room_bed_nm = dept[0] + room_bed

                # Filter by department and room
                filt_adt_df = adt_df[adt_df["BedLabelNM"] == room_bed_nm]
                # Filter by transfer in and out times (note that one hour
                # margin is considered as BM files sometimes start a little
                # before than the transfer of the patient)
                filt_adt_df = filt_adt_df[
                    (filt_adt_df["TransferInDTS"] - 3600) <= float(start_time)
                ]
                filt_adt_df = filt_adt_df[
                    float(start_time) <= filt_adt_df["TransferOutDTS"]
                ]

                if len(filt_adt_df.index) > 0:
                    mrn = int(filt_adt_df["MRN"].values[0])
                    csn = int(filt_adt_df["PatientEncounterID"].values[0])
                    t_start = filt_adt_df["TransferInDTS"].values[0]
                    t_end = filt_adt_df["TransferOutDTS"].values[0]

                    self.table_dic["MRN"].append(mrn)
                    self.table_dic["PatientEncounterID"].append(csn)
                    self.table_dic["transferIn"].append(t_start)
                    self.table_dic["transferOut"].append(t_end)
                    self.table_dic["fileID"].append(bm_file[:-4])
                    self.table_dic["unixFileStartTime"].append(start_time)
                    self.table_dic["unixFileEndTime"].append(t_end)
                    self.table_dic["Department"].append(self.folder_to_dept[dept])
                elif store_unmatched:
                    self.table_dic["MRN"].append(None)
                    self.table_dic["PatientEncounterID"].append(None)
                    self.table_dic["transferIn"].append(None)
                    self.table_dic["transferOut"].append(None)
                    self.table_dic["fileID"].append(bm_file[:-4])
                    self.table_dic["unixFileStartTime"].append(start_time)
                    self.table_dic["unixFileEndTime"].append(t_end)
                    self.table_dic["Department"].append(self.folder_to_dept[dept])
                else:
                    unmatched_files.append(bm_file)

        if len(unmatched_files) > 0:
            logging.warning(
                f"Unmatched files: {sorted(unmatched_files)}. Those "
                "files couldn't be matched with patient's MRN and CSN.",
            )

        if xref_path:
            xref_table = pd.DataFrame(self.table_dic)
            xref_table.to_csv(xref_path, index=False)
