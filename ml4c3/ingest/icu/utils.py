# Imports: standard library
import os
import shutil
import warnings
import multiprocessing
from typing import Dict, List

# Imports: third party
import pandas as pd

# Imports: first party
from ml4c3.definitions.icu import MAPPING_DEPARTMENTS


class FileManager:
    """
    Class to find the desired EDW and BM Files in LM4 related with a list of
    patients (MRNs) and copy back the resulting tensorized files to MAD3.
    """

    def __init__(
        self,
        xref_file: str,
        adt_file: str,
        output_dir: str = "/data/icu",
    ):
        """
        Init File Manager.

        :param xref_file: <str> File containing the cross-referenced MRNs
                                and BM files.
        :param adt_file: <str> File containing the ADT table.
        :param output_dir: <str> Directory where the output files will be saved.
        """
        self.xref_file = xref_file
        self.adt_file = adt_file
        self.output_dir = output_dir
        self.mapping_folders: Dict[str, List[str]] = MAPPING_DEPARTMENTS

    def get_patients(
        self,
        init_patient: int = 0,
        last_patient: int = None,
        overwrite_hd5: bool = True,
        hd5_dir: str = None,
    ):
        """
        Get list of patients from xref_file.

        :param init_patient: <int> First index of the MRNs desired.
        :param last_patient: <int> Last index of the MRNs desired.
        :param overwrite_hd5: <bool> Bool to overwrite already tensorized hd5 files.
        :param hd5_dir: <str> Path where hd5 files are stored.
        """
        df_adt = pd.read_csv(self.adt_file).sort_values(by=["MRN"], ascending=True)
        patients = df_adt[["MRN", "PatientEncounterID"]].drop_duplicates().dropna()
        mrns = patients["MRN"].drop_duplicates()[init_patient:last_patient]
        desired_patients = patients[patients["MRN"].isin(mrns)]
        if not overwrite_hd5 and os.path.isdir(hd5_dir):
            hd5_mrns = [
                int(hd5_mrn.split(".")[0])
                for hd5_mrn in os.listdir(hd5_dir)
                if hd5_mrn.endswith(".hd5")
            ]
            desired_patients = desired_patients[~desired_patients["MRN"].isin(hd5_mrns)]
        desired_patients.to_csv(
            os.path.join(self.output_dir, "patient_list.csv"),
            index=False,
        )

    def find_save_bm_alarms(
        self,
        mad3_dir: str = "/media/mad3/bedmaster_alarms",
        patient_list: str = "/data/icu/patient_list.csv",
    ):
        """
        Find bm alarms and copy them to local folder.

        :param mad3_dir: <str> Path to MAD3 directory.
        :param patient_list: <str> List of desired patients to take their respective
                            BM alarms (.csv).
        """
        df_adt = pd.read_csv(self.adt_file).sort_values(by=["MRN"], ascending=True)
        df_patient = pd.read_csv(patient_list)
        mrns = df_patient["MRN"].drop_duplicates()
        df_adt_filt = df_adt[df_adt["MRN"].isin(mrns)]

        departments = df_adt_filt["DepartmentDSC"].drop_duplicates()
        flag_found = []
        dept_names = []
        for department in departments:
            try:
                short_names = self.mapping_folders[department]
            except KeyError:
                continue
            for short_name in short_names:
                dept_names.append(short_name)
                source_path = os.path.join(mad3_dir, f"bm_alarms_{short_name}.csv")
                destination_path = os.path.join(self.output_dir, "bm_alarms_temp")
                try:
                    shutil.copy(source_path, destination_path)
                    flag_found.append("FOUND")
                except FileNotFoundError:
                    flag_found.append("NOT FOUND")

        report = pd.DataFrame(
            {"Alarm from department": dept_names, "fileStatus": flag_found},
        )
        report.to_csv(
            os.path.join(self.output_dir, "bm_alarms_temp", "report.csv"),
            index=False,
        )

    def find_save_edw_files(
        self,
        mad3_dir: str = "/media/mad3/edw",
        patient_list: str = "/data/icu/patient_list.csv",
        parallelize: bool = False,
        n_workers: int = None,
    ):
        """
        Find edw files and copy them to local folder.

        :param mad3_dir: <str> Path to MAD3 directory.
        :param patient_list: <str> List of desired patients to take their respective
                            EDW files (.csv).
        :param parallelize: <bool> bool indicating whether the bm files copy
                            process is parallelized or not.
        :param n_workers: <int> Integer indicating the number of cores used in
                            the bm files copy process when parallelized.
        """
        df_patient = pd.read_csv(patient_list)
        mrns = list(df_patient["MRN"].drop_duplicates())

        list_mrns = []
        flag_found = []

        # Enable number of cores only if parallelized is enabled
        if not parallelize:
            n_workers = 1
        # Check number of workers used is not higher than the number of cpus
        if os.cpu_count():
            if (n_workers or 0) > os.cpu_count():  # type: ignore
                n_workers = os.cpu_count()
                warnings.warn(
                    f"Workers are higher than number of cpus."
                    f" Number of workers is reduced to {os.cpu_count()}, "
                    f"the number of cpus your computer have.",
                )
        else:
            warnings.warn(
                "Couldn't determine the number of cpus. Blindly "
                "accepting the n_workers option",
            )

        with multiprocessing.Pool(processes=n_workers) as pool:
            results_list = pool.starmap(
                self._copy_files_edw,
                [(mad3_dir, int(mrn)) for mrn in mrns],
            )

        # Finally copy adt and filtered xref table
        shutil.copy(
            os.path.join(self.adt_file),
            os.path.join(self.output_dir, "edw_temp", "adt.csv"),
        )

        df_xref = pd.read_csv(self.xref_file)
        df_xref_filt = df_xref[df_xref["MRN"].isin(mrns)]
        df_xref_filt.to_csv(
            os.path.join(self.output_dir, "edw_temp", "xref.csv"),
            index=False,
        )

        for result in results_list:
            list_mrns.append(result[0])
            flag_found.append(result[1])
        report = pd.DataFrame({"MRN": list_mrns, "fileStatus": flag_found})
        report = report.sort_values(by="MRN", ascending=True)
        report.to_csv(
            os.path.join(self.output_dir, "edw_temp", "report.csv"),
            index=False,
        )

    def find_save_bm_files(
        self,
        lm4_dir: str = "/media/lm4-bedmaster",
        patient_list: str = "/data/icu/blake8/bedmaster/patient_list.csv",
        parallelize: bool = False,
        n_workers: int = None,
    ):
        """
        Find bm files and copy them to local folder.

        :param lm4_dir: <str> Path to LM4 directory.
        :param patient_list: <str> List of desired patients to take their respective
                            BM files (.csv).
        :param parallelize: <bool> bool indicating whether the bm files copy
                            process is parallelized or not.
        :param n_workers: <int> Integer indicating the number of cores used in
                            the bm files copy process when parallelized.
        """
        df_xref = pd.read_csv(self.xref_file).sort_values(by=["MRN"], ascending=True)
        df_patient = pd.read_csv(patient_list)
        mrns = df_patient["MRN"].drop_duplicates()
        df_xref_filt = df_xref[df_xref["MRN"].isin(mrns)]

        list_bm_files = []
        folder = []
        flag_found = []

        # Enable number of cores only if parallelized is enabled
        if not parallelize:
            n_workers = 1
        # Check number of workers used is not higher than the number of cpus
        if os.cpu_count():
            if (n_workers or 0) > os.cpu_count():  # type: ignore
                n_workers = os.cpu_count()
                warnings.warn(
                    f"Workers are higher than number of cpus."
                    f" Number of workers is reduced to {os.cpu_count()}, "
                    f"the number of cpus your computer have.",
                )
        else:
            warnings.warn(
                "Couldn't determine the number of cpus. Blindly "
                "accepting the n_workers option",
            )

        with multiprocessing.Pool(processes=n_workers) as pool:
            results_list = pool.starmap(
                self._copy_files_bm,
                [(lm4_dir, row) for _, row in df_xref_filt.iterrows()],
            )

        for result in results_list:
            list_bm_files.extend(result[0])
            folder.extend(result[1])
            flag_found.extend(result[2])
        report = pd.DataFrame(
            {"fileID": list_bm_files, "folder": folder, "fileStatus": flag_found},
        )
        report = report.sort_values(by="fileID", ascending=True)
        report.to_csv(
            os.path.join(self.output_dir, "bedmaster_temp", "report.csv"),
            index=False,
        )

    def _copy_files_bm(self, lm4_dir, row):
        flag_found = []
        list_bm_files = []
        folder = []
        subfolders = self.mapping_folders[row.Department]
        for subfolder in subfolders:
            if os.path.isdir(os.path.join(lm4_dir, subfolder)):
                source_path = os.path.join(lm4_dir, subfolder, row.fileID + ".mat")
                destination_path = os.path.join(self.output_dir, "bedmaster_temp")
                list_bm_files.append(row.fileID)
                folder.append(os.path.join(lm4_dir, subfolder))
                try:
                    shutil.copy(source_path, destination_path)
                    flag_found.append("FOUND")
                except FileNotFoundError:
                    flag_found.append("NOT FOUND")
        return list_bm_files, folder, flag_found

    def _copy_files_edw(self, mad3_dir, mrn):
        source_path = os.path.join(mad3_dir, str(mrn))
        destination_path = os.path.join(self.output_dir, "edw_temp", str(mrn))
        try:
            shutil.copytree(source_path, destination_path)
            flag_found = "FOUND"
        except FileNotFoundError:
            flag_found = "NOT FOUND"
        return mrn, flag_found
