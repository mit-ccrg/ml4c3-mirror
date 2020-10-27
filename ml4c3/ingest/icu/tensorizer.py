# Imports: standard library
import os
import math
import time
import shutil
import logging
import multiprocessing
from typing import Dict, List

# Imports: third party
import pandas as pd

# Imports: first party
from ml4c3.definitions.icu import LM4_DIR, MAD3_DIR, ICU_SCALE_UNITS
from ml4c3.ingest.icu.utils import FileManager
from ml4c3.ingest.icu.readers import (
    BMReader,
    EDWReader,
    BMAlarmsReader,
    CrossReferencer,
)
from ml4c3.ingest.icu.writers import Writer
from ml4c3.ingest.icu.matchers import PatientBMMatcher


class Tensorizer:
    """
    Main class to tensorize the Bedmaster data and the EDW data.
    """

    def __init__(self, bm_dir: str, alarms_dir: str, edw_dir: str, cross_ref_path: str):
        """
        Init a Tensorizer object.

        :param bm_dir: <str> Directory containing all the Bedmaster data.
        :param alarms_dir: <str> Directory containing all the Bedmaster alarms data.
        :param edw_dir: <str> Directory containing all the EDW data.
        :param cross_ref_path: <str> Full path of the file containing
                              cross reference between EDW and Bedmaster.
        """
        self.bm_dir = bm_dir
        self.alarms_dir = alarms_dir
        self.edw_dir = edw_dir
        self.cross_ref_path = cross_ref_path

    def tensorize(
        self,
        tensors: str,
        mrns: List[str] = None,
        starting_time: int = None,
        ending_time: int = None,
        overwrite_hd5: bool = True,
        n_patients: int = None,
        num_workers: int = None,
        flag_one_source: bool = True,
    ):
        """
        Tensorizes Bedmaster and EDW data.

        It will create a new HD5 for each MRN with the integrated data
        from both sources according to the following structure:

        <BedMaster>
            <visitID>/
                <signal name>/
                    data and metadata
                ...
            ...
        <EDW>
            <visitID>/
                <signal name>/
                    data and metadata
                ...
            ...
        ...

        :param tensors: <str> directory where the output HD5 will be saved.
        :param mrns: <List[str]> a list containing MRNs. The rest will be
                    filtered out. If None, all the MRN are taken
        :param starting_time: <int> starting time in Unix format.
                             If None, timestamps will be taken from
                             the first one.
        :param ending_time: <int> ending time in Unix format.
                            If None, timestamps will be taken until
                            the last one.
        :param overwrite_hd5: <bool> bool indicates whether the existing
                              hd5 files should be overwritten
        :param n_patients: <int> max number of patients to tensorize.
        :param num_workers: <int> Integer indicating the number of cores used in
                            the tensorization process when parallelized.
        :param flag_one_source: <bool> bool indicating whether a patient with
                                just one type of data will be tensorized or not.
        """

        # Get scaling factor and units
        scaling_and_units = ICU_SCALE_UNITS

        # No options specified: get all the cross-referenced files
        files_per_mrn = CrossReferencer(
            self.bm_dir,
            self.edw_dir,
            self.cross_ref_path,
        ).get_xref_files(
            mrns,
            starting_time,
            ending_time,
            overwrite_hd5,
            n_patients,
            tensors,
            flag_one_source,
        )

        os.makedirs(tensors, exist_ok=True)
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(
                self._main_write,
                [
                    (tensors, mrn, visits, scaling_and_units)
                    for mrn, visits in files_per_mrn.items()
                ],
            )

    def _main_write(
        self,
        tensors: str,
        mrn: str,
        visits: Dict,
        scaling_and_units: Dict,
    ):
        try:
            # Open the writer: one file per MRN
            output_file = os.path.join(tensors, f"{mrn}.hd5")
            with Writer(output_file) as writer:
                writer.write_completed_flag("bedmaster", False)
                writer.write_completed_flag("edw", False)
                for visit_id, bm_files in visits.items():
                    # Set the visit ID
                    writer.set_visit_id(visit_id)

                    # Write the data
                    self._write_bm_data(
                        bm_files,
                        writer=writer,
                        scaling_and_units=scaling_and_units,
                    )
                    self._write_bm_alarms_data(
                        self.alarms_dir,
                        self.edw_dir,
                        mrn,
                        visit_id,
                        writer=writer,
                    )
                    writer.write_completed_flag("bedmaster", True)
                    self._write_edw_data(self.edw_dir, mrn, visit_id, writer=writer)
                    writer.write_completed_flag("edw", True)
                    logging.info(
                        f"Tensorization completed for MRN {mrn}, CSN {visit_id}.",
                    )
                logging.info(f"Tensorization completed for MRN {mrn}.")
        except Exception as error:
            logging.exception(f"Tensorization failed for MRN {mrn} with CSNs {visits}")
            raise error

    @staticmethod
    def _write_bm_data(bm_files: List[str], writer: Writer, scaling_and_units: Dict):

        previous_max = None
        for bm_file in bm_files:
            with BMReader(bm_file, scaling_and_units) as reader:
                if previous_max:
                    reader.get_interbundle_correction(previous_max)
                # These blocks can be easily parallelized with MPI:
                # >>> rank = MPI.COMM_WORLD.rank
                # >>> if rank == 1:
                vs_signals = reader.list_vs()
                for vs_signal_name in vs_signals:
                    vs_signal = reader.get_vs(vs_signal_name)
                    if vs_signal:
                        writer.write_signal(vs_signal)

                # >>> if rank == 2
                wv_signals = reader.list_wv()
                for wv_signal_name, channel in wv_signals.items():
                    wv_signal = reader.get_wv(channel, wv_signal_name)
                    if wv_signal:
                        writer.write_signal(wv_signal)
                previous_max = reader.max_segment

    @staticmethod
    def _write_bm_alarms_data(
        alarms_path: str,
        edw_path: str,
        mrn: str,
        visit_id: str,
        writer: Writer,
    ):

        reader = BMAlarmsReader(alarms_path, edw_path, mrn, visit_id)
        alarms = reader.list_alarms()
        for alarm in alarms:
            alarm_instance = reader.get_alarm(alarm)
            writer.write_signal(alarm_instance)

    @staticmethod
    def _write_edw_data(path: str, mrn: str, visit_id: str, writer: Writer):

        reader = EDWReader(path, mrn, visit_id)
        if not os.path.isdir(os.path.join(path, mrn, visit_id)):
            return

        # These blocks can be easily parallelized:
        # >>> rank = MPI.COMM_WORLD.rank
        # >>> if rank == 1:
        medications = reader.list_medications()
        for med_name in medications:
            doses = reader.get_med_doses(med_name)
            writer.write_signal(doses)

        # >>> if rank == 2:
        vitals = reader.list_vitals()
        for vital_name in vitals:
            vital_signal = reader.get_vitals(vital_name)
            writer.write_signal(vital_signal)

        # >>> if rank == 3:
        labs = reader.list_labs()
        for lab_name in labs:
            lab_signal = reader.get_labs(lab_name)
            writer.write_signal(lab_signal)

        # >>> if rank == 4:
        surgery_types = reader.list_surgery()
        for surgery_type in surgery_types:
            surgery = reader.get_surgery(surgery_type)
            writer.write_signal(surgery)

        # >>> if rank == 5:
        other_procedures_types = reader.list_other_procedures()
        for other_procedures_type in other_procedures_types:
            other_procedures = reader.get_other_procedures(other_procedures_type)
            writer.write_signal(other_procedures)

        # >>> if rank == 6:
        transf_types = reader.list_transfusions()
        for transf_type in transf_types:
            transfusions = reader.get_transfusions(transf_type)
            writer.write_signal(transfusions)

        # >>> if rank == 7:
        event_types = reader.list_events()
        for event_type in event_types:
            events = reader.get_events(event_type)
            writer.write_signal(events)

        # >>> if rank == 8:
        static_data = reader.get_static_data()
        writer.write_static_data(static_data)


def create_folders(staging_dir: str):
    """
    Create temp folders for tensorization.

    :param staging_dir: <str> Path to temporary local directory.
    """
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(os.path.join(staging_dir, "bm_alarms_temp"), exist_ok=True)
    os.makedirs(os.path.join(staging_dir, "edw_temp"), exist_ok=True)
    os.makedirs(os.path.join(staging_dir, "bedmaster_temp"), exist_ok=True)


def remove_folders(staging_dir: str):
    """
    Remove temp folders used for tensorization.

    :param staging_dir: <str> Path to temporary local directory.
    """
    shutil.rmtree(os.path.join(staging_dir, "bm_alarms_temp"))
    shutil.rmtree(os.path.join(staging_dir, "edw_temp"))
    shutil.rmtree(os.path.join(staging_dir, "bedmaster_temp"))
    shutil.rmtree(os.path.join(staging_dir, "results_temp"))
    os.remove(os.path.join(staging_dir, "patient_list.csv"))


def copy_source_data(file_manager: FileManager, staging_dir: str, workers: int):
    """
    Copy files from desired patients.

    :param file_manager: <FileManager> Class to copy files.
    :param staging_dir: <str> Path to temporary local directory.
    :param workers: <int> Number of workers for parallelization.
    """
    # Copy BM alarms from those patients
    init = time.time()
    file_manager.find_save_bm_alarms(
        os.path.join(MAD3_DIR, "bedmaster_alarms"),
        os.path.join(staging_dir, "patient_list.csv"),
    )
    elapsed_time = time.time() - init
    logging.info(f"Alarms copied. Process took {round(elapsed_time/60, 4)} minutes.")

    # Copy EDW files from those patients
    init = time.time()
    file_manager.find_save_edw_files(
        os.path.join(MAD3_DIR, "edw"),
        os.path.join(staging_dir, "patient_list.csv"),
        parallelize=True,
        n_workers=workers,
    )
    elapsed_time = time.time() - init
    logging.info(f"EDW files copied. Process took {round(elapsed_time/60, 4)} minutes.")

    # Copy BM files from those patients
    init = time.time()
    file_manager.find_save_bm_files(
        LM4_DIR,
        os.path.join(staging_dir, "patient_list.csv"),
        parallelize=True,
        n_workers=workers,
    )
    elapsed_time = time.time() - init
    logging.info(f"BM files copied. Process took {round(elapsed_time/60, 4)} minutes.")


def copy_hd5(staging_dir: str, destination_tensors: str, workers: int):
    """
    Copy tensorized files to MAD3.

    :param staging_dir: <str> Path to temporary local directory.
    :param destination_tensors: <str> Path to MAD3 directory.
    :param workers: <int> Number of workers to use.
    """
    init_time = time.time()
    list_files = os.listdir(staging_dir)

    with multiprocessing.Pool(processes=workers) as pool:
        pool.starmap(
            _copy_hd5,
            [(staging_dir, destination_tensors, file) for file in list_files],
        )

    elapsed_time = time.time() - init_time
    logging.info(
        f"HD5 files copied. Process took {round(elapsed_time / 60, 4)} minutes.",
    )


def _copy_hd5(staging_dir, destination_dir, file):
    source_path = os.path.join(staging_dir, file)
    shutil.copy(source_path, destination_dir)


def tensorize(args):
    tens = Tensorizer(
        args.path_bedmaster,
        args.path_alarms,
        args.path_edw,
        args.path_xref,
    )
    tens.tensorize(
        tensors=args.tensors,
        overwrite_hd5=args.overwrite_hd5,
        n_patients=args.num_patients_to_tensorize,
        num_workers=args.num_workers,
        flag_one_source=args.allow_one_source,
    )


def tensorize_batched(args):
    # Create crossreference table if needed
    if not os.path.isfile(args.path_xref):
        adt_table_name = os.path.split(args.path_adt)[-1]
        # Match bedmaster files from lm4 with path_adt
        matcher = PatientBMMatcher(
            flag_lm4=True,
            bm_dir=LM4_DIR,
            edw_dir=os.path.join(MAD3_DIR, "cohorts_lists"),
            adt_file=adt_table_name,
        )
        matcher.match_files(args.path_xref)

    # Loop for batch of patients
    missed_patients = []
    total_files_tensorized = []
    if not args.adt_end_index:
        args.adt_end_index = len(pd.read_csv(args.path_adt)["MRN"].drop_duplicates())
    patients = range(args.adt_start_index, args.adt_end_index, args.staging_batch_size)
    total_files = args.adt_end_index - args.adt_start_index
    total_batch = math.ceil(total_files / args.staging_batch_size)
    for idx_batch, num_batch in enumerate(patients):
        start_batch_time = time.time()

        # Compute first and last patient
        first_patient = num_batch
        if num_batch + args.staging_batch_size < args.adt_end_index:
            last_patient = num_batch + args.staging_batch_size
        else:
            last_patient = args.adt_end_index

        # Create necessary local folders
        create_folders(args.staging_dir)

        # Get desired number of patients
        get_files = FileManager(args.path_xref, args.path_adt, args.staging_dir)
        get_files.get_patients(
            init_patient=first_patient,
            last_patient=last_patient,
            overwrite_hd5=args.overwrite_hd5,
            hd5_dir=args.tensors,
        )

        # Copy files from those patients
        copy_source_data(get_files, args.staging_dir, args.num_workers)

        # Run tensorization
        local_path = lambda f: os.path.join(args.staging_dir, f)
        path_bedmaster = local_path("bedmaster_temp")
        path_edw = local_path("edw_temp")
        path_alarms = local_path("bm_alarms_temp")
        xref_file = local_path("edw_temp/xref.csv")
        local_tensors = local_path("results_temp")

        tens = Tensorizer(path_bedmaster, path_alarms, path_edw, xref_file)
        tens.tensorize(
            tensors=local_tensors,
            num_workers=args.num_workers,
            flag_one_source=args.allow_one_source,
        )

        # Check tensorized files
        files_to_tensorize = [int(mrn) for mrn in get_files.mrns]
        files_tensorized = [
            int(hd5_mrn.split(".")[0])
            for hd5_mrn in os.listdir(local_tensors)
            if hd5_mrn.endswith(".hd5")
        ]
        total_files_tensorized.extend(files_tensorized)
        missed_files = list(set(files_to_tensorize) - set(files_tensorized))
        missed_patients.extend(missed_files)

        # Copy tensorized files to MAD3
        if not os.path.isdir(args.tensors):
            os.makedirs(args.tensors)
        copy_hd5(local_tensors, args.tensors, args.num_workers)

        # Remove folders
        remove_folders(args.staging_dir)

        # Measure batch patient time
        end_batch_time = time.time()
        elapsed_time = end_batch_time - start_batch_time
        logging.info(
            f"Processed batch {idx_batch + 1}/{total_batch} of "
            f"{last_patient - first_patient} patients in {elapsed_time:.2f} seconds.",
        )
        if missed_files:
            logging.info(
                f"From {last_patient - first_patient} patients, {len(missed_files)} "
                f"HD5 files are missing. MRN of those patients: {missed_files}.",
            )
        else:
            logging.info(f"All {last_patient - first_patient} patients are tensorized.")

    logging.info(f"HD5 Files tensorized and moved to {args.tensors}")
    logging.info(
        f"{len(total_files_tensorized)} out of {total_files} " f"patients tensorized.",
    )
    if missed_patients:
        logging.warning(
            f"{len(missed_patients)} HD5 files are missing. MRN of "
            f"those patients: {missed_patients}",
        )
