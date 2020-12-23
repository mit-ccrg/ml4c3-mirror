# pylint: disable=broad-except
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
from definitions.icu import ICU_SCALE_UNITS
from ingest.icu.utils import (
    stage_edw_files,
    stage_bedmaster_files,
    save_mrns_and_csns_csv,
    stage_bedmaster_alarms,
)
from ingest.icu.readers import (
    EDWReader,
    BedmasterReader,
    CrossReferencer,
    BedmasterAlarmsReader,
)
from ingest.icu.writers import Writer
from ingest.icu.match_patient_bedmaster import PatientBedmasterMatcher


class Tensorizer:
    """
    Main class to tensorize the Bedmaster data and the EDW data.
    """

    def __init__(
        self,
        bedmaster_dir: str,
        alarms_dir: str,
        edw_dir: str,
        xref_path: str,
        adt_path: str,
    ):
        """
        Initialize Tensorizer object.

        :param bedmaster_dir: <str> Directory containing all the Bedmaster data.
        :param alarms_dir: <str> Directory containing all the Bedmaster alarms data.
        :param edw_dir: <str> Directory containing all the EDW data.
        :param xref_path: <str> Full path of the file containing
               cross reference between EDW and Bedmaster.
        :param adt_path: <str> Full path of the file containing
               the adt patients' info to be tensorized.
        """
        self.bedmaster_dir = bedmaster_dir
        self.alarms_dir = alarms_dir
        self.edw_dir = edw_dir
        self.xref_path = xref_path
        self.adt_path = adt_path
        self.untensorized_files: Dict[str, List[str]] = {"file": [], "error": []}

    def tensorize(
        self,
        tensors: str,
        mrns: List[str] = None,
        starting_time: int = None,
        ending_time: int = None,
        overwrite_hd5: bool = True,
        n_patients: int = None,
        num_workers: int = None,
        allow_one_source: bool = True,
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

        :param tensors: <str> Directory where the output HD5 will be saved.
        :param mrns: <List[str]> MGH MRNs. The rest will be filtered out.
               If None, all the MRN are taken
        :param starting_time: <int> Start time in Unix format.
               If None, timestamps will be taken from the first one.
        :param ending_time: <int> End time in Unix format.
               If None, timestamps will be taken until the last one.
        :param overwrite_hd5: <bool> Overwrite existing HD5 files during tensorization
               should be overwritten.
        :param n_patients: <int> Max number of patients to tensorize.
        :param num_workers: <int> Number of cores used to parallelize tensorization
        :param allow_one_source: <bool> Indicates whether a patient with just one
               type of data will be tensorized or not.
        """
        # No options specified: get all the cross-referenced files
        files_per_mrn = CrossReferencer(
            self.bedmaster_dir,
            self.edw_dir,
            self.xref_path,
        ).get_xref_files(
            mrns=mrns,
            starting_time=starting_time,
            ending_time=ending_time,
            overwrite_hd5=overwrite_hd5,
            n_patients=n_patients,
            tensors=tensors,
            allow_one_source=allow_one_source,
        )
        os.makedirs(tensors, exist_ok=True)

        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(
                self._main_write,
                [
                    (tensors, mrn, visits, ICU_SCALE_UNITS)
                    for mrn, visits in files_per_mrn.items()
                ],
            )

        df = pd.DataFrame.from_dict(self.untensorized_files)
        df.to_csv(
            os.path.join(tensors, "untensorized_bedmaster_files.csv"),
            index=False,
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
                for visit_id, bedmaster_files in visits.items():
                    # Set the visit ID
                    writer.set_visit_id(visit_id)

                    # Write Bedmaster data
                    all_files, untensorized_files = self._write_bedmaster_data(
                        bedmaster_files,
                        writer=writer,
                        scaling_and_units=scaling_and_units,
                    )
                    self.untensorized_files["file"].append(untensorized_files["file"])
                    self.untensorized_files["error"].append(untensorized_files["error"])

                    # Write alarms data
                    self._write_bedmaster_alarms_data(
                        self.alarms_dir,
                        self.edw_dir,
                        self.adt_path,
                        mrn,
                        visit_id,
                        writer=writer,
                    )
                    writer.write_completed_flag("bedmaster", all_files)
                    self._write_edw_data(self.edw_dir, mrn, visit_id, writer=writer)
                    writer.write_completed_flag("edw", True)
                    logging.info(
                        f"Tensorized data from MRN {mrn}, CSN {visit_id} into "
                        f"{output_file}.",
                    )

        except Exception as error:
            logging.exception(f"Tensorization failed for MRN {mrn} with CSNs {visits}")
            raise error

    @staticmethod
    def _write_bedmaster_data(
        bedmaster_files: List[str],
        writer: Writer,
        scaling_and_units: Dict,
    ):
        all_files = True
        previous_max = None
        untensorized_files: Dict[str, List[str]] = {"file": [], "error": []}
        for bedmaster_file in bedmaster_files:
            try:
                with BedmasterReader(bedmaster_file, scaling_and_units) as reader:
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
            except Exception as error:
                untensorized_files["file"].append(bedmaster_file)
                untensorized_files["error"].append(repr(error))
        if len(untensorized_files["file"]) > 0:
            all_files = False
        return all_files, untensorized_files

    @staticmethod
    def _write_bedmaster_alarms_data(
        alarms_path: str,
        edw_path: str,
        adt_path: str,
        mrn: str,
        visit_id: str,
        writer: Writer,
    ):

        reader = BedmasterAlarmsReader(alarms_path, edw_path, mrn, visit_id, adt_path)
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


def create_folders(path_staging_dir: str):
    """
    Create temp folders for tensorization.

    :param path_staging_dir: <str> Path to temporary local directory.
    """
    os.makedirs(path_staging_dir, exist_ok=True)
    os.makedirs(os.path.join(path_staging_dir, "bedmaster_alarms_temp"), exist_ok=True)
    os.makedirs(os.path.join(path_staging_dir, "edw_temp"), exist_ok=True)
    os.makedirs(os.path.join(path_staging_dir, "bedmaster_temp"), exist_ok=True)


def cleanup_staging_files(path_staging_dir: str):
    """
    Remove temp folders used for tensorization.

    :param path_staging_dir: <str> Path to temporary local directory.
    """
    shutil.rmtree(os.path.join(path_staging_dir, "bedmaster_alarms_temp"))
    shutil.rmtree(os.path.join(path_staging_dir, "edw_temp"))
    shutil.rmtree(os.path.join(path_staging_dir, "bedmaster_temp"))
    shutil.rmtree(os.path.join(path_staging_dir, "results_temp"))
    os.remove(os.path.join(path_staging_dir, "patients.csv"))


def copy_hd5(path_staging_dir: str, destination_tensors: str, num_workers: int):
    """
    Copy tensorized files to MAD3.

    :param path_staging_dir: <str> Path to temporary local directory.
    :param destination_tensors: <str> Path to MAD3 directory.
    :param num_workers: <int> Number of workers to use.
    """
    init_time = time.time()
    list_files = os.listdir(path_staging_dir)

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(
            _copy_hd5,
            [(path_staging_dir, destination_tensors, file) for file in list_files],
        )

    elapsed_time = time.time() - init_time
    logging.info(
        f"HD5 files copied to {destination_tensors}. "
        f"Process took {elapsed_time:.2f} sec",
    )


def _copy_hd5(path_staging_dir, destination_dir, file):
    source_path = os.path.join(path_staging_dir, file)
    shutil.copy(source_path, destination_dir)


def tensorize(args):
    # Cross reference ADT table against Bedmaster metadata;
    # this results in the creation of xref.csv
    if not os.path.isfile(args.path_xref):
        matcher = PatientBedmasterMatcher(
            path_bedmaster=args.path_bedmaster,
            path_adt=args.path_adt,
        )
        matcher.match_files(path_xref=args.path_xref)

    # Iterate over batch of patients
    missed_patients = []
    total_files_tensorized = []
    if not args.adt_end_index:
        adt_df = pd.read_csv(args.path_adt)
        mrns = adt_df["MRN"].drop_duplicates()
        args.adt_end_index = len(mrns)
    patients = range(args.adt_start_index, args.adt_end_index, args.staging_batch_size)
    total_files = args.adt_end_index - args.adt_start_index
    total_batch = math.ceil(total_files / args.staging_batch_size)

    for idx_batch, num_batch in enumerate(patients):
        start_batch_time = time.time()

        # Compute first and last patient
        first_mrn_index = num_batch
        if num_batch + args.staging_batch_size < args.adt_end_index:
            last_mrn_index = num_batch + args.staging_batch_size
        else:
            last_mrn_index = args.adt_end_index

        # Create staging directory
        create_folders(args.path_staging_dir)

        """
        # Get desired number of patients
        get_files = FileManager(args.path_xref, args.path_adt, args.path_staging_dir)
        """

        # Get unique MRNs and CSNs from ADT and save to patients.csv.
        save_mrns_and_csns_csv(
            path_staging_dir=args.path_staging_dir,
            hd5_dir=args.tensors,
            path_adt=args.path_adt,
            first_mrn_index=first_mrn_index,
            last_mrn_index=last_mrn_index,
            overwrite_hd5=args.overwrite,
        )

        # Copy Bedmaster alarms from those patients
        init = time.time()
        stage_bedmaster_alarms(
            path_staging_dir=args.path_staging_dir,
            path_adt=args.path_adt,
            path_alarms=args.path_alarms,
        )
        elapsed_time = time.time() - init
        logging.info(
            f"Alarms copied to {args.path_alarms} in {elapsed_time:.2f} sec",
        )

        # Copy EDW files from those patients
        init = time.time()
        stage_edw_files(
            path_staging_dir=args.path_staging_dir,
            path_edw=args.path_edw,
            path_adt=args.path_adt,
            path_xref=args.path_xref,
        )
        elapsed_time = time.time() - init
        logging.info(
            f"EDW files (including ADT and xref tables) copied to "
            f"{args.path_edw} in {elapsed_time:.2f} sec",
        )

        # Copy Bedmaster files from those patients
        init = time.time()
        stage_bedmaster_files(
            path_staging_dir=args.path_staging_dir,
            path_xref=args.path_xref,
            path_bedmaster=args.path_bedmaster,
        )
        elapsed_time = time.time() - init
        logging.info(
            f"Bedmaster files copied to {args.path_bedmaster} "
            f"in {elapsed_time:.2f} sec",
        )

        # Get paths to staging directories
        get_path_to_staging_dir = lambda f: os.path.join(args.path_staging_dir, f)
        path_bedmaster_staging = get_path_to_staging_dir("bedmaster_temp")
        path_edw_staging = get_path_to_staging_dir("edw_temp")
        path_alarms_staging = get_path_to_staging_dir("bedmaster_alarms_temp")
        path_xref_staging = get_path_to_staging_dir("edw_temp/xref.csv")
        path_adt_staging = get_path_to_staging_dir("edw_temp/adt.csv")
        path_tensors_staging = get_path_to_staging_dir("results_temp")

        # Run tensorization
        tensorizer = Tensorizer(
            bedmaster_dir=path_bedmaster_staging,
            alarms_dir=path_alarms_staging,
            edw_dir=path_edw_staging,
            xref_path=path_xref_staging,
            adt_path=path_adt_staging,
        )
        tensorizer.tensorize(
            tensors=path_tensors_staging,
            num_workers=args.num_workers,
            allow_one_source=args.allow_one_source,
        )

        # Check tensorized files
        path_patients_staging = get_path_to_staging_dir("patients.csv")
        mrns_and_csns = pd.read_csv(path_patients_staging)
        files_to_tensorize = mrns_and_csns["MRN"]
        files_tensorized = [
            int(hd5_filename.split(".")[0])
            for hd5_filename in os.listdir(path_tensors_staging)
            if hd5_filename.endswith(".hd5")
        ]
        total_files_tensorized.extend(files_tensorized)
        missed_files = sorted(set(files_to_tensorize) - set(files_tensorized))
        missed_patients.extend(missed_files)

        # Copy newly created HD5 files to final tensor location
        if not os.path.isdir(args.tensors):
            os.makedirs(args.tensors)
        copy_hd5(
            path_staging_dir=path_tensors_staging,
            destination_tensors=args.tensors,
            num_workers=args.num_workers,
        )
        cleanup_staging_files(args.path_staging_dir)

        # Measure batch patient time
        end_batch_time = time.time()
        elapsed_time = end_batch_time - start_batch_time
        logging.info(
            f"Processed batch {idx_batch + 1}/{total_batch} of "
            f"{last_mrn_index - first_mrn_index} patients in "
            f"{elapsed_time:.2f} seconds.",
        )
        if missed_files:
            logging.info(
                f"From {last_mrn_index - first_mrn_index} patients, {len(missed_files)}"
                f"HD5 files are not tensorized. MRN of those patients: {missed_files}.",
            )
        else:
            logging.info(
                f"All {last_mrn_index - first_mrn_index} patients are tensorized.",
            )

    logging.info(f"HD5 Files tensorized and moved to {args.tensors}")
    logging.info(
        f"{len(total_files_tensorized)} out of {total_files} " f"patients tensorized.",
    )
    if missed_patients:
        logging.warning(
            f"{len(missed_patients)} HD5 files are not tensorized. MRN of "
            f"those patients: {sorted(missed_patients)}",
        )
    os.rmdir(args.path_staging_dir)
