# pylint: disable=broad-except
# Imports: standard library
import os
import math
import time
import shutil
import logging
import multiprocessing
from typing import List

# Imports: third party
import pandas as pd

# Imports: first party
from tensorize.utils import save_mrns_and_csns_csv
from tensorize.edw.readers import EDWReader, CrossReferencer
from tensorize.edw.writers import Writer


class Tensorizer:
    """
    Main class to tensorize EDW data.
    """

    def __init__(
        self,
        edw: str,
        adt: str,
    ):
        """
        Initialize Tensorizer object.

        :param edw: <str> Directory containing all the EDW data.
        :param adt: <str> Full path of the file containing
               the adt patients' info to be tensorized.
        """
        self.edw = edw
        self.adt = adt

    def tensorize(
        self,
        tensors: str,
        mrns: List[str] = None,
        starting_time: int = None,
        ending_time: int = None,
        overwrite_hd5: bool = True,
        n_patients: int = None,
        num_workers: int = None,
    ):
        """
        Tensorizes EDW data.

        It will create a new HD5 for each MRN with the integrated data
        from both sources according to the following structure:

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
        files_per_mrn = CrossReferencer(self.edw, self.adt).get_xref_files(
            mrns=mrns,
            starting_time=starting_time,
            ending_time=ending_time,
            overwrite_hd5=overwrite_hd5,
            n_patients=n_patients,
            tensors=tensors,
        )
        os.makedirs(tensors, exist_ok=True)

        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(
                self._main_write,
                [(tensors, mrn, visits) for mrn, visits in files_per_mrn.items()],
            )

    def _main_write(
        self,
        tensors: str,
        mrn: str,
        visits: List,
    ):
        try:
            # Open the writer: one file per MRN
            output_file = os.path.join(tensors, f"{mrn}.hd5")
            with Writer(output_file) as writer:
                writer.write_completed_flag(False)
                for visit_id in visits:
                    # Set the visit ID
                    writer.set_visit_id(visit_id)

                    self._write_edw_data(self.edw, mrn, visit_id, writer=writer)
                    writer.write_completed_flag(True)
                    logging.info(
                        f"Tensorized data from MRN {mrn}, CSN {visit_id} into "
                        f"{output_file}.",
                    )
        except Exception as error:
            logging.exception(f"Tensorization failed for MRN {mrn} with CSNs {visits}")
            raise error

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
    os.makedirs(os.path.join(staging_dir, "edw_temp"), exist_ok=True)


def stage_edw_files(
    staging_dir: str,
    edw: str,
    adt: str,
):
    """
    Find EDW files and copy them to local folder.

    :param staging_dir: <str> Path to temporary staging directory.
    :param edw: <str> Path to directory with EDW data.
    :param adt: <str> Path to adt.csv.
    """
    path_patients = os.path.join(staging_dir, "patients.csv")
    mrns_and_csns = pd.read_csv(path_patients)
    mrns = mrns_and_csns["MRN"].drop_duplicates()

    for mrn in mrns:
        source_path = os.path.join(edw, str(mrn))
        destination_path = os.path.join(staging_dir, "edw_temp", str(mrn))
        try:
            shutil.copytree(source_path, destination_path)
        except FileNotFoundError as e:
            logging.warning(f"{source_path} not found. Error given: {e}")

    # Copy ADT table
    adt_df = pd.read_csv(adt)
    adt_subset = adt_df[adt_df["MRN"].isin(mrns)]
    path_adt_new = os.path.join(staging_dir, "edw_temp", "adt.csv")
    adt_subset.to_csv(path_adt_new, index=False)


def cleanup_staging_files(staging_dir: str):
    """
    Remove temp folders used for tensorization.

    :param staging_dir: <str> Path to temporary local directory.
    """
    shutil.rmtree(os.path.join(staging_dir, "edw_temp"))
    shutil.rmtree(os.path.join(staging_dir, "results_temp"))
    os.remove(os.path.join(staging_dir, "patients.csv"))


def copy_hd5(staging_dir: str, destination_tensors: str, num_workers: int):
    """
    Copy tensorized files to MAD3.

    :param staging_dir: <str> Path to temporary local directory.
    :param destination_tensors: <str> Path to MAD3 directory.
    :param num_workers: <int> Number of workers to use.
    """
    init_time = time.time()
    list_files = os.listdir(staging_dir)

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(
            _copy_hd5,
            [(staging_dir, destination_tensors, file) for file in list_files],
        )

    elapsed_time = time.time() - init_time
    logging.info(
        f"HD5 files copied to {destination_tensors}. "
        f"Process took {elapsed_time:.2f} sec",
    )


def _copy_hd5(staging_dir, destination_dir, file):
    source_path = os.path.join(staging_dir, file)
    shutil.copy(source_path, destination_dir)


def tensorize(args):

    if not os.path.isdir(args.edw) or len(os.listdir(args.edw)) == 0:
        raise ValueError(f"Server with EDW data is not mounted in {args.edw}")

    # Iterate over batch of patients
    missed_patients = []
    num_mrns_tensorized = []

    # If user does not set the end index,
    if args.mrn_end_index is None:
        adt_df = pd.read_csv(args.adt)
        mrns = adt_df["MRN"].drop_duplicates()
        args.mrn_end_index = len(mrns)

    mrn_indices_batched = range(
        args.mrn_start_index,
        args.mrn_end_index,
        args.staging_batch_size,
    )
    num_mrns = args.mrn_end_index - args.mrn_start_index
    num_batches = math.ceil(num_mrns / args.staging_batch_size)

    # Iterate over batches of MRN indices
    for idx_batch, first_mrn_index in enumerate(mrn_indices_batched):
        start_batch_time = time.time()

        # For this batch, determine index of last MRN
        if first_mrn_index + args.staging_batch_size < args.mrn_end_index:
            last_mrn_index = first_mrn_index + args.staging_batch_size
        else:
            last_mrn_index = args.mrn_end_index

        # Create staging directory
        create_folders(args.staging_dir)

        # Get unique MRNs and CSNs of this batch from ADT and save to patients.csv
        save_mrns_and_csns_csv(
            staging_dir=args.staging_dir,
            hd5_dir=args.tensors,
            adt=args.adt,
            first_mrn_index=first_mrn_index,
            last_mrn_index=last_mrn_index,
            overwrite_hd5=args.overwrite,
        )

        # Copy EDW files from this batch of patients
        init = time.time()
        stage_edw_files(
            staging_dir=args.staging_dir,
            edw=args.edw,
            adt=args.adt,
        )
        elapsed_time = time.time() - init
        logging.info(
            f"EDW files and ADT table copied from {args.edw} and {args.adt} "
            f"respectively in {elapsed_time:.2f} sec",
        )

        # Get paths to staging directories
        get_path_to_staging_dir = lambda f: os.path.join(args.staging_dir, f)
        edw_staging = get_path_to_staging_dir("edw_temp")
        adt_staging = get_path_to_staging_dir("edw_temp/adt.csv")
        path_tensors_staging = get_path_to_staging_dir("results_temp")

        # Run tensorization
        tensorizer = Tensorizer(
            edw=edw_staging,
            adt=adt_staging,
        )
        tensorizer.tensorize(
            tensors=path_tensors_staging,
            num_workers=args.num_workers,
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
        num_mrns_tensorized.extend(files_tensorized)
        missed_files = sorted(set(files_to_tensorize) - set(files_tensorized))
        missed_patients.extend(missed_files)

        # Copy newly created HD5 files to final tensor location
        if not os.path.isdir(args.tensors):
            os.makedirs(args.tensors)
        copy_hd5(
            staging_dir=path_tensors_staging,
            destination_tensors=args.tensors,
            num_workers=args.num_workers,
        )
        cleanup_staging_files(args.staging_dir)

        # Measure batch patient time
        end_batch_time = time.time()
        elapsed_time = end_batch_time - start_batch_time
        logging.info(
            f"Processed batch {idx_batch + 1}/{num_batches} of "
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
        f"{len(num_mrns_tensorized)} out of {num_mrns} " f"patients tensorized.",
    )
    if missed_patients:
        logging.warning(
            f"{len(missed_patients)} HD5 files are not tensorized. MRN of "
            f"those patients: {sorted(missed_patients)}",
        )
    os.rmdir(args.staging_dir)
