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
from tensorize.utils import save_mrns_and_csns_csv
from tensorize.bedmaster.readers import BedmasterReader, CrossReferencer
from tensorize.bedmaster.writers import Writer
from tensorize.bedmaster.match_patient_bedmaster import PatientBedmasterMatcher


class Tensorizer:
    """
    Main class to tensorize the Bedmaster data.
    """

    def __init__(
        self,
        bedmaster: str,
        xref: str,
        adt: str,
    ):
        """
        Initialize Tensorizer object.

        :param bedmaster: <str> Directory containing all the Bedmaster data.
        :param xref: <str> Full path of the file containing
               cross reference between EDW and Bedmaster.
        :param adt: <str> Full path of the file containing
               the adt patients' info to be tensorized.
        """
        self.bedmaster = bedmaster
        self.xref = xref
        self.adt = adt
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
    ):
        """
        Tensorizes Bedmaster data.

        It will create a new HD5 for each MRN with the integrated data
        from both sources according to the following structure:

        <BedMaster>
            <visitID>/
                <signal name>/
                    data and metadata
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
        """
        # No options specified: get all the cross-referenced files
        files_per_mrn = CrossReferencer(
            self.bedmaster,
            self.xref,
            self.adt,
        ).get_xref_files(
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
                [
                    (tensors, mrn, visits, ICU_SCALE_UNITS)
                    for mrn, visits in files_per_mrn.items()
                ],
            )

        df = pd.DataFrame.from_dict(self.untensorized_files)
        if not df.empty:
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
                writer.write_completed_flag(False)
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

                    writer.write_completed_flag(all_files)
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


def create_folders(staging_dir: str):
    """
    Create temp folders for tensorization.

    :param staging_dir: <str> Path to temporary local directory.
    """
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(os.path.join(staging_dir, "bedmaster_temp"), exist_ok=True)


def stage_bedmaster_files(
    staging_dir: str,
    adt: str,
    xref: str,
):
    """
    Find Bedmaster files and copy them to local folder.

    :param staging_dir: <str> Path to temporary staging directory.
    :param adt: <str> Path to CSN containing ADT table.
    :param xref: <str> Path to xref.csv with Bedmaster metadata.
    """
    path_patients = os.path.join(staging_dir, "patients.csv")
    mrns_and_csns = pd.read_csv(path_patients)
    mrns = mrns_and_csns["MRN"].drop_duplicates()

    # Copy ADT table
    adt_df = pd.read_csv(adt)
    adt_subset = adt_df[adt_df["MRN"].isin(mrns)]
    path_adt_new = os.path.join(staging_dir, "bedmaster_temp", "adt.csv")
    adt_subset.to_csv(path_adt_new, index=False)

    # Copy xref table
    xref_df = pd.read_csv(xref).sort_values(by=["MRN"], ascending=True)
    xref_subset = xref_df[xref_df["MRN"].isin(mrns)]
    path_xref_new = os.path.join(staging_dir, "bedmaster_temp", "xref.csv")
    xref_subset.to_csv(path_xref_new, index=False)

    # Iterate over all Bedmaster file paths to copy to staging directory
    path_destination_dir = os.path.join(staging_dir, "bedmaster_temp")
    for path_source_file in xref_subset["Path"]:
        if os.path.exists(path_source_file):
            try:
                shutil.copy(path_source_file, path_destination_dir)
            except FileNotFoundError as e:
                logging.warning(f"{path_source_file} not found. Error given: {e}")
        else:
            logging.warning(f"{path_source_file} not found.")


def cleanup_staging_files(staging_dir: str):
    """
    Remove temp folders used for tensorization.

    :param staging_dir: <str> Path to temporary local directory.
    """
    shutil.rmtree(os.path.join(staging_dir, "bedmaster_temp"))
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

    if not os.path.isdir(args.bedmaster) or len(os.listdir(args.bedmaster)) == 0:
        raise ValueError(
            f"Server with Bedmaster data is not mounted in {args.bedmaster}",
        )

    # Cross reference ADT table against Bedmaster metadata;
    # this results in the creation of xref.csv
    if not os.path.isfile(args.xref):
        matcher = PatientBedmasterMatcher(
            bedmaster=args.bedmaster,
            adt=args.adt,
        )
        matcher.match_files(xref=args.xref)

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

        # Copy Bedmaster files from this batch of patients
        init = time.time()
        stage_bedmaster_files(
            staging_dir=args.staging_dir,
            adt=args.adt,
            xref=args.xref,
        )
        elapsed_time = time.time() - init
        logging.info(
            f"Bedmaster files copied from {args.bedmaster} in {elapsed_time:.2f} sec",
        )

        # Get paths to staging directories
        get_path_to_staging_dir = lambda f: os.path.join(args.staging_dir, f)

        bedmaster_staging = get_path_to_staging_dir("bedmaster_temp")
        xref_staging = get_path_to_staging_dir("bedmaster_temp/xref.csv")
        adt_staging = get_path_to_staging_dir("bedmaster_temp/adt.csv")
        path_tensors_staging = get_path_to_staging_dir("results_temp")

        # Run tensorization
        tensorizer = Tensorizer(
            bedmaster=bedmaster_staging,
            xref=xref_staging,
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
