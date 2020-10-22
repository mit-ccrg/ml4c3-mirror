# Imports: standard library
import os
import sys
import logging
import argparse
import datetime
import operator
import multiprocessing
from typing import Dict

# Imports: third party
import numpy as np

# Imports: first party
import ml4c3.definitions
from ml4c3.logger import load_config
from ml4c3.models import BottleneckType
from ml4c3.definitions.icu import MAD3_DIR
from ml4c3.definitions.sts import STS_DATA_CSV
from ml4c3.tensormap.TensorMap import TensorMap, update_tmaps

BOTTLENECK_STR_TO_ENUM = {
    "flatten_restructure": BottleneckType.FlattenRestructure,
    "global_average_pool": BottleneckType.GlobalAveragePoolStructured,
    "variational": BottleneckType.Variational,
}

# pylint: disable=import-outside-toplevel, unnecessary-comprehension
# pylint: disable=too-many-lines, unused-variable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(
        title="ml4c3 modes",
        description="Select one of the following modes: \n"
        "\t * train: ADD DESCRIPTION. \n"
        "\t * train_shallow: ADD DESCRIPTION. \n"
        "\t * train_simclr: ADD DESCRIPTION. \n"
        "\t * infer: ADD DESCRIPTION. \n"
        "\t * hyperoptimize: ADD DESCRIPTION. \n"
        "\t * build: ADD DESCRIPTION. \n"
        "\t * explore: ADD DESCRIPTION. \n"
        "\t * explore_icu: ADD DESCRIPTION. \n"
        "\t * plot_ecg: ADD DESCRIPTION. \n"
        "\t * tensorize_ecg: ADD DESCRIPTION. \n"
        "\t * tensorize_icu: ADD DESCRIPTION. \n"
        "\t * assess_icu_coverage: ADD DESCRIPTION. \n"
        "\t * check_icu_structure: ADD DESCRIPTION. \n"
        "\t * pre_tensorize_summary: ADD DESCRIPTION. \n"
        "\t * match_patient_bm: ADD DESCRIPTION. \n",
        dest="mode",
    )

    # Tensor Map arguments
    tmap_parser = argparse.ArgumentParser(add_help=False)
    tmap_parser.add_argument("--input_tensors", default=[], nargs="+")
    tmap_parser.add_argument("--output_tensors", default=[], nargs="+")
    tmap_parser.add_argument(
        "--tensor_maps_in",
        default=[],
        help="Do not set this directly. Use input_tensors",
    )
    tmap_parser.add_argument(
        "--tensor_maps_out",
        default=[],
        help="Do not set this directly. Use output_tensors",
    )
    tmap_parser.add_argument(
        "--mrn_column_name",
        default="medrecn",
        help="Name of MRN column in tensors_all*.csv",
    )

    # Input and Output files and directories
    io_parser = argparse.ArgumentParser(add_help=False)
    io_parser.add_argument(
        "--sample_csv",
        help="Path to CSV with Sample IDs to restrict tensor paths",
    )
    io_parser.add_argument(
        "--tensors",
        help="Path to folder containing tensors, or where tensors will be written.",
    )
    io_parser.add_argument(
        "--output_folder",
        default="./recipes-output",
        help="Path to output folder for recipes.py runs.",
    )
    io_parser.add_argument(
        "--model_file",
        help="Path to a saved model architecture and weights (hd5).",
    )
    io_parser.add_argument(
        "--model_files",
        nargs="*",
        default=[],
        help="List of paths to saved model architectures and weights (hd5).",
    )
    io_parser.add_argument(
        "--donor_layers",
        help=(
            "Path to a model file (hd5) which will be loaded by layer, useful for"
            " transfer learning."
        ),
    )
    io_parser.add_argument(
        "--remap_layer",
        action="append",
        nargs=2,
        help="For transfer layer, manually remap layer from pretrained model to layer"
        " in new model. For example: --rename_layer pretrained_layer_name "
        "new_layer_name. Layers are remapped using this argument one at a time, "
        "repeat for multiple layers.",
    )
    io_parser.add_argument(
        "--freeze_donor_layers",
        action="store_true",
        help="Whether to freeze the layers from donor_layers.",
    )
    io_parser.add_argument(
        "--sts_csv",
        default=STS_DATA_CSV,
        help="Path to STS data csv file.",
    )

    # Run specific and debugging arguments
    run_parser = argparse.ArgumentParser(add_help=False)
    run_parser.add_argument(
        "--id",
        default="no_id",
        help=(
            "Identifier for this run, user-defined string to keep experiments"
            " organized."
        ),
    )
    run_parser.add_argument(
        "--random_seed",
        default=12878,
        type=int,
        help="Random seed to use throughout run.  Always use np.random.",
    )
    run_parser.add_argument(
        "--eager",
        default=False,
        action="store_true",
        help=(
            "Run tensorflow functions in eager execution mode (helpful for debugging)."
        ),
    )
    run_parser.add_argument(
        "--plot_mode",
        default="clinical",
        choices=["clinical", "full"],
        help="ECG view to plot.",
    )

    # Config arguments
    run_parser.add_argument(
        "--logging_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=(
            "Logging level. Overrides any configuration given in the logging"
            " configuration file."
        ),
    )

    # Image file format arguments
    run_parser.add_argument(
        "--image_ext",
        default=".pdf",
        choices=[".pdf", ".eps", ".svg", ".png"],
        help="File format extension to save images as."
        "Note this includes a leading period.",
    )

    # Training optimization options
    run_parser.add_argument(
        "--num_workers",
        default=multiprocessing.cpu_count(),
        type=int,
        help="Number of workers to use for every tensor generator.",
    )

    # ECG Tensorize arguments
    ecg_tens_parser = subparser.add_parser(
        name="tensorize_ecg",
        description="TODO",
        parents=[io_parser, run_parser],
    )
    ecg_tens_parser.add_argument(
        "--bad_xml_dir",
        default=os.path.expanduser("~/bad-xml"),
        help="Path to directory to store XML files that fail tensorization.",
    )
    ecg_tens_parser.add_argument(
        "--bad_hd5_dir",
        default=os.path.expanduser("~/bad-hd5"),
        help="Path to directory to store HD5 files that fail tensorization.",
    )
    ecg_tens_parser.add_argument(
        "--xml_folder",
        help="Path to folder of XMLs of ECG data.",
    )

    # ICU Tensorize arguments
    icu_parser = argparse.ArgumentParser(add_help=False)
    icu_parser.add_argument(
        "--path_bedmaster",
        default="/media/lm4-bedmaster/",
        help="Directory with Bedmaster .mat files.",
    )
    icu_parser.add_argument(
        "--path_alarms",
        default="/media/mad3/bedmaster_alarms/",
        help="Directory with Bedmaster alarms .csv files.",
    )
    icu_parser.add_argument(
        "--path_edw",
        default="/media/mad3/edw/",
        help="Directory with EDW .csv files.",
    )
    icu_parser.add_argument(
        "--path_xref",
        default="/media/mad3/xref.csv",
        help="Full path of the file where EDW and Bedmaster "
        "are cross referenced. CSV file which indicates the "
        "corresponding MRN and CSN of each Bedmaster file.",
    )

    icu_tens_parser = subparser.add_parser(
        name="tensorize_icu",
        description="TODO",
        parents=[io_parser, run_parser, icu_parser],
    )
    icu_tens_parser.add_argument(
        "--overwrite_hd5",
        action="store_true",
        help="Bool indicating whether the existing hd5 files should be overwritten.",
    )
    icu_tens_parser.add_argument(
        "--num_patients_to_tensorize",
        default=None,
        type=int,
        help="Maximum number of patients whose data will be tensorized. "
        "Useful for troubleshooting.",
    )
    icu_tens_parser.add_argument(
        "--allow_one_source",
        action="store_false",
        help="If this parameter is set, patients with just one type of data "
        "will be tensorized.",
    )
    icu_tens_parser.add_argument(
        "--adt_file",
        type=str,
        default=f"{os.path.join(MAD3_DIR, 'cohorts_lists', 'adt.csv')}",
        help="Full path of ADT table.",
    )
    icu_tens_parser.add_argument(
        "--adt_start_index",
        type=int,
        default=0,
        help="Index of first patient in ADT table to get data from",
    )
    icu_tens_parser.add_argument(
        "--adt_end_index",
        type=int,
        default=None,
        help="Index of last patient in ADT table to get data from",
    )
    icu_tens_parser.add_argument(
        "--staging_dir",
        type=str,
        default=os.path.expanduser("~/icu-temp"),
        help=(
            "If --tensors is a location mounted over the network, tensorization speed "
            "will be slower than tensorizing to a local directory. To tensorize "
            "faster, tensorize to an intermediate local directory given by "
            "--staging_dir. The pipeline will copy these HD5 files to --tensors, then "
            "delete the HD5 files from --staging_dir."
        ),
    )
    icu_tens_parser.add_argument(
        "--staging_batch_size",
        default=None,
        type=int,
        help=(
            "Number of patients whose data to tensorize locally to --staging_dir "
            "before being moved to --tensors"
        ),
    )

    # Model Architecture Parameters
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument(
        "--activation",
        default="relu",
        help="Activation function for hidden units in neural nets dense layers.",
    )
    model_parser.add_argument(
        "--block_size",
        default=3,
        type=int,
        help="Number of convolutional layers within a block.",
    )
    model_parser.add_argument(
        "--bottleneck_type",
        type=str,
        default=list(BOTTLENECK_STR_TO_ENUM)[0],
        choices=list(BOTTLENECK_STR_TO_ENUM),
    )
    model_parser.add_argument(
        "--conv_layers",
        nargs="*",
        default=[32],
        type=int,
        help="List of number of kernels in convolutional layers.",
    )
    model_parser.add_argument(
        "--conv_x",
        default=[3],
        nargs="*",
        type=int,
        help=(
            "X dimension of convolutional kernel. Filter sizes are specified per layer"
            " given by conv_layers and per block given by dense_blocks. Filter sizes"
            " are repeated if there are less than the number of layers/blocks."
        ),
    )
    model_parser.add_argument(
        "--conv_y",
        default=[3],
        nargs="*",
        type=int,
        help=(
            "Y dimension of convolutional kernel. Filter sizes are specified per layer"
            " given by conv_layers and per block given by dense_blocks. Filter sizes"
            " are repeated if there are less than the number of layers/blocks."
        ),
    )
    model_parser.add_argument(
        "--conv_z",
        default=[2],
        nargs="*",
        type=int,
        help=(
            "Z dimension of convolutional kernel. Filter sizes are specified per layer"
            " given by conv_layers and per block given by dense_blocks. Filter sizes"
            " are repeated if there are less than the number of layers/blocks."
        ),
    )
    model_parser.add_argument(
        "--conv_dilate",
        default=False,
        action="store_true",
        help="Dilate the convolutional layers.",
    )
    model_parser.add_argument(
        "--conv_dropout",
        default=0.0,
        type=float,
        help="Dropout rate of convolutional kernels must be in [0.0, 1.0].",
    )
    model_parser.add_argument(
        "--conv_type",
        default="conv",
        choices=["conv", "separable", "depth"],
        help="Type of convolutional layer",
    )
    model_parser.add_argument(
        "--conv_normalize",
        choices=["", "batch_norm"],
        help="Type of normalization layer for convolutions",
    )
    model_parser.add_argument(
        "--conv_regularize",
        choices=["dropout", "spatial_dropout"],
        help="Type of regularization layer for convolutions.",
    )
    model_parser.add_argument(
        "--dense_blocks",
        nargs="*",
        default=[32, 24, 16],
        type=int,
        help="List of number of kernels in convolutional layers.",
    )
    model_parser.add_argument(
        "--dense_layers",
        nargs="*",
        default=[16, 64],
        type=int,
        help="List of number of hidden units in neural nets dense layers.",
    )
    model_parser.add_argument(
        "--directly_embed_and_repeat",
        type=int,
        help="If set, directly embed input tensors (without passing to a dense layer)"
        " into concatenation layer, and repeat each input N times, where N is this"
        " argument's value. To directly embed a feature without repetition, set to 1.",
    )
    model_parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="Dropout rate of dense layers must be in [0.0, 1.0].",
    )
    model_parser.add_argument(
        "--layer_order",
        nargs=3,
        default=["activation", "regularization", "normalization"],
        choices=["activation", "normalization", "regularization"],
        help=(
            "Order of activation, regularization, and normalization layers following"
            " convolutional layers."
        ),
    )
    model_parser.add_argument(
        "--nest_model",
        nargs=2,
        action="append",
        help="Embed a nested model ending at the specified layer before the bottleneck"
        " layer of the current model. Repeat this argument to embed multiple models."
        " For example --nest_model /path/to/model_weights.h5 embed_layer",
    )
    model_parser.add_argument(
        "--padding",
        default="same",
        help="Valid or same border padding on the convolutional layers.",
    )
    model_parser.add_argument(
        "--pool_after_final_dense_block",
        default=True,
        action="store_false",
        help="Pool the last layer of all dense blocks.",
    )
    model_parser.add_argument(
        "--pool_type",
        default="max",
        choices=["max", "average"],
        help="Type of pooling layers.",
    )
    model_parser.add_argument(
        "--pool_x",
        default=2,
        type=int,
        help="Pooling size in the x-axis, if 1 no pooling will be performed.",
    )
    model_parser.add_argument(
        "--pool_y",
        default=2,
        type=int,
        help="Pooling size in the y-axis, if 1 no pooling will be performed.",
    )
    model_parser.add_argument(
        "--pool_z",
        default=1,
        type=int,
        help="Pooling size in the z-axis, if 1 no pooling will be performed.",
    )

    # Training Parameters
    training_parser = argparse.ArgumentParser(add_help=False)
    training_parser.add_argument(
        "--epochs",
        default=12,
        type=int,
        help="Number of training epochs.",
    )
    training_parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Mini batch size for stochastic gradient descent algorithms.",
    )
    training_parser.add_argument(
        "--train_csv",
        help="Path to CSV with Sample IDs to reserve for training.",
    )
    training_parser.add_argument(
        "--valid_csv",
        help=(
            "Path to CSV with Sample IDs to reserve for validation. Takes precedence"
            " over valid_ratio."
        ),
    )
    training_parser.add_argument(
        "--test_csv",
        help=(
            "Path to CSV with Sample IDs to reserve for testing. Takes precedence over"
            " test_ratio."
        ),
    )
    training_parser.add_argument(
        "--valid_ratio",
        default=0.2,
        type=float,
        help=(
            "Rate of training tensors to save for validation must be in [0.0, 1.0]. If"
            " any of train/valid/test csv is specified, split by ratio is applied on"
            " the remaining tensors after reserving tensors given by csvs. If not"
            " specified, default 0.2 is used. If default ratios are used with"
            " train_csv, some tensors may be ignored because ratios do not sum to 1."
        ),
    )
    training_parser.add_argument(
        "--test_ratio",
        default=0.1,
        type=float,
        help=(
            "Rate of training tensors to save for testing must be in [0.0, 1.0]. If any"
            " of train/valid/test csv is specified, split by ratio is applied on the"
            " remaining tensors after reserving tensors given by csvs. If not"
            " specified, default 0.1 is used. If default ratios are used with"
            " train_csv, some tensors may be ignored because ratios do not sum to 1."
        ),
    )
    training_parser.add_argument(
        "--learning_rate",
        default=0.0002,
        type=float,
        help="Learning rate during training.",
    )
    training_parser.add_argument(
        "--learning_rate_patience",
        default=8,
        type=int,
        help="Number of epochs without validation loss improvement to wait before"
        " reducing learning rate by multiplying by the learning_rate_reduction scale"
        " factor.",
    )
    training_parser.add_argument(
        "--learning_rate_reduction",
        default=0.5,
        type=float,
        help="Scale factor to reduce learning rate by.",
    )
    training_parser.add_argument(
        "--mixup_alpha",
        default=0,
        type=float,
        help="If non-zero, mixup batches with this alpha parameter for mixup.",
    )
    training_parser.add_argument(
        "--patience",
        default=24,
        type=int,
        help=(
            "Early Stopping parameter: Maximum number of epochs to run without"
            " validation loss improvements."
        ),
    )
    training_parser.add_argument(
        "--balance_csvs",
        default=[],
        nargs="*",
        help=(
            "Balances batches with representation from sample IDs in this list of CSVs"
        ),
    )
    training_parser.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        help="Optimizer for model training",
    )
    training_parser.add_argument(
        "--learning_rate_schedule",
        type=str,
        choices=["triangular", "triangular2"],
        help="Adjusts learning rate during training.",
    )
    training_parser.add_argument(
        "--anneal_rate",
        default=0.0,
        type=float,
        help="Annealing rate in epochs of loss terms during training",
    )
    training_parser.add_argument(
        "--anneal_shift",
        default=0.0,
        type=float,
        help="Annealing offset in epochs of loss terms during training",
    )
    training_parser.add_argument(
        "--anneal_max",
        default=2.0,
        type=float,
        help="Annealing maximum value",
    )

    # Training modes parsers
    train_parser = subparser.add_parser(
        name="train",
        description="TODO",
        parents=[model_parser, training_parser, io_parser, run_parser, tmap_parser],
    )
    train_shallow_parser = subparser.add_parser(
        name="train_shallow",
        description="TODO",
        parents=[model_parser, training_parser, io_parser, run_parser, tmap_parser],
    )
    train_simclr_parser = subparser.add_parser(
        name="train_simclr",
        description="TODO",
        parents=[model_parser, training_parser, io_parser, run_parser, tmap_parser],
    )
    hyperoptimize_parser = subparser.add_parser(
        name="hyperoptimize",
        description="TODO",
        parents=[model_parser, training_parser, io_parser, run_parser, tmap_parser],
    )

    # Hyperoptimize arguments
    hyperoptimize_parser.add_argument(
        "--max_parameters",
        default=9000000,
        type=int,
        help="Maximum number of trainable parameters in a model during "
        "hyperoptimization.",
    )
    hyperoptimize_parser.add_argument(
        "--max_evals",
        default=16,
        type=int,
        help=(
            "Maximum number of models for the hyperparameter optimizer to evaluate"
            " before returning."
        ),
    )
    hyperoptimize_parser.add_argument(
        "--l1",
        default=0.0,
        type=float,
        help="L1 value for regularization in shallow model.",
    )
    hyperoptimize_parser.add_argument(
        "--l2",
        default=0.0,
        type=float,
        help="L2 value for regularization in shallow model.",
    )

    # Explore arguments
    explore_parser = subparser.add_parser(
        name="explore",
        description="TODO",
        parents=[io_parser, run_parser, tmap_parser, training_parser],
    )
    explore_parser.add_argument(
        "--explore_disable_saving_output",
        action="store_true",
        help="Disable saving outputs from explore: histograms, summary statistics, "
        "and tensors.",
    )
    explore_parser.add_argument(
        "--explore_export_error",
        action="store_true",
        help="Export error_type in tensors_all_*.csv generated by explore.",
    )
    explore_parser.add_argument(
        "--explore_export_fpath",
        action="store_true",
        help="Export path to HD5 in tensors_all_*.csv generated by explore.",
    )
    explore_parser.add_argument(
        "--explore_export_generator",
        action="store_true",
        help="Export generator (e.g. train, valid, or test split) in "
        "tensors_all_*.csv generated by explore.",
    )
    explore_parser.add_argument(
        "--explore_stratify_label",
        help=(
            "TensorMap or column name of value in CSV to stratify distribution around,"
            " e.g. mortality. Optional."
        ),
    )
    explore_parser.add_argument(
        "--source_name",
        default="ecg",
        help=(
            "Name of source dataset at tensors, e.g. ECG. "
            "Adds contextual detail to summary CSV and plots."
        ),
    )
    explore_parser.add_argument(
        "--join_tensors",
        nargs="+",
        help=(
            "TensorMap or column name in csv of value in tensors used in join with"
            " reference. Can be more than 1 join value."
        ),
    )
    explore_parser.add_argument(
        "--time_tensor",
        help=(
            "TensorMap or column name in csv of value in tensors to perform time"
            " cross-ref on. Time cross referencing is optional."
        ),
    )
    explore_parser.add_argument(
        "--reference_tensors",
        help="Either a csv or directory of hd5 containing a reference dataset.",
    )
    explore_parser.add_argument(
        "--reference_name",
        default="Reference",
        help=(
            "Name of dataset at reference, e.g. STS. "
            "Adds contextual detail to summary CSV and plots."
        ),
    )
    explore_parser.add_argument(
        "--reference_join_tensors",
        nargs="+",
        help=(
            "TensorMap or column name in csv of value in reference used in join in"
            " tensors. Can be more than 1 join value."
        ),
    )
    explore_parser.add_argument(
        "--reference_start_time_tensor",
        action="append",
        nargs="+",
        help=(
            "TensorMap or column name in csv of start of time window in reference."
            " Define multiple time windows by using this argument more than once. The"
            " number of time windows must match across all time window arguments. An"
            " integer can be provided as a second argument to specify an offset to the"
            " start time. e.g. tStart -30"
        ),
    )
    explore_parser.add_argument(
        "--reference_end_time_tensor",
        action="append",
        nargs="+",
        help=(
            "TensorMap or column name in csv of end of time window in reference. Define"
            " multiple time windows by using this argument more than once. The number"
            " of time windows must match across all time window arguments. An integer"
            " can be provided as a second argument to specify an offset to the end"
            " time. e.g. tEnd 30"
        ),
    )
    explore_parser.add_argument(
        "--window_name",
        action="append",
        help=(
            "Name of time window. By default, name of window is index of window."
            " Define multiple time windows by using this argument multiple times."
            " The number of time windows must match across all time window arguments."
        ),
    )
    explore_parser.add_argument(
        "--order_in_window",
        action="append",
        choices=["newest", "oldest", "random"],
        help=(
            "If specified, exactly --number_in_window rows with join tensor are used in"
            " time window. Defines which source tensors in a time series to use in time"
            " window. Define multiple time windows by using this argument more than"
            " once. The number of time windows must match across all time window"
            " arguments."
        ),
    )
    explore_parser.add_argument(
        "--number_per_window",
        type=int,
        default=1,
        help=(
            "Minimum number of rows with join tensor to use in each time window. "
            "By default, 1 tensor is used for each window."
        ),
    )
    explore_parser.add_argument(
        "--match_any_window",
        action="store_true",
        help=(
            "If specified, join tensor does not need to be found in every time window."
            " Join tensor needs only be found in at least 1 time window. Default only"
            " use rows with join tensor that appears across all time windows."
        ),
    )
    explore_parser.add_argument(
        "--reference_labels",
        nargs="+",
        help=(
            "TensorMap or column name of values in csv to report distribution on, e.g."
            " mortality. Label distribution reporting is optional. Can list multiple"
            " labels to report."
        ),
    )

    # Check structure parser
    check_structure_parser = subparser.add_parser(
        "check_structure",
        description="Verify EDW and BM files and directories structure "
        "before tensorizing.",
        parents=[icu_parser],
    )
    check_structure_parser.add_argument(
        "--check_edw",
        action="store_true",
        help="If this parameter is set, the EDW files and directory structure "
        "are verified.",
    )
    check_structure_parser.add_argument(
        "--check_bm",
        action="store_true",
        help="If this parameter is set, the Bedmaster files and directory "
        "structure are verified.",
    )

    # Explore parser
    explore_icu_parser = subparser.add_parser(
        "explore_icu",
        description="Calculate summary statistics after tensorizing.",
        parents=[io_parser, run_parser, tmap_parser],
    )
    explore_icu_parser.add_argument(
        "--output_files_prefix",
        default="post_tensorize",
        help="Base name of the summary stats .csv files. Default: post_tensorize",
    )
    explore_icu_parser.add_argument(
        "--no_csns",
        action="store_true",
        help="If the parameter is set, the explore mode won't loop through csns.",
    )

    # Match patient parser
    match_patient_parser = subparser.add_parser(
        "match_patient_bm",
        description="Create a cross-reference "
        "table between the available BM files and EDW data from an ADT table.",
        parents=[io_parser, run_parser, icu_parser],
    )
    match_patient_parser.add_argument(
        "--lm4",
        action="store_true",
        help="If argument set, the matching is performed in LM4 sever.",
    )
    match_patient_parser.add_argument(
        "--desired_depts",
        nargs="+",
        default=None,
        help="List indicating all the desired departments in which BedMaster"
        "files are matched with patients.",
    )

    # Pre tensorize summary parser
    pre_tensorize_summary_parser = subparser.add_parser(
        "pre_tensorize_summary",
        description="Calculate summary statistics before tensorizing.",
        parents=[io_parser, run_parser, icu_parser],
    )
    pre_tensorize_summary_parser.add_argument(
        "--summary_stats_base_name",
        default="pre_tensorize",
        help="Base name of the summary stats .csv files. " "By default: pre_tensorize",
    )
    pre_tensorize_summary_parser.add_argument(
        "--signals",
        nargs="+",
        default=None,
        help="List of BM signals to calculate their summary_statistics. To "
        "calculate statistics for all signals select 'all'. It always "
        "calculates statistics for all EDW signals.",
    )
    pre_tensorize_summary_parser.add_argument(
        "--detailed_bm",
        action="store_true",
        help="Generate detailed statistics for Bedmaster, like time "
        "irregularities frequence. This option can take some time"
        "to complete.",
    )
    pre_tensorize_summary_parser.add_argument(
        "--no_xref",
        action="store_true",
        help="Don't cross-reference files. Enable along with --detailed_bm "
        "to get statistics from bedmaster files without needing "
        "an edw and xref path.",
    )

    # Assess coverage parser
    assess_coverage_parser = subparser.add_parser(
        "assess_coverage",
        description="Assess BM and HD5 coverage by means of MRNs and CSNs.",
        parents=[io_parser, run_parser, icu_parser],
    )
    assess_coverage_parser.add_argument(
        "--cohort_query",
        type=str,
        default=None,
        help="Name of the query to obtain list of patients and ADT table.",
    )
    assess_coverage_parser.add_argument(
        "--event_column",
        type=str,
        default=None,
        help="Name of the event column (if exists) in --cohort_query/--cohort_csv.",
    )
    assess_coverage_parser.add_argument(
        "--time_column",
        type=str,
        default=None,
        help="Name of the event time column (if exists) in "
        "--cohort_query/--cohort_csv.",
    )
    assess_coverage_parser.add_argument(
        "--cohort_csv",
        type=str,
        default=None,
        help="Full path of the .csv file containing a list of patients. "
        "If --cohort_query is set, this parameter will be ignored.",
    )
    assess_coverage_parser.add_argument(
        "--adt_csv",
        type=str,
        default=None,
        help="Full path of the ADT table of the list of patients in --cohort_csv. "
        "If --cohort_query is set, this parameter will be ignored.",
    )
    assess_coverage_parser.add_argument(
        "--desired_depts",
        nargs="+",
        default=None,
        help="List of department names.",
    )
    assess_coverage_parser.add_argument(
        "--count",
        action="store_true",
        help="Count the number of unique rows (events) in --cohort_query/--cohort_csv.",
    )

    # Additional subparsers
    infer_parser = subparser.add_parser(
        name="infer",
        description="TODO",
        parents=[model_parser, training_parser, io_parser, run_parser, tmap_parser],
    )
    plot_parser = subparser.add_parser(
        name="plot_ecg",
        description="TODO",
        parents=[model_parser, training_parser, io_parser, run_parser, tmap_parser],
    )
    build_parser = subparser.add_parser(
        name="build",
        description="TODO",
        parents=[model_parser, training_parser, io_parser, run_parser, tmap_parser],
    )

    args = parser.parse_args()
    _process_args(args)
    return args


def _process_args(args: argparse.Namespace):
    now_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    args_file = os.path.join(
        args.output_folder,
        args.id,
        "arguments_" + now_string + ".txt",
    )
    command_line = f"\n./scripts/tf.sh {' '.join(sys.argv)}\n"
    if not os.path.exists(os.path.dirname(args_file)):
        os.makedirs(os.path.dirname(args_file))
    with open(args_file, "w") as f:
        f.write(command_line)
        for k, v in sorted(args.__dict__.items(), key=operator.itemgetter(0)):
            f.write(k + " = " + str(v) + "\n")
    load_config(
        args.logging_level,
        os.path.join(args.output_folder, args.id),
        "log_" + now_string,
    )

    ml4c3.definitions.STS_DATA_CSV = args.sts_csv

    # Create list of names of all needed TMaps
    if "input_tensors" in args and args.mode != "explore_icu":
        needed_tmaps_names = args.input_tensors + args.output_tensors

        # Update dict of tmaps to include all needed tmaps
        tmaps: Dict[str, TensorMap] = {}
        for tmap_name in needed_tmaps_names:
            tmaps = update_tmaps(tmap_name=tmap_name, tmaps=tmaps)

        # Update args with TMaps
        args.tensor_maps_in = [tmaps[tmap_name] for tmap_name in args.input_tensors]
        args.tensor_maps_out = [tmaps[tmap_name] for tmap_name in args.output_tensors]
    if "bottleneck_type" in args:
        args.bottleneck_type = BOTTLENECK_STR_TO_ENUM[args.bottleneck_type]
    if "learning_rate_schedule" in args:
        if args.learning_rate_schedule is not None and args.patience < args.epochs:
            raise ValueError(
                "learning_rate_schedule is not compatible with ReduceLROnPlateau. "
                "Set patience > epochs.",
            )

    np.random.seed(args.random_seed)

    # Replace tildes with full path to home dirs
    if "bad_xml_dir" in args and args.bad_xml_dir:
        args.bad_xml_dir = os.path.expanduser(args.bad_xml_dir)
    if "bad_hd5_dir" in args and args.bad_hd5_dir:
        args.bad_hd5_dir = os.path.expanduser(args.bad_hd5_dir)

    logging.info(f"Command Line was: {command_line}")
    if "input_tensors" in args and args.mode != "explore_icu":
        logging.info(f"Total TensorMaps: {len(tmaps)} Arguments are {args}")

    if args.eager:
        # Imports: third party
        import tensorflow as tf

        tf.config.experimental_run_functions_eagerly(True)

    if "layer_order" in args and len(set(args.layer_order)) != 3:
        raise ValueError(
            "Activation, normalization, and regularization layers must each be listed"
            f" exactly once for valid ordering. Got : {args.layer_order}",
        )

    if args.num_workers <= 0:
        raise ValueError(
            f"num_workers must be a positive integer, got: {args.num_workers}",
        )

    if args.remap_layer is not None:
        args.remap_layer = {
            pretrained_layer: new_layer
            for pretrained_layer, new_layer in args.remap_layer
        }
