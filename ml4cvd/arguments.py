# Imports: standard library
import os
import sys
import logging
import argparse
import datetime
import operator
import multiprocessing

# Imports: third party
import numpy as np

# Imports: first party
import ml4cvd.definitions
from ml4cvd.logger import load_config
from ml4cvd.models import BottleneckType
from ml4cvd.definitions import STS_DATA_CSV
from ml4cvd.tensormap.TensorMap import update_tmaps

BOTTLENECK_STR_TO_ENUM = {
    "flatten_restructure": BottleneckType.FlattenRestructure,
    "global_average_pool": BottleneckType.GlobalAveragePoolStructured,
    "variational": BottleneckType.Variational,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", help="What would you like to do?")

    # Config arguments
    parser.add_argument(
        "--logging_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=(
            "Logging level. Overrides any configuration given in the logging"
            " configuration file."
        ),
    )

    # Tensorize arguments
    parser.add_argument(
        "--bad_xml_dir",
        default="~/bad-xml",
        help="Path to directory to store XML files that fail tensorization.",
    )
    parser.add_argument(
        "--bad_hd5_dir",
        default="~/bad-hd5",
        help="Path to directory to store HD5 files that fail tensorization.",
    )
    parser.add_argument(
        "--xml_folder", default="", help="Path to folder of XMLs of ECG data.",
    )
    parser.add_argument(
        "--mrn_key_in_xml",
        default="PatientID",
        help="Name of key (tag) in XML file encoding MRN (e.g. patient ID)",
    )
    parser.add_argument(
        "--tensorize_dry_run",
        action="store_true",
        help="Do not convert XMLs to HD5s; only map XMLs to MRNs",
    )

    # Tensor Map arguments
    parser.add_argument("--input_tensors", default=[], nargs="+")
    parser.add_argument("--output_tensors", default=[], nargs="+")
    parser.add_argument(
        "--tensor_maps_in",
        default=[],
        help="Do not set this directly. Use input_tensors",
    )
    parser.add_argument(
        "--tensor_maps_out",
        default=[],
        help="Do not set this directly. Use output_tensors",
    )
    parser.add_argument(
        "--mrn_column_name",
        default="medrecn",
        help="Name of MRN column in tensors_all*.csv",
    )

    # Input and Output files and directories
    parser.add_argument(
        "--sample_csv", help="Path to CSV with Sample IDs to restrict tensor paths",
    )
    parser.add_argument(
        "--tensors",
        help="Path to folder containing tensors, or where tensors will be written.",
    )
    parser.add_argument(
        "--output_folder",
        default="./ml4cvd-output",
        help="Path to output folder for recipes.py runs.",
    )
    parser.add_argument(
        "--model_file", help="Path to a saved model architecture and weights (hd5).",
    )
    parser.add_argument(
        "--model_files",
        nargs="*",
        default=[],
        help="List of paths to saved model architectures and weights (hd5).",
    )
    parser.add_argument(
        "--donor_layers",
        help=(
            "Path to a model file (hd5) which will be loaded by layer, useful for"
            " transfer learning."
        ),
    )
    parser.add_argument(
        "--remap_layer",
        action="append",
        nargs=2,
        help="For transfer layer, manually remap layer from pretrained model to layer"
        " in new model. For example: --rename_layer pretrained_layer_name new_layer_name."
        " Layers are remapped using this argument one at a time, repeat for multiple layers.",
    )
    parser.add_argument(
        "--freeze_donor_layers",
        action="store_true",
        help="Whether to freeze the layers from donor_layers.",
    )

    # Model Architecture Parameters
    parser.add_argument(
        "--activation",
        default="relu",
        help="Activation function for hidden units in neural nets dense layers.",
    )
    parser.add_argument(
        "--block_size",
        default=3,
        type=int,
        help="Number of convolutional layers within a block.",
    )
    parser.add_argument(
        "--bottleneck_type",
        type=str,
        default=list(BOTTLENECK_STR_TO_ENUM)[0],
        choices=list(BOTTLENECK_STR_TO_ENUM),
    )
    parser.add_argument(
        "--conv_layers",
        nargs="*",
        default=[32],
        type=int,
        help="List of number of kernels in convolutional layers.",
    )
    parser.add_argument(
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
    parser.add_argument(
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
    parser.add_argument(
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
    parser.add_argument(
        "--conv_dilate",
        default=False,
        action="store_true",
        help="Dilate the convolutional layers.",
    )
    parser.add_argument(
        "--conv_dropout",
        default=0.0,
        type=float,
        help="Dropout rate of convolutional kernels must be in [0.0, 1.0].",
    )
    parser.add_argument(
        "--conv_type",
        default="conv",
        choices=["conv", "separable", "depth"],
        help="Type of convolutional layer",
    )
    parser.add_argument(
        "--conv_normalize",
        choices=["", "batch_norm"],
        help="Type of normalization layer for convolutions",
    )
    parser.add_argument(
        "--conv_regularize",
        choices=["dropout", "spatial_dropout"],
        help="Type of regularization layer for convolutions.",
    )
    parser.add_argument(
        "--dense_blocks",
        nargs="*",
        default=[32, 24, 16],
        type=int,
        help="List of number of kernels in convolutional layers.",
    )
    parser.add_argument(
        "--dense_layers",
        nargs="*",
        default=[16, 64],
        type=int,
        help="List of number of hidden units in neural nets dense layers.",
    )
    parser.add_argument(
        "--directly_embed_and_repeat",
        type=int,
        help="If set, directly embed input tensors (without passing to a dense layer)"
        " into concatenation layer, and repeat each input N times, where N is this"
        " argument's value. To directly embed a feature without repetition, set to 1.",
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="Dropout rate of dense layers must be in [0.0, 1.0].",
    )
    parser.add_argument(
        "--language_layer",
        default="ecg_rest_text",
        help="Name of TensorMap for learning language models (eg train_char_model).",
    )
    parser.add_argument(
        "--language_prefix",
        default="partners_ecg_rest",
        help=(
            "Path prefix for a TensorMap to learn language models (eg train_char_model)"
        ),
    )
    parser.add_argument(
        "--layer_order",
        nargs=3,
        default=["activation", "regularization", "normalization"],
        choices=["activation", "normalization", "regularization"],
        help=(
            "Order of activation, regularization, and normalization layers following"
            " convolutional layers."
        ),
    )
    parser.add_argument(
        "--nest_model",
        nargs=2,
        action="append",
        help="Embed a nested model ending at the specified layer before the bottleneck"
        " layer of the current model. Repeat this argument to embed multiple models."
        " For example --nest_model /path/to/model_weights.h5 embed_layer",
    )
    parser.add_argument(
        "--padding",
        default="same",
        help="Valid or same border padding on the convolutional layers.",
    )
    parser.add_argument(
        "--pool_after_final_dense_block",
        default=True,
        action="store_false",
        help="Pool the last layer of all dense blocks.",
    )
    parser.add_argument(
        "--pool_type",
        default="max",
        choices=["max", "average"],
        help="Type of pooling layers.",
    )
    parser.add_argument(
        "--pool_x",
        default=2,
        type=int,
        help="Pooling size in the x-axis, if 1 no pooling will be performed.",
    )
    parser.add_argument(
        "--pool_y",
        default=2,
        type=int,
        help="Pooling size in the y-axis, if 1 no pooling will be performed.",
    )
    parser.add_argument(
        "--pool_z",
        default=1,
        type=int,
        help="Pooling size in the z-axis, if 1 no pooling will be performed.",
    )

    # Training Parameters
    parser.add_argument(
        "--epochs", default=12, type=int, help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Mini batch size for stochastic gradient descent algorithms.",
    )
    parser.add_argument(
        "--train_csv", help="Path to CSV with Sample IDs to reserve for training.",
    )
    parser.add_argument(
        "--valid_csv",
        help=(
            "Path to CSV with Sample IDs to reserve for validation. Takes precedence"
            " over valid_ratio."
        ),
    )
    parser.add_argument(
        "--test_csv",
        help=(
            "Path to CSV with Sample IDs to reserve for testing. Takes precedence over"
            " test_ratio."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
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
    parser.add_argument(
        "--learning_rate",
        default=0.0002,
        type=float,
        help="Learning rate during training.",
    )
    parser.add_argument(
        "--learning_rate_patience",
        default=8,
        type=int,
        help="Number of epochs without validation loss improvement to wait before"
        " reducing learning rate by multiplying by the learning_rate_reduction scale"
        " factor.",
    )
    parser.add_argument(
        "--learning_rate_reduction",
        default=0.5,
        type=float,
        help="Scale factor to reduce learning rate by.",
    )
    parser.add_argument(
        "--mixup_alpha",
        default=0,
        type=float,
        help=(
            "If positive apply mixup and sample from a Beta with this value as shape"
            " parameter alpha."
        ),
    )
    parser.add_argument(
        "--patience",
        default=24,
        type=int,
        help=(
            "Early Stopping parameter: Maximum number of epochs to run without"
            " validation loss improvements."
        ),
    )
    parser.add_argument(
        "--balance_csvs",
        default=[],
        nargs="*",
        help=(
            "Balances batches with representation from sample IDs in this list of CSVs"
        ),
    )
    parser.add_argument(
        "--optimizer", default="adam", type=str, help="Optimizer for model training",
    )
    parser.add_argument(
        "--learning_rate_schedule",
        type=str,
        choices=["triangular", "triangular2"],
        help="Adjusts learning rate during training.",
    )
    parser.add_argument(
        "--anneal_rate",
        default=0.0,
        type=float,
        help="Annealing rate in epochs of loss terms during training",
    )
    parser.add_argument(
        "--anneal_shift",
        default=0.0,
        type=float,
        help="Annealing offset in epochs of loss terms during training",
    )
    parser.add_argument(
        "--anneal_max", default=2.0, type=float, help="Annealing maximum value",
    )

    # Hyperoptimize arguments
    parser.add_argument(
        "--max_parameters",
        default=9000000,
        type=int,
        help="Maximum number of trainable parameters in a model during hyperoptimization.",
    )
    parser.add_argument(
        "--max_evals",
        default=16,
        type=int,
        help=(
            "Maximum number of models for the hyperparameter optimizer to evaluate"
            " before returning."
        ),
    )
    parser.add_argument(
        "--l1",
        default=0.01,
        type=float,
        help="L1 value for regularization in shallow model.",
    )
    parser.add_argument(
        "--l2",
        default=0.01,
        type=float,
        help="L2 value for regularization in shallow model.",
    )

    # Run specific and debugging arguments
    parser.add_argument(
        "--id",
        default="no_id",
        help=(
            "Identifier for this run, user-defined string to keep experiments"
            " organized."
        ),
    )
    parser.add_argument(
        "--random_seed",
        default=12878,
        type=int,
        help="Random seed to use throughout run.  Always use np.random.",
    )
    parser.add_argument(
        "--eager",
        default=False,
        action="store_true",
        help=(
            "Run tensorflow functions in eager execution mode (helpful for debugging)."
        ),
    )
    parser.add_argument(
        "--plot_mode",
        default="clinical",
        choices=["clinical", "full"],
        help="ECG view to plot.",
    )

    # Training optimization options
    parser.add_argument(
        "--num_workers",
        default=multiprocessing.cpu_count(),
        type=int,
        help="Number of workers to use for every tensor generator.",
    )

    # Explore arguments
    parser.add_argument(
        "--explore_disable_saving_output",
        action="store_true",
        help="Disable saving outputs from explore: histograms, summary statistics, and tensors.",
    )
    parser.add_argument(
        "--explore_export_error",
        action="store_true",
        help="Export error_type in tensors_all_*.csv generated by explore.",
    )
    parser.add_argument(
        "--explore_export_fpath",
        action="store_true",
        help="Export path to HD5 in tensors_all_*.csv generated by explore.",
    )
    parser.add_argument(
        "--explore_export_generator",
        action="store_true",
        help="Export generator (e.g. train, valid, or test split) in tensors_all_*.csv generated by explore.",
    )
    parser.add_argument(
        "--explore_stratify_label",
        help=(
            "TensorMap or column name of value in CSV to stratify distribution around,"
            " e.g. mortality. Optional."
        ),
    )
    parser.add_argument(
        "--source_name",
        default="ecg",
        help=(
            "Name of source dataset at tensors, e.g. ECG. "
            "Adds contextual detail to summary CSV and plots."
        ),
    )
    parser.add_argument(
        "--join_tensors",
        nargs="+",
        help=(
            "TensorMap or column name in csv of value in tensors used in join with"
            " reference. Can be more than 1 join value."
        ),
    )
    parser.add_argument(
        "--time_tensor",
        help=(
            "TensorMap or column name in csv of value in tensors to perform time"
            " cross-ref on. Time cross referencing is optional."
        ),
    )
    parser.add_argument(
        "--reference_tensors",
        help="Either a csv or directory of hd5 containing a reference dataset.",
    )
    parser.add_argument(
        "--reference_name",
        default="Reference",
        help=(
            "Name of dataset at reference, e.g. STS. "
            "Adds contextual detail to summary CSV and plots."
        ),
    )
    parser.add_argument(
        "--reference_join_tensors",
        nargs="+",
        help=(
            "TensorMap or column name in csv of value in reference used in join in"
            " tensors. Can be more than 1 join value."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
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
    parser.add_argument(
        "--window_name",
        action="append",
        help=(
            "Name of time window. By default, name of window is index of window."
            " Define multiple time windows by using this argument multiple times."
            " The number of time windows must match across all time window arguments."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--number_per_window",
        type=int,
        default=1,
        help=(
            "Minimum number of rows with join tensor to use in each time window. "
            "By default, 1 tensor is used for each window."
        ),
    )
    parser.add_argument(
        "--match_any_window",
        action="store_true",
        help=(
            "If specified, join tensor does not need to be found in every time window."
            " Join tensor needs only be found in at least 1 time window. Default only"
            " use rows with join tensor that appears across all time windows."
        ),
    )
    parser.add_argument(
        "--reference_labels",
        nargs="+",
        help=(
            "TensorMap or column name of values in csv to report distribution on, e.g."
            " mortality. Label distribution reporting is optional. Can list multiple"
            " labels to report."
        ),
    )

    parser.add_argument(
        "--sts_csv", default=STS_DATA_CSV, help="Path to STS data csv file.",
    )

    args = parser.parse_args()
    _process_args(args)
    return args


def _process_args(args: argparse.Namespace):
    now_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    args_file = os.path.join(
        args.output_folder, args.id, "arguments_" + now_string + ".txt",
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

    ml4cvd.definitions.STS_DATA_CSV = args.sts_csv

    # Create list of names of all needed TMaps
    needed_tmaps_names = args.input_tensors + args.output_tensors

    # Update dict of tmaps to include all needed tmaps
    tmaps = {}
    for tmap_name in needed_tmaps_names:
        tmaps = update_tmaps(tmap_name=tmap_name, tmaps=tmaps)

    # Update args with TMaps
    args.tensor_maps_in = [tmaps[tmap_name] for tmap_name in args.input_tensors]
    args.tensor_maps_out = [tmaps[tmap_name] for tmap_name in args.output_tensors]

    args.bottleneck_type = BOTTLENECK_STR_TO_ENUM[args.bottleneck_type]

    if args.learning_rate_schedule is not None and args.patience < args.epochs:
        raise ValueError(
            f"learning_rate_schedule is not compatible with ReduceLROnPlateau. Set"
            f" patience > epochs.",
        )

    np.random.seed(args.random_seed)

    # Replace tildes with full path to home dirs
    if args.bad_xml_dir:
        args.bad_xml_dir = os.path.expanduser(args.bad_xml_dir)
    if args.bad_hd5_dir:
        args.bad_hd5_dir = os.path.expanduser(args.bad_hd5_dir)

    logging.info(f"Command Line was: {command_line}")
    logging.info(f"Total TensorMaps: {len(tmaps)} Arguments are {args}")

    if args.eager:
        # Imports: third party
        import tensorflow as tf

        tf.config.experimental_run_functions_eagerly(True)

    if len(set(args.layer_order)) != 3:
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
