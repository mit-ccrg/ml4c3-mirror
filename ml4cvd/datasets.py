# Imports: standard library
import os
import math
import logging
from typing import Set, Dict, List, Tuple, Union, Callable, Optional, Generator
from collections import Counter, defaultdict
from multiprocessing import Event, Queue, Process

# Imports: third party
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

# Imports: first party
from ml4cvd.definitions import CSV_EXT, TENSOR_EXT, MRN_COLUMNS
from ml4cvd.tensormap.TensorMap import TensorMap, id_from_filename

SampleGenerator = Generator[
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]], None, None,
]
Cleanup = Callable[[], None]
MAX_QUEUE_SIZE = 2048
PATH_SUCCEEDED = 0
PATH_FAILED = 1
SAMPLE_FAILED = 2

BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX, BATCH_PATHS_INDEX = (
    0,
    1,
    2,
)


def _get_sample(
    tmaps: List[TensorMap],
    sample_tensors: Dict[str, np.ndarray],
    is_input: bool,
    augment: bool,
    hd5: h5py.File,
) -> Dict[str, np.ndarray]:
    sample = dict()
    for tm in tmaps:
        name = tm.input_name if is_input else tm.output_name
        sample[name] = tm.postprocess_tensor(
            tensor=sample_tensors[tm.name], hd5=hd5, augment=augment,
        )
    return sample


def _tensor_worker(
    worker_name: str,
    paths: List[str],
    start_signal: Event,
    tensor_queue: Queue,
    input_maps: List[TensorMap],
    output_maps: List[TensorMap],
    augment: bool = False,
):
    tmaps = input_maps + output_maps
    is_input_list = [True] * len(input_maps) + [False] * len(output_maps)
    while True:
        start_signal.wait()
        start_signal.clear()
        np.random.shuffle(paths)
        for path in paths:
            try:
                # A path may contain many samples. TMaps fetch all the samples from a
                # path but generator should yield individual samples.
                #
                # tensors = [
                #     [sample_1_voltage, sample_2_voltage, ...],
                #     [sample_1_age,     sample_2_age,     ...],
                #     [sample_1_sex,     sample_2_sex,     ...],
                #     ...
                # ]
                tensors = []
                with h5py.File(path, "r") as hd5:
                    # load all samples found at path
                    for tm in tmaps:
                        _tensor = tm.tensor_from_file(tm, hd5)
                        # if tensor is not dynamically shaped,
                        # wrap in extra dimension to simulate time series with 1 sample
                        if tm.time_series_limit is None:
                            _tensor = np.array([_tensor])
                        tensors.append(_tensor)

                    # individually yield samples
                    for i in range(len(tensors[0])):
                        sample_tensors = {
                            tm.name: tensor[i] for tm, tensor in zip(tmaps, tensors)
                        }
                        try:
                            in_tensors = _get_sample(
                                tmaps=input_maps,
                                sample_tensors=sample_tensors,
                                is_input=True,
                                augment=augment,
                                hd5=hd5,
                            )
                            out_tensors = _get_sample(
                                tmaps=output_maps,
                                sample_tensors=sample_tensors,
                                is_input=False,
                                augment=augment,
                                hd5=hd5,
                            )
                            tensor_queue.put((in_tensors, out_tensors, path))
                        except (
                            IndexError,
                            KeyError,
                            ValueError,
                            OSError,
                            RuntimeError,
                        ) as e:
                            # sample failed postprocessing
                            tensor_queue.put(SAMPLE_FAILED)
                            continue
                tensor_queue.put(PATH_SUCCEEDED)
            except (IndexError, KeyError, ValueError, OSError, RuntimeError) as e:
                # could not load samples from path
                tensor_queue.put(PATH_FAILED)


class StatsWrapper:
    def __init__(self):
        self.stats = Counter()


def make_data_generator_factory(
    data_split: str,
    num_workers: int,
    input_maps: List[TensorMap],
    output_maps: List[TensorMap],
    paths: List[str],
    augment: bool = False,
    keep_paths: bool = False,
) -> Tuple[Callable[[], SampleGenerator], StatsWrapper, Cleanup]:
    tensor_queue = Queue()

    processes = []
    worker_paths = np.array_split(paths, num_workers)
    for i, _paths in enumerate(worker_paths):
        name = f"{data_split}_worker_{i}"
        start_signal = Event()
        process = Process(
            target=_tensor_worker,
            name=name,
            args=(
                name,
                _paths,
                start_signal,
                tensor_queue,
                input_maps,
                output_maps,
                augment,
            ),
        )
        process.start()
        process.start_signal = start_signal
        processes.append(process)
    logging.info(f"Started {num_workers} {data_split} workers.")

    def cleanup_workers():
        for process in processes:
            process.terminate()
        tensor_queue.close()
        logging.info(f"Stopped {num_workers} {data_split} workers.")

    stats_wrapper = StatsWrapper()
    epoch_counter = 0
    name = f"{data_split}_dataset"

    def data_generator_factory() -> SampleGenerator:
        nonlocal epoch_counter
        epoch_counter += 1
        for process in processes:
            process.start_signal.set()

        stats = Counter()
        num_paths = len(paths)
        while stats["paths_completed"] < num_paths:
            sample = tensor_queue.get()
            if sample == PATH_SUCCEEDED:
                stats["paths_succeeded"] += 1
                stats["paths_completed"] += 1
            elif sample == PATH_FAILED:
                stats["paths_failed"] += 1
                stats["paths_completed"] += 1
            elif sample == SAMPLE_FAILED:
                stats["samples_failed"] += 1
                stats["samples_completed"] += 1
            else:
                stats["samples_succeeded"] += 1
                stats["samples_completed"] += 1
                _collect_sample_stats(stats, sample, input_maps, output_maps)
                yield sample if keep_paths else sample[:BATCH_PATHS_INDEX]

        logging.info(
            f"{get_stats_string(name, stats, epoch_counter)}"
            f"{get_verbose_stats_string({data_split: stats}, input_maps, output_maps)}",
        )

        nonlocal stats_wrapper
        stats_wrapper.stats = stats

    return data_generator_factory, stats_wrapper, cleanup_workers


def make_dataset(
    data_split: str,
    input_maps: List[TensorMap],
    output_maps: List[TensorMap],
    paths: List[str],
    batch_size: int,
    num_workers: int,
    augment: bool = False,
    keep_paths: bool = False,
) -> Tuple[tf.data.Dataset, StatsWrapper, Cleanup]:
    output_types = (
        {tm.input_name: tf.float32 for tm in input_maps},
        {tm.output_name: tf.float32 for tm in output_maps},
    )
    output_shapes = (
        {tm.input_name: tm.shape for tm in input_maps},
        {tm.output_name: tm.shape for tm in output_maps},
    )
    if keep_paths:
        output_types += (tf.string,)
        output_shapes += (tuple(),)
    data_generator_factory, stats_wrapper, cleanup = make_data_generator_factory(
        data_split=data_split,
        num_workers=num_workers,
        input_maps=input_maps,
        output_maps=output_maps,
        paths=paths,
        augment=augment,
        keep_paths=keep_paths,
    )
    dataset = (
        tf.data.Dataset.from_generator(
            generator=data_generator_factory,
            output_types=output_types,
            output_shapes=output_shapes,
        )
        .batch(batch_size)
        .prefetch(16)
    )

    return dataset, stats_wrapper, cleanup


def train_valid_test_datasets(
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    tensors: str,
    batch_size: int,
    num_workers: int,
    keep_paths: bool = False,
    keep_paths_test: bool = True,
    sample_csv: str = None,
    valid_ratio: float = None,
    test_ratio: float = None,
    train_csv: str = None,
    valid_csv: str = None,
    test_csv: str = None,
    no_empty_paths_allowed: bool = True,
    output_folder: str = None,
    run_id: str = None,
) -> Tuple[
    Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    Tuple[StatsWrapper, StatsWrapper, StatsWrapper],
    Tuple[Cleanup, Cleanup, Cleanup],
]:
    """
    Get tensorflow.data.Datasets for training, validation and testing data.

    :param tensor_maps_in: list of TensorMaps that are input names to a model
    :param tensor_maps_out: list of TensorMaps that are output from a model
    :param tensors: directory containing tensors
    :param batch_size: number of samples in each batch
    :param num_workers: number of worker processes used to feed each dataset
    :param keep_paths: bool to return the path to each sample's source file
    :param keep_paths_test:  bool to return the path to each sample's source file for the test set
    :param sample_csv: CSV file of sample ids, sample ids are considered for train/valid/test only if it is in sample_csv
    :param valid_ratio: rate of tensors to use for validation, mutually exclusive with valid_csv
    :param test_ratio: rate of tensors to use for testing, mutually exclusive with test_csv
    :param train_csv: CSV file of sample ids to use for training
    :param valid_csv: CSV file of sample ids to use for validation, mutually exclusive with valid_ratio
    :param test_csv: CSV file of sample ids to use for testing, mutually exclusive with test_ratio
    :param no_empty_paths_allowed: if true, all data splits must contain paths, otherwise only one split needs to be non-empty
    :param output_folder: output folder of output files
    :param run_id: id of experiment
    :return: tuple of three tensorflow Datasets, three StatsWrapper objects, and three callbacks to cleanup worker processes
    """
    if len(tensor_maps_in) == 0 or len(tensor_maps_out) == 0:
        raise ValueError(f"input and output tensors must both be given")

    train_paths, valid_paths, test_paths = get_train_valid_test_paths(
        tensors=tensors,
        sample_csv=sample_csv,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
        no_empty_paths_allowed=no_empty_paths_allowed,
    )

    if output_folder is not None and run_id is not None:

        def save_paths(paths: List[str], split: str):
            fpath = os.path.join(output_folder, run_id, f"{split}{CSV_EXT}")
            ids = [id_from_filename(path) for path in paths]
            df = pd.DataFrame({"sample_id": ids})
            df.to_csv(fpath, index=False)
            logging.info(f"--{split}_csv was not provided; saved sample IDs to {fpath}")

        if train_csv is None:
            save_paths(paths=train_paths, split="train")
        if valid_csv is None:
            save_paths(paths=valid_paths, split="valid")
        if test_csv is None:
            save_paths(paths=test_paths, split="test")

    train_dataset, train_stats, train_cleanup = make_dataset(
        data_split="train",
        input_maps=tensor_maps_in,
        output_maps=tensor_maps_out,
        paths=train_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=True,
        keep_paths=keep_paths,
    )
    valid_dataset, valid_stats, valid_cleanup = make_dataset(
        data_split="valid",
        input_maps=tensor_maps_in,
        output_maps=tensor_maps_out,
        paths=valid_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        keep_paths=keep_paths,
    )
    test_dataset, test_stats, test_cleanup = make_dataset(
        data_split="test",
        input_maps=tensor_maps_in,
        output_maps=tensor_maps_out,
        paths=test_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        keep_paths=keep_paths or keep_paths_test,
    )

    return (
        (train_dataset, valid_dataset, test_dataset),
        (train_stats, valid_stats, test_stats),
        (train_cleanup, valid_cleanup, test_cleanup),
    )


def _collect_sample_stats(
    stats: Counter,
    sample: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], str],
    input_maps: List[TensorMap],
    output_maps: List[TensorMap],
) -> None:
    in_tensors, out_tensors, path = sample

    def _update_stats(tmaps, tensors, is_input):
        for tm in tmaps:
            stats[f"{tm.name}_n"] += 1
            if tm.axes == 1:
                tensor = tensors[tm.input_name if is_input else tm.output_name]
                if tm.is_categorical:
                    stats[f"{tm.name}_index_{np.argmax(tensor):.0f}"] += 1
                elif tm.is_continuous:
                    value = tensor[0]
                    min_key = f"{tm.name}_min"
                    max_key = f"{tm.name}_max"
                    if min_key not in stats or value < stats[min_key]:
                        stats[min_key] = value
                    if max_key not in stats or value > stats[max_key]:
                        stats[max_key] = value
                    stats[f"{tm.name}_sum"] += value
                    stats[f"{tm.name}_squared_sum"] += value ** 2

    _update_stats(tmaps=input_maps, tensors=in_tensors, is_input=True)
    _update_stats(tmaps=output_maps, tensors=out_tensors, is_input=False)


def get_stats_string(name: str, stats: Counter, epoch_count: int) -> str:
    # fmt: off
    return (
        f"\n"
        f"------------------- {name} completed true epoch {epoch_count} -------------------\n"
        f"\tGenerator shuffled {stats['paths_completed']} paths. {stats['paths_succeeded']} paths succeeded and {stats['paths_failed']} paths failed.\n"
        f"\tFrom {stats['paths_succeeded']} paths, {stats['samples_completed']} samples were extracted.\n"
        f"\tFrom {stats['samples_completed']} samples, {stats['samples_succeeded']} yielded tensors and {stats['samples_failed']} samples failed.\n"
    )
    # fmt: on


def get_verbose_stats_string(
    split_stats: Dict[str, Counter],
    input_maps: List[TensorMap],
    output_maps: List[TensorMap],
) -> str:
    if len(split_stats) == 1:
        stats = list(split_stats.values())[0]
        dataframes = _get_stats_as_dataframes(stats, input_maps, output_maps)
    else:
        dataframes = _get_stats_as_dataframes_from_multiple_datasets(
            split_stats, input_maps, output_maps,
        )
    continuous_tm_df, categorical_tm_df, other_tm_df = dataframes

    with pd.option_context("display.max_columns", None, "display.max_rows", None):
        continuous_tm_string = (
            f">>>>>>>>>> Continuous Tensor Maps\n{continuous_tm_df}"
            if len(continuous_tm_df) != 0
            else ""
        )

        categorical_tm_strings = []
        for tm in categorical_tm_df.index.get_level_values(
            "TensorMap",
        ).drop_duplicates():
            tm_df = categorical_tm_df.loc[tm]
            categorical_tm_strings.append(
                f">>>>>>>>>> Categorical Tensor Map: [{tm}]\n{tm_df}",
            )

        other_tm_string = (
            f">>>>>>>>>> Other Tensor Maps\n{other_tm_df}"
            if len(other_tm_df) != 0
            else ""
        )

    tensor_stats_string = "\n\n".join(
        [
            s
            for s in [continuous_tm_string] + categorical_tm_strings + [other_tm_string]
            if s != ""
        ],
    )

    return f"\n{tensor_stats_string}\n"


def _get_stats_as_dataframes(
    stats: Counter, input_maps: List[TensorMap], output_maps: List[TensorMap],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    continuous_tmaps = []
    categorical_tmaps = []
    other_tmaps = []
    for tm in input_maps + output_maps:
        if tm.axes == 1 and tm.is_continuous:
            continuous_tmaps.append(tm)
        elif tm.axes == 1 and tm.is_categorical:
            categorical_tmaps.append(tm)
        else:
            other_tmaps.append(tm)

    _stats = defaultdict(list)
    for tm in continuous_tmaps:
        count = stats[f"{tm.name}_n"]
        mean = stats[f"{tm.name}_sum"] / count
        std = np.sqrt((stats[f"{tm.name}_squared_sum"] / count) - (mean ** 2))
        _stats["count"].append(f"{count:.0f}")
        _stats["mean"].append(f"{mean:.2f}")
        _stats["std"].append(f"{std:.2f}")
        _stats["min"].append(f"{stats[f'{tm.name}_min']:.2f}")
        _stats["max"].append(f"{stats[f'{tm.name}_max']:.2f}")
    continuous_tm_df = pd.DataFrame(_stats, index=[tm.name for tm in continuous_tmaps])
    continuous_tm_df.index.name = "TensorMap"

    _stats = defaultdict(list)
    for tm in categorical_tmaps:
        total = stats[f"{tm.name}_n"]
        for channel, index in tm.channel_map.items():
            count = stats[f"{tm.name}_index_{index}"]
            _stats["count"].append(f"{count:.0f}")
            _stats["percent"].append(f"{count / total * 100:.2f}")
            _stats["TensorMap"].append(tm.name)
            _stats["Label"].append(channel)
        _stats["count"].append(f"{total:.0f}")
        _stats["percent"].append("100.00")
        _stats["TensorMap"].append(tm.name)
        _stats["Label"].append("total")
    categorical_tm_df = pd.DataFrame(
        {key: val for key, val in _stats.items() if key in {"count", "percent"}},
        index=pd.MultiIndex.from_tuples(
            zip(_stats["TensorMap"], _stats["Label"]), names=["TensorMap", "Label"],
        ),
    )
    categorical_tm_df.index.name = "TensorMap"

    other_tm_df = pd.DataFrame(
        {"count": [f"{stats[f'{tm.name}_n']:.0f}" for tm in other_tmaps]},
        index=[tm.name for tm in other_tmaps],
    )
    other_tm_df.index.name = "TensorMap"

    return continuous_tm_df, categorical_tm_df, other_tm_df


def _get_stats_as_dataframes_from_multiple_datasets(
    split_stats: Dict[str, Counter],
    input_maps: List[TensorMap],
    output_maps: List[TensorMap],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_to_dataframes = {
        split: _get_stats_as_dataframes(
            stats=stats, input_maps=input_maps, output_maps=output_maps,
        )
        for split, stats in split_stats.items()
    }

    def combine_split_dataframes(split_to_dataframe, index=["TensorMap"]):
        df = (
            pd.concat(split_to_dataframe, names=["Split"])
            .reorder_levels(index + ["Split"])
            .reset_index()
        )
        df["Split"] = pd.Categorical(df["Split"], split_to_dataframe.keys())
        if "Label" in index:
            labels = df["Label"].drop_duplicates()
            df["Label"] = pd.Categorical(df["Label"], labels)
        df = df.set_index(index + ["Split"]).sort_index()
        return df

    continuous_tm_df = combine_split_dataframes(
        {
            split: continuous_df
            for split, (continuous_df, _, _) in split_to_dataframes.items()
        },
    )
    categorical_tm_df = combine_split_dataframes(
        {
            split: categorical_df
            for split, (_, categorical_df, _) in split_to_dataframes.items()
        },
        ["TensorMap", "Label"],
    )
    other_tm_df = combine_split_dataframes(
        {split: other_df for split, (_, _, other_df) in split_to_dataframes.items()},
    )

    return continuous_tm_df, categorical_tm_df, other_tm_df


def _get_train_valid_test_discard_ratios(
    valid_ratio: float,
    test_ratio: float,
    train_csv: str,
    valid_csv: str,
    test_csv: str,
) -> Tuple[int, int, int, int]:

    if valid_csv is not None:
        valid_ratio = 0
    if test_csv is not None:
        test_ratio = 0
    if train_csv is not None:
        train_ratio = 0
        discard_ratio = 1.0 - valid_ratio - test_ratio
    else:
        train_ratio = 1.0 - valid_ratio - test_ratio
        discard_ratio = 0

    if not math.isclose(train_ratio + valid_ratio + test_ratio + discard_ratio, 1.0):
        raise ValueError(
            "ratios do not sum to 1, train/valid/test/discard ="
            f" {train_ratio}/{valid_ratio}/{test_ratio}/{discard_ratio}",
        )
    logging.debug(
        "train/valid/test/discard ratios:"
        f" {train_ratio}/{valid_ratio}/{test_ratio}/{discard_ratio}",
    )

    return train_ratio, valid_ratio, test_ratio, discard_ratio


def _sample_csv_to_set(
    sample_csv: Optional[str] = None, mrn_col_name: Optional[str] = None,
) -> Union[None, Set[str]]:

    if sample_csv is None:
        return None

    # Read CSV to dataframe and assume no header
    df = pd.read_csv(sample_csv, header=None, low_memory=False)

    # If first row and column is castable to int, there is no header
    try:
        int(df.iloc[0].values[0])

    # If fails, must be header; overwrite column name with first row and remove first row
    except ValueError:
        df.columns = df.iloc[0]
        df = df[1:]
    if mrn_col_name is None:
        df.columns = [
            col.lower() if isinstance(col, str) else col for col in df.columns
        ]

        # Find intersection between CSV columns and possible MRN column names
        matches = set(df.columns).intersection(MRN_COLUMNS)

        # If no matches, assume the first column is MRN
        if not matches:
            mrn_col_name = df.columns[0]
        else:
            # Get first string from set of matches to use as column name
            mrn_col_name = next(iter(matches))

        if len(matches) > 1:
            logging.warning(
                f"{sample_csv} has more than one potential column for MRNs. Inferring most"
                " likely column name, but recommend explicitly setting MRN column name.",
            )

    # Isolate MRN column from dataframe, cast to float -> int -> string
    sample_ids = df[mrn_col_name].astype(float).astype(int).apply(str)

    return set(sample_ids)


def get_train_valid_test_paths(
    tensors: str,
    sample_csv: str,
    valid_ratio: float,
    test_ratio: float,
    train_csv: str,
    valid_csv: str,
    test_csv: str,
    no_empty_paths_allowed: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Return 3 disjoint lists of tensor paths.

    The paths are split in training, validation, and testing lists.
    If no arguments are given, paths are split into train/valid/test in the ratio 0.7/0.2/0.1.
    Otherwise, at least 2 arguments are required to specify train/valid/test sets.

    :param tensors: path to directory containing tensors
    :param sample_csv: path to csv containing sample ids, only consider sample ids for splitting
                       into train/valid/test sets if they appear in sample_csv
    :param valid_ratio: rate of tensors in validation list, mutually exclusive with valid_csv
    :param test_ratio: rate of tensors in testing list, mutually exclusive with test_csv
    :param train_csv: path to csv containing sample ids to reserve for training list
    :param valid_csv: path to csv containing sample ids to reserve for validation list, mutually exclusive with valid_ratio
    :param test_csv: path to csv containing sample ids to reserve for testing list, mutually exclusive with test_ratio
    :param no_empty_paths_allowed: If true, all data splits must contain paths, otherwise only one split needs to be non-empty

    :return: tuple of 3 lists of hd5 tensor file paths
    """
    train_paths = []
    valid_paths = []
    test_paths = []
    discard_paths = []

    (
        train_ratio,
        valid_ratio,
        test_ratio,
        discard_ratio,
    ) = _get_train_valid_test_discard_ratios(
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
    )

    choices = {
        "train": (train_paths, train_ratio),
        "valid": (valid_paths, valid_ratio),
        "test": (test_paths, test_ratio),
        "discard": (discard_paths, discard_ratio),
    }

    sample_set = _sample_csv_to_set(sample_csv)
    train_set = _sample_csv_to_set(train_csv)
    valid_set = _sample_csv_to_set(valid_csv)
    test_set = _sample_csv_to_set(test_csv)

    if (
        train_set is not None
        and valid_set is not None
        and not train_set.isdisjoint(valid_set)
    ):
        raise ValueError("train and validation samples overlap")
    if (
        train_set is not None
        and test_set is not None
        and not train_set.isdisjoint(test_set)
    ):
        raise ValueError("train and test samples overlap")
    if (
        valid_set is not None
        and test_set is not None
        and not valid_set.isdisjoint(test_set)
    ):
        raise ValueError("validation and test samples overlap")

    # Find tensors and split them among train/valid/test
    for root, dirs, files in os.walk(tensors):
        for fname in files:
            if not fname.endswith(TENSOR_EXT):
                continue

            path = os.path.join(root, fname)
            split = os.path.splitext(fname)
            sample_id = split[0]

            if sample_set is not None and sample_id not in sample_set:
                continue
            elif train_set is not None and sample_id in train_set:
                train_paths.append(path)
            elif valid_set is not None and sample_id in valid_set:
                valid_paths.append(path)
            elif test_set is not None and sample_id in test_set:
                test_paths.append(path)
            else:
                choice = np.random.choice(
                    [k for k in choices], p=[choices[k][1] for k in choices],
                )
                choices[choice][0].append(path)

    logging.info(
        f"Found {len(train_paths)} train, {len(valid_paths)} validation, and"
        f" {len(test_paths)} testing tensors at: {tensors}",
    )
    logging.debug(f"Discarded {len(discard_paths)} tensors due to given ratios")
    if (
        no_empty_paths_allowed
        and (len(train_paths) == 0 or len(valid_paths) == 0 or len(test_paths) == 0)
    ) or (len(train_paths) == 0 and len(valid_paths) == 0 and len(test_paths) == 0):
        raise ValueError(
            f"Not enough tensors at {tensors}\n"
            f"Found {len(train_paths)} training,"
            f" {len(valid_paths)} validation, and"
            f" {len(test_paths)} testing tensors\n"
            f"Discarded {len(discard_paths)} tensors",
        )

    return train_paths, valid_paths, test_paths