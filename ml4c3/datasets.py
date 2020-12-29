# Imports: standard library
import os
import math
import logging
from typing import Set, Dict, List, Tuple, Union, Callable, Optional, Generator
from threading import Thread
from collections import Counter, defaultdict
from multiprocessing import Event, Queue, Process

# Imports: third party
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_probability import distributions as tfd

# Imports: first party
from definitions.globals import CSV_EXT, TENSOR_EXT, MRN_COLUMNS
from tensormap.TensorMap import TensorMap, PatientData, find_negative_label_and_channel

SampleGenerator = Generator[
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
    None,
    None,
]
Cleanup = Callable[[], None]
MAX_QUEUE_SIZE = 2048
ID_SUCCEEDED = 0
ID_FAILED = 1
SAMPLE_FAILED = 2

BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX, BATCH_IDS_INDEX = (
    0,
    1,
    2,
)

# pylint: disable=line-too-long


def infer_mrn_column(df: pd.DataFrame, patient_csv: str) -> str:
    matches = []
    for col in df.columns.astype(str):
        if col.lower() in MRN_COLUMNS:
            matches.append(col)
    if len(matches) == 0:
        return df.columns[0]
    if len(matches) > 1:
        logging.warning(
            f"{patient_csv} has more than one potential column for MRNs. "
            "Inferring most likely column name, but recommend verifying "
            "data columns or not using column inference.",
        )
    return matches[0]


def _get_sample(
    tmaps: List[TensorMap],
    sample_tensors: Dict[str, np.ndarray],
    is_input: bool,
    augment: bool,
    validate: bool,
    normalize: bool,
    return_nan: bool,
    data: PatientData,
) -> Dict[str, np.ndarray]:
    sample = dict()
    for tm in tmaps:
        name = tm.input_name if is_input else tm.output_name
        try:
            sample[name] = tm.postprocess_tensor(
                tensor=sample_tensors[tm.name],
                data=data,
                augment=augment,
                validate=validate,
                normalize=normalize,
            )
        except (
            TypeError,
            IndexError,
            KeyError,
            ValueError,
            OSError,
            RuntimeError,
        ) as exception:
            if return_nan:
                nans = np.empty(tm.shape)
                nans.fill("" if tm.is_language else np.nan)
                sample[name] = nans
            else:
                raise exception

    return sample


def _tensor_worker(
    worker_name: str,
    hd5_sources: List[List[str]],
    csv_data: List[Tuple[str, pd.DataFrame, str]],
    patient_ids: List[int],
    start_signal: Event,
    tensor_queue: Queue,
    input_tmaps: List[TensorMap],
    output_tmaps: List[TensorMap],
    augment: bool = False,
    validate: bool = True,
    normalize: bool = True,
    return_nan: bool = False,
):
    tmaps = input_tmaps + output_tmaps

    while True:
        start_signal.wait()
        start_signal.clear()
        np.random.shuffle(patient_ids)
        for patient_id in patient_ids:
            num_linked = 1
            open_hd5s = []
            try:
                # An ID may contain many samples. TMaps fetch all the samples from an
                # ID but generator should yield individual samples.
                #
                # tensors = [
                #     [sample_1_voltage, sample_2_voltage, ...],
                #     [sample_1_age,     sample_2_age,     ...],
                #     [sample_1_sex,     sample_2_sex,     ...],
                #     ...
                # ]
                tensors = []
                data = PatientData(patient_id=patient_id)

                # Add top level groups in hd5s to patient dictionary;
                # iterate over all subdirectories in the hd5 source directory
                # and check if the file exists at that full path; if it does,
                # append open hd5 file object to PatientData and move to next
                # hd5_source
                for hd5_source in hd5_sources:
                    for subdir in hd5_source:
                        hd5_path = os.path.join(subdir, f"{patient_id}.hd5")
                        if not os.path.isfile(hd5_path):
                            continue
                        hd5 = h5py.File(hd5_path, "r")
                        for key in hd5:
                            data[key] = hd5[key]
                        open_hd5s.append(hd5)
                        break
                # Add rows in csv with patient data accessible in patient dictionary
                for csv_name, df, mrn_col in csv_data:
                    mask = df[mrn_col] == patient_id
                    if not mask.any():
                        continue
                    data[csv_name] = df[mask]

                # Load all samples found at ID
                max_len = 1
                for tm in tmaps:
                    try:
                        _tensor = tm.tensor_from_file(tm, data)
                    except (
                        TypeError,
                        IndexError,
                        KeyError,
                        ValueError,
                        OSError,
                        RuntimeError,
                    ) as exception:
                        if return_nan:
                            tensors.append(None)
                            continue
                        raise exception

                    # If tensor is not dynamically shaped,
                    # wrap in extra dimension to simulate time series with 1 sample
                    if tm.time_series_limit is None:
                        _tensor = np.array([_tensor])
                    if len(_tensor) > max_len:
                        max_len = len(_tensor)
                    if tm.linked_tensors:
                        num_linked = len(_tensor)
                    tensors.append(_tensor)

                # If returning nans, replace None with lists of nans of length max_len
                for i, tm in enumerate(tmaps):
                    if tensors[i] is None:
                        nans = np.empty((max_len,) + tm.shape)
                        nans.fill("" if tm.is_language else np.nan)
                        tensors[i] = nans

                # Individually yield samples
                for i in range(max_len):
                    sample_tensors = {
                        tm.name: tensor[i] for tm, tensor in zip(tmaps, tensors)
                    }
                    try:
                        in_tensors = _get_sample(
                            tmaps=input_tmaps,
                            sample_tensors=sample_tensors,
                            is_input=True,
                            augment=augment,
                            validate=validate,
                            normalize=normalize,
                            return_nan=return_nan,
                            data=data,
                        )
                        out_tensors = _get_sample(
                            tmaps=output_tmaps,
                            sample_tensors=sample_tensors,
                            is_input=False,
                            augment=augment,
                            validate=validate,
                            normalize=normalize,
                            return_nan=return_nan,
                            data=data,
                        )
                        tensor_queue.put(
                            ((in_tensors, out_tensors), patient_id, num_linked, None),
                        )
                    except (
                        TypeError,
                        IndexError,
                        KeyError,
                        ValueError,
                        OSError,
                        RuntimeError,
                    ) as exception:
                        # Sample failed postprocessing;
                        # if sample was linked, fail all samples from this ID
                        if num_linked != 1:
                            raise ValueError("Linked sample failed")
                        tensor_queue.put(
                            (SAMPLE_FAILED, patient_id, num_linked, exception),
                        )
                        continue
                tensor_queue.put((ID_SUCCEEDED, patient_id, num_linked, None))
            except (
                TypeError,
                IndexError,
                KeyError,
                ValueError,
                OSError,
                RuntimeError,
            ) as exception:
                # Could not load samples from ID
                tensor_queue.put((ID_FAILED, patient_id, num_linked, exception))
            finally:
                for hd5 in open_hd5s:
                    hd5.close()


class StatsWrapper:
    def __init__(self):
        self.stats = Counter()


def _csv_data_source_name_among_tmap_path_prefixes(
    tmaps: List[TensorMap],
    csv_sources: List[Tuple[str, str]],
):
    """Iterate over CSV data source names, and check if name is among TMap path
    prefixes. If not, either a) no TMaps require the CSV source, or b) the CSV source
    name is wrong in --tensors"""
    tmap_path_prefixes = [tm.path_prefix for tm in tmaps]
    for csv_source in csv_sources:
        csv_source_name = csv_source[1]
        if csv_source_name not in tmap_path_prefixes:
            logging.warning(
                f"CSV source name {csv_source_name} is not found in any TMap path "
                f"prefixes. This CSV data source may not be needed. Alternatively, "
                f"the CSV source name given to --tensors is incorrect.",
            )


def make_data_generator_factory(
    data_split: str,
    num_workers: int,
    hd5_sources: List[List[str]],
    csv_sources: List[Tuple[str, str]],
    patient_ids: Set[int],
    input_tmaps: List[TensorMap],
    output_tmaps: List[TensorMap],
    augment: bool = False,
    validate: bool = True,
    normalize: bool = True,
    keep_ids: bool = False,
    debug: bool = False,
    verbose: bool = True,
    return_nan: bool = False,
) -> Tuple[Callable[[], SampleGenerator], StatsWrapper, Cleanup]:
    Method = Thread if debug else Process
    tensor_queue: Queue = Queue(MAX_QUEUE_SIZE)

    # Load all CSVs into dataframes
    csv_data = []
    for csv_source, csv_name in csv_sources:
        df = pd.read_csv(csv_source, low_memory=False)
        mrn_col = infer_mrn_column(df, csv_source)
        df[mrn_col] = df[mrn_col].dropna().astype(int)
        csv_data.append((csv_name, df, mrn_col))
        logging.info(f"Loaded dataframe with shape {df.shape} from {csv_source}")

    processes = []
    worker_ids = np.array_split(list(patient_ids), num_workers)
    for i, _ids in enumerate(worker_ids):
        name = f"{data_split}_worker_{i}"
        start_signal = Event()
        process = Method(
            target=_tensor_worker,
            name=name,
            args=(
                name,
                hd5_sources,
                csv_data,
                _ids,
                start_signal,
                tensor_queue,
                input_tmaps,
                output_tmaps,
                augment,
                validate,
                normalize,
                return_nan,
            ),
        )
        process.start()
        process.start_signal = start_signal
        processes.append(process)
    logging.info(f"Started {num_workers} {data_split} workers.")

    def cleanup_workers():
        for process in processes:
            if isinstance(process, Process):
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

        stats: Counter = Counter()
        num_ids = len(patient_ids)
        linked_tensor_buffer: defaultdict = defaultdict(list)
        while stats["ids_completed"] < num_ids or len(linked_tensor_buffer) > 0:
            sample, patient_id, num_linked, exception = tensor_queue.get()
            if exception is not None:
                exception_message = str(exception).strip("'")
                stats[
                    f"exception - {type(exception).__name__} - {exception_message}"
                ] += 1
            if sample == ID_SUCCEEDED:
                stats["ids_succeeded"] += 1
                stats["ids_completed"] += 1
            elif sample == ID_FAILED:
                stats["ids_failed"] += 1
                stats["ids_completed"] += 1
                if patient_id in linked_tensor_buffer:
                    del linked_tensor_buffer[patient_id]
            elif sample == SAMPLE_FAILED:
                stats["samples_failed"] += 1
                stats["samples_completed"] += 1
            elif num_linked != 1:
                linked_tensor_buffer[patient_id].append(sample)
                if len(linked_tensor_buffer[patient_id]) == num_linked:
                    for sample in linked_tensor_buffer[patient_id]:
                        stats["samples_succeeded"] += 1
                        stats["samples_completed"] += 1
                        _collect_sample_stats(stats, sample, input_tmaps, output_tmaps)
                        yield sample if not keep_ids else sample + (patient_id,)
                    del linked_tensor_buffer[patient_id]
                else:
                    continue
            else:
                stats["samples_succeeded"] += 1
                stats["samples_completed"] += 1
                _collect_sample_stats(stats, sample, input_tmaps, output_tmaps)
                yield sample if not keep_ids else sample + (patient_id,)

        # If no successful tensors were obtained, iterate over stats, find exceptions,
        # and log the exception message and the count into an error message
        if stats["samples_succeeded"] == 0:
            error_message = "No samples succeeded. Errors collected during parsing:\n"
            for key in stats:
                if "exception" in key:
                    error_message += f"{key}: {stats[key]} instances occurred\n"
            raise ValueError(error_message)

        if verbose:
            logging.info(
                f"{get_stats_string(name, stats, epoch_counter)}"
                f"{get_verbose_stats_string({data_split: stats}, input_tmaps, output_tmaps)}",
            )
        nonlocal stats_wrapper
        stats_wrapper.stats = stats

    return data_generator_factory, stats_wrapper, cleanup_workers


def make_dataset(
    data_split: str,
    hd5_sources: List[List[str]],
    csv_sources: List[Tuple[str, str]],
    patient_ids: Set[int],
    input_tmaps: List[TensorMap],
    output_tmaps: List[TensorMap],
    batch_size: int,
    num_workers: int,
    augment: bool = False,
    validate: bool = True,
    normalize: bool = True,
    keep_ids: bool = False,
    cache: bool = True,
    debug: bool = False,
    verbose: bool = True,
    return_nan: bool = False,
) -> Tuple[tf.data.Dataset, StatsWrapper, Cleanup]:
    output_types = (
        {
            tm.input_name: tf.string if tm.is_language else tf.float32
            for tm in input_tmaps
        },
        {
            tm.output_name: tf.string if tm.is_language else tf.float32
            for tm in output_tmaps
        },
    )
    output_shapes = (
        {tm.input_name: tm.shape for tm in input_tmaps},
        {tm.output_name: tm.shape for tm in output_tmaps},
    )
    if keep_ids:
        output_types += (tf.int64,)
        output_shapes += (tuple(),)

    data_generator_factory, stats_wrapper, cleanup = make_data_generator_factory(
        data_split=data_split,
        num_workers=num_workers,
        hd5_sources=hd5_sources,
        csv_sources=csv_sources,
        patient_ids=patient_ids,
        input_tmaps=input_tmaps,
        output_tmaps=output_tmaps,
        augment=augment,
        validate=validate,
        normalize=normalize,
        keep_ids=keep_ids,
        debug=debug,
        verbose=verbose,
        return_nan=return_nan,
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

    if cache:
        dataset = dataset.cache()

    return dataset, stats_wrapper, cleanup


def mixup_dataset(dataset: tf.data.Dataset, mixup_alpha: float) -> tf.data.Dataset:

    dist = tfd.Beta(mixup_alpha, mixup_alpha)

    def mixup(*batch):
        """
        Augments a batch of samples by overlaying consecutive samples weighted by
        samples taken from a beta distribution.
        """
        in_batch, out_batch = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX]
        # last batch in epoch may not be exactly batch_size, get actual _size
        _size = tf.shape(in_batch[next(iter(in_batch))])[:1]
        # roll samples for masking [1,2,3] -> [3,1,2]
        in_roll = {k: tf.roll(in_batch[k], 1, 0) for k in in_batch}
        out_roll = {k: tf.roll(out_batch[k], 1, 0) for k in out_batch}
        # sample from beta distribution
        lambdas = dist.sample(_size)
        for k in in_batch:
            # lambdas is shape (_size,), reshape to match rank of tensor for math
            _dims = [_size, tf.ones(tf.rank(in_batch[k]) - 1, tf.int32)]
            _shape = tf.concat(_dims, 0)
            _lambdas = tf.reshape(lambdas, _shape)
            # augment samples with mixup
            in_batch[k] = in_batch[k] * _lambdas + in_roll[k] * (1 - _lambdas)
        for k in out_batch:
            _dims = [_size, tf.ones(tf.rank(out_batch[k]) - 1, tf.int32)]
            _shape = tf.concat(_dims, 0)
            _lambdas = tf.reshape(lambdas, _shape)
            out_batch[k] = out_batch[k] * _lambdas + out_roll[k] * (1 - _lambdas)
        return batch

    dataset = dataset.map(mixup)
    return dataset


def tensors_to_sources(
    tensors: Union[str, List[Union[str, Tuple[str, str]]]],
    tmaps: List[TensorMap],
) -> Tuple[List[str], List[Tuple[str, str]]]:
    if not isinstance(tensors, list):
        tensors = [tensors]
    csv_sources = []
    hd5_sources = []
    for source in tensors:
        if isinstance(source, tuple):
            csv_sources.append(source)
        elif source.endswith(CSV_EXT):
            csv_name = os.path.splitext(os.path.basename(source))[0]
            csv_sources.append((source, csv_name))
        else:
            hd5_subdirs = [root_dir[0] for root_dir in os.walk(source)]
            hd5_sources.append(hd5_subdirs)
    _csv_data_source_name_among_tmap_path_prefixes(
        tmaps=tmaps,
        csv_sources=csv_sources,
    )
    return hd5_sources, csv_sources


def train_valid_test_datasets(
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    tensors: Union[str, List[Union[str, Tuple[str, str]]]],
    batch_size: int,
    num_workers: int,
    keep_ids: bool = False,
    keep_ids_test: bool = True,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    mrn_column_name: Optional[str] = None,
    patient_csv: Optional[str] = None,
    train_csv: Optional[str] = None,
    valid_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    allow_empty_split: bool = False,
    output_folder: Optional[str] = None,
    cache: bool = True,
    mixup_alpha: float = 0.0,
    debug: bool = False,
) -> Tuple[
    Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
    Tuple[StatsWrapper, StatsWrapper, StatsWrapper],
    Tuple[Cleanup, Cleanup, Cleanup],
]:
    """
    Get tensorflow.data.Datasets for training, validation and testing data.

    :param tensor_maps_in: list of TensorMaps that are input names to a model
    :param tensor_maps_out: list of TensorMaps that are output from a model
    :param tensors: list of paths or tuples to directories or CSVs containing tensors
    :param batch_size: number of samples in each batch
    :param num_workers: number of worker processes used to feed each dataset
    :param keep_ids: if true, return the patient ID for each sample
    :param keep_ids_test: if true, return the patient ID for each sample in the test set
    :param mrn_column_name: name of column in csv files with MRNs
    :param patient_csv: CSV file of sample ids, sample ids are considered for
                       train/valid/test only if it is in patient_csv
    :param valid_ratio: rate of tensors to use for validation, mutually exclusive
                        with valid_csv
    :param test_ratio: rate of tensors to use for testing, mutually exclusive with
                       test_csv
    :param train_csv: CSV file of sample ids to use for training
    :param valid_csv: CSV file of sample ids to use for validation, mutually exclusive
                      with valid_ratio
    :param test_csv: CSV file of sample ids to use for testing, mutually exclusive
                     with test_ratio
    :param allow_empty_split: If true, allow one or more data splits to be empty
    :param output_folder: output folder of output files
    :param cache: if true, cache dataset in memory
    :param mixup_alpha: if non-zero, mixup batches with this alpha parameter for mixup
    :param debug: if true, use threads to allow pdb debugging
    :return: tuple of three tensorflow Datasets, three StatsWrapper objects, and
             three callbacks to cleanup worker processes
    """

    train_ids, valid_ids, test_ids = get_train_valid_test_ids(
        tensors=tensors,
        mrn_column_name=mrn_column_name,
        patient_csv=patient_csv,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
        allow_empty_split=allow_empty_split,
    )

    # Save train/validation/test splits if not given
    if output_folder is not None:

        def save_ids(ids: Set[int], split: str):
            fpath = os.path.join(output_folder, f"{split}{CSV_EXT}")
            df = pd.DataFrame({"patient_id": list(ids)})
            df.to_csv(fpath, index=False)
            logging.info(f"--{split}_csv was not provided; saved sample IDs to {fpath}")

        if train_csv is None:
            save_ids(ids=train_ids, split="train")
        if valid_csv is None:
            save_ids(ids=valid_ids, split="valid")
        if test_csv is None:
            save_ids(ids=test_ids, split="test")

    # Parse tensors into hd5 sources or csv sources with a csv name
    hd5_sources, csv_sources = tensors_to_sources(
        tensors=tensors,
        tmaps=tensor_maps_in + tensor_maps_out,
    )

    train_dataset, train_stats, train_cleanup = make_dataset(
        data_split="train",
        hd5_sources=hd5_sources,
        csv_sources=csv_sources,
        patient_ids=train_ids,
        input_tmaps=tensor_maps_in,
        output_tmaps=tensor_maps_out,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=True,
        keep_ids=keep_ids,
        cache=cache,
        debug=debug,
    )
    valid_dataset, valid_stats, valid_cleanup = make_dataset(
        data_split="valid",
        hd5_sources=hd5_sources,
        csv_sources=csv_sources,
        patient_ids=valid_ids,
        input_tmaps=tensor_maps_in,
        output_tmaps=tensor_maps_out,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        keep_ids=keep_ids,
        cache=cache,
        debug=debug,
    )
    test_dataset, test_stats, test_cleanup = make_dataset(
        data_split="test",
        hd5_sources=hd5_sources,
        csv_sources=csv_sources,
        patient_ids=test_ids,
        input_tmaps=tensor_maps_in,
        output_tmaps=tensor_maps_out,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        keep_ids=keep_ids or keep_ids_test,
        cache=cache,
        debug=debug,
    )

    if mixup_alpha != 0:
        train_dataset = mixup_dataset(dataset=train_dataset, mixup_alpha=mixup_alpha)

    return (
        (train_dataset, valid_dataset, test_dataset),
        (train_stats, valid_stats, test_stats),
        (train_cleanup, valid_cleanup, test_cleanup),
    )


def _collect_sample_stats(
    stats: Counter,
    sample: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
    input_tmaps: List[TensorMap],
    output_tmaps: List[TensorMap],
) -> None:
    in_tensors, out_tensors = sample

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

    _update_stats(tmaps=input_tmaps, tensors=in_tensors, is_input=True)
    _update_stats(tmaps=output_tmaps, tensors=out_tensors, is_input=False)


def get_stats_string(name: str, stats: Counter, epoch_count: int) -> str:
    # fmt: off
    return (
        f"\n"
        f"------------------- {name} completed true epoch {epoch_count} -------------------\n"
        f"\tGenerator shuffled {stats['ids_completed']} IDs. {stats['ids_succeeded']} IDs succeeded and {stats['ids_failed']} IDs failed.\n"
        f"\tFrom {stats['ids_succeeded']} IDs, {stats['samples_completed']} samples were extracted.\n"
        f"\tFrom {stats['samples_completed']} samples, {stats['samples_succeeded']} yielded tensors and {stats['samples_failed']} samples failed.\n"
    )
    # fmt: on


def get_verbose_stats_string(
    split_stats: Dict[str, Counter],
    input_tmaps: List[TensorMap],
    output_tmaps: List[TensorMap],
) -> str:
    if len(split_stats) == 1:
        stats = list(split_stats.values())[0]
        dataframes = _get_stats_as_dataframes(stats, input_tmaps, output_tmaps)
    else:
        dataframes = _get_stats_as_dataframes_from_multiple_datasets(
            split_stats,
            input_tmaps,
            output_tmaps,
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
    stats: Counter,
    input_tmaps: List[TensorMap],
    output_tmaps: List[TensorMap],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    continuous_tmaps = []
    categorical_tmaps = []
    other_tmaps = []
    for tm in input_tmaps + output_tmaps:
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
            zip(_stats["TensorMap"], _stats["Label"]),
            names=["TensorMap", "Label"],
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
    input_tmaps: List[TensorMap],
    output_tmaps: List[TensorMap],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_to_dataframes = {
        split: _get_stats_as_dataframes(
            stats=stats,
            input_tmaps=input_tmaps,
            output_tmaps=output_tmaps,
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
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    train_csv: Optional[str] = None,
    valid_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
) -> Tuple[float, float, float, float]:

    if valid_csv is not None:
        valid_ratio = 0.0
    if test_csv is not None:
        test_ratio = 0.0
    if train_csv is not None:
        train_ratio = 0.0
        discard_ratio = 1.0 - valid_ratio - test_ratio
    else:
        train_ratio = 1.0 - valid_ratio - test_ratio
        discard_ratio = 0.0

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


def patient_csv_to_set(
    patient_csv: Optional[str] = None,
    mrn_column_name: Optional[str] = None,
) -> Union[None, Set[int]]:

    if patient_csv is None:
        return None

    # Read CSV to dataframe and assume no header
    df = pd.read_csv(patient_csv, header=None, low_memory=False)

    # If first row and column is castable to int, there is no header
    try:
        int(df.iloc[0].values[0])

    # If fails, must be header; overwrite column name with first row and remove
    # first row
    except ValueError:
        df.columns = df.iloc[0]
        df = df[1:]

    if mrn_column_name is None:
        df.columns = [
            col.lower() if isinstance(col, str) else col for col in df.columns
        ]
        mrn_column_name = infer_mrn_column(df, patient_csv)

    # Isolate MRN column from dataframe, cast to int
    patient_ids = df[mrn_column_name].dropna().astype(int)

    return set(patient_ids)


def get_train_valid_test_ids(
    tensors: Union[str, List[Union[str, Tuple[str, str]]]],
    mrn_column_name: Optional[str] = None,
    patient_csv: Optional[str] = None,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    train_csv: Optional[str] = None,
    valid_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    allow_empty_split: bool = False,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Return 3 disjoint sets of IDs.

    The IDs are gathered from sources in tensors and are split into training,
    validation, and testing splits. By default, IDs are split into train/valid/test
    in the ratio 0.7/0.2/0.1. Otherwise, at least 2 arguments are required to specify
    train/valid/test sets.

    :param tensors: list of paths or tuples to directories or CSVs containing tensors
    :param mrn_column_name: name of column in csv files with MRNs
    :param patient_csv: path to csv containing sample ids, only consider sample ids
                       for splitting into train/valid/test sets if they appear in
                       patient_csv
    :param valid_ratio: rate of tensors in validation list, mutually exclusive with
                        valid_csv
    :param test_ratio: rate of tensors in testing list, mutually exclusive with
                       test_csv
    :param train_csv: path to csv containing sample ids to reserve for training list
    :param valid_csv: path to csv containing sample ids to reserve for validation
                      list, mutually exclusive with valid_ratio
    :param test_csv: path to csv containing sample ids to reserve for testing list,
                     mutually exclusive with test_ratio
    :param allow_empty_split: If true, allow one or more data splits to be empty

    :return: tuple of 3 sets of sample IDs
    """
    train_ids: Set[int] = set()
    valid_ids: Set[int] = set()
    test_ids: Set[int] = set()
    discard_ids: Set[int] = set()
    unassigned_ids: Set[int] = set()

    if not isinstance(tensors, list):
        tensors = [tensors]

    train_ratio, valid_ratio, test_ratio, _ = _get_train_valid_test_discard_ratios(
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
    )

    sample_set = patient_csv_to_set(
        patient_csv=patient_csv,
        mrn_column_name=mrn_column_name,
    )
    train_set = patient_csv_to_set(
        patient_csv=train_csv,
        mrn_column_name=mrn_column_name,
    )
    valid_set = patient_csv_to_set(
        patient_csv=valid_csv,
        mrn_column_name=mrn_column_name,
    )
    test_set = patient_csv_to_set(patient_csv=test_csv, mrn_column_name=mrn_column_name)

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

    # Get list of intersect of all IDs from all sources
    all_source_ids = []
    for source in tensors:
        if isinstance(source, tuple):
            source = source[0]

        if os.path.isdir(source):
            source_ids = set()
            for root, _, files in os.walk(source):
                for fname in files:
                    if not fname.endswith(TENSOR_EXT):
                        continue
                    try:
                        patient_id = int(os.path.splitext(fname)[0])
                    except ValueError:
                        continue
                    source_ids.add(patient_id)
        elif source.endswith(CSV_EXT):
            source_ids = patient_csv_to_set(
                patient_csv=source,
                mrn_column_name=mrn_column_name,
            )
        else:
            raise ValueError(f"Cannot get IDs from source: {source}")
        all_source_ids.append(source_ids)

    all_ids = set.intersection(*all_source_ids)

    # Split IDs among train/valid/test
    for patient_id in all_ids:
        if sample_set is not None and patient_id not in sample_set:
            continue

        if train_set is not None and patient_id in train_set:
            train_ids.add(patient_id)
        elif valid_set is not None and patient_id in valid_set:
            valid_ids.add(patient_id)
        elif test_set is not None and patient_id in test_set:
            test_ids.add(patient_id)
        else:
            unassigned_ids.add(patient_id)

    unassigned_ids: List[int] = list(unassigned_ids)
    np.random.shuffle(unassigned_ids)
    n = len(unassigned_ids)
    n_train, n_valid, n_test = (
        round(train_ratio * n),
        round(valid_ratio * n),
        round(test_ratio * n),
    )
    indices = [n_train, n_train + n_valid, n_train + n_valid + n_test]
    _train, _valid, _test, _discard = np.split(unassigned_ids, indices)
    train_ids |= set(_train)
    valid_ids |= set(_valid)
    test_ids |= set(_test)

    logging.info(
        f"Split IDs from sources at tensors into {len(train_ids)} training, "
        f"{len(valid_ids)} validation, and {len(test_ids)} testing IDs.",
    )
    logging.debug(f"Discarded {len(discard_ids)} tensors due to given ratios")
    if (
        not allow_empty_split
        and len(train_ids) == 0
        and len(valid_ids) == 0
        and len(test_ids) == 0
    ):
        raise ValueError(
            f"Cannot have empty split\n"
            f"Found {len(train_ids)} training,"
            f" {len(valid_ids)} validation, and"
            f" {len(test_ids)} testing IDs\n"
            f"Discarded {len(discard_ids)} IDs",
        )
    return train_ids, valid_ids, test_ids


def get_dicts_of_arrays_from_dataset(
    dataset: tf.data.Dataset,
) -> Union[
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray],
]:
    input_data_batches = defaultdict(list)
    output_data_batches = defaultdict(list)
    id_batches = []

    for i, batch in enumerate(dataset):
        if (i + 1) % 1000 == 0:
            logging.info(f"Completed batch {i + 1}")
        input_data_batch = batch[BATCH_INPUT_INDEX]
        for input_name, input_tensor in input_data_batch.items():
            input_data_batches[input_name].append(input_tensor.numpy())

        output_data_batch = batch[BATCH_OUTPUT_INDEX]
        for output_name, output_tensor in output_data_batch.items():
            output_data_batches[output_name].append(output_tensor.numpy())

        if len(batch) == 3:
            id_batches.append(batch[BATCH_IDS_INDEX].numpy().astype(int))

    input_data = {
        input_name: np.concatenate(input_data_batches[input_name])
        for input_name in input_data_batches
    }
    output_data = {
        output_name: np.concatenate(output_data_batches[output_name])
        for output_name in output_data_batches
    }
    patient_ids = None if len(id_batches) == 0 else np.concatenate(id_batches).tolist()

    return (
        (input_data, output_data)
        if not patient_ids
        else (input_data, output_data, patient_ids)
    )


def get_array_from_dict_of_arrays(
    tensor_maps: List[TensorMap],
    data: Dict[str, np.ndarray],
    drop_redundant_columns: bool = False,
) -> np.ndarray:
    array = np.array([])

    # Determine if we are parsing input or output tensor maps
    tmap_type = "input" if np.all(["input" in key for key in data]) else "output"

    # Iterate over tmaps
    for tm in tensor_maps:
        if tmap_type == "input":
            tensor = data[tm.input_name]
        else:
            tensor = data[tm.output_name]

        # For categorical tmaps, drop the redundant negative column
        # However, there is no need to do this because we regularize:
        # https://inmachineswetrust.com/posts/drop-first-columns/
        if drop_redundant_columns:
            if tm.is_categorical:
                _, neg_idx = find_negative_label_and_channel(tm.channel_map)
                tensor = np.delete(tensor, neg_idx, axis=1)
        array = np.append(array, tensor, axis=1) if len(array) > 0 else tensor

    # If the array is 1D, reshape to contiguous flattened array
    # e.g. change shape from (n, 1) to (n,)
    if array.shape[1] == 1:
        array = array.ravel()
    return array
