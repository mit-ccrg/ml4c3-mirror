# Imports: standard library
import os
import csv
import argparse
from typing import Set, List, Tuple, Iterable, Optional
from collections import defaultdict

# Imports: third party
import numpy as np
import pytest

# Imports: first party
from ml4c3.datasets import (
    BATCH_IDS_INDEX,
    make_dataset,
    patient_csv_to_set,
    get_train_valid_test_ids,
)
from definitions.globals import TENSOR_EXT

DATA_SPLIT = Tuple[str, Set[str]]
DATA_SPLITS = Tuple[DATA_SPLIT, DATA_SPLIT, DATA_SPLIT]
ID_SPLIT = Set[str]
ID_SPLITS = Tuple[ID_SPLIT, ID_SPLIT, ID_SPLIT]


def _write_samples(
    csv_path: str,
    patient_ids: Iterable[str],
    use_header: bool = False,
    write_dupes: bool = False,
):
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if use_header:
            csv_writer.writerow(["patient_id"])
        for patient_id in patient_ids:
            csv_writer.writerow([patient_id])
            if write_dupes:
                csv_writer.writerow([patient_id])


@pytest.fixture(scope="function")
def patient_csv(tmpdir_factory, request) -> DATA_SPLIT:
    use_header = getattr(request, "param", None) == "header"
    write_dupes = getattr(request, "param", None) == "duplicates"
    csv_path = tmpdir_factory.mktemp("csvs").join("sample.csv")
    patient_ids = {
        str(patient_id)
        for patient_id in np.random.choice(
            range(pytest.N_TENSORS),
            size=np.random.randint(pytest.N_TENSORS * 3 / 5, pytest.N_TENSORS * 4 / 5),
            replace=False,
        )
    }
    _write_samples(csv_path, patient_ids, use_header, write_dupes)
    return csv_path, patient_ids


@pytest.fixture(scope="function")
def train_valid_test_csv(tmpdir_factory, request) -> DATA_SPLITS:
    overlap = getattr(request, "param", "")
    csv_dir = tmpdir_factory.mktemp("csvs")
    train_path = csv_dir.join("train.csv")
    valid_path = csv_dir.join("valid.csv")
    test_path = csv_dir.join("test.csv")
    # the total number of train/valid/test sets should be < sample set to test scenario when all 3 csv are used
    n = int(pytest.N_TENSORS / 2)
    n1 = int(n / 3)
    n2 = int(n * 2 / 3)
    patient_ids = [str(patient_id) for patient_id in range(n)]
    np.random.shuffle(patient_ids)
    train_ids, valid_ids, test_ids = (
        patient_ids[:n1],
        patient_ids[n1:n2],
        patient_ids[n2:],
    )
    if "train" in overlap and "valid" in overlap:
        train_ids.append(valid_ids[0])
    elif "train" in overlap and "test" in overlap:
        train_ids.append(test_ids[0])
    elif "valid" in overlap and "test" in overlap:
        valid_ids.append(test_ids[0])
    _write_samples(train_path, train_ids)
    _write_samples(valid_path, valid_ids)
    _write_samples(test_path, test_ids)
    return (
        (train_path, set(train_ids)),
        (valid_path, set(valid_ids)),
        (test_path, set(test_ids)),
    )


@pytest.fixture(scope="function")
def train_valid_test_ids(
    default_arguments: argparse.Namespace,
    train_valid_test_csv: DATA_SPLITS,
) -> ID_SPLITS:
    args = default_arguments
    (
        (train_csv, train_ids),
        (valid_csv, valid_ids),
        (test_csv, test_ids),
    ) = train_valid_test_csv
    return get_train_valid_test_ids(
        tensors=args.tensors,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        patient_csv=None,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
    )


@pytest.fixture(scope="function")
def train_ids(train_valid_test_ids: ID_SPLITS) -> ID_SPLIT:
    return train_valid_test_ids[0]


@pytest.fixture(scope="function")
def valid_ids(train_valid_test_ids: ID_SPLITS) -> ID_SPLIT:
    return train_valid_test_ids[1]


@pytest.fixture(scope="function")
def test_ids(train_valid_test_ids: ID_SPLITS) -> ID_SPLIT:
    return train_valid_test_ids[2]


# Ugly meta fixtures because fixtures cannot be
# used as parameters in pytest.mark.parametrize
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(scope="function")
def sample_set(request, patient_csv: DATA_SPLIT) -> Optional[DATA_SPLIT]:
    if request.param is None:
        return None
    return patient_csv


@pytest.fixture(scope="function")
def train_set(request, train_valid_test_csv: DATA_SPLITS) -> Optional[DATA_SPLIT]:
    if request.param is None:
        return None
    return train_valid_test_csv[0]


@pytest.fixture(scope="function")
def valid_set(request, train_valid_test_csv: DATA_SPLITS) -> Optional[DATA_SPLIT]:
    if request.param is None:
        return None
    return train_valid_test_csv[1]


@pytest.fixture(scope="function")
def test_set(request, train_valid_test_csv: DATA_SPLITS) -> Optional[DATA_SPLIT]:
    if request.param is None:
        return None
    return train_valid_test_csv[2]


@pytest.fixture(scope="function")
def valid_test_ratio() -> Tuple[float, float]:
    valid_ratio = np.random.randint(1, 5) / 10
    test_ratio = np.random.randint(1, 5) / 10
    return valid_ratio, test_ratio


@pytest.fixture(scope="function")
def valid_ratio(request, valid_test_ratio: Tuple[float, float]) -> Optional[float]:
    if request.param is None:
        return None
    return valid_test_ratio[0]


@pytest.fixture(scope="function")
def test_ratio(request, valid_test_ratio: Tuple[float, float]) -> Optional[float]:
    if request.param is None:
        return None
    return valid_test_ratio[1]


class TestDataset:
    def test_get_true_epoch(
        self,
        default_arguments: argparse.Namespace,
        train_ids: ID_SPLIT,
    ):
        num_workers = 2
        num_tensors = len(train_ids)  # 8 paths by default
        batch_size = 2
        repeat_test = 3

        # the test should currently fail but is flaky
        for _ in range(repeat_test):
            dataset, stats, cleanup = make_dataset(
                data_split="train",
                tensors=default_arguments.tensors,
                patient_ids=train_ids,
                input_tmaps=default_arguments.tensor_maps_in,
                output_tmaps=default_arguments.tensor_maps_out,
                batch_size=batch_size,
                num_workers=num_workers,
                keep_ids=True,
            )

            rets = []
            for batch in dataset:
                rets.append(batch)

            patient_ids = []
            for ret in rets:
                patient_ids.extend(ret[BATCH_IDS_INDEX])
            patient_ids = [patient_id.numpy().decode() for patient_id in patient_ids]
            unique_ids, counts = np.unique(patient_ids, return_counts=True)
            unique_counts = np.unique(counts)

            try:
                # make sure the tensors visited were seen the same number of times
                # make sure all the tensors are visited
                # make sure the tensors are visited the expected number of times
                assert len(unique_counts) == 1
                assert set(unique_ids) == set(train_ids)
            finally:
                cleanup()


class TestSampleCsvToSet:
    def test_patient_csv(self, patient_csv: DATA_SPLIT):
        csv_path, patient_ids = patient_csv
        sample_set = patient_csv_to_set(csv_path)

        assert open(csv_path).readline() != "patient_id\n"
        assert all([patient_id in sample_set for patient_id in patient_ids])
        assert len(patient_ids) == len(sample_set)

    @pytest.mark.parametrize("patient_csv", ["header"], indirect=["patient_csv"])
    def test_patient_csv_header(self, patient_csv: DATA_SPLIT):
        csv_path, patient_ids = patient_csv
        sample_set = patient_csv_to_set(csv_path)
        assert open(csv_path).readline() == "patient_id\n"
        assert all([patient_id in sample_set for patient_id in patient_ids])
        assert len(patient_ids) == len(sample_set)

    @pytest.mark.parametrize("patient_csv", ["duplicates"], indirect=["patient_csv"])
    def test_patient_csv_duplicates(self, patient_csv: DATA_SPLIT):
        csv_path, patient_ids = patient_csv
        sample_set = patient_csv_to_set(csv_path)
        assert open(csv_path).readline() != "patient_id\n"
        assert all([patient_id in sample_set for patient_id in patient_ids])
        assert len(patient_ids) == len(sample_set)

        with open(csv_path) as csv_file:
            dupe_set = set()
            has_dupe = False
            for line in csv_file:
                if line in dupe_set:
                    has_dupe = True
                dupe_set.add(line)
            assert has_dupe


class TestGetTrainValidTestPaths:
    @pytest.mark.parametrize("sample_set", [None, "patient_csv"], indirect=True)
    @pytest.mark.parametrize("train_set", [None, "train_csv"], indirect=True)
    @pytest.mark.parametrize("valid_set", [None, "valid_csv"], indirect=True)
    @pytest.mark.parametrize("test_set", [None, "test_csv"], indirect=True)
    def test_get_ids(
        self,
        default_arguments: argparse.Namespace,
        sample_set: Optional[DATA_SPLIT],
        train_set: Optional[DATA_SPLIT],
        valid_set: Optional[DATA_SPLIT],
        test_set: Optional[DATA_SPLIT],
    ):
        args = default_arguments

        def _ids_equal_samples(all_ids: Set[str], samples: Set[str]):
            assert len(all_ids) == len(samples)
            assert len(all_ids - samples) == 0
            return True

        patient_csv, patient_ids = sample_set or (None, None)
        train_csv, train_ids = train_set or (None, None)
        valid_csv, valid_ids = valid_set or (None, None)
        test_csv, test_ids = test_set or (None, None)
        train, valid, test = get_train_valid_test_ids(
            tensors=args.tensors,
            patient_csv=patient_csv,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            train_csv=train_csv,
            valid_csv=valid_csv,
            test_csv=test_csv,
        )

        # make sure paths are disjoint and unique
        all_ids = train | valid | test
        counts = defaultdict(int)
        for patient_id in all_ids:
            counts[patient_id] += 1
        assert all(count == 1 for count in counts.values())

        # if sample csv was not given, find the files, just like how tensor_generator does
        if patient_ids is None:
            patient_ids = set()
            for root, dirs, files in os.walk(default_arguments.tensors):
                for name in files:
                    if os.path.splitext(name)[-1].lower() != TENSOR_EXT:
                        continue
                    patient_ids.add(os.path.splitext(name)[0])

        if train_ids is not None:
            # this block handles the cases where samples are discarded, which happens if train_csv is supplied
            assert len(all_ids) <= len(patient_ids)
            assert all(patient_id in patient_ids for patient_id in all_ids)
        else:
            assert _ids_equal_samples(all_ids, patient_ids)

        if train_ids is not None:
            train_ids &= patient_ids
            assert _ids_equal_samples(train, train_ids)

        if valid_ids is not None:
            valid_ids &= patient_ids
            assert _ids_equal_samples(valid, valid_ids)

        if test_ids is not None:
            test_ids &= patient_ids
            assert _ids_equal_samples(test, test_ids)

    @pytest.mark.parametrize(
        "train_valid_test_csv",
        ["train-valid", "train-test", "valid-test"],
        indirect=True,
    )
    def test_get_paths_overlap(
        self,
        default_arguments: argparse.Namespace,
        train_valid_test_csv: DATA_SPLITS,
    ):
        args = default_arguments
        (
            (train_csv, train_ids),
            (valid_csv, valid_ids),
            (test_csv, test_ids),
        ) = train_valid_test_csv
        with pytest.raises(
            ValueError,
            match=(
                r"(train|validation|test) and (train|validation|test) samples overlap"
            ),
        ):
            train, valid, test = get_train_valid_test_ids(
                tensors=args.tensors,
                valid_ratio=args.valid_ratio,
                test_ratio=args.test_ratio,
                patient_csv=None,
                train_csv=train_csv,
                valid_csv=valid_csv,
                test_csv=test_csv,
            )
