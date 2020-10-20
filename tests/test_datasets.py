# Imports: standard library
import os
import csv
from collections import defaultdict

# Imports: third party
import numpy as np
import pytest

# Imports: first party
from ml4c3.datasets import (
    BATCH_PATHS_INDEX,
    make_dataset,
    _sample_csv_to_set,
    get_train_valid_test_paths,
)
from ml4c3.definitions.globals import TENSOR_EXT


def _write_samples(csv_path, sample_ids, use_header=False, write_dupes=False):
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if use_header:
            csv_writer.writerow(["sample_id"])
        for sample_id in sample_ids:
            csv_writer.writerow([sample_id])
            if write_dupes:
                csv_writer.writerow([sample_id])


@pytest.fixture(scope="function")
def sample_csv(tmpdir_factory, request):
    use_header = getattr(request, "param", None) == "header"
    write_dupes = getattr(request, "param", None) == "duplicates"
    csv_path = tmpdir_factory.mktemp("csvs").join("sample.csv")
    sample_ids = {
        str(sample_id)
        for sample_id in np.random.choice(
            range(pytest.N_TENSORS),
            size=np.random.randint(pytest.N_TENSORS * 3 / 5, pytest.N_TENSORS * 4 / 5),
            replace=False,
        )
    }
    _write_samples(csv_path, sample_ids, use_header, write_dupes)
    return csv_path, sample_ids


@pytest.fixture(scope="function")
def train_valid_test_csv(tmpdir_factory, request):
    overlap = getattr(request, "param", "")
    csv_dir = tmpdir_factory.mktemp("csvs")
    train_path = csv_dir.join("train.csv")
    valid_path = csv_dir.join("valid.csv")
    test_path = csv_dir.join("test.csv")
    # the total number of train/valid/test sets should be < sample set to test scenario when all 3 csv are used
    n = int(pytest.N_TENSORS / 2)
    n1 = int(n / 3)
    n2 = int(n * 2 / 3)
    sample_ids = [str(sample_id) for sample_id in range(n)]
    np.random.shuffle(sample_ids)
    train_ids, valid_ids, test_ids = sample_ids[:n1], sample_ids[n1:n2], sample_ids[n2:]
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
def train_valid_test_paths(default_arguments, train_valid_test_csv):
    args = default_arguments
    (
        (train_csv, train_ids),
        (valid_csv, valid_ids),
        (test_csv, test_ids),
    ) = train_valid_test_csv
    return get_train_valid_test_paths(
        tensors=args.tensors,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        sample_csv=None,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
    )


@pytest.fixture(scope="function")
def train_paths(train_valid_test_paths):
    return train_valid_test_paths[0]


@pytest.fixture(scope="function")
def valid_paths(train_valid_test_paths):
    return train_valid_test_paths[1]


@pytest.fixture(scope="function")
def test_paths(train_valid_test_paths):
    return train_valid_test_paths[2]


# Ugly meta fixtures because fixtures cannot be
# used as parameters in pytest.mark.parametrize
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(scope="function")
def sample_set(request, sample_csv):
    if request.param is None:
        return None
    return sample_csv


@pytest.fixture(scope="function")
def train_set(request, train_valid_test_csv):
    if request.param is None:
        return None
    return train_valid_test_csv[0]


@pytest.fixture(scope="function")
def valid_set(request, train_valid_test_csv):
    if request.param is None:
        return None
    return train_valid_test_csv[1]


@pytest.fixture(scope="function")
def test_set(request, train_valid_test_csv):
    if request.param is None:
        return None
    return train_valid_test_csv[2]


@pytest.fixture(scope="function")
def valid_test_ratio():
    valid_ratio = np.random.randint(1, 5) / 10
    test_ratio = np.random.randint(1, 5) / 10
    return valid_ratio, test_ratio


@pytest.fixture(scope="function")
def valid_ratio(request, valid_test_ratio):
    if request.param is None:
        return None
    return valid_test_ratio[0]


@pytest.fixture(scope="function")
def test_ratio(request, valid_test_ratio):
    if request.param is None:
        return None
    return valid_test_ratio[1]


class TestDataset:
    def test_get_true_epoch(self, default_arguments, train_paths):
        num_workers = 2
        num_tensors = len(train_paths)  # 8 paths by default
        batch_size = 2
        repeat_test = 3

        # the test should currently fail but is flaky
        for _ in range(repeat_test):
            dataset, stats, cleanup = make_dataset(
                data_split="train",
                input_maps=default_arguments.tensor_maps_in,
                output_maps=default_arguments.tensor_maps_out,
                paths=train_paths,
                batch_size=batch_size,
                num_workers=num_workers,
                keep_paths=True,
            )

            rets = []
            for batch in dataset:
                rets.append(batch)

            paths = []
            for ret in rets:
                paths.extend(ret[BATCH_PATHS_INDEX])
            paths = [path.numpy().decode() for path in paths]
            unique_paths, counts = np.unique(paths, return_counts=True)
            unique_counts = np.unique(counts)

            try:
                # make sure the tensors visited were seen the same number of times
                # make sure all the tensors are visited
                # make sure the tensors are visited the expected number of times
                assert len(unique_counts) == 1
                assert set(unique_paths) == set(train_paths)
            finally:
                cleanup()


class TestSampleCsvToSet:
    def test_sample_csv(self, sample_csv):
        csv_path, sample_ids = sample_csv
        sample_set = _sample_csv_to_set(csv_path)

        assert open(csv_path).readline() != "sample_id\n"
        assert all([sample_id in sample_set for sample_id in sample_ids])
        assert len(sample_ids) == len(sample_set)

    @pytest.mark.parametrize("sample_csv", ["header"], indirect=["sample_csv"])
    def test_sample_csv_header(self, sample_csv):
        csv_path, sample_ids = sample_csv
        sample_set = _sample_csv_to_set(csv_path)
        assert open(csv_path).readline() == "sample_id\n"
        assert all([sample_id in sample_set for sample_id in sample_ids])
        assert len(sample_ids) == len(sample_set)

    @pytest.mark.parametrize("sample_csv", ["duplicates"], indirect=["sample_csv"])
    def test_sample_csv_duplicates(self, sample_csv):
        csv_path, sample_ids = sample_csv
        sample_set = _sample_csv_to_set(csv_path)
        assert open(csv_path).readline() != "sample_id\n"
        assert all([sample_id in sample_set for sample_id in sample_ids])
        assert len(sample_ids) == len(sample_set)

        with open(csv_path) as csv_file:
            dupe_set = set()
            has_dupe = False
            for line in csv_file:
                if line in dupe_set:
                    has_dupe = True
                dupe_set.add(line)
            assert has_dupe


class TestGetTrainValidTestPaths:
    @pytest.mark.parametrize("sample_set", [None, "sample_csv"], indirect=True)
    @pytest.mark.parametrize("train_set", [None, "train_csv"], indirect=True)
    @pytest.mark.parametrize("valid_set", [None, "valid_csv"], indirect=True)
    @pytest.mark.parametrize("test_set", [None, "test_csv"], indirect=True)
    def test_get_paths(
        self, default_arguments, sample_set, train_set, valid_set, test_set,
    ):
        args = default_arguments

        def _path_2_sample(path):
            return os.path.splitext(os.path.basename(path))[0]

        def _paths_equal_samples(paths, samples):
            assert len(paths) == len(samples)
            assert all(_path_2_sample(path) in samples for path in paths)
            return True

        sample_csv, sample_ids = sample_set or (None, None)
        train_csv, train_ids = train_set or (None, None)
        valid_csv, valid_ids = valid_set or (None, None)
        test_csv, test_ids = test_set or (None, None)

        train_paths, valid_paths, test_paths = get_train_valid_test_paths(
            tensors=args.tensors,
            sample_csv=sample_csv,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            train_csv=train_csv,
            valid_csv=valid_csv,
            test_csv=test_csv,
        )

        # make sure paths are disjoint and unique
        all_paths = train_paths + valid_paths + test_paths
        counts = defaultdict(int)
        for path in all_paths:
            counts[path] += 1
        assert all(count == 1 for count in counts.values())

        # if sample csv was not given, find the files, just like how tensor_generator does
        if sample_ids is None:
            sample_paths = []
            for root, dirs, files in os.walk(default_arguments.tensors):
                for name in files:
                    if os.path.splitext(name)[-1].lower() != TENSOR_EXT:
                        continue
                    sample_paths.append(os.path.join(root, name))
            sample_ids = {_path_2_sample(path) for path in sample_paths}

        if train_ids is not None:
            # this block handles the cases where samples are discarded, which happens if train_csv is supplied
            assert len(all_paths) <= len(sample_ids)
            assert all(_path_2_sample(path) in sample_ids for path in all_paths)
        else:
            assert _paths_equal_samples(all_paths, sample_ids)

        if train_ids is not None:
            train_ids &= sample_ids
            assert _paths_equal_samples(train_paths, train_ids)

        if valid_ids is not None:
            valid_ids &= sample_ids
            assert _paths_equal_samples(valid_paths, valid_ids)

        if test_ids is not None:
            test_ids &= sample_ids
            assert _paths_equal_samples(test_paths, test_ids)

    @pytest.mark.parametrize(
        "train_valid_test_csv",
        ["train-valid", "train-test", "valid-test"],
        indirect=True,
    )
    def test_get_paths_overlap(self, default_arguments, train_valid_test_csv):
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
            train_paths, valid_paths, test_paths = get_train_valid_test_paths(
                tensors=args.tensors,
                valid_ratio=args.valid_ratio,
                test_ratio=args.test_ratio,
                sample_csv=None,
                train_csv=train_csv,
                valid_csv=valid_csv,
                test_csv=test_csv,
            )
