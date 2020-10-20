# Imports: standard library
import os

# Imports: third party
import numpy as np
import pandas as pd
import pytest

# Imports: first party
from ml4c3.recipes import infer_multimodal_multitask, train_multimodal_multitask
from ml4c3.explorations import (
    explore,
    continuous_explore_header,
    categorical_explore_header,
    _tmap_requires_modification_for_explore,
)
from ml4c3.tensormap.TensorMap import TensorMap, Interpretation


class TestRecipes:
    def test_infer(self, default_arguments):
        infer_multimodal_multitask(default_arguments)
        path = os.path.join(
            default_arguments.output_folder,
            default_arguments.id,
            f"predictions_test.csv",
        )
        predictions = pd.read_csv(path)
        test_samples = pd.read_csv(
            os.path.join(
                default_arguments.output_folder,
                default_arguments.id,
                "test.csv",
            ),
        )
        assert len(set(predictions["sample_id"])) == len(test_samples)

    def test_explore(self, default_arguments, tmpdir_factory, utils):
        temp_dir = tmpdir_factory.mktemp("explore_tensors")
        default_arguments.tensors = str(temp_dir)
        tmaps = pytest.TMAPS_UP_TO_4D[:]
        tmaps.append(
            TensorMap(
                f"scalar",
                shape=(1,),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=pytest.TFF,
            ),
        )
        explore_expected = utils.build_hdf5s(temp_dir, tmaps, n=pytest.N_TENSORS)
        default_arguments.num_workers = 3
        default_arguments.tensor_maps_in = tmaps
        default_arguments.explore_export_fpath = True
        explore(default_arguments)

        csv_path = os.path.join(
            default_arguments.output_folder,
            default_arguments.id,
            "tensors_union.csv",
        )
        explore_result = pd.read_csv(csv_path)

        for row in explore_result.iterrows():
            row = row[1]
            for tm in tmaps:
                row_expected = explore_expected[(row["fpath"], tm)]
                if _tmap_requires_modification_for_explore(tm):
                    actual = getattr(row, continuous_explore_header(tm))
                    assert not np.isnan(actual)
                    continue
                if tm.is_continuous:
                    actual = getattr(row, continuous_explore_header(tm))
                    assert actual == row_expected
                    continue
                if tm.is_categorical:
                    for channel, idx in tm.channel_map.items():
                        channel_val = getattr(
                            row,
                            categorical_explore_header(tm, channel),
                        )
                        assert channel_val == row_expected[idx]

    def test_train(self, default_arguments):
        train_multimodal_multitask(default_arguments)
