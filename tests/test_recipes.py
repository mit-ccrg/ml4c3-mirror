# Imports: standard library
import os
import argparse

# Imports: third party
import numpy as np
import pandas as pd
import pytest

# Imports: first party
from ml4c3.recipes import train_model, infer_multimodal_multitask
from ml4c3.explorations import (
    explore,
    continuous_explore_header,
    categorical_explore_header,
    _tmap_requires_modification_for_explore,
)
from ml4c3.tensormap.TensorMap import TensorMap, Interpretation


class TestRecipes:
    @staticmethod
    def test_infer(default_arguments_infer: argparse.Namespace):
        infer_multimodal_multitask(default_arguments_infer)
        path = os.path.join(
            default_arguments_infer.output_folder,
            default_arguments_infer.id,
            "predictions_test.csv",
        )
        predictions = pd.read_csv(path)
        test_samples = pd.read_csv(
            os.path.join(
                default_arguments_infer.output_folder,
                default_arguments_infer.id,
                "test.csv",
            ),
        )
        assert len(set(predictions["patient_id"])) == len(test_samples)

    @staticmethod
    def test_explore(
        default_arguments_explore: argparse.Namespace,
        tmpdir_factory,
        utils,
    ):
        temp_dir = tmpdir_factory.mktemp("explore_tensors")
        default_arguments_explore.tensors = str(temp_dir)
        tmaps = pytest.TMAPS_UP_TO_4D[:]
        tmaps.append(
            TensorMap(
                "scalar",
                shape=(1,),
                interpretation=Interpretation.CONTINUOUS,
                tensor_from_file=pytest.TFF,
            ),
        )
        explore_expected = utils.build_hdf5s(temp_dir, tmaps, n=pytest.N_TENSORS)
        default_arguments_explore.num_workers = 3
        default_arguments_explore.tensor_maps_in = tmaps
        default_arguments_explore.explore_export_fpath = True
        explore(default_arguments_explore)

        csv_path = os.path.join(
            default_arguments_explore.output_folder,
            default_arguments_explore.id,
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

    @staticmethod
    def test_train(default_arguments: argparse.Namespace):
        train_model(default_arguments)
