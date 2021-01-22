# Imports: standard library
import os
import copy
import logging
import argparse
import multiprocessing as mp
from timeit import default_timer as timer
from typing import Dict

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.arguments import parse_args
from definitions.globals import MODEL_EXT

# pylint: disable=redefined-outer-name, broad-except, import-outside-toplevel


def run(args: argparse.Namespace):
    start_time = timer()  # Keep track of elapsed execution time
    try:
        # fmt: off
        if args.recipe in [
            "train",
            "train_keras_logreg",
            "train_sklearn_logreg",
            "train_sklearn_randomforest",
            "train_sklearn_svm",
            "train_sklearn_xbost",
        ]:
            train_model(args)
        elif args.recipe == "train_simclr":
            train_simclr_model(args)
        elif args.recipe == "infer":
            infer_multimodal_multitask(args)
        elif args.recipe == "hyperoptimize":
            from ml4c3.hyperoptimizers import hyperoptimize  # isort: skip
            hyperoptimize(args)

        elif args.recipe == "tensorize_ecg":
            from ingest.ecg.tensorizer import tensorize as tensorize_ecg  # isort: skip
            tensorize_ecg(args)

        elif args.recipe == "pull_adt":
            from ingest.edw.pipeline import pull_edw_data  # isort: skip
            pull_edw_data(args, only_adt=True)

        elif args.recipe == "pull_edw":
            from ingest.edw.pipeline import pull_edw_data  # isort: skip
            pull_edw_data(args)

        elif args.recipe == "tensorize_icu_no_edw_pull":
            from ingest.icu.tensorizer import tensorize as tensorize_icu  # isort: skip
            tensorize_icu(args)

        elif args.recipe == "tensorize_icu":
            from ingest.edw.pipeline import pull_edw_data  # isort: skip
            from ingest.icu.tensorizer import tensorize as tensorize_icu  # isort: skip
            pull_edw_data(args)
            tensorize_icu(args)

        elif args.recipe == "tensorize_edw":
            from tensorize.edw.tensorizer import tensorize as tensorize_edw  # isort: skip, pylint: disable=line-too-long
            tensorize_edw(args)

        elif args.recipe == "tensorize_bedmaster":
            from tensorize.bedmaster.tensorizer import tensorize as tensorize_bedmaster  # isort: skip, pylint: disable=line-too-long
            tensorize_bedmaster(args)

        elif args.recipe == "explore":
            from ml4c3.explorations import explore  # isort: skip
            explore(args=args, disable_saving_output=args.explore_disable_saving_output)

        elif args.recipe == "plot_ecg":
            from ml4c3.plots import plot_ecg  # isort: skip
            plot_ecg(args)

        elif args.recipe == "build":
            build_multimodal_multitask(args)

        elif args.recipe == "assess_coverage":
            from ingest.icu.assess_coverage import assess_coverage  # isort: skip
            assess_coverage(args)

        elif args.recipe == "check_edw_structure":
            from tensorize.edw.check_structure import check_edw_structure  # isort: skip
            check_edw_structure(args)

        elif args.recipe == "check_bedmaster_structure":
            from tensorize.bedmaster.check_structure import check_bedmaster_structure  # isort: skip, pylint: disable=line-too-long
            check_bedmaster_structure(args)

        elif args.recipe == "pre_tensorize_explore":
            from ingest.icu.pre_tensorize_explorations import pre_tensorize_explore  # isort: skip, pylint: disable=line-too-long
            pre_tensorize_explore(args)

        elif args.recipe == "match_patient_bedmaster":
            from tensorize.bedmaster.match_patient_bedmaster import match_data  # isort: skip, pylint: disable=line-too-long
            match_data(args)

        elif args.recipe == "visualize":
            from visualizer.run import run_visualizer  # isort: skip
            run_visualizer(args)

        elif args.recipe == "extract_ecg_features":
            from tensorize.bedmaster.ecg_features_extraction import extract_ecg_features  # isort: skip, pylint: disable=line-too-long
            extract_ecg_features(args)

        else:
            raise ValueError("Unknown recipe:", args.recipe)
        # fmt: on

    except Exception as error:
        logging.exception(error)
    finally:
        for child in mp.active_children():
            child.terminate()

    end_time = timer()
    elapsed_time = end_time - start_time
    logging.info(f"Executed {args.recipe} operation in {elapsed_time:.2f} sec")


def build_multimodal_multitask(args: argparse.Namespace):
    from ml4c3.models import make_model  # isort: skip

    model = make_model(args)
    model_file = os.path.join(args.output_folder, "model_weights" + MODEL_EXT)
    model.save(model_file)
    logging.info(f"Model saved to {model_file}")
    return model


def train_model(args: argparse.Namespace) -> Dict[str, float]:
    if args.recipe != "train":
        args.mixup_alpha = 0

    from ml4c3.datasets import (  # isort: skip
        get_split_stats,
        get_verbose_stats_string,
        train_valid_test_datasets,
    )

    # Create datasets
    datasets, stats, cleanups = train_valid_test_datasets(
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        tensors=args.tensors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patient_csv=args.patient_csv,
        mrn_column_name=args.mrn_column_name,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        output_folder=args.output_folder,
        cache=args.cache,
        mixup_alpha=args.mixup_alpha,
        debug=args.debug,
    )
    train_dataset, valid_dataset, test_dataset = datasets

    from ml4c3.models import make_model, train_model_from_datasets  # isort: skip

    model = make_model(args)

    # Train model using datasets
    train_results = train_model_from_datasets(
        model=model,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        epochs=args.epochs,
        image_ext=args.image_ext,
        learning_rate_patience=args.learning_rate_patience,
        learning_rate_reduction=args.learning_rate_reduction,
        num_workers=args.num_workers,
        output_folder=args.output_folder,
        patience=args.patience,
        plot=True,
        return_history=True,
    )
    from tensorflow.keras.models import Model  # isort: skip

    if isinstance(model, Model):
        model, history = train_results
    else:
        model = train_results
    from ml4c3.evaluations import predict_and_evaluate  # isort: skip

    # Evaluate trained model
    plot_path = args.output_folder
    train_results = {}
    if args.mixup_alpha == 0:
        train_results = predict_and_evaluate(
            model=model,
            data=train_dataset,
            tensor_maps_in=args.tensor_maps_in,
            tensor_maps_out=args.tensor_maps_out,
            plot_path=plot_path,
            data_split="train",
            image_ext=args.image_ext,
        )

    test_results = predict_and_evaluate(
        model=model,
        data=test_dataset,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=plot_path,
        data_split="test",
        image_ext=args.image_ext,
        save_coefficients=args.save_coefficients,
        top_features_to_plot=args.top_features_to_plot,
    )

    for cleanup in cleanups:
        cleanup()

    if args.recipe == "train":
        logging.info(f"Model trained for {len(history.history['loss'])} epochs")

    split_stats = get_split_stats(stats_all=stats)
    verbose_stats_string = get_verbose_stats_string(
        split_stats=split_stats,
        input_tmaps=args.tensor_maps_in,
        output_tmaps=args.tensor_maps_out,
    )
    logging.info(verbose_stats_string)

    performance_metrics = {}
    performance_metrics.update(
        {f"auc_train_{key}": value for key, value in train_results.items()},
    )
    performance_metrics.update(
        {f"auc_test_{key}": value for key, value in test_results.items()},
    )
    return performance_metrics


def infer_multimodal_multitask(args: argparse.Namespace) -> Dict[str, float]:
    # Create datasets
    from ml4c3.datasets import train_valid_test_datasets  # isort: skip

    datasets, _, cleanups = train_valid_test_datasets(
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        tensors=args.tensors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patient_csv=args.patient_csv,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        allow_empty_split=True,
        output_folder=args.output_folder,
        cache=args.cache,
        debug=args.debug,
    )
    _, _, test_dataset = datasets

    from ml4c3.models import make_model  # isort: skip
    from ml4c3.evaluations import predict_and_evaluate  # isort: skip

    model = make_model(args)

    data_split = "test"
    if args.test_csv is not None:
        data_split = os.path.splitext(os.path.basename(args.test_csv))[0]
    performance_metrics = predict_and_evaluate(
        model=model,
        data=test_dataset,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=args.output_folder,
        data_split=data_split,
        save_predictions=True,
        image_ext=args.image_ext,
    )

    for cleanup in cleanups:
        cleanup()
    return performance_metrics


def train_simclr_model(args: argparse.Namespace):
    if args.dropout != 0 or args.conv_dropout != 0 or args.conv_regularize is not None:
        raise ValueError("SimCLR model fails to converge with regularization.")

    if args.tensor_maps_out:
        raise ValueError("Cannot give output tensors when training SimCLR model.")
    shape = (args.dense_layers[-1],)

    def make_double_tff(tm):
        def tff(_tm, hd5):
            tensor = tm.tensor_from_file(tm, hd5)
            return np.array([tensor, tensor])

        return tff

    simclr_tensor_maps_in = []
    for tm in args.tensor_maps_in:
        if tm.time_series_limit is not None:
            raise ValueError("SimCLR inputs should only return 1 sample per path.")

        simclr_tm = copy.deepcopy(tm)
        simclr_tm.tensor_from_file = make_double_tff(tm)
        simclr_tm.time_series_limit = 2
        simclr_tm.linked_tensors = True
        simclr_tensor_maps_in.append(simclr_tm)
    if all(map(lambda tm: not tm.augmenters, simclr_tensor_maps_in)):
        raise ValueError("At least one SimCLR input should have augmentations.")
    args.tensor_maps_in = simclr_tensor_maps_in

    from ml4c3.metrics import simclr_loss, simclr_accuracy  # isort: skip
    from tensormap.TensorMap import TensorMap  # isort: skip

    projection_tm = TensorMap(
        name="projection",
        shape=shape,
        loss=simclr_loss,
        metrics=[simclr_accuracy],
        tensor_from_file=lambda tm, hd5: np.zeros((2,) + shape),
        time_series_limit=2,
    )
    args.tensor_maps_out = [projection_tm]
    from ml4c3.datasets import train_valid_test_datasets  # isort: skip

    datasets, _, cleanups = train_valid_test_datasets(
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        tensors=args.tensors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patient_csv=args.patient_csv,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        output_folder=args.output_folder,
        cache=args.cache,
        mixup_alpha=args.mixup_alpha,
        debug=args.debug,
    )
    train_dataset, valid_dataset, _ = datasets

    from ml4c3.models import make_model, train_model_from_datasets  # isort: skip

    model = make_model(args)
    model, history = train_model_from_datasets(
        model=model,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate_patience=args.learning_rate_patience,
        learning_rate_reduction=args.learning_rate_reduction,
        output_folder=args.output_folder,
        image_ext=args.image_ext,
        return_history=True,
        plot=True,
    )

    for cleanup in cleanups:
        cleanup()
    logging.info(f"Model trained for {len(history.history['loss'])} epochs")


if __name__ == "__main__":
    args = parse_args()
    run(args)
