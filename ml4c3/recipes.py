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
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.models import Model

# Imports: first party
from ingest.ecg import tensorize as tensorize_ecg
from ingest.icu import tensorize_batched as tensorize_icu_batched
from ml4c3.plots import plot_ecg, plot_architecture_diagram
from ml4c3.models import make_model, train_model_from_datasets
from ml4c3.metrics import simclr_loss, simclr_accuracy
from ml4c3.datasets import get_verbose_stats_string, train_valid_test_datasets
from visualizer.run import run_server
from ml4c3.arguments import parse_args
from ml4c3.evaluations import predict_and_evaluate
from ml4c3.explorations import explore
from definitions.globals import MODEL_EXT
from ingest.edw.pipeline import pull_edw_data
from ml4c3.hyperoptimizers import hyperoptimize
from ml4c3.tensormap.TensorMap import TensorMap
from ingest.icu.assess_coverage import assess_coverage
from ingest.icu.check_structure import check_icu_structure
from ingest.icu.ecg_features_extraction import extract_ecg_features
from ingest.icu.match_patient_bedmaster import match_data
from ingest.icu.pre_tensorize_explorations import pre_tensorize_explore

# pylint: disable=redefined-outer-name, broad-except


def run(args: argparse.Namespace):
    start_time = timer()  # Keep track of elapsed execution time
    try:
        if args.mode in [
            "train",
            "train_keras_logreg",
            "train_sklearn_logreg",
            "train_sklearn_randomforest",
            "train_sklearn_svm",
            "train_sklearn_xbost",
        ]:
            train_model(args)
        elif args.mode == "train_simclr":
            train_simclr_model(args)
        elif args.mode == "infer":
            infer_multimodal_multitask(args)
        elif args.mode == "hyperoptimize":
            hyperoptimize(args)
        elif args.mode == "tensorize_ecg":
            tensorize_ecg(args)
        elif args.mode == "pull_adt":
            pull_edw_data(args, only_adt=True)
        elif args.mode == "pull_edw":
            pull_edw_data(args)
        elif args.mode == "tensorize_icu_no_edw_pull":
            tensorize_icu_batched(args)
        elif args.mode == "tensorize_icu":
            pull_edw_data(args)
            tensorize_icu_batched(args)
        elif args.mode == "explore":
            explore(args=args, disable_saving_output=args.explore_disable_saving_output)
        elif args.mode == "plot_ecg":
            plot_ecg(args)
        elif args.mode == "build":
            build_multimodal_multitask(args)
        elif args.mode == "assess_coverage":
            assess_coverage(args)
        elif args.mode == "check_icu_structure":
            check_icu_structure(args)
        elif args.mode == "pre_tensorize_explore":
            pre_tensorize_explore(args)
        elif args.mode == "match_patient_bedmaster":
            match_data(args)
        elif args.mode == "visualize":
            run_server(args)
        elif args.mode == "extract_ecg_features":
            extract_ecg_features(args)
        else:
            raise ValueError("Unknown mode:", args.mode)

    except Exception as error:
        logging.exception(error)
        for child in mp.active_children():
            child.terminate()

    end_time = timer()
    elapsed_time = end_time - start_time
    logging.info(
        "Executed the '{}' operation in {:.2f} seconds".format(args.mode, elapsed_time),
    )


def build_multimodal_multitask(args: argparse.Namespace) -> Model:
    model = make_model(args)
    model_file = os.path.join(args.output_folder, "model_weights" + MODEL_EXT)
    model.save(model_file)
    plot_architecture_diagram(
        model_to_dot(model, show_shapes=True, expand_nested=True),
        os.path.join(
            args.output_folder,
            "architecture_graph" + args.image_ext,
        ),
    )
    logging.info(f"Model saved to {model_file}")
    return model


def train_model(args: argparse.Namespace) -> Dict[str, float]:
    if args.mode != "train":
        args.mixup_alpha = 0

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

    model = make_model(args)

    # Train model using datasets
    train_results = train_model_from_datasets(
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
    if isinstance(model, Model):
        model, history = train_results
    else:
        model = train_results

    # Evaluate trained model
    plot_path = args.output_folder
    if args.mixup_alpha == 0:
        predict_and_evaluate(
            model=model,
            data=train_dataset,
            tensor_maps_in=args.tensor_maps_in,
            tensor_maps_out=args.tensor_maps_out,
            plot_path=plot_path,
            data_split="train",
            image_ext=args.image_ext,
        )

    performance_metrics = predict_and_evaluate(
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

    if args.mode == "train":
        logging.info(f"Model trained for {len(history.history['loss'])} epochs")
    logging.info(
        get_verbose_stats_string(
            split_stats={
                "train": stats[0].stats,
                "valid": stats[1].stats,
                "test": stats[2].stats,
            },
            input_tmaps=args.tensor_maps_in,
            output_tmaps=args.tensor_maps_out,
        ),
    )
    return performance_metrics


def infer_multimodal_multitask(args: argparse.Namespace) -> Dict[str, float]:
    # Create datasets
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

    projection_tm = TensorMap(
        name="projection",
        shape=shape,
        loss=simclr_loss,
        metrics=[simclr_accuracy],
        tensor_from_file=lambda tm, hd5: np.zeros((2,) + shape),
        time_series_limit=2,
    )
    args.tensor_maps_out = [projection_tm]

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
