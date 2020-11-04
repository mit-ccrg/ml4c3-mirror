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
from ml4c3.plots import plot_ecg, plot_architecture_diagram
from ml4c3.models import (
    make_shallow_model,
    make_sklearn_model,
    train_model_from_datasets,
    make_multimodal_multitask_model,
)
from ml4c3.metrics import simclr_loss, simclr_accuracy
from ml4c3.datasets import get_verbose_stats_string, train_valid_test_datasets
from ml4c3.arguments import parse_args
from ml4c3.ingest.ecg import tensorize as tensorize_ecg
from ml4c3.ingest.icu import tensorize as tensorize_icu
from ml4c3.ingest.icu import tensorize_batched as tensorize_icu_batched
from ml4c3.ingest.sts import tensorize as tensorize_sts
from ml4c3.evaluations import predict_and_evaluate
from ml4c3.explorations import explore
from ml4c3.hyperoptimizers import hyperoptimize
from ml4c3.assess_icu_coverage import assess_icu_coverage
from ml4c3.definitions.globals import MODEL_EXT
from ml4c3.tensormap.TensorMap import TensorMap
from ml4c3.ecg_features_extraction import extract_ecg_features
from ml4c3.ingest.icu.matchers.match_data import match_data
from ml4c3.ingest.icu.summarizers.summarizer import pre_tensorize_summary
from ml4c3.ingest.icu.check_icu_structure.check_icu_structure import check_icu_structure

# pylint: disable=redefined-outer-name, broad-except


def run(args: argparse.Namespace):
    start_time = timer()  # Keep track of elapsed execution time
    try:
        if args.mode == "train":
            train_multimodal_multitask(args)
        elif args.mode == "train_shallow":
            train_shallow_model(args)
        elif args.mode == "train_simclr":
            train_simclr_model(args)
        elif args.mode == "infer":
            infer_multimodal_multitask(args)
        elif args.mode == "hyperoptimize":
            hyperoptimize(args)
        elif args.mode == "tensorize_ecg":
            tensorize_ecg(args)
        elif args.mode == "tensorize_icu":
            if args.staging_batch_size:
                tensorize_icu_batched(args)
            else:
                tensorize_icu(args)
        elif args.mode == "tensorize_sts":
            tensorize_sts(args)
        elif args.mode == "explore":
            explore(args=args, disable_saving_output=args.explore_disable_saving_output)
        elif args.mode == "plot_ecg":
            plot_ecg(args)
        elif args.mode == "build":
            build_multimodal_multitask(args)
        elif args.mode == "assess_icu_coverage":
            assess_icu_coverage(args)
        elif args.mode == "check_icu_structure":
            check_icu_structure(args)
        elif args.mode == "pre_tensorize_summary":
            pre_tensorize_summary(args)
        elif args.mode == "match_patient_bm":
            match_data(args)
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
    model = make_multimodal_multitask_model(**args.__dict__)
    model_file = os.path.join(args.output_folder, args.id, "model_weights" + MODEL_EXT)
    model.save(model_file)
    plot_architecture_diagram(
        model_to_dot(model, show_shapes=True, expand_nested=True),
        os.path.join(
            args.output_folder,
            args.id,
            "architecture_graph" + args.image_ext,
        ),
    )
    logging.info(f"Model saved to {model_file}")
    return model


def train_multimodal_multitask(args: argparse.Namespace) -> Dict[str, float]:
    datasets, stats, cleanups = train_valid_test_datasets(
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        tensors=args.tensors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_csv=args.sample_csv,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        output_folder=args.output_folder,
        run_id=args.id,
        cache=args.cache,
        mixup_alpha=args.mixup_alpha,
    )
    train_dataset, valid_dataset, test_dataset = datasets
    model = make_multimodal_multitask_model(**args.__dict__)

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
        run_id=args.id,
        image_ext=args.image_ext,
        return_history=True,
        plot=True,
    )

    out_path = os.path.join(args.output_folder, args.id + "/")
    if args.mixup_alpha == 0:
        predict_and_evaluate(
            model=model,
            data=train_dataset,
            tensor_maps_in=args.tensor_maps_in,
            tensor_maps_out=args.tensor_maps_out,
            plot_path=out_path,
            data_split="train",
            image_ext=args.image_ext,
        )

    performance_metrics = predict_and_evaluate(
        model=model,
        data=test_dataset,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=out_path,
        data_split="test",
        image_ext=args.image_ext,
        save_coefficients=args.save_coefficients,
    )

    for cleanup in cleanups:
        cleanup()

    logging.info(f"Model trained for {len(history.history['loss'])} epochs")
    logging.info(
        get_verbose_stats_string(
            split_stats={
                "train": stats[0].stats,
                "valid": stats[1].stats,
                "test": stats[2].stats,
            },
            input_maps=args.tensor_maps_in,
            output_maps=args.tensor_maps_out,
        ),
    )
    return performance_metrics


def infer_multimodal_multitask(args: argparse.Namespace) -> Dict[str, float]:
    datasets, _, cleanups = train_valid_test_datasets(
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        tensors=args.tensors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_csv=args.sample_csv,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        no_empty_paths_allowed=False,
        output_folder=args.output_folder,
        run_id=args.id,
        cache=args.cache,
    )
    _, _, test_dataset = datasets

    model = make_multimodal_multitask_model(**args.__dict__)
    out_path = os.path.join(args.output_folder, args.id + "/")
    data_split = "test"
    if args.test_csv is not None:
        data_split = os.path.splitext(os.path.basename(args.test_csv))[0]
    performance_metrics = predict_and_evaluate(
        model=model,
        data=test_dataset,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=out_path,
        data_split=data_split,
        save_predictions=True,
        image_ext=args.image_ext,
    )

    for cleanup in cleanups:
        cleanup()
    return performance_metrics


def train_shallow_model(args: argparse.Namespace) -> Dict[str, float]:
    """
    Train shallow model (e.g. linear or logistic regression) and return
    performance metrics.
    """

    # Create datasets
    datasets, _, cleanups = train_valid_test_datasets(
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        tensors=args.tensors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_csv=args.sample_csv,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        output_folder=args.output_folder,
        run_id=args.id,
        cache=args.cache,
        mixup_alpha=args.mixup_alpha,
    )
    train_dataset, valid_dataset, test_dataset = datasets

    # Initialize shallow model
    if args.sklearn_model_type is None:
        model = make_shallow_model(
            tensor_maps_in=args.tensor_maps_in,
            tensor_maps_out=args.tensor_maps_out,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            learning_rate_schedule=args.learning_rate_schedule,
            model_file=args.model_file,
            donor_layers=args.donor_layers,
            l1=args.l1,
            l2=args.l2,
        )
    else:
        # SKLearn only works with one output tmap
        assert len(args.tensor_maps_out) == 1

        hyperparameters = {}
        if args.sklearn_model_type == "logreg":
            if args.l1 == 0 and args.l2 == 0:
                c = 1e7
            else:
                c = 1 / (args.l1 + args.l2)
            hyperparameters["c"] = args.c
            hyperparameters["l1_ratio"] = args.c * args.l1
        elif args.sklearn_model_type == "svm":
            hyperparameters["c"] = args.c
        elif args.sklearn_model_type == "randomforest":
            hyperparameters["n_estimators"] = args.n_estimators
            hyperparameters["max_depth"] = args.max_depth
            hyperparameters["min_samples_split"] = args.min_samples_split
            hyperparameters["min_samples_leaf"] = args.min_samples_leaf
        elif args.sklearn_model_type == "xgboost":
            hyperparameters["n_estimators"] = args.n_estimators
            hyperparameters["max_depth"] = args.max_depth
        model = make_sklearn_model(
            model_type=args.sklearn_model_type,
            hyperparameters=hyperparameters,
        )

    # Train model using datasets
    model = train_model_from_datasets(
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
        run_id=args.id,
        image_ext=args.image_ext,
    )

    # Evaluate trained model
    plot_path = os.path.join(args.output_folder, args.id + "/")
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
        save_coefficients=True,
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
        sample_csv=args.sample_csv,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        output_folder=args.output_folder,
        run_id=args.id,
        cache=args.cache,
        mixup_alpha=args.mixup_alpha,
    )
    train_dataset, valid_dataset, _ = datasets
    model = make_multimodal_multitask_model(**args.__dict__)
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
        run_id=args.id,
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
