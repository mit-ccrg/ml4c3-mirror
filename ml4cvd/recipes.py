# Imports: standard library
import os
import logging
import argparse
from timeit import default_timer as timer
from typing import Dict

# Imports: third party
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.models import Model

# Imports: first party
from ml4cvd.plots import plot_ecg
from ml4cvd.models import (
    make_shallow_model,
    train_model_from_datasets,
    _save_architecture_diagram,
    make_multimodal_multitask_model,
)
from ml4cvd.datasets import get_verbose_stats_string, train_valid_test_datasets
from ml4cvd.arguments import parse_args
from ml4cvd.definitions import IMAGE_EXT, MODEL_EXT
from ml4cvd.evaluations import predict_and_evaluate
from ml4cvd.explorations import explore
from ml4cvd.tensorizer_ecg import tensorize_ecg
from ml4cvd.hyperoptimizers import hyperoptimize


def run(args: argparse.Namespace):
    start_time = timer()  # Keep track of elapsed execution time
    try:
        if "train" == args.mode:
            train_multimodal_multitask(args)
        elif "tensorize" == args.mode:
            tensorize_ecg(args)
        elif "explore" == args.mode:
            explore(args=args, disable_saving_output=args.explore_disable_saving_output)
        elif "infer" == args.mode:
            infer_multimodal_multitask(args)
        elif "plot_ecg" == args.mode:
            plot_ecg(args)
        elif "train_shallow" == args.mode:
            train_shallow_model(args)
        elif "hyperoptimize" == args.mode:
            hyperoptimize(args)
        elif "hyperoptimize_shallow" == args.mode:
            hyperoptimize_shallow(args)
        elif "build" == args.mode:
            build_multimodal_multitask(args)
        else:
            raise ValueError("Unknown mode:", args.mode)

    except Exception as e:
        logging.exception(e)

    end_time = timer()
    elapsed_time = end_time - start_time
    logging.info(
        "Executed the '{}' operation in {:.2f} seconds".format(args.mode, elapsed_time),
    )


def build_multimodal_multitask(args: argparse.Namespace) -> Model:
    model = make_multimodal_multitask_model(**args.__dict__)
    model_file = os.path.join(args.output_folder, args.id, "model_weights" + MODEL_EXT)
    model.save(model_file)
    _save_architecture_diagram(
        model_to_dot(model, show_shapes=True, expand_nested=True),
        os.path.join(args.output_folder, args.id, "architecture_graph" + IMAGE_EXT),
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
    )
    train_dataset, valid_dataset, test_dataset = datasets
    model = make_multimodal_multitask_model(**args.__dict__)
    model, history = train_model_from_datasets(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate_patience=args.learning_rate_patience,
        learning_rate_reduction=args.learning_rate_reduction,
        output_folder=args.output_folder,
        run_id=args.id,
        return_history=True,
    )

    out_path = os.path.join(args.output_folder, args.id + "/")
    predict_and_evaluate(
        model=model,
        data=train_dataset,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=out_path,
        data_split="train",
    )
    performance_metrics = predict_and_evaluate(
        model=model,
        data=test_dataset,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=out_path,
        data_split="test",
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
        no_empty_paths_allowed=False,
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
    )

    for cleanup in cleanups:
        cleanup()
    return performance_metrics


def train_shallow_model(args: argparse.Namespace) -> Dict[str, float]:
    """
    Train shallow model (e.g. linear or logistic regression) and return performance metrics.
    """
    # Create datasets
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
    )
    train_dataset, valid_dataset, test_dataset = datasets

    # Initialize shallow model
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

    # Train model using datasets
    model = train_model_from_datasets(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate_patience=args.learning_rate_patience,
        learning_rate_reduction=args.learning_rate_reduction,
        output_folder=args.output_folder,
        run_id=args.id,
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
    )
    performance_metrics = predict_and_evaluate(
        model=model,
        data=test_dataset,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=plot_path,
        data_split="test",
        save_coefficients=True,
    )
    for cleanup in cleanups:
        cleanup()
    return performance_metrics


if __name__ == "__main__":
    args = parse_args()
    run(args)
