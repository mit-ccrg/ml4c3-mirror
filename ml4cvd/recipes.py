# Imports: standard library
import os
import logging
import argparse
from timeit import default_timer as timer
from typing import Dict

# Imports: third party
import numpy as np
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.models import Model

# Imports: first party
from ml4cvd.plots import plot_ecg, plot_metric_history
from ml4cvd.models import (
    make_shallow_model,
    _save_architecture_diagram,
    train_model_from_generators,
    make_multimodal_multitask_model,
)
from ml4cvd.arguments import parse_args
from ml4cvd.definitions import IMAGE_EXT, MODEL_EXT
from ml4cvd.evaluations import predict_and_evaluate
from ml4cvd.explorations import explore
from ml4cvd.tensorizer_ecg import tensorize_ecg
from ml4cvd.hyperparameters import hyperoptimize, sample_random_hyperparameter
from ml4cvd.tensor_generators import (
    TensorGenerator,
    get_verbose_stats_string,
    train_valid_test_tensor_generators,
)


def run(args: argparse.Namespace):
    start_time = timer()  # Keep track of elapsed execution time
    try:
        if "train" == args.mode:
            train_multimodal_multitask(args)
        elif "tensorize" == args.mode:
            tensorize_ecg(args)
        elif "explore" == args.mode:
            explore(args=args, save_output=args.explore_save_output)
        elif "infer" == args.mode:
            infer_multimodal_multitask(args)
        elif "plot_ecg" == args.mode:
            plot_ecg(args)
        elif "train_shallow" == args.mode:
            train_shallow_model(args)
        elif "hyperoptimize" == args.mode:
            hyperoptimize(args)
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
    generate_train, generate_valid, generate_test = train_valid_test_tensor_generators(
        **args.__dict__
    )
    model = make_multimodal_multitask_model(**args.__dict__)
    model, history = train_model_from_generators(
        model=model,
        generate_train=generate_train,
        generate_valid=generate_valid,
        training_steps=args.training_steps,
        validation_steps=args.validation_steps,
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
        data=generate_train,
        steps=args.training_steps,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=out_path,
        data_split="train",
    )
    performance_metrics = predict_and_evaluate(
        model=model,
        data=generate_test,
        steps=args.test_steps,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=out_path,
        data_split="test",
    )

    generate_train.kill_workers()
    generate_valid.kill_workers()
    generate_test.kill_workers()

    logging.info(f"Model trained for {len(history.history['loss'])} epochs")

    if isinstance(generate_train, TensorGenerator):
        logging.info(
            get_verbose_stats_string(
                {
                    "train": generate_train,
                    "valid": generate_valid,
                    "test": generate_test,
                },
            ),
        )
    return performance_metrics


def infer_multimodal_multitask(args: argparse.Namespace) -> Dict[str, float]:
    _, _, generate_test = train_valid_test_tensor_generators(
        no_empty_paths_allowed=False, **args.__dict__
    )
    model = make_multimodal_multitask_model(**args.__dict__)
    out_path = os.path.join(args.output_folder, args.id + "/")
    data_split = "test"
    if args.test_csv is not None:
        data_split = os.path.splitext(os.path.basename(args.test_csv))[0]
    return predict_and_evaluate(
        model=model,
        data=generate_test,
        steps=args.test_steps,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=out_path,
        data_split=data_split,
        save_predictions=True,
    )


def train_shallow_model(args: argparse.Namespace) -> Dict[str, float]:
    """
    Train a shallow model (e.g. linear or logistic regression) and return performance metrics.
    """
    generate_train, generate_valid, generate_test = train_valid_test_tensor_generators(
        **args.__dict__
    )

    if args.shallow_model_regularization is not None:
        # if using l1/l2/l1l2 regularization,
        # generate l1 l2 combinations,
        # train model on training data,
        # assess trained model on validation data,
        # select model that achieves lowest validation loss to apply to test data

        l1_l2_combos = set()
        low = args.L_range[0]
        high = args.L_range[1]
        while len(l1_l2_combos) < args.max_evals:
            l1 = None
            l2 = None
            if args.shallow_model_regularization == "l1l2":
                l1 = sample_random_hyperparameter(low, high, "logarithmic")
                l2 = sample_random_hyperparameter(low, high, "logarithmic")
            elif args.shallow_model_regularization == "l1":
                l1 = sample_random_hyperparameter(low, high, "logarithmic")
            elif args.shallow_model_regularization == "l2":
                l2 = sample_random_hyperparameter(low, high, "logarithmic")
            else:
                raise ValueError(
                    f"Unknown regularization method for shallow model: {args.shallow_model_regularization}",
                )
            l1_l2_combos.add((l1, l2))

        model = None
        best_loss = np.finfo(float).max
        best_l1 = -1
        best_l2 = -1
        for l1, l2 in l1_l2_combos:
            _model = make_shallow_model(
                tensor_maps_in=args.tensor_maps_in,
                tensor_maps_out=args.tensor_maps_out,
                optimizer=args.optimizer,
                learning_rate=args.learning_rate,
                learning_rate_schedule=args.learning_rate_schedule,
                training_steps=args.training_steps,
                model_file=args.model_file,
                donor_layers=args.donor_layers,
                l1=l1,
                l2=l2,
            )
            _model, history = train_model_from_generators(
                model=_model,
                generate_train=generate_train,
                generate_valid=None,
                training_steps=args.training_steps,
                validation_steps=None,
                epochs=args.epochs,
                patience=args.patience,
                learning_rate_patience=args.learning_rate_patience,
                learning_rate_reduction=args.learning_rate_reduction,
                output_folder=args.output_folder,
                run_id=args.id,
                return_history=True,
                plot=False,
            )
            title = f"l1_{l1:.5}_l2_{l2:.5}".replace(".", "-")
            trial_path = os.path.join(args.output_folder, args.id, "trials", title)
            plot_metric_history(
                history=history,
                training_steps=args.training_steps,
                title="",
                prefix=trial_path,
            )
            train_aucs = predict_and_evaluate(
                model=_model,
                data=generate_train,
                steps=args.training_steps,
                tensor_maps_in=args.tensor_maps_in,
                tensor_maps_out=args.tensor_maps_out,
                plot_path=trial_path,
                data_split="train",
            )
            valid_aucs = predict_and_evaluate(
                model=_model,
                data=generate_valid,
                steps=args.validation_steps,
                tensor_maps_in=args.tensor_maps_in,
                tensor_maps_out=args.tensor_maps_out,
                plot_path=trial_path,
                data_split="valid",
            )
            # fmt: off
            logging.info(
                f"\nModel with L1 = {l1:.5} and L2 = {l2:.5} achieved the following AUCs:\n"
                f"\tTrain AUC: {', '.join([f'{label} = {auc:.5}' for label, auc in train_aucs.items()])}\n"
                f"\tValid AUC: {', '.join([f'{label} = {auc:.5}' for label, auc in valid_aucs.items()])}\n",
            )
            # fmt: on

            generate_valid.reset()
            val_loss = _model.evaluate(generate_valid, steps=args.validation_steps)[0]
            if val_loss < best_loss:
                best_loss = val_loss
                best_l1 = l1
                best_l2 = l2
                model = _model

        model_file = os.path.join(
            args.output_folder, args.id, "model_weights" + MODEL_EXT,
        )
        model.save(model_file, overwrite=True)
        _save_architecture_diagram(
            model_to_dot(model, show_shapes=True, expand_nested=True),
            os.path.join(
                args.output_folder, args.id, "architecture_graph" + IMAGE_EXT,
            ),
        )
        # fmt: off
        logging.info(
            f"\nTried {len(l1_l2_combos)} L1 and L2 combinations.\n"
            f"\tModel with L1 = {best_l1:.5} and L2 = {best_l2:.5} achieved best loss of {best_loss:.5} on validation data.\n"
            f"\tModel saved to {model_file} and applied to test data.\n",
        )
        # fmt: on
    else:
        model = make_shallow_model(
            tensor_maps_in=args.tensor_maps_in,
            tensor_maps_out=args.tensor_maps_out,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            learning_rate_schedule=args.learning_rate_schedule,
            training_steps=args.training_steps,
            model_file=args.model_file,
            donor_layers=args.donor_layers,
        )
        model = train_model_from_generators(
            model=model,
            generate_train=generate_train,
            generate_valid=generate_valid,
            training_steps=args.training_steps,
            validation_steps=args.validation_steps,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate_patience=args.learning_rate_patience,
            learning_rate_reduction=args.learning_rate_reduction,
            output_folder=args.output_folder,
            run_id=args.id,
        )

    # Evaluate trained model on test data
    plot_path = os.path.join(args.output_folder, args.id + "/")
    predict_and_evaluate(
        model=model,
        data=generate_train,
        steps=args.training_steps,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=plot_path,
        data_split="train",
    )
    performance_metrics = predict_and_evaluate(
        model=model,
        data=generate_test,
        steps=args.test_steps,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        plot_path=plot_path,
        data_split="test",
        save_coefficients=True,
    )

    generate_train.kill_workers()
    generate_valid.kill_workers()
    generate_test.kill_workers()
    return performance_metrics


if __name__ == "__main__":
    args = parse_args()
    run(args)
