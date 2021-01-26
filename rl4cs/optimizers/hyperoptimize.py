# Imports: standard library
import os
import copy
import json
import math
import time
import logging
import argparse
from typing import Dict

# Imports: third party
import ray
import numpy as np
import pandas as pd
import hyperopt
from ray import tune
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch

# Imports: first party
from rl4cs.utils.element_saver import ElementSaver
from rl4cs.model_training.memory import Memory
from rl4cs.model_training.nn_model import NNModel
from rl4cs.model_training.dqn_algorithm import DQNAlgorithm
from rl4cs.model_training.main_training import main_training


class HyperOptRL:
    """
    Class to optimize hyperparamters from the RL algorithm.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Init class.

        :param args: <argparse.Namespace>
        """
        self.args = args
        self.path_table = os.path.join(args.tables_dir, args.params_tab)
        self.space = self._define_space()
        self.memory = Memory(self.args.max_memory)

    def _define_space(self) -> Dict:
        """
        Define search space of hyperparameters to optimize from a table (.csv).

        :return: <Dict> Space, parameters and their range to be optimized.
        """
        space: Dict[str, hyperopt.pyll.base.Apply] = {}
        hyperparams_df = pd.read_csv(self.path_table)
        if self.args.params_subspace:
            hyperparams_df = hyperparams_df[
                hyperparams_df["Field"] == self.args.params_subspace
            ]
        for _, row in hyperparams_df.iterrows():
            if row.Name not in self.args:
                raise ValueError(
                    f"Parameter {row.Name} in table does not exist in args.",
                )
            if row.Type == "uniform":
                space[row.Name] = hp.uniform(row.Name, row.Val_1, row.Val_2)
            elif row.Type == "quniform":
                space[row.Name] = hp.quniform(row.Name, row.Val_1, row.Val_2, row.Val_3)
            elif row.Type == "loguniform":
                space[row.Name] = hp.loguniform(
                    row.Name,
                    math.log(row.Val_1),
                    math.log(row.Val_2),
                )
            elif row.Type == "choice":
                if row.Name == "hidden_architecture":
                    options = json.loads("[" + row.Choices.replace("-", ",") + "]")
                else:
                    options = row.Choices.split(" - ")
                space[row.Name] = hp.choice(row.Name, options)
            else:
                raise ValueError(f"{row.Type} does not exist as type for space.")
        return space

    def _adapt_arguments(
        self,
        args: argparse.Namespace,
        config: Dict,
    ) -> argparse.Namespace:
        """
        Adapt command line arguments to the search space.

        :param args: <argparse.Namespace> Default arguments
        :param config: <Dict> Configuration of arguments on the specific trial.
        :return: <argparse.Namespace> Modified arguments.
        """
        for hyperparam in self.space:
            if hyperparam not in args:
                raise ValueError(
                    f"Parameter {hyperparam} in table does not exist in args.",
                )
            vars(args)[hyperparam] = config[hyperparam]
        return args

    def _get_data(self):
        """
        Get data from file and load it to the memory.
        """
        loader = ElementSaver(self.args.save_dir)
        _, data = loader.load_states(self.args.save_name)
        # Randomize data
        np.random.shuffle(data)
        for row in data:
            self.memory.add_sample(tuple(row))

    def optimize_rl(self, config: Dict):
        """
        Base function to optimize ALL hyperparameters or RL related ones.

        :param config: <Dict> Configuration of arguments on the specific trial.
        """
        # Make copy to avoid issues among workers.
        args = copy.copy(self.args)
        args = self._adapt_arguments(args, config)
        reward_obtained = main_training(args)
        tune.report(mean_last_reward=reward_obtained)

    def optimize_nn(self, config: Dict):
        """
        Base function to optimize NN related hyperparameters.

        :param config: <Dict> Configuration of arguments on the specific trial.
        """
        # Make copy to avoid issues among workers.
        args = copy.copy(self.args)
        args = self._adapt_arguments(args, config)

        # Compute from memory parameters for nn and training
        num_states = int(len(self.memory.take_samples(1)[0][0]))
        num_actions = int(
            max(np.array(self.memory.take_samples(args.max_memory))[:, 1]) + 1,
        )
        num_batches = int((len(self.memory.samples) / args.policy_batch_size) / 4)
        # Initialize Q networks
        q_nn_model = NNModel(
            num_states,
            num_actions,
            args.policy_batch_size,
            args.hidden_architecture,
            args.activation,
            args.policy_learning_rate,
            args.beta1,
            args.beta2,
        )
        q_target = NNModel(
            num_states,
            num_actions,
            args.policy_batch_size,
            args.hidden_architecture,
            args.activation,
        )

        # Initialize nn variables (weights and biases)
        rl_algorithm = DQNAlgorithm(
            q_nn_model,
            q_target,
            None,
            self.memory,
            args.max_epsilon,
            args.min_epsilon,
            args.lmbda,
            args.discount_gamma,
            args.reply_start_size,
            args.sync_steps,
        )
        # Train neural network
        steps = 0
        loss = 0.0
        init_time = time.time()
        for epoch in range(args.num_epochs + 1):
            # Loop over batches
            for _ in range(num_batches):
                if steps % args.sync_steps == 0:
                    rl_algorithm.sync_q_networks()
                loss = rl_algorithm.replay()
            # Save logs after some 10 epochs
            if epoch % 10 == 0:
                elapsed_time = time.time() - init_time
                logging.info(
                    f"Epoch: {epoch}, Loss = {round(loss, 4)}, "
                    f"Elapsed time = {round(elapsed_time, 2)} seconds.",
                )
            if loss <= args.terminate_loss:
                break
        tune.report(loss=loss)

    def optimize_hyperparameters(
        self,
    ) -> ray.tune.analysis.experiment_analysis.ExperimentAnalysis:
        """
        Optimize hyperparameters using ray wrapping hyperopt library.

        :return: <ray.tune.analysis.experiment_analysis.ExperimentAnalysis>
                 Result analysis after hyperparameter optimization.
        """
        if self.args.params_subspace == "nn":
            # Load data to the memory
            self._get_data()
            base_function = self.optimize_nn
            metric = "loss"
        else:
            base_function = self.optimize_rl
            metric = "mean_last_reward"

        hyperopt_search = HyperOptSearch(
            self.space,
            metric=metric,
            mode=self.args.extreme_type,
        )
        analysis = tune.run(
            base_function,
            num_samples=self.args.num_samples,
            search_alg=hyperopt_search,
            name=self.args.results_name[:-4],
            local_dir=os.path.join(self.args.save_dir, "hyperopt_results"),
        )
        return analysis


def run_hyperoptimizer(args: argparse.Namespace):
    """
    Run hyperparameter optimization and save results to table (.csv).

    :param args: <argparse.Namespace> Arguments.
    """
    rl_hyperopt = HyperOptRL(args)
    save_path = os.path.join(args.save_dir, "hyperopt_results")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    result = rl_hyperopt.optimize_hyperparameters()
    df_result = result.results_df
    df_result.to_csv(os.path.join(save_path, args.results_name))
