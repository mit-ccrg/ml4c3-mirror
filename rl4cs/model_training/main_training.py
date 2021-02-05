# Imports: standard library
import os
import time
import logging
import argparse

# Imports: third party
import gym
import numpy as np
from gym.envs.registration import register

# Imports: first party
from rl4cs.utils.plotter import RLPlotter
from rl4cs.utils.element_saver import ElementSaver
from rl4cs.model_training.memory import Memory
from rl4cs.model_training.nn_model import NNModel
from rl4cs.model_training.dqn_algorithm import DQNAlgorithm

# pylint: disable=invalid-name,too-many-branches


def main_training(args: argparse.Namespace):
    """
    Training policy with DQN & experience replay algorithm.

    :param args: <argparse.Namespace> Arguments.
    """
    # Initialize environment
    if args.environment == "cv_model":
        register(id="CV_Typed-v0", entry_point="rl4cs.environments:CVEnvTyped")
        env = gym.make(
            "CV_Typed-v0",
            init_conditions=args.init_conditions,
            paths=[
                os.path.join(args.tables_dir, "discrete_action_space_v2.csv"),
                os.path.join(args.tables_dir, "reward_params.json"),
                args.save_dir,
            ],
            save_flag=args.save,
            render=args.render,
        )
    elif args.environment == "cartpole":
        register(id="CartPole_own-v0", entry_point="rl4cs.environments:CartPoleEnv")
        env = gym.make("CartPole_own-v0")
    else:
        raise ValueError(
            f"Environment {args.enviroment} doesn't exist. "
            f"Possible nevironments are: cv_model and cartpole.",
        )

    # Get parameters from environment
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Set Domain randomization training
    if args.environment == "cv_model" and args.recipe == "training":
        env.set_domain_randomization(args.patient_state, args.randomization)

    # Create q and target nn model, memory and set max steps per episode
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

    mem = Memory(args.max_memory)
    env.max_sim_time = args.max_episode_steps

    # Initialize DQN RL algorithm
    rl_algorithm = DQNAlgorithm(
        q_nn_model,
        q_target,
        env,
        mem,
        args.max_epsilon,
        args.min_epsilon,
        args.lmbda,
        args.discount_gamma,
        args.reply_start_size,
        args.sync_steps,
        args.dqn_method,
    )
    cnt = 0
    start_time = time.time()
    start_time2 = None
    while cnt < args.num_episodes:
        # Every 10 eposides show PV Loop, training time, states and reward
        env.model.episode = cnt
        if cnt % 10 == 0:
            logging.info(f"Episode {cnt} of {args.num_episodes}.")
            logging.info(f"Current states: {env.observations}.")
            end_time = time.time()
            if start_time2:
                logging.info(
                    f"Episode time: {round(end_time - start_time2, 2)} seconds.",
                )
            if rl_algorithm.reward_store:
                logging.info(f"Obtained reward: {rl_algorithm.reward_store[-1]}.")
            start_time2 = time.time()
        if cnt % 10 == 0:
            rl_algorithm.render = True and args.render
        else:
            env.close()
            rl_algorithm.render = False
        rl_algorithm.run()
        cnt += 1
    end_time = time.time()
    logging.info(f"Total time {round((end_time - start_time) / 60, 2)} minutes.")

    # Save data
    if args.save:
        env.saver.save_states(args.save_name + ".npy")
        env.saver.save_params(args.save_name + ".npy")
    saver = ElementSaver(args.save_dir)
    saver.save_qfunc(args.save_name + ".h5", q_nn_model.model)

    # Print total reward evolution over episodes
    if args.render:
        title = "Reward evolution over episodes"
        axis_labels = ("Number of episodes", "Total reward")
        axis_lims = None
        plot = RLPlotter(title, axis_labels, axis_lims, args.save_dir)
        data = np.array(
            [
                list(range(len(rl_algorithm.reward_store))),
                rl_algorithm.reward_store,
            ],
        ).transpose()
        plot.update_plot(data)
        plot.save_figure("plot_reward200_volume_newmethod.png")
    if args.recipe == "hyperoptimize":
        return sum(rl_algorithm.reward_store[-args.len_reward :]) / args.len_reward
