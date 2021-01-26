# Imports: standard library
import os
import logging
import argparse

# Imports: third party
import gym
import numpy as np
from gym.envs.registration import register

# Imports: first party
from definitions.cv import PARAMETER_NAMES as PARAM_NAME
from definitions.cv import PARAMETER_UNITS as PARAM_UNIT
from definitions.cv import OBSERVATIONS_NAME as OBS_NAME
from definitions.cv import OBSERVATIONS_MAPPING as OBS_MAP
from definitions.cv import OBSERVATIONS_TO_UNITS as OBS_UNIT
from rl4cs.utils.plotter import RLPlotter
from rl4cs.utils.element_saver import ElementSaver
from rl4cs.model_training.nn_model import NNModel

# pylint: disable=invalid-name,too-many-branches


def main_simulation(args: argparse.Namespace):
    """
    Simulate model with or without policy.

    :param args: <argparse.Namespace> Arguments.
    """
    # Initialize environment
    if args.environment == "cv_model":
        register(id="CV_Typed-v0", entry_point="rl4cs.environments:CVEnvTyped")
        env = gym.make(
            "CV_Typed-v0",
            init_conditions=args.init_conditions,
            paths=[
                os.path.join(args.tables_dir, "discrete_action_space.csv"),
                args.save_dir,
            ],
            save_flag=True,
            render=args.render,
        )
    elif args.environment == "cartpole":
        register(id="CartPole_own-v0", entry_point="rl4cs.environments:CartPoleEnv")
        env = gym.make("CartPole_own-v0")
    else:
        raise ValueError(
            f"Environment {args.enviroment} doesn't exist. "
            f"Possible environments are: cv_model and cartpole.",
        )
    # Run simulation, either transient or steady-state
    # Initialize savers
    saver = ElementSaver(args.save_dir)
    saver_rew = ElementSaver(args.save_dir)
    saver_act = ElementSaver(args.save_dir)

    logging.info(
        f"Start {args.sim_type} simulation of {args.environment} model with policy "
        f"called: {args.policy_name}. Initial time: {args.start_time}, "
        f"final time: {args.end_time}.",
    )

    if args.sim_type == "transient":
        # Policy won't have any effect in this case
        if args.policy_name:
            logging.warning(
                "This type of simulation is not compatible with"
                "a policy. Simulation will be done without policy.",
            )
            # Run simulation and store states
            time, observations = env.model.simulation(args.end_time, args.start_time)
            saver.buff_time = time
            saver.buff_states = observations
            # Save states
            saver.save_states(args.save_name + "_states" + ".npy")

    elif args.sim_type == "steady-state":
        # Load policy as a Q neural network
        if args.policy_name:
            num_states = env.observation_space.shape[0]
            num_actions = env.action_space.n
            q_nn = NNModel(num_states, num_actions)
            loader_poli = ElementSaver(args.save_dir)
            q_nn.model = loader_poli.load_qfunc(args.policy_name)

        # Initialize reward and reset environment
        total_reward = 0
        observations = env.reset()
        # Run simulation
        while env.time_step <= args.end_time:
            if args.policy_name:
                action = int(np.argmax(q_nn.predict_one(observations).numpy()))
            else:
                action = 0
            saver_act.buff_states.append((action,))
            saver_act.buff_time.append(env.time_step)
            observations, reward, _, _ = env.step(action)
            saver_rew.buff_states.append((reward,))
            total_reward += reward
        saver_rew.buff_states.append((total_reward,))

        # Save actions, observations, reward and parameters
        saver_rew.save_states(args.save_name + "_reward" + ".npy")
        saver_act.save_states(args.save_name + "_action" + ".npy")
        env.saver.save_states(args.save_name + "_states" + ".npy")
        env.saver.save_params(args.save_name + "_params" + ".npy")

    else:
        raise ValueError(
            f"Simulation {args.sim_type} doesn't exist. "
            f"Possible simulation types are: transient and steady-state.",
        )

    if args.render:
        # Plot results
        logging.info(
            f"Start Plotting results. Plotting states: {args.plot_list_states}, "
            f"parameters: {args.plot_list_params} "
            f"and extras: {args.plot_list_others}.",
        )

        if args.sim_type == "transient":
            # Load data
            time, states = saver.load_states(args.save_name + "_states" + ".npy")
            # Plot states
            for state in args.plot_list_states:
                if state not in OBS_NAME:
                    raise ValueError(f"{state} state/observation does not exist.")
                title = f"{OBS_NAME[state]} Evolution"
                axis_labels = ("Time [s]", f"{OBS_NAME[state]} [{OBS_UNIT[state]}]")
                axis_lims = None
                plot = RLPlotter(title, axis_labels, axis_lims, args.save_dir)
                plot.update_plot(
                    np.array([time, states[:, OBS_MAP[state]]]).transpose(),
                )
                plot.save_figure(args.save_name + "_" + state + ".png")
            # Plot PV Loop
            if "pv_loop" in args.plot_list_others:
                title = "PV Loop"
                axis_labels = (
                    f"LV Volume' [{OBS_UNIT['v_lv']}]",
                    f"LV Pressure [{OBS_UNIT['p_lv']}]",
                )
                axis_lims = None
                plot = RLPlotter(title, axis_labels, axis_lims, args.save_dir)
                plot.update_plot(
                    np.array(
                        [states[:, OBS_MAP["v_lv"]], states[:, OBS_MAP["p_lv"]]],
                    ).transpose(),
                )
                plot.save_figure(args.save_name + "_pv_loop" + ".png")

        if args.sim_type == "steady-state":
            # Load data
            _, rewards = saver_rew.load_states(args.save_name + "_reward" + ".npy")
            _, actions = saver_act.load_states(args.save_name + "_action" + ".npy")
            time, obs = env.saver.load_states(args.save_name + "_states" + ".npy")
            params = env.saver.load_params(args.save_name + "_params" + ".npy")
            # Plot states/observations
            for state in args.plot_list_states:
                if state not in OBS_NAME:
                    raise ValueError(f"{state} state/observation does not exist.")
                title = f"Mean {OBS_NAME[state]} Evolution"
                axis_labels = ("Time [min]", f"Mean {state} [{OBS_UNIT[state]}]")
                axis_lims = None
                plot = RLPlotter(title, axis_labels, axis_lims, args.save_dir)
                plot.update_plot(np.array([time, obs[:, OBS_MAP[state]]]).transpose())
                plot.save_figure(args.save_name + "_" + state + ".png")
            # Plot parameters
            for parameter in args.plot_list_params:
                if parameter not in PARAM_NAME:
                    raise ValueError(f"{parameter} parameter does not exist.")
                title = f"{PARAM_NAME[parameter]} Evolution"
                axis_labels = ("Time [min]", f"{parameter} [{PARAM_UNIT[parameter]}]")
                axis_lims = None
                plot = RLPlotter(title, axis_labels, axis_lims, args.save_dir)
                plot.update_plot(np.array([time, params[parameter]]).transpose())
                plot.save_figure(args.save_name + "_" + parameter + ".png")
            # Plot other fields (actions, reward, etc.)
            if "actions" in args.plot_list_others:
                title = "Actions Taken Over Time"
                axis_labels = ("Time [min]", "Action [#]")
                axis_lims = None
                plot = RLPlotter(title, axis_labels, axis_lims, args.save_dir)
                plot.update_plot(
                    np.array([time, actions.reshape(len(actions))]).transpose(),
                )
                plot.save_figure(args.save_name + "_actions" + ".png")
            if "reward" in args.plot_list_others:
                title = "Reward Over Time"
                axis_labels = ("Time [min]", "Reward [u]")
                axis_lims = None
                plot = RLPlotter(title, axis_labels, axis_lims, args.save_dir)
                rewards = rewards.reshape(len(rewards))
                data = np.array([list(range(len(rewards[:-1]))), rewards[:-1]])
                text = f"Total reward: {rewards[-1]}"
                plot.update_plot(data.transpose(), text)
                plot.save_figure(args.save_name + "_reward" + ".png")

    if not args.save:
        # Remove files if they aren't meant to be stored
        logging.info("Remove stored files.")

        saver_rew.remove_states(args.save_name + "_reward" + ".npy")
        saver_act.remove_states(args.save_name + "_action" + ".npy")
        env.saver.remove_states(args.save_name + "_states" + ".npy")
        env.saver.remove_params(args.save_name + "_params" + ".npy")
