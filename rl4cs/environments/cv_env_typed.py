# Imports: standard library
from abc import ABC
from typing import Dict, List, Tuple, Optional

# Imports: third party
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

# Imports: first party
from rl4cs.utils.element_saver import ElementSaver
from rl4cs.environments.rewards import Reward
from rl4cs.environments.burkhoff_model.cv_model import CVModel


class CVEnvTyped(gym.Env, ABC):
    """
    Description:
        Implement gym environment for typed CV model.

    Observation:
        Type: Box(13)
        Num    Observation                 Min         Max         Units
        0   V_lv_mean                   0           750         [ml]
        1   V_cas_mean                  0           750         [ml]
        2   V_cvs_mean                  0           750         [ml]
        3   V_rv_mean                   0           750         [ml]
        4   V_cap_mean                  0           750         [ml]
        5   V_cvp_mean                  0           750         [ml]
        6   P_lv_mean                   0           200         [mmHg]
        7   P_cas mean                  0           200         [mmHg]
        8   P_cvs mean                  0           200         [mmHg]
        9   P_rv mean                   0           200         [mmHg]
        10  P_cap mean                  0           200         [mmHg]
        11  P_cvp mean                  0           200         [mmHg]
        12  Card_output                 0           7           [L/min]

    Actions:
        Type: Discrete(8)
        Num    Action
        0   Nothing.
        1   No V, no HR, decrease Elastance.
        2   No V, no HR, increase Elastance.
        3   No V, decrease HR, no Elastance.
        4   No V, decrease HR, decrease Elastance.
        5   No V, decrease HR, increase Elastance.
        6   No V, increase HR, no Elastance.
        7   No V, increase HR, decrease Elastance.
        8   No V, increase HR, increase Elastance.
        9   Decrease V, no HR, no Elastance.
        10   Decrease V, no HR, decrease Elastance.
        11   Decrease V, no HR, increase Elastance.
        12   Decrease V, decrease HR, no Elastance.
        13   Decrease V, decrease HR, decrease Elastance.
        14   Decrease V, decrease HR, increase Elastance.
        15   Decrease V, increase HR, no Elastance.
        16   Decrease V, increase HR, decrease Elastance.
        17   Decrease V, increase HR, increase Elastance.
        18   Increase V, no HR, no Elastance.
        19   Increase V, no HR, decrease Elastance.
        20   Increase V, no HR, increase Elastance.
        21   Increase V, decrease HR, no Elastance.
        22   Increase V, decrease HR, decrease Elastance.
        23   Increase V, decrease HR, increase Elastance.
        24   Increase V, increase HR, no Elastance.
        25   Increase V, increase HR, decrease Elastance.
        26   Increase V, increase HR, increase Elastance.

    Starting State:
        Different illness states: normal, mild, moderate, severe.
    """

    def __init__(
        self,
        init_conditions: List,
        paths: List[str],
        save_flag: bool = False,
        render: bool = False,
    ):
        """
        Init Gym environment with CV model.

        :param init_conditions: <List> Initial conditions for the CV model.
        :param save_flag: <bool> Flag to determine whether saving states or not.
        :param render: <bool> Flag to determine whether showing PV loop or not.
        """
        # Init
        self.init_conditions = init_conditions
        self.path_saver = paths[2]
        self.np_random = None
        self.seed()
        self.time_step = 0
        self.max_time_rl = 2880  # 48 hours of stay

        # Patient state
        self.patient_state: Optional[str] = None
        self.randomization: Optional[str] = None

        # Loading cv model
        self.model = CVModel(init_conditions, paths, save_flag, render)

        # Set limits
        obs_limits = self._set_limits_observations()
        self.obs_min = np.array(obs_limits["lower"])
        self.obs_max = np.array(obs_limits["upper"])

        # Define reward
        self.reward = Reward(
            paths[1],
            type_rew="continuous",
            cardiac_output=True,
            drug_penalty=False,
            power=True,
        )

        self.params_lims: Dict[str, List] = {
            "Vs": [250, 4000],
            "HR": [10, 199],
            "Ees_rv": [0.02, 6.0],
            "Ees_lv": [0.1, 12.0],
        }

        # Gym interface
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(self.obs_min, self.obs_max)
        self.observations = np.ones(self.observation_space.shape[0])

        # Saver
        self.save_flag = save_flag
        if self.save_flag:
            self.saver = ElementSaver(self.path_saver)

    @staticmethod
    def _set_limits_observations() -> Dict:
        """
        Set lower and upper limits for observations. List order is: V_lv_mean,
        V_cas_mean, V_cvs_mean, V_rv_mean, V_cap_mean, V_cvp_mean, P_lv_mean,
        P_cas mean , P_cvs mean, P_rv mean, P_cap mean, P_cvp mean and
        Card_output.

        :return: <Dict> Lower and upper bounds of observations.
        """
        limits: Dict[str, List[float]] = {
            "lower": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "upper": [750, 750, 750, 750, 750, 750, 200, 200, 200, 200, 200, 200, 8],
        }
        return limits

    def seed(self, seed=None) -> List:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple:
        """
        Simulate step forward for training RL algorithm.

        :param action: <int> Index with the action to take.
        :return: <Tuple> Array with observations, reward, done status and information.
        """
        # Save data
        if self.save_flag:
            self.saver.buff_states.append(self.observations)
            self.saver.buff_time.append(self.time_step)
            self.saver.store_dict(self.model.params)

        # Get step forward
        self.time_step, self.observations = self.model.step_training_rl(action)

        # Get reward and terminate condition
        reward, done = self._reward_and_terminate_condition(action)

        return self.observations, reward, done, {}

    def _reward_and_terminate_condition(self, action) -> Tuple:
        """
        Compute reward and determine terminate condition.

        :return: <Tuple> Reward and terminate condition
        """
        # Reward function
        reward = self.reward.compute_reward(self.observations, action)
        card_output = self.observations[-1]

        # Terminate condition
        done = (
            (self.time_step >= self.max_time_rl)
            or (card_output < 2)
            or any(self.observations < self.obs_min)
            or any(self.observations > self.obs_max)
            or self.model.params["Ees_rv"] > max(self.params_lims["Ees_rv"])
            or self.model.params["Ees_lv"] > max(self.params_lims["Ees_lv"])
            or self.model.params["HR"] > max(self.params_lims["HR"])
            or self.model.params["Vs"] > max(self.params_lims["Vs"])
            or self.model.params["Vs"] < min(self.params_lims["Vs"])
        )
        return reward, done

    def set_domain_randomization(
        self,
        patient_state: Optional[str] = None,
        randomization: Optional[str] = None,
    ):
        """
        Set domain randomization characteristics.

        :param patient_state: <Optional[str]> Patient health state
                                    (normal, mild, moderate, severe).
        :param randomization: <Optional[str]> Randomization type. None or Uniform.
        """
        self.patient_state = patient_state
        self.randomization = randomization

    def reset(self) -> np.ndarray:
        """
        Reset environment and CV model.

        :return: Reset observations.
        """
        # Reset parameters
        self.model.params = self.model.init_params()
        if self.patient_state:
            self.model.set_patient_state(self.patient_state, self.randomization)

        # Reset model
        self.model.reset(initial_conditions=self.init_conditions, first_time=True)
        self.time_step = 0

        # Get your initial states
        states = self.model.states
        extra_obs = self.model.extra_obss
        self.observations = np.array(states)
        self.observations = np.append(self.observations, extra_obs)

        return self.observations

    def render(self, mode: str = "human"):
        """
        Activate render for visualizing PV loop.

        :param mode: <str> Visualize mode.
        """
        self.model.render = True

    def close(self):
        """
        Close visualization.
        """
        self.model.render = False
