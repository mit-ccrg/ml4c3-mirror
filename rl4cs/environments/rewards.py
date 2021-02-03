# Imports: standard library
import json

# Imports: third party
import numpy as np


class Reward:
    """
    Reward functions.
    """

    def __init__(
        self,
        params_reward: str,
        type_rew: str = "discrete",
        cardiac_output: bool = True,
        drug_penalty: bool = False,
        power: bool = False,
    ):
        """
        Init reward class.

        :param params_reward: <str> Path to table with parameters to fit reward.
        :param type_rew: <str> Type of reward: 'continuous' or 'discrete'.
        :param cardiac_output: <bool> If param set, reward considers cardiac output.
        :param drug_penalty: <bool> If param set, reward considers drug penalty.
        :param power: <bool> If param set, reward considers power.
        """
        # Parameters
        with open(params_reward) as params:
            self.params_reward = json.load(params)

        # Type of reward (continuous or discrete)
        if type_rew not in ("continuous", "discrete"):
            raise ValueError(
                f"type_rew is {type_rew}, however it must be either "
                f"'continuous' or 'discrete'.",
            )
        self.type_rew = type_rew

        # Reward flags
        self.cardiac_output = cardiac_output
        self.drug_penalty = drug_penalty
        self.power = power

        # States and action
        self.states = np.empty(0)
        self.action = 0

    def compute_reward(self, states: np.ndarray, action: int) -> float:
        """
        Compute the reward as the addition of the terms considered: cardiac output,
        drug penalty and power penalty.

        :param states: <np.ndarray> States of the current step.
        :param action: <int> Action taken in the current step.
        :return: <float> Computed reward as addition of the different terms.
        """
        self.states = states
        self.action = action

        # Initialize reward and compute it as desired
        reward = 0.0
        if self.cardiac_output:
            reward += self._rew_cardiac_output()
        if self.drug_penalty:
            reward += self._rew_drug_penalty()
        if self.power:
            reward += self._rew_power_penalty()

        return reward

    def _rew_cardiac_output(self) -> float:
        """
        Compute reward related to cardiac output.
        :return: <float> reward obtained.
        """
        reward = 0.0
        # Get the cardiac output from states
        card_out = self.states[-1]

        # Get params
        params = self.params_reward["caridac_output"]

        # Compute reward continuous or discrete
        if self.type_rew == "continuous":
            l_param = params["continuous"]["l"]
            k_param = params["continuous"]["k"]
            off_param = params["continuous"]["off"]
            # Sigmoid continuous parametrized function
            reward = l_param / (1 + np.exp(-k_param * (card_out - off_param)))

        if self.type_rew == "discrete":
            if card_out <= params["discrete"]["low_threshold"]:
                reward = params["discrete"]["low_rew"]
            elif (
                params["discrete"]["low_threshold"]
                < card_out
                < params["discrete"]["high_threshold"]
            ):
                reward = params["discrete"]["intermed_rew"]
            else:
                reward = params["discrete"]["high_rew"]
        return reward

    def _rew_drug_penalty(self) -> float:
        """
        Compute reward related to drug penalty.
        :return: <float> reward obtained.
        """

        # Get params
        params = self.params_reward["drug_penalty"]

        # Compute reward, only discrete
        reward = params["low_rew"]
        if 0 < self.action <= 3 or self.action == 6:
            reward = params["intermed_rew"]
        if 3 < self.action < 6 or self.action > 6:
            reward = params["high_rew"]
        return reward

    def _rew_power_penalty(self) -> float:
        """
        Compute reward related to power penalty.
        :return: <float> reward obtained.
        """
        reward = 0.0
        # Get cardiac output and mean arterial pressure from states
        card_out = self.states[-1]
        martp = self.states[7]
        # Compute power
        power = martp * card_out * 0.0022

        # Get params
        params = self.params_reward["power_penalty"]

        # Compute reward continuous or discrete
        if self.type_rew == "continuous":
            l_param = params["continuous"]["l"]
            m_param = params["continuous"]["m"]
            off_param = params["continuous"]["off"]
            # ReLu continuous parametrized function
            reward = m_param * (
                -1 / l_param * np.log(1 + np.exp(l_param * (power - off_param)))
            )

        if self.type_rew == "discrete":
            if power < params["discrete"]["low_threshold"]:
                reward = params["discrete"]["low_rew"]
            elif (
                params["discrete"]["low_threshold"]
                <= power
                < params["discrete"]["high_threshold"]
            ):
                reward = params["discrete"]["intermed_rew"]
            else:
                reward = params["discrete"]["high_rew"]
        return reward
