# Imports: standard library
import re
import math
from typing import Any, Dict, List


class SetAction:
    """
    Configuration of an action.
    """

    def __init__(self, sim_step: int):
        """
        Init action configurator.
        """
        self.action_list: Dict[str, Dict[str, Any]] = {}
        self.action_count: Dict[str, List] = {}
        self.sim_step = sim_step  # [min]

    def set_action(
        self,
        action_type: str,
        dose: float,
        duration: float,
        cumulative_flag: bool = False,
    ):
        """
        Save new action with its configuration parameters.

        :param action_type: <str> Type of medication given.
        :param dose: <int> Quantity of medication given.
        :param duration: <int> Duration of medication effect.
        :param cumulative_flag: <bool> If flag set, action is accumulated
        """
        if action_type in self.action_count and cumulative_flag:
            self.action_count[action_type].append(
                self.action_count[action_type][-1] + 1,
            )
        else:
            self.action_count[action_type] = [1]

        self.action_list[
            action_type + "_" + str(self.action_count[action_type][-1])
        ] = {
            "Dose": dose,
            "Duration": duration,
            "Time": self.sim_step,
        }

    def reset_queues(self):
        """
        Reset all queues of actions.
        """
        self.action_list = {}
        self.action_count = {}

    def remove_action(self, action_name: str):
        """
        Remove action from the list.

        :param action_name: <str> Type of medication given.
        """
        del self.action_list[action_name]
        action_type, number = re.split("_", action_name)

        if not self.action_count[action_type]:
            raise ValueError(f"Error with {action_type} already non existing.")

        self.action_count[action_type].remove(int(number))
        if not self.action_count[action_type]:
            del self.action_count[action_type]

    def __iter__(self):
        return self

    def __next__(self, action_type: str):
        """
        Set next step, iterate over desired active actions.

        :param action_type: <str> Type of medication given.
        :return:
        """
        effect = 0.0
        list_finished = []
        for idx in self.action_count[action_type]:
            action = action_type + "_" + str(idx)
            dose = self.action_list[action]["Dose"]
            duration = self.action_list[action]["Duration"]
            time = self.action_list[action]["Time"]

            if time > duration:
                raise StopIteration("Iteration limit reached.")

            effect += self._temporal_evolution("linear", dose, time, duration)
            self.action_list[action]["Time"] += self.sim_step

            if time >= duration:
                list_finished.append(action)

        return effect, list_finished

    @staticmethod
    def _hill_equation(action_type: str, dose: float) -> float:
        """
        Determine effect of drug depending on the dose by Hill equation.

        :param action_type: <str> Type of action. Now only Vs.
        :param dose: <float> Quantity of drug administered.
        :return: <float> Quantity of drug effect.
        """
        if action_type:
            n_pow = 5
            k_a = 2
        else:
            n_pow = 5
            k_a = 2
        output = 1 / (1 + (k_a / (2 * dose)) ** n_pow)
        return output

    @staticmethod
    def _temporal_evolution(
        ev_type: str,
        dose: float,
        time: float,
        duration: float,
    ) -> float:
        """
        Evolution of the drug effect over time.

        :param ev_type: <str> Evolution type: linear or exponential.
        :param dose: <float> Quantity of the drug effect.
        :param time: <float> Time.
        :param duration: <float> Duration of the drug effect.
        :return: <float> Effect of the drug over time.
        """
        if ev_type == "linear":
            effect = (dose / duration) * time
        elif ev_type == "exponential":
            tau = duration / 4
            effect = dose * (1 - math.exp(-time / tau))
        else:
            raise ValueError(f"{ev_type} is not a type of temporal evolution.")
        return effect
