# Imports: standard library
import math
import logging
from typing import Dict, List, Tuple, Union, Optional

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from rl4cs.utils.plotter import RLPlotter
from rl4cs.utils.element_saver import ElementSaver
from rl4cs.environments.burkhoff_model.action import SetAction

# pylint: disable=unbalanced-tuple-unpacking


class CVModel:
    """
    Implementation of Burkhoff hemodynamics model.
    """

    def __init__(
        self,
        init_conditions: List[float],
        paths: List[str],
        save_flag: bool = False,
        render: bool = False,
    ):
        """
        Init CV Model.

        :param init_conditions: <List> Initial conditions of the model.
        :param save_flag: <bool> Flag to determine whether saving states or not.
        """
        # Init
        self.init_conditions = init_conditions
        self.params = self.init_params()

        self.episode = 0

        self.path_act_space = paths[0]
        self.path_saver = paths[2]

        self.warm_up = 10  # [s]
        self.tstep = 0.001  # [s]
        self.tstep_rl = 10  # [min]
        self.time = 0
        self.time_rl = 0

        self.states: Tuple = ()  # v_lv, v_cas, v_cvs, v_rv, v_cap, v_cvp
        self.extra_obss: Tuple = ()  # p_lv, p_cas, p_cvs, p_rv, p_cap, p_cvp, hr*v_lv

        self.action_setter = SetAction(self.tstep_rl)

        # Saver
        self.save_flag = save_flag
        if self.save_flag:
            self.saver = ElementSaver(self.path_saver)

        # Render
        self.render = render
        if self.render:
            title = "PV Loop"
            axis_labels = ("LV Volume [mmL]", "LV Pressure [mmHg]")
            axis_lims = [40, 120, 0, 120]
            self.plot = RLPlotter(title, axis_labels, axis_lims, self.path_saver)

        # Setup previous simulation
        self.adapt_step_size()
        self.reset(initial_conditions=init_conditions, first_time=True)

    @staticmethod
    def init_params() -> Dict[str, float]:
        """
        Initialize parameters to healthy state.

        :return: <Dict[str, float]> Dictionary with all model parameters.
        """
        params: Dict[str, float] = {
            # Heart parameters
            "Ees_rv": 0.7,  # mmHg / mL
            "Ees_lv": 1.0,  # 3.0,  # mmHg / mL
            "Vo_rv": 0,  # mL
            "Vo_lv": 0,  # mL
            "Tes_rv": 0.175,  # s
            "Tes_lv": 0.175,  # s
            "Tes": 0.175,  # s
            "tau_rv": 0.025,  # s
            "tau_lv": 0.025,  # s
            "tau": 0.025,  # s
            "A_rv": 0.35,  # mmHg
            "A_lv": 0.35,  # mmHg
            "B_rv": 0.023,  # mL ^ -1
            "B_lv": 0.033,  # mL ^ -1
            # Circulation parameters
            "Ra_pul": round(40 / 1333.22, 4),  # dyn * s * cm ^ -5 --> mmHg * s / mL
            "Ra_sys": round(1200 / 1333.22, 4),  # dyn * s * cm ^ -5 --> mmHg * s / mL
            "Rc_pul": round(27 / 1333.22, 4),  # dyn * s * cm ^ -5 --> mmHg * s / mL
            "Rc_sys": round(40 / 1333.22, 4),  # dyn * s * cm ^ -5 --> mmHg * s / mL
            "Rv_pul": round(20 / 1333.22, 4),  # dyn * s * cm ^ -5 --> mmHg * s / mL
            "Rv_sys": round(20 / 1333.22, 4),  # dyn * s * cm ^ -5 --> mmHg * s / mL
            "Rt_pul": round(87 / 1333.22, 4),  # dyn * s * cm ^ -5 --> mmHg * s / mL
            "Rt_sys": round(1260 / 1333.22, 4),  # dyn * s * cm ^ -5 --> mmHg * s / mL
            "Ca_pul": 13,  # 13,  # mL / mmHg
            "Ca_sys": 1.32,  # 1.32,  # mL / mmHg
            "Cv_pul": 8,  # mL / mmHg
            "Cv_sys": 70,  # mL / mmHg
            # Common parameters
            "HR": 75,  # 75,  # bpm
            "Vt": 5500,  # mL
            "Vs": 750,  # mL
            "Vu": 4750,  # mL
        }
        return params

    def adapt_step_size(self):
        """
        Adapt step size depending on the heart rate with table.
        """
        if self.params["HR"] < 75:
            self.tstep = 0.008
            self.warm_up = 9
        elif 75 <= self.params["HR"] < 90:
            self.tstep = 0.006
            self.warm_up = 6
        elif 90 <= self.params["HR"] < 140:
            self.tstep = 0.005
            self.warm_up = 5
        elif 140 <= self.params["HR"] < 160:
            self.tstep = 0.004
            self.warm_up = 4
        elif 160 <= self.params["HR"] < 180:
            self.tstep = 0.003
            self.warm_up = 3
        elif 180 <= self.params["HR"] < 200:
            self.tstep = 0.001
            self.warm_up = 3
        elif 200 <= self.params["HR"] < 250:
            self.tstep = 0.0005
            self.warm_up = 3
        else:
            self.tstep = 0.0001
            self.warm_up = 3

    def circulatory_system(self):
        """
        Computes next step of hemodynamics simulation.
        """
        # Get previous state
        v_lv_k, v_cas_k, v_cvs_k, v_rv_k, v_cap_k, v_cvp_k = self.states

        p_cas_k = v_cas_k / self.params["Ca_sys"]
        p_cvs_k = v_cvs_k / self.params["Cv_sys"]
        p_cap_k = v_cap_k / self.params["Ca_pul"]
        p_cvp_k = v_cvp_k / self.params["Cv_pul"]

        # Get elastance
        elastance = self._control_elastance()

        # Left ventricle
        p_eslv = self.params["Ees_lv"] * (v_lv_k - self.params["Vo_lv"])

        p_edlv = self.params["A_lv"] * (
            math.exp(self.params["B_lv"] * (v_lv_k - self.params["Vo_lv"])) - 1
        )

        p_lv_k = (p_eslv - p_edlv) * elastance + p_edlv

        alpha_lv = self._lv_valves_alpha(p_lv_k, p_cvp_k)
        beta_lv = self._lv_valves_beta(p_lv_k, p_cas_k)

        q_lv_i = (p_cvp_k - p_lv_k) / self.params["Rv_pul"]
        q_lv_o = (p_lv_k - p_cas_k) / self.params["Rc_sys"]

        v_lv_k1 = v_lv_k + self.tstep * (q_lv_i * alpha_lv - q_lv_o * beta_lv)

        # Right ventricle
        p_esrv = self.params["Ees_rv"] * (v_rv_k - self.params["Vo_rv"])

        p_edrv = self.params["A_rv"] * (
            math.exp(self.params["B_rv"] * (v_rv_k - self.params["Vo_rv"])) - 1
        )

        p_rv_k = (p_esrv - p_edrv) * elastance + p_edrv

        alpha_rv = self._rv_valves_alpha(p_rv_k, p_cvs_k)
        beta_rv = self._rv_valves_beta(p_rv_k, p_cap_k)

        q_rv_i = (p_cvs_k - p_rv_k) / self.params["Rv_sys"]
        q_rv_o = (p_rv_k - p_cap_k) / self.params["Rc_pul"]

        v_rv_k1 = v_rv_k + self.tstep * (q_rv_i * alpha_rv - q_rv_o * beta_rv)

        # Arterial systemic circulation
        q_cas_i = (p_lv_k - p_cas_k) / self.params["Rc_sys"]
        q_cas_o = (p_cas_k - p_cvs_k) / self.params["Ra_sys"]

        v_cas_k1 = v_cas_k + self.tstep * (q_cas_i * beta_lv - q_cas_o)

        # Venous systemic circulation
        q_cvs_i = (p_cas_k - p_cvs_k) / self.params["Ra_sys"]
        q_cvs_o = (p_cvs_k - p_rv_k) / self.params["Rv_sys"]

        v_cvs_k1 = v_cvs_k + self.tstep * (q_cvs_i - q_cvs_o * alpha_rv)

        # Arterial pulmonary circulation
        q_cap_i = (p_rv_k - p_cap_k) / self.params["Rc_pul"]
        q_cap_o = (p_cap_k - p_cvp_k) / self.params["Ra_pul"]

        v_cap_k1 = v_cap_k + self.tstep * (q_cap_i * beta_rv - q_cap_o)

        # Venous pulmonary circulation
        q_cvp_i = (p_cap_k - p_cvp_k) / self.params["Ra_pul"]
        q_cvp_o = (p_cvp_k - p_lv_k) / self.params["Rv_pul"]

        v_cvp_k1 = v_cvp_k + self.tstep * (q_cvp_i - q_cvp_o * alpha_lv)

        # Update state and time
        self.time += self.tstep
        self.states = (v_lv_k1, v_cas_k1, v_cvs_k1, v_rv_k1, v_cap_k1, v_cvp_k1)
        self.extra_obss = (
            p_lv_k,
            p_cas_k,
            p_cvs_k,
            p_rv_k,
            p_cap_k,
            p_cvp_k,
            v_lv_k1 * self.params["HR"] / 1000,
        )

    def _control_elastance(self):
        period = 60 / self.params["HR"]
        cyclic_time = self.time % period

        if cyclic_time < 3 * self.params["Tes"] / 2:
            elastance = 0.5 * (
                math.sin(math.pi * cyclic_time / self.params["Tes"] - math.pi / 2) + 1
            )
        else:
            elastance = 0.5 * math.exp(
                -(cyclic_time - 3 * self.params["Tes"] / 2) / self.params["tau"],
            )
        return elastance

    @staticmethod
    def _lv_valves_alpha(p_lv, p_cvp):
        if p_lv < p_cvp:
            alpha_lv = 1
        else:
            alpha_lv = 0
        return alpha_lv

    @staticmethod
    def _lv_valves_beta(p_lv, p_cas):
        if p_lv > p_cas:
            beta_lv = 1
        else:
            beta_lv = 0
        return beta_lv

    @staticmethod
    def _rv_valves_alpha(p_rv, p_cvs):
        if p_rv < p_cvs:
            alpha_rv = 1
        else:
            alpha_rv = 0
        return alpha_rv

    @staticmethod
    def _rv_valves_beta(p_rv, p_cap):
        if p_rv > p_cap:
            beta_rv = 1
        else:
            beta_rv = 0
        return beta_rv

    def apply_action(self, action: int):
        """
        Apply action to the model. Actions are listed in the table:
        discrete_action_space.csv.

        :param action: <int> Action to be applied
        """
        if action > 26:
            raise ValueError("Action out of scope (larger than 26), not allowed.")

        act_df = pd.read_csv(self.path_act_space)  # Discrete action space
        actions_considered = ["Vs", "HR", "Ees_lv", "Ees_rv"]

        oneshot_act: Dict[str, float] = {}
        cont_act: Dict[str, List[float]] = {}

        for act in actions_considered:
            if not np.isnan(act_df[act][action]):
                if act == "Vs":
                    cont_act[act] = [act_df[act][action], act_df[act + "_time"][action]]
                else:
                    oneshot_act[act] = act_df[act][action]

        if cont_act:
            for act in cont_act:
                self.action_setter.set_action(act, cont_act[act][0], cont_act[act][1])
        self._change_parameters(oneshot_act, cont_act)

    def _change_parameters(self, var_params: Dict, cont_action: Optional[Dict]):
        if var_params:
            for parameter in var_params:
                self.params[parameter] *= var_params[parameter]

        if cont_action:
            list_finished: List = []
            for action in self.action_setter.action_count:
                effect, list_finished = self.action_setter.__next__(action)
                if action == "Vs":
                    self.params["Vs"] += effect
                else:
                    raise ValueError(f"Action {action} does not exist.")
            # Remove all actions from the object that have already finished in prev iter
            if list_finished:
                for action_finished in list_finished:
                    self.action_setter.remove_action(action_finished)

    def simulation(self, final_time: float, start_time: float = 0) -> Tuple:
        """
        Simulate hemodynamics model for a period of time.

        :param final_time: <float> Initial time of the simulation.
        :param start_time: <float> Final time of the simulation.
        :return: <Tuple> Tuple with lists of simulated time,
                         states and extra observations.
        """
        results_obs: Union[np.ndarray, List[List[float]]] = [
            [] for _ in range(len(self.states) + len(self.extra_obss))
        ]
        result_time: List[float] = []
        while self.time <= final_time:
            self.circulatory_system()
            if self.time >= start_time:
                result_time.append(self.time)
                for idx, state in enumerate(self.states):
                    results_obs[idx].append(state)
                for idx, extra_obs in enumerate(self.extra_obss):
                    results_obs[idx + len(self.states)].append(extra_obs)
        return result_time, results_obs

    def step_simulation(self, action: int):
        """
        Simulate hemodynamics model for a step forward.

        :param action: <int> Index with the action to be applied.
        :return: <Tuple> Return next time state and extra observations.
        """
        self.apply_action(action)
        self.circulatory_system()
        return self.time, self.states, self.extra_obss

    def step_training_rl(self, action: int):
        """
        Simulate hemodynamics model for a training RL.

        :param action: <int> Index with the action to be applied.
        :return: <Tuple> Return next time and desired observations.
        """

        self.apply_action(action)
        self.adapt_step_size()
        self.reset(initial_conditions=self.init_conditions, first_time=False)

        observations: Union[np.ndarray, List[List[float]]] = [
            [] for _ in range(len(self.states) + len(self.extra_obss))
        ]

        # Let simulator run for some seconds to stabilize to the new setpoint
        for _ in range(int(self.warm_up / self.tstep)):
            self.circulatory_system()

        # Compute observations within one heart beat
        for _ in range(int(round(60 / self.params["HR"] / self.tstep)) + 1):
            self.circulatory_system()
            for idx, state in enumerate(self.states):
                observations[idx].append(state)
            for idx, extra_obs in enumerate(self.extra_obss):
                observations[idx + len(self.states)].append(extra_obs)

        card_output = (
            (max(observations[0]) - min(observations[0])) * self.params["HR"] / 1000
        )
        observations_mean = np.mean(np.array(observations), axis=1)
        observations_mean[-1] = card_output

        # Plot PV loop if enabled every 100 minutes
        if self.render and self.time_rl % 100 == 0:
            data = np.array([observations[0], observations[6]]).transpose()
            text = (
                f"Cardiac Output [L/min]: {round(card_output, 2)} \n"
                f"Heart Rate [bpm]: {round(self.params['HR'], 2)} \n"
                f"Volume [mml]: {round(self.params['Vs'], 2)} \n"
                f"Elastances LV, RV [mmHg/mmL]: \n {round(self.params['Ees_lv'], 2)}, "
                f"{round(self.params['Ees_rv'], 2)}."
            )
            self.plot.update_plot(data, text)
            if self.save_flag:
                self.plot.save_figure(f"pv_loop_{self.episode}_{self.time_rl}.png")

        # Update time
        self.time_rl += self.tstep_rl

        return self.time_rl, observations_mean

    def reset(self, initial_conditions: List, first_time: bool):
        """
        Reset time, states and extra observations.

        :param initial_conditions: <List> Initial conditions of the simulation.
        :param first_time: <bool> Whether it is the first reset or not.
        """
        # Check initial conditions accomplish stressed blood volume
        total_volume = 0
        for condition in initial_conditions:
            total_volume += condition
        if total_volume != self.params["Vs"]:
            diff = self.params["Vs"] - total_volume
            for idx, condition in enumerate(initial_conditions):
                initial_conditions[idx] = condition + diff / len(initial_conditions)

        # Reset times and get initial states
        self.time = 0
        self.states = tuple(initial_conditions)
        self.circulatory_system()
        self.time = 0
        self.states = tuple(initial_conditions)

        # Reset queues in continuous actions
        self.action_setter.reset_queues()

        # If reset is a start of a new episode
        if first_time:
            # Set RL time to 0
            self.time_rl = 0
            # Log info about initial conditions and parameters
            if self.episode % 10 == 0:
                logging.info(
                    f"New episode starts with the following states: {self.states} "
                    f"(Total volume of: {sum(initial_conditions)} [mmL]); "
                    f"and parameters: {self.params}.",
                )

    def set_patient_state(self, illness_state: str, randomization: Optional[str]):
        """
        Set parameter values and initial conditions for different illness
        states.

        :param illness_state: <str> Initial state of the patient. Options are:
                                    normal, mild, moderate, severe.
        :param randomization: <str> Randomization method. It can be: None or uniform.
        """

        def _set_random_val(val_state, val_orig):
            if not randomization:
                return val_state
            if randomization == "uniform":
                min_val = min(val_state, val_orig)
                max_val = max(val_state, val_orig)
                return round(np.random.uniform(min_val, max_val), 4)
            if randomization == "triangular":
                min_val = min(val_state, val_orig)
                max_val = max(val_state, val_orig)
                mid_val = (min_val + max_val) / 2
                return round(np.random.triangular(min_val, mid_val, max_val), 4)

            raise ValueError(
                "This randomization option does not exist. "
                "Existing options are: None, uniform or triangular",
            )

        in_par = self.init_params()

        if illness_state == "normal":
            self.params["HR"] = _set_random_val(70, in_par["HR"])  # bpm
            self.params["Vs"] = _set_random_val(650, in_par["Vs"])  # ml
            self.params["Ees_rv"] = _set_random_val(0.61, in_par["Ees_rv"])  # mmHg / ml
            self.params["A_rv"] = _set_random_val(0.35, in_par["A_rv"])  # mmHg
            self.params["tau"] = _set_random_val(0.03, in_par["tau"])  # s
            self.params["Ees_lv"] = _set_random_val(4.5, in_par["Ees_lv"])  # mmHg / ml
            self.params["A_lv"] = _set_random_val(1.3, in_par["A_lv"])  # mmHg
            self.params["Rc_sys"] = _set_random_val(
                0.03,
                in_par["Rc_sys"],
            )  # mmHg * s / ml
            self.params["Ra_sys"] = _set_random_val(
                0.8,
                in_par["Ra_sys"],
            )  # mmHg * s / ml
            self.params["Rc_pul"] = _set_random_val(
                0.03,
                in_par["Rc_pul"],
            )  # mmHg * s / ml
            self.params["Ra_pul"] = _set_random_val(
                0.03,
                in_par["Ra_pul"],
            )  # mmHg * s / ml
            self.params["Ca_sys"] = _set_random_val(1.5, in_par["Ca_sys"])  # ml / s

        elif illness_state == "mild":
            self.params["HR"] = _set_random_val(80, in_par["HR"])  # bpm
            self.params["Vs"] = _set_random_val(650, in_par["Vs"])  # ml
            self.params["Ees_rv"] = _set_random_val(0.50, in_par["Ees_rv"])  # mmHg / ml
            self.params["A_rv"] = _set_random_val(0.25, in_par["A_rv"])  # mmHg
            self.params["tau"] = _set_random_val(0.05, in_par["tau"])  # s
            self.params["Ees_lv"] = _set_random_val(1.25, in_par["Ees_lv"])  # mmHg / ml
            self.params["A_lv"] = _set_random_val(0.53, in_par["A_lv"])  # mmHg
            self.params["Rc_sys"] = _set_random_val(
                0.04,
                in_par["Rc_sys"],
            )  # mmHg * s / ml
            self.params["Ra_sys"] = _set_random_val(
                1.1,
                in_par["Ra_sys"],
            )  # mmHg * s / ml
            self.params["Rc_pul"] = _set_random_val(
                0.02,
                in_par["Rc_pul"],
            )  # mmHg * s / ml
            self.params["Ra_pul"] = _set_random_val(
                0.02,
                in_par["Ra_pul"],
            )  # mmHg * s / ml
            self.params["Ca_sys"] = _set_random_val(1.0, in_par["Ca_sys"])  # ml / s

        elif illness_state == "moderate":
            self.params["HR"] = _set_random_val(80, in_par["HR"])  # bpm
            self.params["Vs"] = _set_random_val(1750, in_par["Vs"])  # ml
            self.params["Ees_rv"] = _set_random_val(0.42, in_par["Ees_rv"])  # mmHg / ml
            self.params["A_rv"] = _set_random_val(0.15, in_par["A_rv"])  # mmHg
            self.params["tau"] = _set_random_val(0.07, in_par["tau"])  # s
            self.params["Ees_lv"] = _set_random_val(0.75, in_par["Ees_lv"])  # mmHg / ml
            self.params["A_lv"] = _set_random_val(0.16, in_par["A_lv"])  # mmHg
            self.params["Rc_sys"] = _set_random_val(
                0.04,
                in_par["Rc_sys"],
            )  # mmHg * s / ml
            self.params["Ra_sys"] = _set_random_val(
                1.2,
                in_par["Ra_sys"],
            )  # mmHg * s / ml
            self.params["Rc_pul"] = _set_random_val(
                0.04,
                in_par["Rc_pul"],
            )  # mmHg * s / ml
            self.params["Ra_pul"] = _set_random_val(
                0.04,
                in_par["Ra_pul"],
            )  # mmHg * s / ml
            self.params["Ca_sys"] = _set_random_val(1.0, in_par["Ca_sys"])  # ml / s

        elif illness_state == "severe":
            self.params["HR"] = _set_random_val(80, in_par["HR"])  # bpm
            self.params["Vs"] = _set_random_val(1750, in_par["Vs"])  # ml
            self.params["Ees_rv"] = _set_random_val(0.40, in_par["Ees_rv"])  # mmHg / ml
            self.params["A_rv"] = _set_random_val(0.10, in_par["A_rv"])  # mmHg
            self.params["tau"] = _set_random_val(0.09, in_par["tau"])  # s
            self.params["Ees_lv"] = _set_random_val(0.57, in_par["Ees_lv"])  # mmHg / ml
            self.params["A_lv"] = _set_random_val(0.05, in_par["A_lv"])  # mmHg
            self.params["Rc_sys"] = _set_random_val(
                0.04,
                in_par["Rc_sys"],
            )  # mmHg * s / ml
            self.params["Ra_sys"] = _set_random_val(
                1.4,
                in_par["Ra_sys"],
            )  # mmHg * s / ml
            self.params["Rc_pul"] = _set_random_val(
                0.02,
                in_par["Rc_pul"],
            )  # mmHg * s / ml
            self.params["Ra_pul"] = _set_random_val(
                0.04,
                in_par["Ra_pul"],
            )  # mmHg * s / ml
            self.params["Ca_sys"] = _set_random_val(1.0, in_par["Ca_sys"])  # ml / s

        else:
            raise ValueError(
                "This illness state does not exist. Existing illness states are:"
                "normal, mild, moderate and severe.",
            )
