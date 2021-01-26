# Imports: standard library
import os
import pickle
from typing import Dict, List, Tuple, Union

# Imports: third party
import numpy as np
import tensorflow as tf


class ElementSaver:
    """
    Save and load different elements of RL training.
    """

    def __init__(self, path_data: str = "./saved_elements"):
        """
        Init saver class.

        :param path_data: <str> Path to directory where data is saved
        """

        def _generate_dir(folder):
            if not os.path.isdir(os.path.join(path_data, folder)):
                os.makedirs(os.path.join(path_data, folder))
            return os.path.join(path_data, folder)

        self.states_path = _generate_dir("states")
        self.time_path = _generate_dir("time")
        self.params_path = _generate_dir("params")
        self.policy_path = _generate_dir("policy")

        self.buff_states: List[Union[Tuple, np.ndarray]] = []
        self.buff_time: List[float] = []
        self.buff_params: Dict[str, List] = {}

        self.create_folders()

    def create_folders(self):
        """
        Create folders to store data.
        """
        paths = [self.states_path, self.time_path, self.params_path, self.policy_path]
        for path in paths:
            if not os.path.isdir(os.path.join(path)):
                os.makedirs(path)

    def clear_buffs(self, buff_name: str):
        if buff_name == "states":
            self.buff_states = []
        elif buff_name == "time":
            self.buff_time = []
        elif buff_name == "params":
            self.buff_params = {}
        else:
            raise ValueError(
                "This buffer does not exist. Buffers are: states, time and params.",
            )

    def store_dict(self, dict_params: Dict):
        """
        Store parameters in a dictionary of lists.

        :param dict_params: <Dict[str, List]> Dictionary with list of parameters.
        """
        if not self.buff_params:
            for element in dict_params:
                self.buff_params[element] = [dict_params[element]]
        else:
            for element in dict_params:
                self.buff_params[element].append(dict_params[element])

    def save_states(self, name: str):
        """
        Save time and state arrays with numpy.

        :param name: <str> Name of the file to store (.npy).
        """
        if not name.endswith(".npy"):
            name += ".npy"

        with open(os.path.join(self.states_path, name), "wb") as file:
            np.save(file, np.array(self.buff_states))

        with open(os.path.join(self.time_path, name), "wb") as file:
            np.save(file, np.array(self.buff_time))

    def load_states(self, name: str):
        """
        Load time and state arrays saved with numpy.

        :param name: name of the file to load (.npy).
        :return: <Tuple[np.ndarray]> Tuple with arrays of states and time.
        """
        if not name.endswith(".npy"):
            name += ".npy"

        with open(os.path.join(self.states_path, name), "rb") as file:
            states = np.load(file, allow_pickle=True)

        with open(os.path.join(self.time_path, name), "rb") as file:
            time = np.load(file, allow_pickle=True)
        return time, states

    def remove_states(self, name: str):
        """
        Remove file with stored states.

        :param name: Name of the file to remove (.npy).
        """
        if not name.endswith(".npy"):
            name += ".npy"

        if not os.path.isfile(os.path.join(self.states_path, name)):
            raise OSError(f"File {name} doesn't exist, thus can't be removed.")

        os.remove(os.path.join(self.states_path, name))
        os.remove(os.path.join(self.time_path, name))

    def save_params(self, name: str):
        """
        Save dictionary with pickle.

        :param name: <str> Name of the file to store (.pkl).
        """
        if not name.endswith(".pkl"):
            name += ".pkl"

        with open(os.path.join(self.params_path, name), "wb") as file:
            pickle.dump(self.buff_params, file, pickle.HIGHEST_PROTOCOL)

    def load_params(self, name: str) -> Dict:
        """
        Load dictionary saved with pickle.

        :param name: <str> Name of the file to load (.pkl).
        :return: <Dict> Dictionary with parameters.
        """
        if not name.endswith(".pkl"):
            name += ".pkl"

        with open(os.path.join(self.params_path, name), "rb") as file:
            return pickle.load(file)

    def remove_params(self, name: str):
        """
        Remove file with stored parameters.

        :param name: <str> Name of the file to remove (.pkl).
        """
        if not name.endswith(".pkl"):
            name += ".pkl"

        if not os.path.isfile(os.path.join(self.params_path, name)):
            raise OSError(f"File {name} doesn't exist, thus can't be removed.")

        os.remove(os.path.join(self.params_path, name))

    def save_qfunc(self, name: str, model):
        """
        Save neural network that represents the Q function.

        :param name: <str> Name of file to store the NN Q function (.h5).
        :param model: <tf.python.keras.engine.training.Model> NN Q model.
        """
        if not name.endswith(".h5"):
            name += ".h5"

        model.save(os.path.join(self.policy_path, name))

    def load_qfunc(self, name: str):
        """
        Load neural network representing a previous trained Q function.

        :param name: <str> name of the folder to store the information.
        :return: <tf.python.keras.engine.training.Model> Loaded NN Q model.
        """
        if not name.endswith(".h5"):
            name += ".h5"

        return tf.keras.models.load_model(os.path.join(self.policy_path, name))

    def remove_qfunc(self, name: str):
        """
        Remove folder with stored policy.

        :param name: Name of the folder to remove.
        """
        if not name.endswith(".h5"):
            name += ".h5"

        if not os.path.isdir(os.path.join(self.policy_path, name)):
            raise OSError(f"File {name} doesn't exist, thus can't be removed.")

        os.remove(os.path.join(self.policy_path, name))
