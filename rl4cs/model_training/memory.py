# Imports: standard library
import random
from typing import List, Tuple


class Memory:
    """
    Memory class for storing samples to train the DQN RL algorithm.
    """

    def __init__(self, max_memory: int):
        """
        Init Memory class.

        :param max_memory: <int> Maximum length of memory buffer.
        """
        self.max_memory = max_memory
        self.samples: List[Tuple] = []

    def add_sample(self, sample: Tuple):
        """
        Add sample to memory.

        :param sample: <Tuple> Sample to be added to memory.
        """
        self.samples.append(sample)
        if len(self.samples) > self.max_memory:
            self.samples.pop(0)

    def take_samples(self, num_samples: int) -> List:
        """
        Return desired number of samples.

        :param num_samples: <int> Number of samples to be returned.
        :return: <List> List of samples.
        """
        if num_samples > len(self.samples):
            return random.sample(self.samples, len(self.samples))
        return random.sample(self.samples, num_samples)
