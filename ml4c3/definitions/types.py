# Imports: standard library
from typing import Dict, List, Tuple, Union

# Imports: third party
import numpy as np

ArgumentList = List[Union[int, float]]
Arguments = Dict[str, Union[int, float, ArgumentList]]
Inputs = Dict[str, np.ndarray]
Outputs = Inputs
Path = str
Paths = List[Path]
Predictions = List[np.ndarray]
SampleIntervalData = Dict[
    int,
    Dict[Tuple[Tuple[str, str], Tuple[str, str]], Dict[str, Union[str, int, float]]],
]
ChannelMap = Dict[str, int]
