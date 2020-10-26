# Imports: standard library
from typing import Dict, List, Union

# Imports: third party
import numpy as np

ArgumentList = List[Union[int, float]]
Arguments = Dict[str, Union[int, float, ArgumentList]]
Inputs = Dict[str, np.ndarray]
Outputs = Inputs
Path = str
Paths = List[Path]
Predictions = List[np.ndarray]
ChannelMap = Dict[str, int]
