# Imports: standard library
from typing import Dict, List, Union

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.ingest.icu.data_objects import (
    Event,
    Procedure,
    Medication,
    StaticData,
    Measurement,
    BedmasterAlarm,
    BedmasterSignal,
)

ArgumentList = List[Union[int, float]]
Arguments = Dict[str, Union[int, float, ArgumentList]]
Inputs = Dict[str, np.ndarray]
Outputs = Inputs
Path = str
Paths = List[Path]
Predictions = List[np.ndarray]
ChannelMap = Dict[str, int]
BedmasterType = (BedmasterSignal, BedmasterAlarm)
EDWType = (StaticData, Measurement, Procedure, Event, Medication)
