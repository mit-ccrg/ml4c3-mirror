# Imports: standard library
from enum import Enum, auto


class BottleneckType(Enum):
    # All decoder outputs are flattened to put into embedding
    FlattenRestructure = auto()

    # Structured (not flat) decoder outputs are global average pooled
    GlobalAveragePoolStructured = auto()

    # All decoder outputs are flattened then variationally sampled to put into embedding
    Variational = auto()
