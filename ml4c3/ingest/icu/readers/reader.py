# Imports: standard library
import logging
from abc import ABC
from typing import Any

# Imports: third party
import numpy as np
import unidecode


class Reader(ABC):
    """
    Parent class for our Readers class.

    As an abstract class, it can't be directly instanced. Its children
    should be used instead.
    """

    @staticmethod
    def _ensure_contiguous(data: np.ndarray) -> np.ndarray:
        if len(data) > 0:
            dtype = Any
            try:
                data = data.astype(float)
                if all(x.is_integer() for x in data):
                    dtype = int
                else:
                    dtype = float
            except ValueError:
                dtype = "S"
            try:
                data = np.ascontiguousarray(data, dtype=dtype)
            except (UnicodeEncodeError, SystemError):
                logging.info("Uknown character. Not ensuring contiguous array")
                new_data = []
                for element in data:
                    new_data.append(unidecode.unidecode(str(element)))
                data = np.ascontiguousarray(new_data, dtype="S")
            except ValueError:
                logging.exception(
                    f"Unknown method to convert np.ndarray of "
                    f"{dtype} objects to numpy contiguous type.",
                )
                raise
        return data
