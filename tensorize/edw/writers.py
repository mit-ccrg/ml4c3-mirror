# Imports: standard library
from typing import Union

# Imports: third party
import h5py
import numpy as np

# Imports: first party
from tensorize.edw.data_objects import (
    EDWType,
    StaticData,
    ICUDiscreteData,
    ICUContinuousData,
)

# pylint: disable=too-many-branches


class Writer(h5py.File):
    """
    Class used to write the signals into the final HD5 file.

    This class is a Wrapper around h5py.File. Thanks to that,
    it can be used with python's 'with' as in the example.

    >>> with Writer('output_file.hd5') as writer:
        do something

    The structure of the output file will be:

    <edw>
        <visitID>/
            <signal name>/
                data and metadata
            ...
        ...
    ...
    """

    edw_dir = "edw"

    def __init__(
        self,
        output_file: str,
        visit_id: str = None,
    ):
        """
        Inherits from h5py.File to profit its features.

        :param output_file: <str> name of the output file.
        :param visit_id: <str> the visit ID of te data that will be logged.
                        If None, it has to be set using set_visit_id(visit_id)
                        before writing on the HD5
        """
        super().__init__(output_file, "w")
        self.create_group(self.edw_dir)
        self.visit_id = visit_id
        if visit_id:
            self.set_visit_id(visit_id)

    def set_visit_id(self, visit_id: str):
        """
        Set a visit ID.

        All the writes will be performed under this visit ID
        on the final HD5 until the visit ID is changed.

        :param visit_id: <str> the visit ID.
        """
        self.visit_id = visit_id
        self[self.edw_dir].create_group(visit_id)

    def write_completed_flag(self, flag: bool):
        """
        Write a flag indicating if all the data from the specified source had
        been tensorized.

        :param flag: <bool> Bool indicating if the tensorization for the specified
                     data source is finished.
        """
        self[self.edw_dir].attrs["completed"] = flag

    def write_static_data(self, static_data: StaticData):
        """
        Write the static information on the correct spot on the HD5.

        :param static_data: <StaticData> the static data to be written.
        """
        # Check that the visitID has been set
        if not self.visit_id:
            raise ValueError("Visit ID not found. Please, check that you have set one.")

        static_group = self[self.edw_dir][self.visit_id]

        for field in dir(static_data):
            if not field.startswith("_"):
                static_group.attrs[field] = getattr(static_data, field)

    def write_signal(self, signal: Union[ICUContinuousData, ICUDiscreteData]):
        """
        Writer for generic fields that all the data share.
        """
        # Check that the visitID has been set
        if not self.visit_id:
            raise ValueError("Visit ID not found. Please, check that you have set one.")

        signal_name = signal.name.lower()
        signal_name = signal_name.replace(" ", "_")
        signal_name = signal_name.replace(",", "")
        signal_name = signal_name.replace("/", "|")
        signal_name = signal_name.replace("(", "").replace(")", "")

        if isinstance(signal, EDWType):
            source = self.edw_dir
        else:
            raise ValueError(f"Source {signal} not recognized")

        base_dir = self[source][self.visit_id]
        source_type = signal._source_type.lower()  # pylint: disable=protected-access
        if source_type not in base_dir.keys():
            base_dir.create_group(source_type)
        base_dir = base_dir[source_type]

        # Add new signal
        if signal_name not in base_dir.keys():
            signal_dir = base_dir.create_group(signal_name)
            metadata = {}
            for field, value in signal.__dict__.items():
                if isinstance(value, dict):
                    metadata.update(value)
                else:
                    self.write_new_data(signal_dir, field, value)
            if metadata:
                meta_dir = signal_dir.create_group("metadata")
                for field, value in metadata.items():
                    self.write_new_data(meta_dir, field, value)
        else:  # Concatenate data into existing signal (just for Bedmaster)
            raise ValueError(f"EDW signal {signal} already exists.")

    @staticmethod
    def write_new_data(signal_dir, field, data):
        if isinstance(data, np.ndarray):
            signal_dir.create_dataset(
                name=field.lower(),
                data=data,
                maxshape=(None,),
                compression=32015,
            )
        else:
            signal_dir.attrs[field] = data
