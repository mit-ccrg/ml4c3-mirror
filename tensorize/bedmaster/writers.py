# Imports: standard library
import logging
from typing import Union

# Imports: third party
import h5py
import numpy as np

# Imports: first party
from tensorize.bedmaster.data_objects import (
    BedmasterType,
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

    <bedmaster>
        <visitID>/
            <signal name>/
                data and metadata
            ...
        ...
    <edw>
        <visitID>/
            <signal name>/
                data and metadata
            ...
        ...
    ...
    """

    bedmaster_dir = "bedmaster"

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
        self.create_group(self.bedmaster_dir)
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
        self[self.bedmaster_dir].create_group(visit_id)

    def write_completed_flag(self, flag: bool):
        """
        Write a flag indicating if all the data from the specified source had
        been tensorized.

        :param flag: <bool> Bool indicating if the tensorization for the specified
                     data source is finished.
        """
        self[self.bedmaster_dir].attrs["completed"] = flag

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

        if isinstance(signal, BedmasterType):
            if signal.value.size == 0 or signal.time.size == 0:
                logging.info(
                    f"Signal {signal.name} not written: time or value is empty",
                )
                return
            source = self.bedmaster_dir
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
            signal_dir = base_dir[signal_name]
            current_length = signal_dir["value"].size
            for field, value in signal.__dict__.items():
                if isinstance(value, np.ndarray):
                    if field == "sample_freq":
                        value = self._aggregate_sample_freq(
                            signal_dir,
                            value,
                            current_length,
                        )
                    self.concatenate_data(signal_dir, field, value)

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

    @classmethod
    def concatenate_data(cls, signal_dir, field, data):
        if data is None:
            return
        field = field.lower()
        new_size = signal_dir[field].shape[0] + data.shape[0]
        signal_dir[field].resize(new_size, axis=0)
        signal_dir[field][-len(data) :] = data.astype(signal_dir[field].dtype)

    @staticmethod
    def _aggregate_sample_freq(signal_dir, sample_freq, idx_shift):
        if signal_dir["sample_freq"][-1][0] == sample_freq[0][0]:
            sample_freq = np.delete(sample_freq, 0)
        if sample_freq.size == 0:
            return None
        sample_freq = np.fromiter(
            map(lambda sf_t: (sf_t[0], sf_t[1] + idx_shift), sample_freq),
            dtype="int,int",
        )
        return sample_freq
