# Imports: standard library
from typing import List, Tuple

# Imports: third party
import h5py
import numpy as np
import pytest

# Imports: first party
from ingest.icu.writers import Writer
from ingest.icu.data_objects import EDWType, BedmasterType, ICUDataObject

# pylint: disable=protected-access


def get_visit_id() -> str:
    return str(np.random.randint(1000000000, 9999999999))


def get_attributes(signal: ICUDataObject) -> Tuple[List[str], List[str]]:
    signal_keys = [attr for attr in dir(signal) if not attr.startswith("_")]
    array_keys = [
        attr
        for attr in signal_keys
        if isinstance(getattr(signal, attr), (np.ndarray, dict))
    ]
    attr_keys = list(set(signal_keys) - set(array_keys))
    return array_keys, attr_keys


def test_set_visit_id(temp_file, fake_signal):
    static_data = fake_signal.get_static_data()
    measurement = fake_signal.get_measurement()

    expected_error_message = "Visit ID not found. Please, check that you have set one."

    visit_id = get_visit_id()
    with Writer(temp_file.name, visit_id=visit_id) as writer:
        writer.write_static_data(static_data)
        writer.write_signal(measurement)

    with Writer(temp_file.name) as writer:
        with pytest.raises(Exception) as e_info:
            writer.write_static_data(static_data)
        assert str(e_info.value) == expected_error_message
        with pytest.raises(Exception) as e_info:
            writer.write_signal(measurement)
        assert str(e_info.value) == expected_error_message

        writer.set_visit_id(visit_id)
        writer.write_static_data(static_data)
        writer.write_signal(measurement)


def test_write_static_data(temp_file, fake_signal):
    static_data = fake_signal.get_static_data()
    visit_id = get_visit_id()

    with Writer(temp_file.name) as writer:
        writer.set_visit_id(visit_id)
        assert len(writer["edw"][visit_id].attrs) == 0
        writer.write_static_data(static_data)

    with h5py.File(temp_file.name, "r") as output_file:
        base_dir = output_file["edw"][visit_id]
        assert len(base_dir.attrs) == 20
        for key in base_dir.attrs.keys():
            if isinstance(base_dir.attrs[key], np.ndarray):
                assert np.array_equal(
                    base_dir.attrs[key],
                    getattr(static_data, key.lower()),
                )
            else:
                assert base_dir.attrs[key] == getattr(static_data, key.lower())


def test_write_data(temp_file, fake_signal):
    measurement = fake_signal.get_measurement()
    medication = fake_signal.get_medication()
    procedure = fake_signal.get_procedure()
    bedmaster_signal = fake_signal.get_bedmaster_signal()

    visit_id = get_visit_id()
    with Writer(temp_file.name) as writer:
        writer.set_visit_id(visit_id)
        writer.write_signal(measurement)

    with h5py.File(temp_file.name, "r") as output_file:
        base_dir = output_file["edw"][visit_id][measurement._source_type.lower()]
        assert list(base_dir.keys()) == [measurement.name.lower()]
        signal_dir = base_dir[measurement.name.lower()]

        arrays, attrs = get_attributes(measurement)
        assert_dir(signal_dir, measurement, arrays, attrs)

    signals = [medication, procedure, bedmaster_signal]
    with Writer(temp_file.name, visit_id=visit_id) as writer:
        for signal in signals:
            writer.write_signal(signal)

    with h5py.File(temp_file.name, "r") as output_file:
        for signal in signals:
            if isinstance(signal, BedmasterType):
                signal_source = "bedmaster"
            elif isinstance(signal, EDWType):
                signal_source = "edw"
            else:
                raise ValueError(f"Signal type {type(signal)} not Bedmaster or EDW")
            base_dir = output_file[signal_source][visit_id]
            signal_base_dir = base_dir[signal._source_type.lower()]
            signal_name = signal.name.replace("/", "|").lower()
            assert list(signal_base_dir.keys()) == [signal_name]

            arrays, attrs = get_attributes(signal)
            signal_dir = signal_base_dir[signal_name]
            assert_dir(signal_dir, signal, arrays, attrs)


def test_name_conflict(temp_file, fake_signal):

    m1_signal = fake_signal.get_measurement()
    m2_signal = fake_signal.get_measurement()

    m2_signal.source = "EDW_med"

    assert m1_signal.name == m2_signal.name
    assert m1_signal.source != m2_signal.source

    visit_id = get_visit_id()
    with Writer(temp_file.name) as writer:
        writer.set_visit_id(visit_id)
        writer.write_signal(m1_signal)
        writer.write_signal(m2_signal)

    with h5py.File(temp_file.name, "r") as output_file:
        m1_dir = output_file["edw"][visit_id][m1_signal._source_type.lower()]
        m2_dir = output_file["edw"][visit_id][m2_signal._source_type.lower()]

        def _test(m_dir, m_signal):
            assert list(m_dir.keys()) == [m_signal.name.lower()]
            signal_dir = m_dir[m_signal.name.lower()]
            arrays, attrs = get_attributes(m_signal)
            assert sorted(list(signal_dir.keys())) == sorted(arrays)
            for key in arrays:
                expected = getattr(m_signal, key.lower())
                if isinstance(expected, dict):
                    for k in expected:
                        assert k.lower() in signal_dir[key]
                else:
                    value = signal_dir[key][()]
                    assert np.array_equal(value, expected)
            for attr in attrs:
                value = signal_dir.attrs[attr]
                expected = getattr(m_signal, attr.lower())
                assert np.array_equal(value, expected)

        _test(m1_dir, m1_signal)
        _test(m2_dir, m2_signal)


def test_write_multiple_files(temp_file, fake_signal):
    visit_id = get_visit_id()

    wv_signal1 = fake_signal.get_bedmaster_signal()
    wv_signal2 = fake_signal.get_bedmaster_signal()
    wv_arrays, wv_attr = get_attributes(wv_signal1)

    with Writer(temp_file.name, visit_id=visit_id) as writer:
        writer.write_signal(wv_signal1)
        writer.write_signal(wv_signal2)

    with h5py.File(temp_file.name, "r") as output_file:
        wv_base_dir = output_file["bedmaster"][visit_id]["waveform"]
        wv_name = wv_signal1.name.replace("/", "|").lower()
        assert sorted(list(wv_base_dir.keys())) == [wv_name]

        wv_signal_dir = wv_base_dir[wv_name]

        assert sorted(list(wv_signal_dir.attrs.keys())) == sorted(wv_attr)

        for field in wv_attr:
            assert wv_signal_dir.attrs[field] == getattr(wv_signal1, field)

        assert sorted(list(wv_signal_dir.keys())) == sorted(wv_arrays)
        for field in wv_signal_dir:
            if field == "sample_freq":
                continue
            value = wv_signal_dir[field][()]
            expected = np.concatenate(
                (
                    getattr(wv_signal1, field.lower()),
                    getattr(wv_signal2, field.lower()),
                ),
            )
            assert len(value) == len(expected)
            assert np.array_equal(value, expected)

        corrected_sf2 = np.fromiter(
            [(sf, idx + wv_signal1.value.size) for sf, idx in wv_signal2.sample_freq],
            dtype="float,int",
        )
        expected_sf = np.concatenate([wv_signal2.sample_freq, corrected_sf2])
        assert np.array_equal(wv_signal_dir["sample_freq"][()], expected_sf)


def assert_dir(
    signal_dir: h5py.Group,
    signal: ICUDataObject,
    names: List[str],
    attrs: List[str],
):
    assert sorted(list(signal_dir.keys())) == sorted(names)
    for field in signal_dir:
        expected = getattr(signal, field.lower())
        if isinstance(expected, dict):
            for k in expected:
                assert k.lower() in signal_dir[field]
        else:
            value = signal_dir[field][()]
            assert np.array_equal(value, expected)

    assert sorted(list(signal_dir.attrs.keys())) == sorted(attrs)
    for field in attrs:
        assert signal_dir.attrs[field] == getattr(signal, field)
