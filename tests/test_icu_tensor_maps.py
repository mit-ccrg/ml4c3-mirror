# type: ignore
# pylint: disable=import-error
# Imports: standard library
from typing import Dict, List, Tuple, Union, Callable

# Imports: third party
import h5py
import numpy as np
import pandas as pd

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from tensormap.TensorMap import TensorMap
from tensormap.icu_signals import get_tmap as get_signal_tmap
from ingest.icu.data_objects import ICUDataObject
from tensormap.icu_around_event import get_tmap as get_around_tmap

TEST_DATA = Tuple[
    h5py.File,
    Dict[str, List[Union[str, ICUDataObject, Dict[str, ICUDataObject]]]],
]
TEST_TENSOR_MAPS = Dict[str, Dict[str, TensorMap]]


def test_alarms_tmaps(hd5_data: TEST_DATA, testing_tmaps: TEST_TENSOR_MAPS):
    hd5, data = hd5_data
    data = data["alarms"]
    testing_tmaps = testing_tmaps["alarms"]

    for _name, tm in testing_tmaps.items():
        if tm.name.replace("_init_date", "") in data[0].keys():
            original = np.array(
                [d[tm.name.replace("_init_date", "")].start_date for d in data],
            )
            tensor = tm.tensor_from_file(tm, hd5)
            assert np.array_equal(original, tensor)

        elif tm.name.replace("_duration", "") in data[0].keys():
            original = np.array(
                [d[tm.name.replace("_duration", "")].duration for d in data],
            )
            tensor = tm.tensor_from_file(tm, hd5)
            assert np.array_equal(original, tensor)

        elif tm.name.replace("_level", "") in data[0].keys():
            original = np.array([d[tm.name.replace("_level", "")].level for d in data])
            tensor = tm.tensor_from_file(tm, hd5)
            assert np.array_equal(original, tensor)
        else:
            raise AssertionError(f"TMap {tm.name} couldn't be tested.")


def test_events_tmaps(hd5_data: TEST_DATA, testing_tmaps: TEST_TENSOR_MAPS):
    hd5, data = hd5_data

    def _test_event_tmaps(data, testing_tmaps):
        for _name, tm in testing_tmaps.items():
            if tm.name.replace("_start_date", "") in data[0].keys():
                original = np.array(
                    [d[tm.name.replace("_start_date", "")].start_date for d in data],
                )
                tensor = tm.tensor_from_file(tm, hd5)
                assert np.array_equal(original, tensor)
            elif tm.name.replace("_end_date", "") in data[0].keys():
                original = np.array(
                    [d[tm.name.replace("_end_date", "")].end_date for d in data],
                )
                tensor = tm.tensor_from_file(tm, hd5)
                assert np.array_equal(original, tensor)
            elif tm.name.endswith("_double"):
                original = None
                for visit in data:
                    if tm.name.replace("_double", "") in visit.keys():
                        if len(visit[tm.name.replace("_double", "")].start_date) > 0:
                            original = np.array([[0, 1], [1, 0]])
                            break
                if original is not None:
                    tensor = tm.tensor_from_file(tm, hd5)
                    assert np.array_equal(original, tensor)
            elif tm.name.endswith("_single"):
                original = np.array([[1, 0]])
                for visit in data:
                    if tm.name.replace("_single", "") in visit.keys():
                        if len(visit[tm.name.replace("_single", "")].start_date) > 0:
                            original = np.array([[0, 1]])
                            break
                tensor = tm.tensor_from_file(tm, hd5)
                assert np.array_equal(original, tensor)
            else:
                raise AssertionError(f"TMap {tm.name} couldn't be tested.")

    _test_event_tmaps(data["events"], testing_tmaps["events"])
    _test_event_tmaps(data["procedures"], testing_tmaps["procedures"])


def test_measurements_tmaps(hd5_data: TEST_DATA, testing_tmaps: TEST_TENSOR_MAPS):
    hd5, data = hd5_data
    data = data["measurements"]
    testing_tmaps = testing_tmaps["measurements"]

    for _name, tm in testing_tmaps.items():
        tm_name = tm.name
        if "blood_pressure" in tm.name:
            tm_name = tm_name.replace("_systolic", "").replace("_diastolic", "")
        if tm_name.replace("_timeseries", "") in data[0].keys():
            name = tm_name.replace("_timeseries", "")
            tensor = tm.tensor_from_file(tm, hd5)
            original = []
            for signal in data:
                if "_systolic" in tm.name:
                    old_values = signal[name].value.astype(str)
                    values = np.zeros(len(old_values))
                    for index, value in enumerate(old_values):
                        values[index] = int(value.split("/")[0])
                elif "_diastolic" in tm.name:
                    old_values = signal[name].value.astype(str)
                    values = np.zeros(len(old_values))
                    for index, value in enumerate(old_values):
                        values[index] = int(value.split("/")[1])
                else:
                    values = signal[name].value
                original.append([values, signal[name].time])
            original = np.array(original)
            assert np.array_equal(original, tensor)

        elif tm_name.replace("_value", "") in data[0].keys():
            name = tm_name.replace("_value", "")
            original = []
            for signal in data:
                if "_systolic" in tm.name:
                    old_values = signal[name].value.astype(str)
                    values = np.zeros(len(old_values))
                    for index, value in enumerate(old_values):
                        values[index] = int(value.split("/")[0])
                elif "_diastolic" in tm.name:
                    old_values = signal[name].value.astype(str)
                    values = np.zeros(len(old_values))
                    for index, value in enumerate(old_values):
                        values[index] = int(value.split("/")[1])
                else:
                    values = signal[name].value
                original.append(values)
            tensor = tm.tensor_from_file(tm, hd5)
            assert np.array_equal(original, tensor)

            visit_id = list(hd5["edw"])[0]
            tensor = tm.tensor_from_file(tm, hd5, visits=visit_id)
            expected = original[0].reshape(1, original[0].size)
            assert np.array_equal(expected, tensor)

        elif tm_name.replace("_time", "") in data[0].keys():
            name = tm_name.replace("_time", "")
            original = []
            for signal in data:
                values = signal[name].time
                original.append(values)
            tensor = tm.tensor_from_file(tm, hd5)
            assert np.array_equal(original, tensor)

        elif tm_name.replace("_units", "") in data[0].keys():
            original = np.array([d[tm_name.replace("_units", "")].units for d in data])
            tensor = tm.tensor_from_file(tm, hd5)
            assert np.array_equal(original, tensor)

        else:
            raise AssertionError(f"TMap {tm.name} couldn't be tested.")


def test_medications_tmaps(hd5_data: TEST_DATA, testing_tmaps: TEST_TENSOR_MAPS):
    hd5, data = hd5_data
    data = data["meds"]
    testing_tmaps = testing_tmaps["meds"]

    for _name, tm in testing_tmaps.items():
        if tm.name.replace("_timeseries", "") in data[0].keys():
            name = tm.name.replace("_timeseries", "")
            tensor = tm.tensor_from_file(tm, hd5)
            original = []
            for signal in data:
                original.append([signal[name].dose, signal[name].start_date])
            original = np.array(original)
            assert np.array_equal(original, tensor)

        elif tm.name.replace("_dose", "") in data[0].keys():
            name = tm.name.replace("_dose", "")
            original = np.array([d[name].dose for d in data])
            tensor = tm.tensor_from_file(tm, hd5)
            assert np.array_equal(original, tensor)

            visit_id = list(hd5["edw"])[0]
            tensor = tm.tensor_from_file(tm, hd5, visits=visit_id)
            expected = original[0].reshape(1, original[0].size)
            assert np.array_equal(expected, tensor)

        elif tm.name.replace("_time", "") in data[0].keys():
            name = tm.name.replace("_time", "")
            original = np.array([d[name].start_date for d in data])
            tensor = tm.tensor_from_file(tm, hd5)
            assert np.array_equal(original, tensor)

        elif tm.name.replace("_units", "") in data[0].keys():
            original = np.array([d[tm.name.replace("_units", "")].units for d in data])
            tensor = tm.tensor_from_file(tm, hd5)
            assert np.array_equal(original, tensor)

        elif tm.name.replace("_route", "") in data[0].keys():
            original = np.array([d[tm.name.replace("_route", "")].route for d in data])
            tensor = tm.tensor_from_file(tm, hd5)
            assert np.array_equal(original, tensor)

        else:
            raise AssertionError(f"TMap {tm.name} couldn't be tested.")


def test_visits_tmap(hd5_data: TEST_DATA):
    hd5, data = hd5_data
    visits = data["visits"]
    tm = get_signal_tmap("visits")
    original = np.array(visits)[:, None]
    tensor = tm.tensor_from_file(tm, hd5)
    assert np.array_equal(original, tensor)


def test_sex_tmap(hd5_data: TEST_DATA):
    hd5, data = hd5_data
    data = data["demo"]

    tm = get_signal_tmap("sex")
    original = np.array([[1, 0] if d.sex == "male" else [0, 1] for d in data])
    tensor = tm.tensor_from_file(tm, hd5)
    assert np.array_equal(original, tensor)


def test_length_of_stay_tmap(hd5_data: TEST_DATA):
    hd5, data = hd5_data
    data = data["demo"]

    tm = get_signal_tmap("length_of_stay")
    end_original = np.array([get_unix_timestamps(d.end_date) for d in data])
    admin_original = np.array([get_unix_timestamps(d.admin_date) for d in data])
    original = (end_original - admin_original) / 60 / 60
    original = original[:, None]
    tensor = tm.tensor_from_file(tm, hd5)
    assert np.array_equal(original, tensor)


def test_age_tmap(hd5_data: TEST_DATA):
    hd5, data = hd5_data
    data = data["demo"]

    tm = get_signal_tmap("age")
    birth_original = np.array([get_unix_timestamps(d.birth_date) for d in data])
    admin_original = np.array([get_unix_timestamps(d.admin_date) for d in data])
    original = (admin_original - birth_original) / 60 / 60 / 24 / 365
    original = original[:, None]
    tensor = tm.tensor_from_file(tm, hd5)
    assert np.array_equal(original, tensor)


def test_named_bedmaster_signals_tmaps(testing_tmaps: TEST_TENSOR_MAPS):
    assert len(testing_tmaps["bedmaster_signals"]) != 0


def test_array_tmaps(hd5_data: TEST_DATA, testing_tmaps: TEST_TENSOR_MAPS):
    hd5, data = hd5_data
    data = data["signals"]

    def test_tm_array(field: str):
        value_tmaps = {
            tmap_name: tmap
            for tmap_name, tmap in testing_tmaps["bedmaster_signals"].items()
            if tmap_name.endswith(f"_{field}")
        }
        assert len(value_tmaps) != 0

        for _name, tm in value_tmaps.items():
            name = tm.name.split("_")[0]
            if len(tm.name.split("_")) > 2:
                sig_name = "_".join(tm.name.split("_")[1:-1])
            else:
                sig_name = "value"
            # Test multiple visits
            original = np.array([getattr(d[name], sig_name) for d in data])
            tensor = tm.tensor_from_file(tm, hd5, raw_values=True)
            assert np.array_equal(original, tensor)

            # Test single visit
            visit_id = list(hd5["bedmaster"])[0]
            tensor = tm.tensor_from_file(tm, hd5, visits=visit_id, raw_values=True)
            expected = original[0].reshape(1, original[0].size)
            assert np.array_equal(expected, tensor)

    test_tm_array("value")


def test_bedmaster_time_interpolation(
    hd5_data: TEST_DATA,
    testing_tmaps: TEST_TENSOR_MAPS,
):
    hd5, data = hd5_data
    data = data["signals"]
    value_tmaps = {
        tmap_name: tmap
        for tmap_name, tmap in testing_tmaps["bedmaster_signals"].items()
        if tmap_name.endswith("_timeseries") and len(tmap_name.split("_")) == 2
    }
    assert len(value_tmaps) != 0

    for _name, tm in value_tmaps.items():
        name = tm.name[: -len("_timeseries")]

        # Raw
        def obtain_raw(orig_time, samples):
            orig_raw = []
            for i, num_samples in enumerate(samples):
                orig_raw.append(orig_time[i])
                orig_raw.extend([np.nan] * (int(num_samples) - 1))
            orig_raw = np.array(orig_raw, dtype=float)
            return orig_raw

        tensor = tm.tensor_from_file(tm, hd5, interpolation="raw", raw_values=True)
        original = np.array(
            [
                [d[name].value, obtain_raw(d[name].time, d[name].samples_per_ts)]
                for d in data
            ],
        )
        assert ((original == tensor) | (pd.isnull(original) & pd.isnull(tensor))).all()

        # Just interpolate waveforms, not vitals
        if data[0][name].source != "bedmaster_waveform":
            continue

        # Linspace
        def obtain_linear(orig_time, values):
            npoints = len(values)
            return np.linspace(orig_time[0], orig_time[-1], npoints)

        tensor = tm.tensor_from_file(tm, hd5, interpolation="linspace", raw_values=True)
        original = np.array(
            [[d[name].value, obtain_linear(d[name].time, d[name].value)] for d in data],
        )
        assert np.array_equal(original, tensor)

        # complete_no_nans
        tensor = tm.tensor_from_file(tm, hd5, interpolation="complete_no_nans")
        for visit_idx, visit_data in enumerate(data):
            signal = visit_data[name]
            assert (
                np.diff(tensor[visit_idx][1])[0] - 0.25 / (signal.sample_freq[0][0] / 4)
                < 0.005
            )
            assert tensor[visit_idx][1][0] == signal.time[0]
            assert tensor[visit_idx][1][-1] == signal.time[-1] + 0.25 - 0.25 / 30

        # complete_nans_end
        tensor = tm.tensor_from_file(tm, hd5, interpolation="complete_nans_end")
        for visit_idx, visit_data in enumerate(data):
            signal = visit_data[name]
            assert signal.value.size + 3 == tensor[visit_idx][0].size
            assert np.array_equal(
                np.unique(np.diff(tensor[visit_idx][1])),
                np.array(
                    [
                        0.004166603088378906,
                        0.004166841506958008,
                        0.008333206176757812,
                        0.008333444595336914,
                    ],
                ),
            )
            assert np.array_equal(
                np.where(np.isnan(tensor[visit_idx][0]))[0],
                np.array([58, 59, 359]),
            )


def test_metadata_tmaps(hd5_data: TEST_DATA, testing_tmaps: TEST_TENSOR_MAPS):
    hd5, data = hd5_data
    data = data["signals"]

    def test_tm_metadata(field: str):
        tmaps = {
            tmap_name: tmap
            for tmap_name, tmap in testing_tmaps["bedmaster_signals"].items()
            if tmap_name.endswith(f"_{field}")
        }
        assert len(tmaps) != 0

        for _name, tm in tmaps.items():
            name = tm.name[: -len(f"_{field}")]
            # Multiple visits
            tensor = tm.tensor_from_file(tm, hd5)
            original = np.array([getattr(d[name], field) for d in data]).reshape(
                tensor.shape,
            )
            assert np.array_equal(original, tensor)

            # Single visit
            visit_id = list(hd5["bedmaster"])[0]
            tensor = tm.tensor_from_file(tm, hd5, visits=visit_id)
            original = np.array([getattr(d[name], field) for d in data])
            expected = original[0].reshape(tensor.shape)
            assert np.array_equal(expected, tensor)

    test_tm_metadata("units")
    test_tm_metadata("sample_freq")
    test_tm_metadata("channel")
    test_tm_metadata("scale_factor")


def test_scaled_values(hd5_data: TEST_DATA, testing_tmaps: TEST_TENSOR_MAPS):
    hd5, data = hd5_data
    data = data["signals"]

    tmaps_continuous = {
        tmap_name: tmap
        for tmap_name, tmap in testing_tmaps["bedmaster_signals"].items()
        if tmap_name.endswith("_value") and len(tmap_name.split("_")) == 2
    }
    assert len(tmaps_continuous) != 0

    for _name, tm in tmaps_continuous.items():
        name = tm.name[: -len("_value")]

        # On continuous data
        visit_id = list(hd5["bedmaster"])[0]
        tensor = tm.tensor_from_file(tm, hd5, visits=visit_id)
        original_signal = [d[name] for d in data][0]
        expected = original_signal.value * original_signal.scale_factor
        expected = expected.reshape(tensor.shape)
        assert np.array_equal(expected, tensor)

    tmaps_timeseries = {
        tmap_name: tmap
        for tmap_name, tmap in testing_tmaps["bedmaster_signals"].items()
        if tmap_name.endswith("_timeseries") and len(tmap_name.split("_")) == 2
    }
    assert len(tmaps_timeseries) != 0

    for _name, tm in tmaps_timeseries.items():
        name = tm.name[: -len("_timeseries")]

        # On timeseries
        tensor = tm.tensor_from_file(tm, hd5, visits=visit_id)
        original_signal = [d[name] for d in data][0]
        expected = original_signal.value * original_signal.scale_factor
        assert np.array_equal(expected, tensor[0][0])

        # If we want raw data
        tensor = tm.tensor_from_file(tm, hd5, visits=visit_id, raw_values=True)
        original_signal = [d[name] for d in data][0]
        expected = original_signal.value
        assert np.array_equal(expected, tensor[0][0])


def test_named_list_signals_tmaps(testing_tmaps: TEST_TENSOR_MAPS):
    assert len(testing_tmaps["list_signals"]) != 0


def test_list_signals_tmaps(hd5_data: TEST_DATA, testing_tmaps: TEST_TENSOR_MAPS):
    hd5, data = hd5_data
    data1 = data["signals"]
    data2 = data["alarms"]
    data3 = data["meds"]

    tm_wv = testing_tmaps["list_signals"]["bedmaster_waveform_signals"]
    tm_vs = testing_tmaps["list_signals"]["bedmaster_vitals_signals"]
    tm_alarms = testing_tmaps["list_signals"]["bedmaster_alarms_signals"]
    tm_med = testing_tmaps["list_signals"]["edw_med_signals"]

    # Check default tensormap
    original1 = np.array([list(d.keys()) for d in data1]).astype(object)
    tensor1_1 = tm_wv.tensor_from_file(tm_wv, hd5)
    tensor1_2 = tm_vs.tensor_from_file(tm_vs, hd5)
    assert np.array_equal(np.sort(original1[:, :-2].flat), np.sort(tensor1_1.flat))
    assert np.array_equal(np.sort(original1[:, -2:].flat), np.sort(tensor1_2.flat))

    # Check with single visit id arg
    visit_id = list(hd5["edw"])[0]
    original2 = np.array([list(d.keys()) for d in data2]).astype(object)
    tensor2 = tm_alarms.tensor_from_file(tm_alarms, hd5, visits=visit_id)
    assert np.array_equal(np.sort(original2[0].flat), np.sort(tensor2.flat))

    # Check with filtered signals arg
    original3 = np.array([list(d.keys()) for d in data3]).astype(object)
    tensor3 = tm_med.tensor_from_file(tm_med, hd5, filtered=True)
    assert np.array_equal(np.sort(original3[:, :-1].flat), np.sort(tensor3.flat))


def test_localtime_tmaps(
    hd5_data: TEST_DATA,
    testing_tmaps: TEST_TENSOR_MAPS,
    test_get_local_time: Callable[[np.ndarray], np.ndarray],
):
    test_time1 = np.array([np.nan, 1496666321.2, 1496676321.0, 1496766321.5])
    test_time2 = np.array([1496666321.2, np.nan, 1496676321.0, 1496766321.5])
    test_time3 = np.array([1496666321.2, 1496676321.0, 1496766321.5, 1516766321.1])

    expected_time1 = np.array(
        [
            np.datetime64("NaT"),
            "2017-06-05T08:38:41.200000",
            "2017-06-05T11:25:21.000000",
            "2017-06-06T12:25:21.500000",
        ],
    ).astype("datetime64[us]")
    expected_time2 = np.array(
        [
            "2017-06-05T08:38:41.200000",
            np.datetime64("NaT"),
            "2017-06-05T11:25:21.000000",
            "2017-06-06T12:25:21.500000",
        ],
    ).astype("datetime64[us]")
    expected_time3 = np.array(
        [
            "2017-06-05T08:38:41.200000",
            "2017-06-05T11:25:21.000000",
            "2017-06-06T12:25:21.500000",
            "2018-01-23T22:58:41.100000",
        ],
    ).astype("datetime64[us]")
    obtained_time1 = test_get_local_time(test_time1)
    obtained_time2 = test_get_local_time(test_time2)
    obtained_time3 = test_get_local_time(test_time3)

    assert (
        (expected_time1 == obtained_time1)
        | (np.isnan(expected_time1) & np.isnan(obtained_time1))
    ).all()
    assert (
        (expected_time2 == obtained_time2)
        | (np.isnan(expected_time2) & np.isnan(obtained_time2))
    ).all()
    assert (
        (expected_time3 == obtained_time3)
        | (np.isnan(expected_time3) & np.isnan(obtained_time3))
    ).all()

    def obtain_raw(orig_time, samples):
        orig_raw = []
        for i, num_samples in enumerate(samples):
            orig_raw.append(orig_time[i])
            orig_raw.extend([np.nan] * (int(num_samples) - 1))
        orig_raw = np.array(orig_raw, dtype=float)
        return orig_raw

    hd5, data = hd5_data
    data1 = data["alarms"]
    data2 = data["signals"]
    tm1 = testing_tmaps["alarms"]["apnea_init_date"]
    tm2 = testing_tmaps["bedmaster_signals"]["i_timeseries"]

    original1 = np.array([d["apnea"].start_date for d in data1])
    original2 = np.array(
        [[d["i"].value, obtain_raw(d["i"].time, d["i"].samples_per_ts)] for d in data2],
    )

    tensor1 = tm1.tensor_from_file(tm1, hd5, readable_dates=True, raw_values=True)
    tensor2 = tm2.tensor_from_file(
        tm2,
        hd5,
        interpolation="raw",
        readable_dates=True,
        raw_values=True,
    )

    converted1 = np.zeros(np.shape(original1), dtype=object)
    for idx in range(np.shape(original1)[0]):
        converted1[idx] = test_get_local_time(original1[idx])

    converted2 = np.zeros(np.shape(original2), dtype=object)
    for idx in range(np.shape(original2)[0]):
        converted2[idx][0] = original2[idx][0]
        converted2[idx][1] = test_get_local_time(original2[idx][1])

    assert np.array_equal(converted1, tensor1)
    assert (
        (converted2 == tensor2) | (pd.isnull(converted2) & pd.isnull(tensor2))
    ).all()


def test_around_event_tmaps(hd5_data: TEST_DATA):
    hd5, data = hd5_data
    measurements = data["measurements"]
    events = data["events"]

    event_time = min(
        np.append(
            events[0]["code_start"].start_date,
            events[0]["rapid_response_start"].start_date,
        ),
    )
    values_array = measurements[0]["blood_pressure"].value
    spliter = lambda x: int(x.split("/")[0])
    vfunc = np.vectorize(spliter)
    values_array = vfunc(values_array.astype(str))
    time_array = measurements[0]["blood_pressure"].time

    tm1 = get_around_tmap(
        "blood_pressure_systolic_value_3_to_6_hrs_pre_arrest_start_date",
    )
    tm2 = get_around_tmap(
        "blood_pressure_systolic_value_3_and_6_hrs_pre_arrest_start_date_2_hrs_window",
    )
    tm3 = get_around_tmap(
        "blood_pressure_systolic_timeseries_3_to_6_hrs_pre_arrest_start_date",
    )
    tm4 = get_around_tmap(
        "blood_pressure_systolic_timeseries_3_and_6_hrs_pre_arrest_start_date_"
        "2_hrs_window",
    )

    tensor1 = tm1.tensor_from_file(tm1, hd5)
    indices1 = np.where(
        np.logical_and(
            time_array < event_time - 3 * 60 * 60,
            time_array > event_time - 6 * 60 * 60,
        ),
    )[0]
    original1 = values_array[indices1]
    assert np.array_equal(original1, tensor1)

    tensor2 = tm2.tensor_from_file(tm2, hd5)
    indices20 = np.where(
        np.logical_and(
            time_array < event_time - 3 * 60 * 60,
            time_array > event_time - 5 * 60 * 60,
        ),
    )[0]
    indices21 = np.where(
        np.logical_and(
            time_array < event_time - 6 * 60 * 60,
            time_array > event_time - 8 * 60 * 60,
        ),
    )[0]
    original2 = np.array([values_array[indices20], values_array[indices21]])
    assert np.array_equal(original2, tensor2)

    tensor3 = tm3.tensor_from_file(tm3, hd5)
    original3 = np.array([original1, time_array[indices1]])
    assert np.array_equal(original3, tensor3)

    tensor4 = tm4.tensor_from_file(tm4, hd5)
    original4 = np.array(
        [[original2[0], time_array[indices20]], [original2[1], time_array[indices21]]],
    )
    assert np.array_equal(original4, tensor4)


def test_sliding_window_tmaps(hd5_data: TEST_DATA):
    hd5, data = hd5_data
    events = data["events"]

    event_time = min(
        np.append(
            events[0]["code_start"].start_date,
            events[0]["rapid_response_start"].start_date,
        ),
    )
    admin_date = get_unix_timestamps(data["demo"][0].admin_date)

    step = 200
    window = 50
    num_of_windows = (
        int((event_time - window * 60 * 60 - admin_date) / 60 / 60 / step) + 1
    )

    tm1 = get_around_tmap(
        f"blood_pressure_systolic_value_{window}_hrs_sliding_window_admin_date_to"
        f"_arrest_start_date_{step}_hrs_step_min",
    )
    tm2 = get_around_tmap(
        f"{window}_hrs_sliding_window_admin_date_to_arrest_start_date"
        f"_{step}_hrs_step_12_hrs_prediction",
    )
    tensor1 = tm1.tensor_from_file(tm1, hd5)
    tensor2 = tm2.tensor_from_file(tm2, hd5)
    assert tensor1.shape[0] == tensor2.shape[0] == num_of_windows


def test_signal_metrics_tmaps(hd5_data: TEST_DATA):
    hd5, data = hd5_data
    measurements = data["measurements"]
    events = data["events"]

    event_time = min(
        np.append(
            events[0]["code_start"].start_date,
            events[0]["rapid_response_start"].start_date,
        ),
    )
    values_array = measurements[0]["blood_pressure"].value
    spliter = lambda x: int(x.split("/")[0])
    vfunc = np.vectorize(spliter)
    values_array = vfunc(values_array.astype(str))
    time_array = measurements[0]["blood_pressure"].time

    tm1 = get_around_tmap(
        "blood_pressure_systolic_value_3_to_6_hrs_pre_arrest_start_date_max",
    )
    tm2 = get_around_tmap(
        "blood_pressure_systolic_value_3_and_6_hrs_pre_arrest_start_date_2_hrs"
        "_window_mean",
    )
    tm3 = get_around_tmap(
        "blood_pressure_systolic_timeseries_3_to_6_hrs_pre_arrest_start_date_median",
    )
    tm4 = get_around_tmap(
        "blood_pressure_systolic_timeseries_3_and_6_hrs_pre_arrest_start_date_"
        "2_hrs_window_last",
    )

    tensor1 = tm1.tensor_from_file(tm1, hd5)
    indices1 = np.where(
        np.logical_and(
            time_array < event_time - 3 * 60 * 60,
            time_array > event_time - 6 * 60 * 60,
        ),
    )[0]
    original1 = np.array([np.nanmax(values_array[indices1])])
    assert np.array_equal(original1, tensor1)

    tensor2 = tm2.tensor_from_file(tm2, hd5)
    indices20 = np.where(
        np.logical_and(
            time_array < event_time - 3 * 60 * 60,
            time_array > event_time - 5 * 60 * 60,
        ),
    )[0]
    indices21 = np.where(
        np.logical_and(
            time_array < event_time - 6 * 60 * 60,
            time_array > event_time - 8 * 60 * 60,
        ),
    )[0]
    original2 = np.array(
        [[np.nanmean(values_array[indices20])], [np.nanmean(values_array[indices21])]],
    )
    assert np.array_equal(original2, tensor2)

    tensor3 = tm3.tensor_from_file(tm3, hd5)
    indice = abs(
        np.flip(values_array[indices1] - np.nanmedian(values_array[indices1])),
    ).argmin()
    indice = len(indices1) - indice - 1
    original3 = np.array(
        [np.nanmedian(values_array[indices1]), time_array[indices1][indice]],
    )
    assert np.array_equal(original3, tensor3)

    tensor4 = tm4.tensor_from_file(tm4, hd5)
    original4 = np.array(
        [
            [values_array[indices20][-1], time_array[indices20[-1]]],
            [values_array[indices21][-1], time_array[indices21][-1]],
        ],
    )
    assert np.array_equal(original4, tensor4)
