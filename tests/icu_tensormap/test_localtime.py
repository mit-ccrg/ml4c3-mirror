# Imports: third party
import numpy as np
import pandas as pd


def test_localtime_tmaps(hd5_data, testing_tmaps, test_get_local_time):

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
    data2 = data["bm_signals"]
    tm1 = testing_tmaps["alarms"]["apnea_init_date"]
    tm2 = testing_tmaps["bm_signals"]["i_timeseries"]

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
