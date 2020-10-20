# Imports: standard library
import copy

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.ingest.icu.readers import BMReader

# pylint: disable=invalid-name


def test_list_vs(bm_reader, empty_matfile):
    expected_vs = ["CO", "CUFF", "HR", "SPO2%", "SPO2R"]
    assert bm_reader.list_vs() == expected_vs

    # No vs on file:
    empty_reader = BMReader(empty_matfile.name)
    assert empty_reader.list_vs() == []


def test_list_wv(bm_reader, empty_matfile):
    expected_dict = {
        "I": "ch7",
        "II": "ch8",
        "III": "ch9",
        "V": "ch10",
        "SPO2": "ch39",
        "RESP": "ch40",
    }
    assert bm_reader.list_wv() == expected_dict

    # No wv on file:
    empty_reader = BMReader(empty_matfile.name)
    assert empty_reader.list_vs() == []


def test_get_vs(bm_reader, matfile):
    def _linearize(arr):
        return np.transpose(arr)[0]

    # Standard case
    heart_rate = bm_reader.get_vs("HR")
    assert heart_rate.name == "HR"
    assert np.array_equal(heart_rate.value, _linearize(matfile["vs/HR"][()]))
    assert np.array_equal(
        heart_rate.time,
        _linearize(matfile["vs_time_corrected/HR/res_vs"][()]),
    )
    assert heart_rate.scale_factor == 0.5
    assert heart_rate.units == "Bpm"
    assert heart_rate.sample_freq == np.array([(0.5, 0)], dtype="float,int")

    # Check that dataevents are collected
    time_corr_arr = np.unpackbits(heart_rate.time_corr_arr, axis=None)
    irregularities = np.where(time_corr_arr)[0]
    expected_irr_idx = np.array([2, 3, 11])
    assert np.array_equal(irregularities, expected_irr_idx)

    # Check that interbundle correction is applied
    bm_reader.interbundle_corr["vs"] = {
        "maxTime": heart_rate.time[10],
        "timeCorr": 3,
    }
    heart_rate_corr = bm_reader.get_vs("HR")

    assert len(heart_rate_corr.time) == len(heart_rate.time) - 11
    assert len(heart_rate_corr.value) == len(heart_rate.value) - 11

    # Case with unknown scale factor and unit
    spo2r = bm_reader.get_vs("SPO2R")
    assert spo2r.scale_factor == 1
    assert spo2r.units == "UNKNOWN"
    assert spo2r.sample_freq == np.array([(0.5, 0)], dtype="float,int")


def test_get_wv(bm_reader, matfile):
    def linearize(arr):
        return np.transpose(arr)[0]

    # Check standard case
    ecgv = bm_reader.get_wv("ch10", "V")
    assert ecgv.name == "V"
    assert np.array_equal(ecgv.value, linearize(matfile["wv/ch10"][()]))
    assert np.array_equal(
        ecgv.time,
        linearize(matfile["wv_time_corrected/ch10/res_wv"][()]),
    )
    assert ecgv.units == "mV"
    assert ecgv.scale_factor == 0.0243
    assert np.array_equal(ecgv.sample_freq, np.array([(240, 0)], dtype="float,int"))

    # Check that it works without specifying signal name
    ecgv_copy = bm_reader.get_wv("ch10")
    assert ecgv_copy.name == ecgv.name
    assert np.array_equal(ecgv_copy.value, ecgv.value)

    # Check dataevents are collected
    time_corr_arr = np.unpackbits(ecgv.time_corr_arr, axis=None)
    irregularities = np.where(time_corr_arr)[0]
    expected_irr_idx = np.array([3, 4, 20, 32, 40, 50, 54])
    assert np.array_equal(irregularities, expected_irr_idx)

    # Case with multiple sample frequency
    ecg2 = bm_reader.get_wv("ch8", "II")
    assert np.array_equal(
        ecg2.sample_freq,
        np.array([(240, 0), (120, 80)], dtype="float,int"),
    )

    # Check that interbundle correction is applied
    overlap = 5
    bm_reader.interbundle_corr["wv"] = {
        "maxTime": ecgv.time[overlap - 1],
        "timeCorr": 8,
    }
    ecg2_corr = bm_reader.get_wv("ch8", "II")

    values_cut = overlap * ecg2_corr.sample_freq[0][0] / 4
    assert len(ecg2_corr.time) == len(ecgv.time) - overlap
    assert len(ecg2_corr.value) == len(ecgv.value) - values_cut
    assert np.array_equal(
        ecg2_corr.sample_freq,
        np.array([(240, 0), (120, 75)], dtype="float,int"),
    )

    # Case with unknown scale factor and unit
    ecg3 = bm_reader.get_wv("ch9", "III")
    assert ecg3.units == "??V"
    assert ecg3.scale_factor == 2.44
    assert np.array_equal(ecg3.sample_freq, np.array([(240, 0)], dtype="float,int"))


def test_format_data(bm_reader):
    expected_data = np.arange(10)

    #  Case column vector [[0], [1]] -> [0,1]
    unformatted_data = np.transpose([expected_data])
    assert np.array_equal(bm_reader.format_data(unformatted_data), expected_data)

    # Case single-row 2D vector [[0,1]] -> [0,1]
    unformatted_data = np.array([expected_data])
    assert np.array_equal(bm_reader.format_data(unformatted_data), expected_data)

    # Case column matrix
    input_2d_data = np.repeat([[48, 46, 50], [49, 46, 50]], 5, axis=0)
    formatted_data = np.repeat([0.2, 1.2], 5)
    assert np.array_equal(bm_reader.format_data(input_2d_data), formatted_data)

    # Case row matrix
    input_2d_data = np.transpose(input_2d_data)
    formatted_data = np.repeat([0.2, 1.2], 5)
    assert np.array_equal(bm_reader.format_data(input_2d_data), formatted_data)


def test_decode_data(bm_reader):
    # float signal
    codified_data = np.repeat([[48, 46, 50], [49, 46, 50]], 5, axis=0)
    decoded_data = np.repeat([0.2, 1.2], 5)
    assert np.array_equal(bm_reader.format_data(codified_data), decoded_data)
    # int signal
    codified_data = np.repeat([[51, 50], [50, 57]], 5, axis=0)
    decoded_data = np.repeat([32, 29], 5)
    assert np.array_equal(bm_reader.format_data(codified_data), decoded_data)

    # string nan 1 and 2
    codified_data = np.repeat([[48, 46, 50, 51], [49, 46, 50, 49]], 5, axis=0)
    codified_data[3] = [78, 111, 110, 101]
    decoded_data = np.repeat([0.23, 1.21], 5)
    decoded_data[3] = np.nan

    obtained = bm_reader.format_data(codified_data)
    assert np.array_equal(
        obtained[~np.isnan(obtained)],
        decoded_data[~np.isnan(decoded_data)],
    )
    assert np.isnan(obtained[3])
    codified_data[3] = [88, 32, 32, 32]
    obtained = bm_reader.format_data(codified_data)
    assert np.array_equal(
        obtained[~np.isnan(obtained)],
        decoded_data[~np.isnan(decoded_data)],
    )
    assert np.isnan(obtained[3])


def test_contiguous_nparrays(bm_reader):
    heart_rate = bm_reader.get_vs("HR")
    ecgv = bm_reader.get_wv("ch10", "V")
    signals = [heart_rate, ecgv]
    for signal in signals:
        assert not signal.value.dtype == object
        assert not signal.time.dtype == object


def test_max_segment(bm_reader):
    def _create_max_seg_dict(seg_no, max_time, signal_name):
        return {
            "segmentNo": seg_no,
            "maxTime": max_time,
            "signalName": signal_name,
        }

    empty_vs_dict = _create_max_seg_dict(0, -1, "")
    empty_wv_dict = _create_max_seg_dict(0, -1, "")
    expected_wv_max = _create_max_seg_dict(
        seg_no=6,
        max_time=1452438403.75,
        signal_name="ch10",
    )
    expected_vs_max = _create_max_seg_dict(
        seg_no=6,
        max_time=1452438402.0,
        signal_name="CO",
    )

    assert bm_reader.max_segment["vs"] == empty_vs_dict
    assert bm_reader.max_segment["wv"] == empty_wv_dict

    wv_signals = bm_reader.list_wv()
    for wv_signal_name, channel in wv_signals.items():
        bm_reader.get_wv(channel, wv_signal_name)

    assert bm_reader.max_segment["vs"] == empty_vs_dict
    assert bm_reader.max_segment["wv"] == expected_wv_max

    vs_signals = bm_reader.list_vs()
    for vs_signal_name in vs_signals:
        bm_reader.get_vs(vs_signal_name)
    assert bm_reader.max_segment["vs"] == expected_vs_max
    assert bm_reader.max_segment["wv"] == expected_wv_max


def test_get_interbundle_correction(bm_reader):
    art1d = bm_reader.get_vs("CO")
    ch10 = bm_reader.get_wv("ch10", "v")

    no_correction = {"vs": None, "wv": None}
    prev_max_vs = {
        "segmentNo": 2,
        "maxTime": art1d.time[4],
        "signalName": "CO",
    }
    prev_max_wv = {
        "segmentNo": 1,
        "maxTime": ch10.time[14],
        "signalName": "ch10",
    }
    prev_max_info = {"vs": prev_max_vs, "wv": prev_max_wv}

    expected_vs = {"maxTime": art1d.time[3], "timeCorr": 2}
    expected_wv = {"maxTime": ch10.time[15], "timeCorr": -0.25}

    def _change_ibcor_dict(source_type, **kwargs):
        new_dict = prev_max_info.copy()
        for source in source_type:
            for key in kwargs:
                new_dict[source][key] = kwargs[key]
        return new_dict

    assert bm_reader.interbundle_corr == no_correction

    # Normal case:
    bm_reader.get_interbundle_correction(prev_max_info)
    assert bm_reader.interbundle_corr["vs"] == expected_vs
    assert bm_reader.interbundle_corr["wv"] == expected_wv

    # Case: signal does not exist on current bundle
    previous = _change_ibcor_dict(["vs", "wv"], signalName="Random")
    bm_reader.get_interbundle_correction(previous)
    assert bm_reader.interbundle_corr == no_correction

    # Case: no overlap between previous and current bundle
    previous = _change_ibcor_dict(["vs", "wv"], segmentNo=-1)
    bm_reader.get_interbundle_correction(previous)
    assert bm_reader.interbundle_corr == no_correction

    # Case: complete overlap between previous and current
    previous = _change_ibcor_dict(["vs", "wv"], segmentNo=100)
    bm_reader.get_interbundle_correction(previous)
    assert bm_reader.interbundle_corr == no_correction


def test_apply_interbundle_correction(bm_reader: BMReader):
    def _create_ib_corr_dict(signal, cut_idx, time_corr):
        source_corr = {
            "maxTime": signal.time[cut_idx],
            "timeCorr": time_corr,
        }
        return source_corr

    def _apply_and_assert_ib(signal, sig_type):
        signal.time_corr_arr = np.unpackbits(signal.time_corr_arr)
        signal_original = copy.deepcopy(signal)
        bm_reader.interbundle_corr[sig_type] = _create_ib_corr_dict(
            signal,
            cut_idx[sig_type],
            time_corr[sig_type],
        )
        bm_reader.apply_ibcorr(signal)
        _assert_ib_correction(
            signal_original,
            signal,
            cut_idx[sig_type],
            time_corr[sig_type],
        )
        bm_reader.interbundle_corr[sig_type] = None

    cut_idx = {"wv": 10, "vs": 4}
    time_corr = {"wv": 5, "vs": 3}

    # Standard case
    ecgi = bm_reader.get_wv("ch7", "I")
    _apply_and_assert_ib(ecgi, "wv")

    cuff = bm_reader.get_vs("CUFF")
    _apply_and_assert_ib(cuff, "vs")

    # With DE
    ecgv = bm_reader.get_wv("ch10", "V")
    _apply_and_assert_ib(ecgv, "wv")

    hr = bm_reader.get_vs("HR")
    _apply_and_assert_ib(hr, "vs")

    # With missing values
    ecgiii = bm_reader.get_wv("ch9", "prova")
    _apply_and_assert_ib(ecgiii, "wv")

    # Multiple sf
    ecgii = bm_reader.get_wv("ch8", "II")
    _apply_and_assert_ib(ecgii, "wv")


def _assert_ib_correction(original, corrected, cut_idx, time_corr):
    # Assert array length
    length_cut = cut_idx + 1
    for array in ["time", "samples_per_ts"]:
        original_arr = getattr(original, array)
        obtained_arr = getattr(corrected, array)
        assert len(obtained_arr) == len(original_arr) - length_cut

    assert len(corrected.value) == len(original.value) - np.sum(
        original.samples_per_ts[:length_cut],
    )

    length_cut = cut_idx + 1
    assert corrected.time[0] == (original.time[length_cut] + time_corr)

    # Assert sample freq:
    expected_first_sf = original.sample_freq[
        [sf[1] < cut_idx for sf in original.sample_freq]
    ][-1][0]
    expected_sf = np.insert(original.sample_freq, 0, (expected_first_sf, length_cut))
    expected_sf = expected_sf[[sf[1] >= cut_idx for sf in expected_sf]]
    expected_sf = np.fromiter(
        map(lambda sf: (sf[0], sf[1] - length_cut), expected_sf),
        dtype="float,int",
    )
    assert np.array_equal(expected_sf, corrected.sample_freq)

    # Assert array value
    de = np.where(original.time_corr_arr)[0]
    if de.size != 0:
        dataevent_idx = de[de > cut_idx][0]
        new_de_idx = dataevent_idx - length_cut
        assert np.array_equal(
            original.time[dataevent_idx:],
            corrected.time[new_de_idx:],
        )
        assert np.array_equal(
            original.time[length_cut:dataevent_idx] + time_corr,
            corrected.time[:new_de_idx],
        )
    else:
        assert corrected.time[-1] == original.time[-1] + time_corr
