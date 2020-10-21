# Imports: third party
import numpy as np
import pandas as pd


def test_named_test_tmaps(testing_tmaps):
    assert len(testing_tmaps["bm_signals"]) != 0


def test_array_tmaps(hd5_data, testing_tmaps):
    hd5, data = hd5_data
    data = data["bm_signals"]

    def test_tm_array(field):
        value_tmaps = {
            tmap_name: tmap
            for tmap_name, tmap in testing_tmaps["bm_signals"].items()
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


def test_bm_time_interpolation(hd5_data, testing_tmaps):
    hd5, data = hd5_data
    data = data["bm_signals"]
    value_tmaps = {
        tmap_name: tmap
        for tmap_name, tmap in testing_tmaps["bm_signals"].items()
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
        if data[0][name].source != "BM_waveform":
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


def test_metadata_tmaps(hd5_data, testing_tmaps):
    hd5, data = hd5_data
    data = data["bm_signals"]

    def test_tm_metadata(field):
        tmaps = {
            tmap_name: tmap
            for tmap_name, tmap in testing_tmaps["bm_signals"].items()
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


def test_scaled_values(hd5_data, testing_tmaps):
    hd5, data = hd5_data
    data = data["bm_signals"]

    tmaps_continuous = {
        tmap_name: tmap
        for tmap_name, tmap in testing_tmaps["bm_signals"].items()
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
        for tmap_name, tmap in testing_tmaps["bm_signals"].items()
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
