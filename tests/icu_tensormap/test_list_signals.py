# Imports: third party
import numpy as np


def test_named_test_tmaps(testing_tmaps):
    assert len(testing_tmaps["list_signals"]) != 0


def test_list_signals_tmaps(hd5_data, testing_tmaps):
    hd5, data = hd5_data
    data1 = data["bm_signals"]
    data2 = data["alarms"]
    data3 = data["meds"]

    tm_wv = testing_tmaps["list_signals"]["bm_waveform_signals"]
    tm_vs = testing_tmaps["list_signals"]["bm_vitals_signals"]
    tm_alarms = testing_tmaps["list_signals"]["bm_alarms_signals"]
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
