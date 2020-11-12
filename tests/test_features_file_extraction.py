# Imports: standard library
from typing import Dict, List, Tuple, Union

# Imports: third party
import h5py

# Imports: first party
from ml4c3.ecg_features_extraction import ECGFeatureFileExtractor
from ml4c3.ingest.icu.data_objects import ICUDataObject

# pylint: disable=unused-argument
# hd5_data fixture is needed to write the data file this test inspects


def test_ecg_features_file_extraction(
    hd5_data: Tuple[
        h5py.File,
        Dict[str, List[Union[str, ICUDataObject, Dict[str, ICUDataObject]]]],
    ],
    temp_file,
):
    ecg_signal = ECGFeatureFileExtractor(temp_file.name)
    assert ecg_signal.list_available_leads() == ["i", "ii", "iii", "v"]
    ecg_signal.extract_features(min_peaks=0)
    ecg_signal.compute_additional_features()
    ecg_signal.save_features()
    expected_features = []
    for peak in ["p", "q", "r", "s", "t"]:
        expected_features.append(f"ecg_{peak}_peaks")
    for wave in ["p", "r", "t"]:
        expected_features.append(f"ecg_{wave}_onsets")
        expected_features.append(f"ecg_{wave}_offsets")
    for segment in ["pr", "tp", "st"]:
        expected_features.append(f"ecg_{segment}_segment")
    for interval in ["pr", "qt", "rr", "qrs"]:
        expected_features.append(f"ecg_{interval}_interval")
    for amplitude in ["qrs"]:
        expected_features.append(f"ecg_{amplitude}_amplitude")
    for height in ["st"]:
        expected_features.append(f"ecg_{height}_height")
    expected_length = len(ecg_signal.r_peaks["ECG_R_Peaks"])

    visit = list(ecg_signal["bedmaster"].keys())[0]
    assert sorted(ecg_signal[f"bedmaster/{visit}/ecg_features/i"].keys()) == sorted(
        expected_features,
    )

    for feature in ecg_signal.r_peaks:
        assert feature.lower() in expected_features
        assert expected_length == len(ecg_signal.r_peaks[feature])
        expected_features.remove(feature.lower())
    for feature in ecg_signal.waves_peaks:
        assert feature.lower() in expected_features
        assert expected_length == len(ecg_signal.waves_peaks[feature])
        expected_features.remove(feature.lower())
    for feature in ecg_signal.other_features:
        assert feature.lower() in expected_features
        if feature in ["ECG_RR_Interval"]:
            assert expected_length - 1 == len(ecg_signal.other_features[feature])
        else:
            assert expected_length == len(ecg_signal.other_features[feature])
        expected_features.remove(feature.lower())
    assert len(expected_features) == 0
