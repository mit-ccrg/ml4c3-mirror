# Imports: third party
import numpy as np
import pytest

# Imports: first party
from ml4c3.utils import get_unix_timestamps
from ml4c3.tensormap.tensor_maps_icu_static import get_tmap


def test_visits_tmap(hd5_data):
    hd5, data = hd5_data

    visits = data["visits"]

    # call tff without visits, get tensor from all visits
    tm = get_tmap("visits")
    original = np.array(visits)
    tensor = tm.tensor_from_file(tm, hd5)
    assert np.array_equal(original, tensor)

    # call tff with visits as list, get tensor from only those in list
    tensor = tm.tensor_from_file(tm, hd5, visits=visits)
    assert np.array_equal(original, tensor)

    # call tff with visits as single visit, only get tensor from that visit
    visit = visits[0]
    original = np.array([visit])
    tensor = tm.tensor_from_file(tm, hd5, visits=visit)
    assert np.array_equal(original, tensor)

    # make sure an invalid visit is not retrieved
    bad_visit = "NOT_A_VISIT"
    visits.append(bad_visit)
    expected_error_message = f"'Visit {bad_visit} not found in hd5'"
    with pytest.raises(KeyError) as e_info:
        tensor = tm.tensor_from_file(tm, hd5, visits=visits)
    assert str(e_info.value) == expected_error_message


def test_sex_tmap(hd5_data):
    hd5, data = hd5_data
    data = data["demo"]

    tm = get_tmap("sex")
    original = np.array([[1, 0] if d.sex == "male" else [0, 1] for d in data])
    tensor = tm.tensor_from_file(tm, hd5)
    assert np.array_equal(original, tensor)


def test_length_of_stay_tmap(hd5_data):
    hd5, data = hd5_data
    data = data["demo"]

    tm = get_tmap("length_of_stay")
    end_original = np.array([get_unix_timestamps(d.end_date) for d in data])
    admin_original = np.array([get_unix_timestamps(d.admin_date) for d in data])
    original = end_original - admin_original
    tensor = tm.tensor_from_file(tm, hd5)
    assert np.array_equal(original, tensor)


def test_age_tmap(hd5_data):
    hd5, data = hd5_data
    data = data["demo"]

    tm = get_tmap("age")
    birth_original = np.array([get_unix_timestamps(d.birth_date) for d in data])
    admin_original = np.array([get_unix_timestamps(d.admin_date) for d in data])
    original = np.round((admin_original - birth_original) / 60 / 60 / 24 / 365).astype(
        int,
    )
    tensor = tm.tensor_from_file(tm, hd5)
    assert np.array_equal(original, tensor)
