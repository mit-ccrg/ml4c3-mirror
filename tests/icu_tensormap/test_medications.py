# Imports: third party
import numpy as np


def test_medications_tmaps(hd5_data, testing_tmaps):
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
