# Imports: third party
import numpy as np


def test_measurements_tmaps(hd5_data, testing_tmaps):
    hd5, data = hd5_data
    data = data["measurements"]
    testing_tmaps = testing_tmaps["measurements"]

    for _name, tm in testing_tmaps.items():
        tm_name = tm.name
        if "blood_pressure" in tm.name:
            tm_name = tm_name.replace("_systolic", "")
            tm_name = tm_name.replace("_diastolic", "")
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
