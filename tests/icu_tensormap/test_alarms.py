# Imports: third party
import numpy as np


def test_alarms_tmaps(hd5_data, testing_tmaps):
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
