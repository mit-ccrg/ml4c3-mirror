# Imports: third party
import numpy as np


def test_events_tmaps(hd5_data, testing_tmaps):
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
