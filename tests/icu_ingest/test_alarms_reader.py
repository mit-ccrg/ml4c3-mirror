# Imports: third party
import numpy as np
import pytest

# Imports: first party
from definitions.icu import ALARMS_FILES
from tensorize.bedmaster.readers import BedmasterAlarmsReader


@pytest.fixture(scope="function")
def alarms_reader() -> BedmasterAlarmsReader:
    reader = BedmasterAlarmsReader(
        pytest.alarms_dir,
        pytest.example_mrn,
        pytest.example_visit_id,
        pytest.adt_path,
    )
    return reader


def test_get_alarms_dfs(alarms_reader: BedmasterAlarmsReader):
    assert len(alarms_reader.alarms_dfs) == 3


def test_list_alarms(alarms_reader: BedmasterAlarmsReader):
    expected_alarms = [
        "TACHY",
        "HR HI 156",
        "ARRHY SUSPEND",
        "HR HI 165",
        "HR HI 160",
        "SPO2 PROBE",
        "PPEAK HIGH",
        "TVEXP LOW",
        "CPP LOW",
    ]
    assert sorted(alarms_reader.list_alarms()) == sorted(expected_alarms)


def test_get_alarm(alarms_reader: BedmasterAlarmsReader):
    alarm1 = alarms_reader.get_alarm("SPO2 PROBE")
    alarm2 = alarms_reader.get_alarm("TVEXP LOW")

    assert alarm1.source == ALARMS_FILES["source"]
    assert alarm2.source == ALARMS_FILES["source"]

    assert alarm1.name == "SPO2 PROBE"
    assert alarm2.name == "TVEXP LOW"

    assert np.array_equal(
        alarm1.start_date,
        np.array(
            [1587614670, 1588739670, 1588847670, 1589037670, 1589237670, 1589337670],
        ),
    )
    assert np.array_equal(alarm2.start_date, np.array([1587400060, 1587400123]))

    assert np.array_equal(alarm1.duration, np.array([16, 2, 0, 6, 4, 0]))
    assert np.array_equal(alarm2.duration, np.array([0, 0]))

    assert alarm1.level == 3
    assert alarm2.level == 6
