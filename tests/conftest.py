# Imports: standard library
import os
import sys
import argparse
import datetime
import tempfile
import multiprocessing as mp
from time import time
from typing import Dict, List, Type, Tuple, Union, Callable
from itertools import product

# Imports: third party
import h5py
import numpy as np
import pytest
import neurokit2 as nk

# Imports: first party
import tensormap
from definitions.icu import EDW_FILES
from ml4c3.arguments import parse_args
from ingest.icu.writers import Writer
from definitions.globals import TENSOR_EXT
from tensormap.TensorMap import TensorMap, Interpretation, get_local_timestamps
from tensormap.icu_signals import get_tmap as GET_SIGNAL_TMAP
from ingest.icu.data_objects import (
    Event,
    Procedure,
    Medication,
    StaticData,
    Measurement,
    ICUDataObject,
    BedmasterAlarm,
    BedmasterSignal,
)
from tensormap.icu_list_signals import get_tmap as GET_LIST_TMAP

# pylint: disable=redefined-outer-name, unused-argument, missing-class-docstring


def pytest_configure():
    def tff(tm, hd5):
        return hd5[f"/{tm.name}"][:]

    pytest.TFF = tff
    pytest.N_TENSORS = 50
    pytest.CONTINUOUS_TMAPS = [
        TensorMap(
            f"{n}d_cont",
            shape=tuple(range(2, n + 2)),
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=tff,
        )
        for n in range(1, 6)
    ]
    pytest.CATEGORICAL_TMAPS = [
        TensorMap(
            f"{n}d_cat",
            shape=tuple(range(2, n + 2)),
            interpretation=Interpretation.CATEGORICAL,
            channel_map={f"c_{i}": i for i in range(n + 1)},
            tensor_from_file=tff,
        )
        for n in range(1, 6)
    ]
    pytest.TMAPS_UP_TO_4D = pytest.CONTINUOUS_TMAPS[:-1] + pytest.CATEGORICAL_TMAPS[:-1]
    pytest.TMAPS_5D = pytest.CONTINUOUS_TMAPS[-1:] + pytest.CATEGORICAL_TMAPS[-1:]
    pytest.MULTIMODAL_UP_TO_4D = [
        list(x)
        for x in product(pytest.CONTINUOUS_TMAPS[:-1], pytest.CATEGORICAL_TMAPS[:-1])
    ]
    pytest.SEGMENT_IN = TensorMap(
        "2d_for_segment_in",
        shape=(32, 32, 1),
        interpretation=Interpretation.CONTINUOUS,
        metrics=["mse"],
        tensor_from_file=tff,
    )
    pytest.SEGMENT_OUT = TensorMap(
        "2d_for_segment_out",
        shape=(32, 32, 2),
        interpretation=Interpretation.CATEGORICAL,
        channel_map={"yes": 0, "no": 1},
        tensor_from_file=tff,
    )
    pytest.MOCK_TMAPS = {
        tmap.name: tmap for tmap in pytest.CONTINUOUS_TMAPS + pytest.CATEGORICAL_TMAPS
    }

    pytest.example_mrn = "123"
    pytest.example_visit_id = "345"
    pytest.run_id = "1234567"
    pytest.run_id_par = "12345678"

    pytest.datadir = os.path.join(os.path.dirname(__file__), "icu_ingest", "data")

    # CrossRef
    pytest.cross_ref_file = os.path.join(pytest.datadir, "xref_file.csv")
    pytest.cross_ref_file_tens = os.path.join(pytest.datadir, "xref_file_tensorize.csv")

    # BedMaster
    pytest.bedmaster_dir = os.path.join(pytest.datadir, "bedmaster")
    pytest.mat_file = os.path.join(pytest.bedmaster_dir, "bedmaster_file-123_5_v4.mat")
    pytest.bedmaster_matching = os.path.join(pytest.datadir, "bedmaster_matching_files")

    # EDW
    pytest.edw_dir = os.path.join(pytest.datadir, "edw")
    pytest.edw_patient_dir = os.path.join(
        pytest.edw_dir,
        pytest.example_mrn,
        pytest.example_visit_id,
    )
    pytest.adt_path = os.path.join(pytest.edw_dir, "adt.csv")

    # Alarms
    pytest.alarms_dir = os.path.join(pytest.datadir, "bedmaster_alarms")


pytest_configure()


class Utils:
    @staticmethod
    def build_hd5s(
        path: str,
        tensor_maps: List[TensorMap],
        n=5,
    ) -> Dict[Tuple[str, TensorMap], np.ndarray]:
        """
        Builds hd5s at path given TensorMaps. Only works for Continuous and
        Categorical TensorMaps.
        """
        out = {}
        for i in range(n):
            hd5_path = os.path.join(path, f"{i}{TENSOR_EXT}")
            with h5py.File(hd5_path, "w") as hd5:
                for tm in tensor_maps:
                    if tm.is_continuous:
                        value = np.full(tm.shape, fill_value=i, dtype=np.float32)
                    elif tm.is_categorical:
                        value = np.zeros(tm.shape, dtype=np.float32)
                        value[..., i % tm.shape[-1]] = 1
                    else:
                        raise NotImplementedError(
                            "Cannot automatically build hd5 from interpretation"
                            f' "{tm.interpretation}"',
                        )
                    hd5.create_dataset(f"/{tm.name}", data=value)
                    out[(hd5_path, tm)] = value
        return out


@pytest.fixture(scope="session")
def utils() -> Type[Utils]:
    return Utils


# The purpose of this fixture is to always use the fake testing TMaps.
# The function which retrieves tmaps is update_tmaps from TensorMap.py;
# However, that function is usually imported directly, i.e.
#
#     from ml4c3.TensorMap import update_tmaps
#
# This import creates a new object with the same name in the importing file,
# and now needs to be mocked too, e.g.
#
#     mock ml4c3.arguments.update_tmaps --> mock_update_tmaps
#
# https://stackoverflow.com/a/45466846
@pytest.fixture(autouse=True)
def use_testing_tmaps(monkeypatch):
    def mock_update_tmaps(tmap_name: str, tmaps: Dict[str, TensorMap]):
        return pytest.MOCK_TMAPS

    monkeypatch.setattr(tensormap.TensorMap, "update_tmaps", mock_update_tmaps)
    monkeypatch.setattr("ml4c3.arguments.update_tmaps", mock_update_tmaps)
    monkeypatch.setattr("ml4c3.hyperoptimizers.update_tmaps", mock_update_tmaps)


baseline_default_arguments = [
    "--input_tensors",
    "3d_cont",
    "--output_tensors",
    "1d_cat",
    "--conv_x",
    "3",
    "--conv_y",
    "3",
    "--conv_z",
    "3",
    "--pool_x",
    "1",
    "--pool_y",
    "1",
    "--pool_z",
    "1",
    "--num_workers",
    "1",
    "--epochs",
    "2",
    "--batch_size",
    "2",
    "--dense_layers",
    "4",
    "--conv_blocks",
    "4",
    "--conv_block_size",
    "3",
    "--optimizer",
    "adam",
    "--activation_layer",
    "relu",
    "--learning_rate",
    "0.001",
]


@pytest.fixture(scope="function")
def default_arguments(tmpdir_factory, utils: Utils) -> argparse.Namespace:
    temp_dir = tmpdir_factory.mktemp("data")
    utils.build_hd5s(temp_dir, pytest.MOCK_TMAPS.values(), n=pytest.N_TENSORS)
    hd5_dir = str(temp_dir)
    sys.argv = [
        ".",
        "train",
        "--tensors",
        hd5_dir,
        "--output_folder",
        hd5_dir,
    ]
    sys.argv.extend(baseline_default_arguments)
    args = parse_args()
    return args


@pytest.fixture(scope="function")
def default_arguments_infer(tmpdir_factory, utils: Utils) -> argparse.Namespace:
    temp_dir = tmpdir_factory.mktemp("data")
    utils.build_hd5s(temp_dir, pytest.MOCK_TMAPS.values(), n=pytest.N_TENSORS)
    hd5_dir = str(temp_dir)
    sys.argv = [
        ".",
        "infer",
        "--tensors",
        hd5_dir,
        "--output_folder",
        hd5_dir,
    ]
    sys.argv.extend(baseline_default_arguments)
    args = parse_args()
    return args


@pytest.fixture(scope="function")
def default_arguments_explore(tmpdir_factory, utils: Utils) -> argparse.Namespace:
    temp_dir = tmpdir_factory.mktemp("data")
    utils.build_hd5s(temp_dir, pytest.MOCK_TMAPS.values(), n=pytest.N_TENSORS)
    hd5_dir = str(temp_dir)
    sys.argv = [
        ".",
        "explore",
        "--tensors",
        hd5_dir,
        "--output_folder",
        hd5_dir,
    ]
    args = parse_args()
    return args


def pytest_exception_interact(node, call, report):
    for child in mp.active_children():
        child.terminate()


@pytest.yield_fixture(scope="function")
def matfile() -> h5py.File:
    with h5py.File(pytest.mat_file, "r") as mat_file:
        yield mat_file


@pytest.yield_fixture(scope="function")
def empty_matfile() -> h5py.File:
    with tempfile.NamedTemporaryFile(delete=False) as _file:
        with h5py.File(_file.name, "w") as mat_file:
            mat_file.create_group("vs")
            mat_file.create_group("wv")
        yield _file
    try:
        os.remove(_file.name)
    except OSError:
        pass


@pytest.yield_fixture(scope="module")
def temp_file():
    with tempfile.NamedTemporaryFile(delete=False) as _file:
        yield _file


@pytest.yield_fixture(scope="function")
def temp_dir():
    with tempfile.TemporaryDirectory() as _tmp_dir:
        yield _tmp_dir


@pytest.fixture(scope="session")
def test_scale_units() -> Dict[str, Dict[str, Union[int, float, str]]]:
    # fmt: off
    return {
        "CUFF": {"scaling_factor": 1, "units": "mmHg"},
        "HR": {"scaling_factor": 0.5, "units": "Bpm"},
        "I": {"scaling_factor": 0.0243, "units": "mV"},
        "II": {"scaling_factor": 0.0243, "units": "mV"},
        "V": {"scaling_factor": 0.0243, "units": "mV"},
        "SPO2": {"scaling_factor": 0.039, "units": "%"},
        "RR": {"scaling_factor": 0.078, "units": "UNKNOWN"},
        "VNT_PRES": {"scaling_factor": 1, "units": "UNKNOWN"},
        "VNT_FLOW": {"scaling_factor": 1, "units": "UNKNOWN"},
        "CO2": {"scaling_factor": 1, "units": "UNKNOWN"},
    }
    # fmt: on


class FakeSignal:
    """
    Mock signal objects for use in testing.
    """

    def __init__(self):
        self.today = datetime.date.today()

    @staticmethod
    def get_bedmaster_signal() -> BedmasterSignal:
        starting_time = int(time())
        sample_freq = 60
        duration_sec = 10
        n_points = duration_sec * sample_freq
        m_signal = BedmasterSignal(
            name="Some_signal",
            source="waveform",
            channel="ch10",
            value=np.array(np.random.randint(40, 100, n_points)),
            time=np.arange(starting_time, starting_time + duration_sec, 0.25),
            units="mmHg",
            sample_freq=np.array(
                [(sample_freq, 0), (120, n_points / 10)],
                dtype="float,int",
            ),
            scale_factor=np.random.randint(0, 5),
            time_corr_arr=np.packbits(np.random.randint(0, 2, 100).astype(np.bool)),
            samples_per_ts=np.array([15] * int(duration_sec / 0.25)),
        )
        return m_signal

    @staticmethod
    def get_static_data() -> StaticData:
        static_data = StaticData(
            department_id=np.array([1234, 12341]),
            department_nm=np.array(["BLAKE1", "BLAKE2"]).astype("S"),
            room_bed=np.array(["123 - 222", "456 - 333"]).astype("S"),
            move_time=np.array(["2021-05-15 06:47:00", "2021-05-25 06:47:00"]).astype(
                "S",
            ),
            weight=np.random.randint(50, 100),
            height=np.random.randint(150, 210) / 100,
            admin_type="testing",
            admin_date="1995-08-06 00:00:00.0000000",
            birth_date="1920-05-06 00:00:00.0000000",
            race=str(np.random.choice(["Asian", "Native American", "Black"])),
            sex=str(np.random.choice(["male", "female"])),
            end_date="2020-07-10 12:00:00.0000000",
            end_stay_type=str(np.random.choice(["discharge", "death"])),
            local_time=["UTC-4:00"],
            medical_hist=np.array(
                ["ID: 245324; NAME: Diabetes; COMMENTS: typeI; DATE: UNKNOWN"],
            ).astype("S"),
            surgical_hist=np.array(
                ["ID: 241324; NAME: VASECTOMY; COMMENTS: Sucessfully; DATE: UNKNOWN"],
            ).astype("S"),
            tobacco_hist="STATUS: Yes - Quit; COMMENT: 10 years ago",
            alcohol_hist="STATUS: Yes; COMMENT: a little",
            admin_diag="aortic valve repair.",
        )
        return static_data

    @staticmethod
    def get_measurement() -> Measurement:
        starting_time = int(time())
        measurement = Measurement(
            name="Some_Measurment",
            source=str(
                np.random.choice(
                    [
                        EDW_FILES["lab_file"]["source"],
                        EDW_FILES["vitals_file"]["source"],
                    ],
                ),
            ),
            value=np.array(np.random.randint(40, 100, 100)),
            time=np.array(list(range(starting_time, starting_time + 100))),
            units=str(np.random.choice(["mmHg", "bpm", "%"])),
            data_type=str(np.random.choice(["categorical", "numerical"])),
            metadata={"Some_Metadata": np.array(np.random.randint(0, 1, 100))},
        )
        return measurement

    @staticmethod
    def get_medication() -> Medication:
        starting_time = int(time())
        medication = Medication(
            name="Some_medication_in_g/ml",
            dose=np.array(np.random.randint(0, 2, 10)),
            units=str(np.random.choice(["g/ml", "mg", "pills"])),
            start_date=np.array(list(range(starting_time, starting_time + 100, 10))),
            action=np.random.choice(["Given", "New bag", "Rate Change"], 10).astype(
                "S",
            ),
            route=str(np.random.choice(["Oral", "Nasal", "Otic"])),
            wt_based_dose=bool(np.random.randint(0, 2)),
        )
        return medication

    @staticmethod
    def get_procedure() -> Procedure:
        starting_time = int(time())
        procedure = Procedure(
            name="Some_procedure",
            source=EDW_FILES["other_procedures_file"]["source"],
            start_date=np.array(list(range(starting_time, starting_time + 100, 5))),
            end_date=np.array(
                list(range(starting_time + 10000, starting_time + 10100, 5)),
            ),
        )
        return procedure

    @staticmethod
    def get_demo() -> StaticData:
        return FakeSignal.get_static_data()

    @staticmethod
    def get_measurements() -> Dict[str, Measurement]:
        starting_time = int(time())
        measurements_dic = {
            "creatinine": EDW_FILES["lab_file"]["source"],
            "ph_arterial": EDW_FILES["lab_file"]["source"],
            "pulse": EDW_FILES["vitals_file"]["source"],
            "r_phs_ob_bp_systolic_outgoing": EDW_FILES["vitals_file"]["source"],
        }
        measurements = {
            measurement_name: Measurement(
                name=measurement_name,
                source=f"{measurements_dic[measurement_name]}",
                value=np.array(np.random.randint(40, 100, 100)),
                time=np.array(list(range(starting_time, starting_time + 100))),
                units=str(np.random.choice(["mmHg", "bpm", "%"])),
                data_type=str(np.random.choice(["categorical", "numerical"])),
            )
            for measurement_name in measurements_dic
        }
        sys = np.random.randint(40, 100, 250)
        dias = np.random.randint(80, 160, 250)
        measurements["blood_pressure"] = Measurement(
            name="blood_pressure",
            source=EDW_FILES["vitals_file"]["source"],
            value=np.array(
                [f"{sys[i]}/{dias[i]}" for i in range(0, len(sys))],
                dtype="S",
            ),
            time=np.array(list(range(starting_time - 50000, starting_time, 200))),
            units="",
            data_type="categorical",
        )
        return measurements

    @staticmethod
    def get_procedures() -> Dict[str, Procedure]:
        starting_time = int(time())
        start_times = np.array(list(range(starting_time, starting_time + 1000, 100)))
        end_times = np.array(
            list(range(starting_time + 100, starting_time + 1100, 100)),
        )
        procedures_dic = {
            "colonoscopy": EDW_FILES["surgery_file"]["source"],
            "hemodialysis": EDW_FILES["other_procedures_file"]["source"],
            "transfuse_red_blood_cells": EDW_FILES["transfusions_file"]["source"],
        }
        procedures = {
            procedure_name: Procedure(
                name=procedure_name,
                source=f"{procedures_dic[procedure_name]}",
                start_date=start_times,
                end_date=end_times,
            )
            for procedure_name in procedures_dic
        }
        return procedures

    @staticmethod
    def get_medications() -> Dict[str, Medication]:
        starting_time = int(time())
        meds_list = [
            "aspirin_325_mg_tablet",
            "cefazolin_2_gram|50_ml_in_dextrose_(iso-osmotic)_intravenous_piggyback",
            "lactated_ringers_iv_bolus",
            "norepinephrine_infusion_syringe_in_swfi_80_mcg|ml_cmpd_central_mgh",
            "sodium_chloride_0.9_%_intravenous_solution",
            "aspirin_500_mg_tablet",
        ]
        medications = {
            med: Medication(
                name=med,
                dose=np.array(np.random.randint(0, 2, 10)),
                units=str(np.random.choice(["g/ml", "mg", "pills"])),
                start_date=np.array(
                    list(range(starting_time, starting_time + 100, 10)),
                ),
                action=np.random.choice(["Given", "New bag", "Rate Change"], 10).astype(
                    "S",
                ),
                route=str(np.random.choice(["Oral", "Nasal", "Otic"])),
                wt_based_dose=bool(np.random.randint(0, 2)),
            )
            for med in meds_list
        }
        return medications

    @staticmethod
    def get_events() -> Dict[str, Event]:
        starting_time = int(time())
        start_times = np.array(list(range(starting_time, starting_time + 1000, 100)))
        events_names = ["code_start", "rapid_response_start"]
        events = {
            event_name: Event(name=event_name, start_date=start_times)
            for event_name in events_names
        }
        return events

    @staticmethod
    def get_alarms() -> Dict[str, BedmasterAlarm]:
        starting_time = int(time())
        start_times = np.array(list(range(starting_time, starting_time + 1000, 100)))
        alarms_names = ["cpp_low", "v_tach", "apnea"]
        alarms = {
            alarm_name: BedmasterAlarm(
                name=alarm_name,
                start_date=start_times,
                duration=np.random.randint(0, 21, size=len(start_times)),
                level=np.random.randint(1, 6),
            )
            for alarm_name in alarms_names
        }
        return alarms

    @staticmethod
    def get_bedmaster_waveforms() -> Dict[str, BedmasterSignal]:
        starting_time = int(time())
        duration = 8
        sample_freq_1 = 240
        sample_freq_2 = 120
        times = np.arange(starting_time, starting_time + duration, 0.25)
        values1 = nk.ecg_simulate(
            duration=int(duration / 2),
            sampling_rate=sample_freq_1,
        )
        values2 = nk.ecg_simulate(
            duration=int(duration / 2),
            sampling_rate=sample_freq_2,
        )
        values = np.concatenate([values1, values2])
        n_samples = np.array(
            [sample_freq_1 * 0.25] * int(duration * 4 / 2)
            + [sample_freq_2 * 0.25] * int(duration * 4 / 2),
        )

        # Remove some samples
        values = np.delete(values, [10, 11, 250])
        n_samples[0] = 58
        n_samples[5] = 59

        leads = {
            lead: BedmasterSignal(
                name=lead,
                source="waveform",
                channel=f"ch{idx}",
                value=values,
                time=times,
                units="mV",
                sample_freq=np.array(
                    [(sample_freq_1, 0), (sample_freq_2, 16)],
                    dtype="float,int",
                ),
                scale_factor=np.random.uniform() * 5,
                time_corr_arr=np.packbits(np.random.randint(0, 2, 100).astype(np.bool)),
                samples_per_ts=n_samples,
            )
            for idx, lead in enumerate(["i", "ii", "iii", "v", "spo2"])
        }
        return leads

    @staticmethod
    def get_bedmaster_vitals() -> Dict[str, BedmasterSignal]:
        starting_time = int(time())
        times = np.array(list(range(starting_time, starting_time + 100)))
        signals = {
            signal: BedmasterSignal(
                name=signal,
                source="vitals",
                channel=signal,
                value=np.array(np.random.randint(40, 100, 100)),
                time=times,
                units="%" if "%" in signal else "",
                sample_freq=np.array([(0.5, 0)], dtype="float,int"),
                scale_factor=np.random.uniform() * 5,
                time_corr_arr=np.packbits(np.random.randint(0, 2, 100).astype(np.bool)),
                samples_per_ts=np.array([0.5] * len(times)),
            )
            for idx, signal in enumerate(["spo2%", "spo2r"])
        }
        return signals


@pytest.fixture(scope="module")
def fake_signal() -> FakeSignal:
    return FakeSignal()


@pytest.fixture(scope="module")
def hd5_data(
    temp_file,
    fake_signal: FakeSignal,
) -> Tuple[
    h5py.File,
    Dict[str, List[Union[str, ICUDataObject, Dict[str, ICUDataObject]]]],
]:
    visits = ["0123456789", "1111111111"]
    data: Dict[str, List[Union[str, ICUDataObject, Dict[str, ICUDataObject]]]] = {
        "measurements": [],
        "procedures": [],
        "meds": [],
        "demo": [],
        "signals": [],
        "events": [],
        "alarms": [],
        "visits": visits,
    }

    with Writer(temp_file.name) as writer:

        def _write(visit_id):
            measurements = fake_signal.get_measurements()
            data["measurements"].append(measurements)
            procedures = fake_signal.get_procedures()
            data["procedures"].append(procedures)
            meds = fake_signal.get_medications()
            data["meds"].append(meds)
            demo = fake_signal.get_demo()
            data["demo"].append(demo)
            waveforms = fake_signal.get_bedmaster_waveforms()
            vitals = fake_signal.get_bedmaster_vitals()
            data["signals"].append({**waveforms, **vitals})
            events = fake_signal.get_events()
            data["events"].append(events)
            alarms = fake_signal.get_alarms()
            data["alarms"].append(alarms)

            writer.set_visit_id(visit_id)
            writer.write_static_data(demo)
            for measurement in measurements:
                writer.write_signal(measurements[measurement])
            for procedure in procedures:
                writer.write_signal(procedures[procedure])
            for med in meds:
                writer.write_signal(meds[med])
            for waveform in waveforms.values():
                writer.write_signal(waveform)
            for vital in vitals.values():
                writer.write_signal(vital)
            for event in events:
                writer.write_signal(events[event])
            for alarm in alarms:
                writer.write_signal(alarms[alarm])

        for visit in visits:
            _write(visit)

    with h5py.File(temp_file, "r") as _hd5:
        yield _hd5, data


@pytest.fixture(scope="function")
def testing_tmaps() -> Dict[str, Dict[str, TensorMap]]:
    test_tmaps = {}

    bedmaster_tmap_test_names = ["i", "ii", "iii", "v", "spo2", "spo2%", "spo2r"]
    test_tmaps["bedmaster_signals"] = {
        name: GET_SIGNAL_TMAP(name)
        for signal in bedmaster_tmap_test_names
        for name in (
            f"{signal}_timeseries",
            f"{signal}_value",
            f"{signal}_samples_per_ts_timeseries",
            f"{signal}_samples_per_ts_value",
            f"{signal}_time_corr_arr_timeseries",
            f"{signal}_time_corr_arr_value",
            f"{signal}_sample_freq",
            f"{signal}_units",
            f"{signal}_scale_factor",
            f"{signal}_channel",
        )
    }
    alarms_tmap_test_names = ["cpp_low", "v_tach", "apnea"]
    test_tmaps["alarms"] = {
        name: GET_SIGNAL_TMAP(name)
        for name in [i + "_init_date" for i in alarms_tmap_test_names]
        + [i + "_duration" for i in alarms_tmap_test_names]
        + [i + "_level" for i in alarms_tmap_test_names]
    }
    meas_tmap_test_names = [
        "creatinine",
        "ph_arterial",
        "pulse",
        "blood_pressure_diastolic",
        "blood_pressure_systolic",
    ]
    test_tmaps["measurements"] = {
        name: GET_SIGNAL_TMAP(name)
        for signal in meas_tmap_test_names
        for name in (
            f"{signal}_timeseries",
            f"{signal}_value",
            f"{signal}_time",
            f"{signal}_units",
        )
    }
    meds_tmap_test_names = [
        "aspirin_325_mg_tablet",
        "cefazolin_2_gram|50_ml_in_dextrose_(iso-osmotic)_intravenous_piggyback",
        "lactated_ringers_iv_bolus",
        "norepinephrine_infusion_syringe_in_swfi_80_mcg|ml_cmpd_central_mgh",
        "sodium_chloride_0.9_%_intravenous_solution",
    ]
    test_tmaps["meds"] = {
        name: GET_SIGNAL_TMAP(name)
        for signal in meds_tmap_test_names
        for name in (
            f"{signal}_timeseries",
            f"{signal}_dose",
            f"{signal}_time",
            f"{signal}_units",
            f"{signal}_route",
        )
    }
    events_tmap_test_names = [
        "code_start",
        "rapid_response_start",
    ]
    test_tmaps["events"] = {
        name: GET_SIGNAL_TMAP(name)
        for signal in events_tmap_test_names
        for name in (f"{signal}_start_date", f"{signal}_double", f"{signal}_single")
    }
    proce_tmap_test_names = ["colonoscopy", "hemodialysis", "transfuse_red_blood_cells"]
    test_tmaps["procedures"] = {
        name: GET_SIGNAL_TMAP(name)
        for signal in proce_tmap_test_names
        for name in (
            f"{signal}_start_date",
            f"{signal}_end_date",
            f"{signal}_double",
            f"{signal}_single",
        )
    }
    list_signals_tmap_test_names = [
        "bedmaster_waveform_signals",
        "bedmaster_vitals_signals",
        "bedmaster_alarms_signals",
        "edw_med_signals",
    ]
    test_tmaps["list_signals"] = {
        name: GET_LIST_TMAP(name) for name in list_signals_tmap_test_names
    }

    return test_tmaps


@pytest.fixture(scope="function")
def test_get_local_time() -> Callable[[np.ndarray], np.ndarray]:
    def _get_local_time(time_array):
        get_function = get_local_timestamps(time_array)
        return get_function

    return _get_local_time
