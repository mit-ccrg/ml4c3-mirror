# Imports: standard library
import os
import math

# Imports: third party
import numpy as np
import pytest

# Imports: first party
from ml4c3.definitions.icu import EDW_FILES
from ml4c3.ingest.icu.readers import EDWReader

# pylint: disable=no-member


@pytest.fixture(scope="function")
def edw_reader() -> EDWReader:
    reader = EDWReader(pytest.edw_dir, "123", "345")
    return reader


def test_infer_full_path(edw_reader: EDWReader):
    expected_path = os.path.join(pytest.edw_patient_dir, "flowsheet.csv")
    assert expected_path == edw_reader.vitals_file


def test_infer_full_path2():
    reader = EDWReader(
        pytest.edw_dir,
        pytest.example_mrn,
        pytest.example_visit_id,
        vitals_file="flowsheet_v2",
    )
    expected_path = os.path.join(pytest.edw_patient_dir, "flowsheet_v2.csv")
    assert expected_path == reader.vitals_file


def test_list_vitals(edw_reader: EDWReader):
    expected_vitals = [
        "BLOOD PRESSURE",
        "PULSE",
        "R PHS OB PULSE OXIMETRY OUTGOING",
        "R PHS OB BP SYSTOLIC OUTGOING",
        "R PHS OB BP DIASTOLIC OUTGOING",
    ]
    assert sorted(edw_reader.list_vitals()) == sorted(expected_vitals)


def test_list_labs(edw_reader: EDWReader):
    expected_labs = [
        "PH, ARTERIAL",
        "LACTATE, BLOOD",
        "MAGNESIUM",
        "CREATININE",
    ]
    assert sorted(edw_reader.list_labs()) == sorted(expected_labs)


def test_list_medications(edw_reader: EDWReader):
    expected_medications = [
        "NOREPINEPHRINE INFUSION SYRINGE IN SWFI 80 MCG/ML CMPD CENTRAL_MGH",
        "SODIUM CHLORIDE 0.9 % INTRAVENOUS SOLUTION",
        "LACTATED RINGERS IV BOLUS",
        "ASPIRIN 325 MG TABLET",
        "CEFAZOLIN 2 GRAM/50 ML IN DEXTROSE (ISO-OSMOTIC) INTRAVENOUS PIGGYBACK",
    ]
    assert sorted(edw_reader.list_medications()) == sorted(expected_medications)


def test_list_surgery(edw_reader: EDWReader):
    expected_surgery = [
        "COLONOSCOPY",
        "CORONARY ARTERY BYPASS GRAFT",
    ]
    assert sorted(edw_reader.list_surgery()) == sorted(expected_surgery)


def test_list_other_procedures(edw_reader: EDWReader):
    expected_procedures = ["HEMODIALYSIS", "HEMODIALYSIS / ULTRAFILTRATION"]
    assert edw_reader.list_other_procedures() == sorted(expected_procedures)


def test_list_transfusions(edw_reader: EDWReader):
    expected_transfusions = [
        "TRANSFUSE RED BLOOD CELLS",
        "TRANSFUSE PLATELETS",
        "TRANSFUSE CRYOPRECIPITATE",
    ]
    assert sorted(edw_reader.list_transfusions()) == sorted(expected_transfusions)


def test_list_events(edw_reader: EDWReader):
    expected_events = ["CODE START", "RAPID RESPONSE START"]
    assert sorted(edw_reader.list_events()) == sorted(expected_events)


def test_get_static_data(edw_reader: EDWReader):
    static_info = edw_reader.get_static_data()
    expected_height = float("5") * 0.3048 + float("2.156") * 0.0254
    assert static_info.height == expected_height
    assert static_info.weight == 1380.0 / 16
    assert static_info.admin_type == "Urgent"
    assert static_info.admin_date == "2015-02-12 06:59:00.0000000"
    assert static_info.birth_date == "1941-06-14 00:00:00.0000000"
    assert static_info.race == "White or Caucasian"
    assert static_info.sex == "Female"
    assert static_info.end_date == "2015-02-12 07:01:15.0000000"
    assert static_info.end_stay_type == "Deceased"
    expected_department_id = np.array([11011, 12121, 13334])
    assert sorted(static_info.department_id) == sorted(expected_department_id)
    expected_department_nm = np.array(
        ["MGH BLAKE 8 CARD SICU", "MGH ELLISON 8 CARDSURG", "MGH ELLISON 8 CARDSURG"],
    ).astype("S")
    assert sorted(static_info.department_nm) == sorted(expected_department_nm)
    expected_room_bed = np.array(["P0860", "812 B", "E0810 A"]).astype("S")
    assert sorted(static_info.room_bed) == sorted(expected_room_bed)
    expected_move_time = np.array(
        [
            "2020-04-20 06:47:00.0000000",
            "2020-04-22 22:57:00.0000000",
            "2020-05-05 22:47:00.0000000",
        ],
    ).astype("S")
    assert sorted(static_info.move_time) == sorted(expected_move_time)
    assert static_info.local_time == np.array(["UTC-5:00"]).astype("S")
    expected_medical_hist = np.array(
        [
            "ID: 3456345; DESCRIPTION: Umbilical hernia without obstruction and "
            "without gangrene; COMMENTS: UNKNOWN; DATE: 12/4/2019",
            "ID: 67364; DESCRIPTION: History of shingles; "
            "COMMENTS: involving neck; DATE: UNKNOWN",
        ],
    ).astype("S")
    assert np.array_equal(static_info.medical_hist, expected_medical_hist)
    expected_surgical_hist = np.array(
        [
            "ID: 3413; DESCRIPTION: UNKNOWN; COMMENTS: Vasectomy; DATE: UNKNOWN",
            "ID: 34232; DESCRIPTION: UMBILICAL HERNIA REPAIR; COMMENTS: "
            "successful repair 2004 - recurred and repaired 11/2016; DATE: UNKNOWN",
            "ID: 23423123; DESCRIPTION: VASECTOMY; COMMENTS: UNKNOWN; DATE: UNKNOWN",
        ],
    ).astype("S")
    assert np.array_equal(static_info.surgical_hist, expected_surgical_hist)
    expected_tobacco_hist = (
        "STATUS: Quit - Yes; COMMENTS: quit 32 yrs ago - smoked for 6 yrs"
    )
    assert static_info.tobacco_hist == expected_tobacco_hist
    expected_alcohol_hist = (
        "STATUS: NONE; COMMENTS: none for past 2 wks "
        " - occasional beer or shot of liquor"
    )
    assert static_info.alcohol_hist == expected_alcohol_hist
    expected_admin_diagnosis = "cad ; NSTEMI"
    assert static_info.admin_diag == expected_admin_diagnosis


def test_get_vitals(edw_reader: EDWReader):
    vitals1 = edw_reader.get_vitals("PULSE")
    vitals2 = edw_reader.get_vitals("R PHS OB BP SYSTOLIC OUTGOING")

    assert vitals1.name == "PULSE"
    assert vitals2.name == "R PHS OB BP SYSTOLIC OUTGOING"

    assert vitals1.source == EDW_FILES["vitals_file"]["source"]
    assert vitals2.source == EDW_FILES["vitals_file"]["source"]

    assert vitals1.data_type == "Numerical"
    assert vitals2.data_type == "Numerical"

    assert vitals1.units == "bpm"
    assert vitals2.units == "mmHg"

    assert hasattr(vitals1, "metadata") and vitals1.metadata is not None
    assert hasattr(vitals2, "metadata") and vitals1.metadata is not None

    expected_time1 = np.array(
        [1423742340.0, 1423742360.0, 1423742380.0, 1423742400.0],
        dtype=float,
    )
    expected_time2 = np.array(
        [1423742350.0, 1423742370.0, 1423742390.0, 1423742410.0],
        dtype=float,
    )
    assert np.array_equal(vitals1.time, expected_time1)
    assert np.array_equal(vitals2.time, expected_time2)

    expected_vals1 = np.array(["62a", 73, 60, 64], dtype="S")
    expected_vals2 = np.array([150, 124, 140, 132])
    assert np.array_equal(vitals1.value, expected_vals1)
    assert np.array_equal(vitals2.value, expected_vals2)


def test_get_labs(edw_reader: EDWReader):
    labs1 = edw_reader.get_labs("PH, ARTERIAL")
    labs2 = edw_reader.get_labs("CREATININE")

    assert labs1.name == "PH, ARTERIAL"
    assert labs2.name == "CREATININE"

    assert labs1.source == EDW_FILES["lab_file"]["source"]
    assert labs2.source == EDW_FILES["lab_file"]["source"]

    assert labs1.data_type == "Numerical"
    assert labs2.data_type == "Numerical"

    assert math.isnan(labs1.units)
    assert labs2.units == "mg/dL"

    assert hasattr(labs1, "metadata") and labs1.metadata is not None
    assert hasattr(labs2, "metadata") and labs2.metadata is not None

    expected_time1 = np.array(
        [1620928800.0, 1620933240.0, 1620940320.0, 1620955080.0, 1620966000.0],
        dtype=float,
    )
    expected_time2 = np.array(
        [1620669060.0, 1620686340.0, 1620689400.0, 1620802020.0, 1620966120.0],
        dtype=float,
    )
    assert np.array_equal(labs1.time, expected_time1)
    assert np.array_equal(labs2.time, expected_time2)

    expected_vals1 = np.array([7.26, 7.45, 7.48, 7.27, 7.14])
    expected_vals2 = np.array([1.94, 2.09, 1.11, 1.43, 4.5])
    assert np.array_equal(labs1.value, expected_vals1)
    assert np.array_equal(labs2.value, expected_vals2)


def test_get_med_doses(edw_reader: EDWReader):
    med1 = edw_reader.get_med_doses("ASPIRIN 325 MG TABLET")
    med2 = edw_reader.get_med_doses("LACTATED RINGERS IV BOLUS")
    med3 = edw_reader.get_med_doses(
        "NOREPINEPHRINE INFUSION SYRINGE IN SWFI 80 MCG/ML CMPD CENTRAL_MGH",
    )
    med4 = edw_reader.get_med_doses(
        "CEFAZOLIN 2 GRAM/50 ML IN DEXTROSE (ISO-OSMOTIC) INTRAVENOUS PIGGYBACK",
    )
    assert med1.name == "ASPIRIN 325 MG TABLET"
    assert med2.source == EDW_FILES["med_file"]["source"]
    assert med1.route == "Oral"
    assert med4.route == "Intravenous"
    assert np.array_equal(med3.dose, np.array([0.75, 0.75, 0]))
    assert np.array_equal(med2.dose, np.array([1000, 0, 1000, 0, 1000, 0]))
    assert med1.units == "mg"
    assert med2.units == "mL/hr"
    assert med3.units == "mL/hr"
    assert med4.units == "g"
    assert np.array_equal(
        med2.start_date,
        np.array(
            [
                323901120.0,
                323902920.0,
                323979300.0,
                323981100.0,
                324066600.0,
                324068400.0,
            ],
            dtype=float,
        ),
    )
    assert not med1.wt_based_dose
    assert np.array_equal(
        med3.action,
        np.array(["Rate Verify", "Rate Verify", "Stopped"], dtype="S"),
    )


def test_get_surgery(edw_reader: EDWReader):
    surgery1 = edw_reader.get_surgery("COLONOSCOPY")
    surgery2 = edw_reader.get_surgery("CORONARY ARTERY BYPASS GRAFT")
    assert surgery1.name == "COLONOSCOPY"
    assert surgery2.name == "CORONARY ARTERY BYPASS GRAFT"
    assert surgery1.source == EDW_FILES["surgery_file"]["source"]
    assert surgery2.source == EDW_FILES["surgery_file"]["source"]
    assert np.array_equal(
        surgery1.start_date,
        np.array([323460900.0, 323640900.0, 323730900.0], dtype=float),
    )
    assert np.array_equal(
        surgery1.end_date,
        np.array([323475300.0, 323658900.0, 323748900.0], dtype=float),
    )
    assert np.array_equal(surgery2.start_date, np.array([323838900.0], dtype=float))
    assert np.array_equal(surgery2.end_date, np.array([323860500.0], dtype=float))


def test_get_other_procedures(edw_reader: EDWReader):
    procedure1 = edw_reader.get_other_procedures("HEMODIALYSIS")
    procedure2 = edw_reader.get_other_procedures("HEMODIALYSIS / ULTRAFILTRATION")
    assert procedure1.name == "HEMODIALYSIS"
    assert procedure2.name == "HEMODIALYSIS / ULTRAFILTRATION"
    assert procedure1.source == EDW_FILES["other_procedures_file"]["source"]
    assert procedure2.source == EDW_FILES["other_procedures_file"]["source"]
    assert np.array_equal(
        procedure1.start_date,
        np.array([323460900.0, 323703900.0, 323838900.0, 323979300.0], dtype=float),
    )
    assert np.array_equal(
        procedure1.end_date,
        np.array([323475300.0, 323720100.0, 323860500.0, 323993760.0], dtype=float),
    )
    assert np.array_equal(
        procedure2.start_date,
        np.array([323548200.0, 323647200.0, 323987520.0], dtype=float),
    )
    assert np.array_equal(
        procedure2.end_date,
        np.array([323567400.0, 323666100.0, 324089820.0], dtype=float),
    )


def test_get_transfusions(edw_reader: EDWReader):
    transfuse1 = edw_reader.get_transfusions("TRANSFUSE RED BLOOD CELLS")
    transfuse2 = edw_reader.get_transfusions("TRANSFUSE PLATELETS")
    assert transfuse1.name == "TRANSFUSE RED BLOOD CELLS"
    assert transfuse2.name == "TRANSFUSE PLATELETS"
    assert transfuse1.source == EDW_FILES["transfusions_file"]["source"]
    assert transfuse2.source == EDW_FILES["transfusions_file"]["source"]
    assert np.array_equal(
        transfuse1.start_date,
        np.array([323640900.0, 323730900.0], dtype=float),
    )
    assert np.array_equal(
        transfuse1.end_date,
        np.array([323658900.0, 323748900.0], dtype=float),
    )
    assert np.array_equal(transfuse2.start_date, np.array([323548200.0], dtype=float))
    assert np.array_equal(transfuse2.end_date, np.array([323561700.0], dtype=float))


def test_get_events(edw_reader: EDWReader):
    event1_name = "CODE START"
    event2_name = "RAPID RESPONSE START"
    event1 = edw_reader.get_events(event1_name)
    event2 = edw_reader.get_events(event2_name)
    assert event1.name == event1_name
    assert event2.name == event2_name
    assert event1.source == EDW_FILES["events_file"]["source"]
    assert event2.source == EDW_FILES["events_file"]["source"]
    assert np.array_equal(
        event1.start_date,
        np.array(
            [
                1270945960,
                1271377960,
                1271809960,
                1272155560,
                1272501160,
                1272673960,
                1272933160,
                1273537960,
                1273969960,
                1274401960,
                1274833960,
            ],
            dtype=float,
        ),
    )
    assert np.array_equal(
        event2.start_date,
        np.array(
            [1271896360, 1272673960, 1273278760, 1274142760, 1275006760],
            dtype=float,
        ),
    )


def test_contiguous_nparrays(edw_reader: EDWReader):
    med1 = edw_reader.get_med_doses("ASPIRIN 325 MG TABLET")
    med2 = edw_reader.get_med_doses("LACTATED RINGERS IV BOLUS")
    med3 = edw_reader.get_med_doses(
        "NOREPINEPHRINE INFUSION SYRINGE IN SWFI 80 MCG/ML CMPD CENTRAL_MGH",
    )
    med4 = edw_reader.get_med_doses(
        "CEFAZOLIN 2 GRAM/50 ML IN DEXTROSE (ISO-OSMOTIC) INTRAVENOUS PIGGYBACK",
    )
    meds = [med1, med2, med3, med4]
    vitals1 = edw_reader.get_vitals("PULSE")
    vitals2 = edw_reader.get_vitals("BLOOD PRESSURE")
    labs = edw_reader.get_labs("PH, ARTERIAL")
    measurements = [vitals1, vitals2, labs]
    surgery = edw_reader.get_surgery("COLONOSCOPY")
    other_procedure = edw_reader.get_other_procedures("HEMODIALYSIS")
    transfuse = edw_reader.get_transfusions("TRANSFUSE RED BLOOD CELLS")
    procedures = [surgery, other_procedure, transfuse]
    for med in meds:
        assert not med.dose.dtype == object
        assert not med.start_date.dtype == object
        assert not med.action.dtype == object
    for measurement in measurements:
        assert not measurement.value.dtype == object
        assert not measurement.time.dtype == object
    for procedure in procedures:
        assert not procedure.start_date.dtype == object
        assert not procedure.end_date.dtype == object
