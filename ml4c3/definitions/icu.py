# Imports: standard library
import os
from typing import Any, Dict, List

MAD3_DIR = "/media/mad3"
LM4_DIR = "/media/lm4-bedmaster"

EDW_PREFIX = "edw"
BEDMASTER_PREFIX = "bedmaster"

MATFILE_EXPECTED_GROUPS: List[str] = [
    "vs",
    "vs_time_corrected",
    "vs_packet",
    "vs_seq",
    "vs_time_corrected",
    "vs_time_original",
    "wv",
    "wv_time_corrected",
    "wv_time_original",
]

BM_SOURCES = {
    "vitals": "BM_vitals",
    "waveform": "BM_waveform",
    "ecg_features": "BM_ecg_features",
}

# TODO list more departments names https://github.com/aguirre-lab/ml4c3/issues/575
MAPPING_DEPARTMENTS: Dict[str, List[Any]] = {
    # EDW dept name: [possible file name]
    "MGH BIGELOW 6 PICU": ["BIG06"],
    "MGH BIGELOW 7": ["BIG7"],
    "MGH BIGELOW 9 MED": ["BIG09", "BIG9", "BIG09PU"],
    "MGH BIGELOW 11 MED": ["BIG11"],
    "MGH BIGELOW 12": ["BIG12"],
    "MGH BIGELOW 13 RACU": ["BIG13"],
    "MGH BIGELOW 14 MED": ["BIG14"],
    "MGH BLAKE 4 ENDO DEPT": [None],
    "MGH BLAKE 6 TRANSPLANT": ["BLAKE6"],
    "MGH BLAKE 7 MICU": ["BLK07"],
    "MGH BLAKE 8 CARD SICU": ["BLK08"],
    "MGH BLAKE 10 NICU": ["BLK10"],
    "MGH BLAKE 12 ICU": ["BLK12"],
    "MGH BLAKE 13 OB": ["BLK13"],
    "MGH ELLISON 4 SICU": ["ELL04", "ELL4"],
    "MGH ELLISON 6 ORTH\\URO": ["ELL6"],
    "MGH ELLISON 7 SURG\\URO": ["ELL-7"],
    "MGH ELLISON 8 CARDSURG": ["ELL08", "ELL8"],
    "MGH ELLISON 9 MED\\CCU": ["ELL09", "ELL9"],
    "MGH ELLISON 10 STP DWN": ["ELL10"],
    "MGH ELLISON11 CARD\\INT": ["ELL11"],
    "MGH ELLISON 12 MED": ["ELL12"],
    "MGH ELLISON13A OB-ANTE": ["ELL13"],
    "MGH ELLISON 14 BRN ICU": ["ELL14"],
    "MGH ELLISON 14 PLASTCS": ["ELL14"],
    "MGH ELLISON 16 MED ONC": ["ELL16"],
    "MGH ELLISON 17 PEDI": ["ELL17"],
    "MGH ELLISON 18 PEDI": ["ELL18"],
    "MGH ELLISON19 THOR/VAS": ["ELL19"],
    "MGH LUNDER 6 NEURO ICU": ["LUN06", "LUN06IM"],
    "MGH LUNDER 7 NEURO": ["LUN07"],
    "MGH LUNDER 8 NEURO": ["LUN08"],
    "MGH LUNDER 9 ONCOLOGY": ["LUN09"],
    "MGH LUNDER 10 ONCOLOGY": ["LUN10"],
    "MGH WHITE 6 ORTHO\\OMF": ["WHT06"],
    "MGH WHITE 7 GEN SURG": ["WHT07"],
    "MGH WHITE 8 MEDICINE": ["WHT08"],
    "MGH WHITE 9 MEDICINE": ["WHT09"],
    "MGH WHITE 10 MEDICINE": ["WHITE10"],
    "MGH WHITE 11 MEDICINE": ["WHITE11"],
    "MGH WHITE 12": ["WHT12"],
    "MGH WHITE 13 PACU": ["WHT13"],
    "MGH CARDIAC CATH LAB": [None],
    "MGH EMERGENCY": [None],
    "MGH PERIOPERATIVE DEPT": [None],
    "MGH EP PACER LAB": [None],
    "MGH PHILLIPS 20 MED": [None],
    "MGH CPC": [None],
}

ALARMS_FILES: Dict[str, Any] = {
    "columns": [
        "UnitBedUID",
        "AlarmStartTime",
        "AlarmLevel",
        "AlarmMessage",
        "AlarmDuration",
        "PatientID",
        "Bed",
        "Unit",
    ],
    "names": MAPPING_DEPARTMENTS,
    "source": "BM_alarms",
}
EDW_FILES: Dict[str, Dict[str, Any]] = {
    "adt_file": {
        "name": "adt.csv",
        "columns": [
            "MRN",
            "PatientEncounterID",
            "BedLabelNM",
            "TransferInDTS",
            "TransferOutDTS",
            "DepartmentID",
            "DepartmentDSC",
        ],
    },
    "move_file": {
        "name": "movements.csv",
        "source": "EDW_static",
        "columns": ["TransferInDTS", "DepartmentID", "DepartmentDSC", "BedLabelNM"],
    },
    "adm_file": {
        "name": "admission-vitals.csv",
        "source": "EDW_static",
        "columns": [
            "WeightPoundNBR",
            "HeightTXT",
            "HospitalAdmitTypeDSC",
            "HospitalAdmitDTS",
            "HospitalDischargeDTS",
            "AdmitDiagnosisTXT",
        ],
    },
    "demo_file": {
        "name": "demographics.csv",
        "source": "EDW_static",
        "columns": ["BirthDTS", "PatientRaceDSC", "SexDSC", "DeathDTS"],
    },
    "med_file": {
        "name": "medications.csv",
        "source": "EDW_med",
        "columns": [
            "MedicationDSC",  # signal_column
            "MARActionDSC",  # status_column
            "MedicationTakenDTS",  # time_column
            "RouteDSC",  # route_column
            "PatientWeightBasedDoseFLG",  # weight_column
            "DiscreteDoseAMT",  # dose_column
            "DoseUnitDSC",  # dose_unit_column
            "InfusionRateNBR",  # infusion_column
            "InfusionRateUnitDSC",  # infusion_units_columns
            "DurationNBR",  # duration_column
            "DurationUnitDSC",  # duration_unit_column
        ],
    },
    "vitals_file": {
        "name": "flowsheet.csv",
        "source": "EDW_flowsheet",
        "columns": [
            "FlowsheetMeasureNM",  # signal_column
            "MeasureTXT",  # result_column
            "UnitsCD",  # units_column
            "RecordedDTS",  # time_column
        ],
    },
    "lab_file": {
        "name": "labs.csv",
        "source": "EDW_labs",
        "columns": [
            "ComponentCommonNM",  # signal_column
            "ResultTXT",  # result_column
            "ReferenceRangeUnitCD",  # units_column
            "SpecimenTakenTimeDTS",  # time_column
        ],
    },
    "surgery_file": {
        "name": "surgery.csv",
        "source": "EDW_surgery",
        "columns": [
            "ProcedureNM",  # signal_column
            "ProgressDSC",  # status_column
            "BeginDTS",  # start_column
            "EndDTS",  # end_column
        ],
    },
    "other_procedures_file": {
        "name": "procedures.csv",
        "source": "EDW_procedures",
        "columns": [
            "ProcedureDSC",  # signal_column
            "OrderStatusDSC",  # status_column
            "StartDTS",  # start_column
            "EndDTS",  # end_column
        ],
    },
    "transfusions_file": {
        "name": "transfusions.csv",
        "source": "EDW_transfusions",
        "columns": [
            "ProcedureDSC",  # signal_column
            "OrderStatusDSC",  # status_column
            "BloodStartInstantDTS",  # start_column
            "BloodEndInstantDTS",  # end_column
        ],
    },
    "events_file": {
        "name": "events.csv",
        "source": "EDW_events",
        "columns": ["EventNM", "EventDTS"],  # signal_column  # time_column
    },
    "medhist_file": {
        "name": "medical-history.csv",
        "source": "EDW_static",
        "columns": [
            "DiagnosisID",  # id_column
            "DiagnosisNM",  # name_column
            "CommentTXT",  # comment_column
            "MedicalHistoryDateTXT",  # date_column
        ],
    },
    "surghist_file": {
        "name": "surgical-history.csv",
        "source": "EDW_static",
        "columns": [
            "ProcedureID",  # id_column
            "ProcedureNM",  # name_column
            "CommentTXT",  # comment_column
            "SurgicalHistoryDateTXT",  # date_column
        ],
    },
    "socialhist_file": {
        "name": "social-history.csv",
        "source": "EDW_static",
        "columns": [
            "TobaccoUserDSC",  # stat_tob_column
            "TobaccoCommentTXT",  # comment_tob_column
            "AlcoholUseDSC",  # stat_alc_column
            "AlcoholCommentTXT",  # comment_alc_column
        ],
    },
    "diagnosis_file": {"name": "diagnosis.csv", "source": "", "columns": []},
}

MED_ACTIONS = [
    "Given",
    "Restarted",
    "Stopped",
    "New Bag",
    "Same Bag",
    "Rate Change",
    "Rate Verify",
]

ML4ICU_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(ML4ICU_PATH)
VISUALIZER_PATH = os.path.join(ML4ICU_PATH, "visualizer")

STATIC_UNITS = {
    "height": "m",
    "weight": "lbs",
}

# fmt: off
ICU_SCALE_UNITS = {
    "I":         {"scaling_factor": 0.0243, "units": "mV"},
    "II":        {"scaling_factor": 0.0243, "units": "mV"},
    "III":       {"scaling_factor": 0.0243, "units": "mV"},
    "V":         {"scaling_factor": 0.0243, "units": "mV"},
    "SPO2":      {"scaling_factor": 0.039,  "units": "%"},
    "RR":        {"scaling_factor": 0.078,  "units": "ohms"},
    "CO2":       {"scaling_factor": 0.25,   "units": "mmHg"},
    "AVR":       {"scaling_factor": 0.0243, "units": "mV"},
    "AVF":       {"scaling_factor": 0.0243, "units": "mV"},
    "AVL":       {"scaling_factor": 0.0243, "units": "mV"},
    "ART1":      {"scaling_factor": 0.2,    "units": "mmHg"},
    "ART6":      {"scaling_factor": 0.2,    "units": "mmHg"},
    "FEM4":      {"scaling_factor": 0.2,    "units": "mmHg"},
    "PA2":       {"scaling_factor": 0.2,    "units": "mmHg"},
    "CVP3":      {"scaling_factor": 0.2,    "units": "mmHg"},
    "CUFF":      {"scaling_factor": 1,      "units": "mmHg"},
    "HR":        {"scaling_factor": 1,      "units": "Bpm"},
    "PVC":       {"scaling_factor": 1,      "units": "Bpm"},
    "ART1S":     {"scaling_factor": 1,      "units": "mmHg"},
    "ART1D":     {"scaling_factor": 1,      "units": "mmHg"},
    "ART1M":     {"scaling_factor": 1,      "units": "mmHg"},
    "ART1R":     {"scaling_factor": 1,      "units": "mmHg"},
    "ART6S":     {"scaling_factor": 1,      "units": "mmHg"},
    "ART6D":     {"scaling_factor": 1,      "units": "mmHg"},
    "ART6M":     {"scaling_factor": 1,      "units": "mmHg"},
    "ART6R":     {"scaling_factor": 1,      "units": "mmHg"},
    "FEM4S":     {"scaling_factor": 1,      "units": "mmHg"},
    "FEM4D":     {"scaling_factor": 1,      "units": "mmHg"},
    "FEM4M":     {"scaling_factor": 1,      "units": "mmHg"},
    "FEM4R":     {"scaling_factor": 1,      "units": "mmHg"},
    "ASBRAMP":   {"scaling_factor": 0.1,    "units": "sec"},
    "BT":        {"scaling_factor": 0.1,    "units": "Celcius"},
    "IT":        {"scaling_factor": 0.1,    "units": "Celcius"},
    "CO":        {"scaling_factor": 0.1,    "units": "L/min"},
    "CO2RR":     {"scaling_factor": 1,      "units": "BrMin"},
    "DYNRES":    {"scaling_factor": 0.1,    "units": "cmH2O/L/sec"},
    "FLWR":      {"scaling_factor": 1,      "units": "L/min"},
    "FLWTRIG":   {"scaling_factor": 1,      "units": "L/min"},
    "I:E":       {"scaling_factor": 1,      "units": "Ratio"},
    "INSPMEAS":  {"scaling_factor": 0.01,   "units": "sec"},
    "INSPTM":    {"scaling_factor": 0.01,   "units": "sec"},
    "IN_HLD":    {"scaling_factor": 0.1,    "units": "sec"},
    "MAWP":      {"scaling_factor": 1,      "units": "cmH2O"},
    "MV":        {"scaling_factor": 0.1,    "units": "L/min"},
    "PA2D":      {"scaling_factor": 1,      "units": "mmHg"},
    "PA2M":      {"scaling_factor": 1,      "units": "mmHg"},
    "PA2S":      {"scaling_factor": 1,      "units": "mmHg"},
    "PEEP":      {"scaling_factor": 1,      "units": "cmH2O"},
    "PIP":       {"scaling_factor": 1,      "units": "cmH2O"},
    "PPLAT":     {"scaling_factor": 1,      "units": "cmH2O"},
    "PRSSUP":    {"scaling_factor": 1,      "units": "cmH2O"},
    "PTRR":      {"scaling_factor": 1,      "units": "BrMin"},
    "RESP":      {"scaling_factor": 1,      "units": "BrMin"},
    "SENS":      {"scaling_factor": 0.1,    "units": "cmH20"},
    "SETFIO2":   {"scaling_factor": 1,      "units": "%"},
    "SETIE":     {"scaling_factor": 1,      "units": "ratio"},
    "SETPCP":    {"scaling_factor": 1,      "units": "cmH2O"},
    "SETTV":     {"scaling_factor": 1,      "units": "ml"},
    "SPO2%":     {"scaling_factor": 1,      "units": "%"},
    "SPO2R":     {"scaling_factor": 1,      "units": "Bpm"},
    "SPONTMV":   {"scaling_factor": 0.1,    "units": "L/min"},
    "STATRES":   {"scaling_factor": 0.1,    "units": "cmH20/L/Sec"},
    "TV":        {"scaling_factor": 1,      "units": "ml"},
    "Vent Rate": {"scaling_factor": 1,      "units": "BrMin"},
    "NBPM":      {"scaling_factor": 1,      "units": "mmHg"},
    "NBPS":      {"scaling_factor": 1,      "units": "mmHg"},
    "NBPD":      {"scaling_factor": 1,      "units": "mmHg"},
    "TMP1":      {"scaling_factor": 0.1,    "units": "Celcius"},
    "TMP2":      {"scaling_factor": 0.1,    "units": "Celcius"},
}
# fmt: on

ICU_TMAPS_METADATA: Dict[str, Dict[str, Any]] = {
    #    "TMAP_NAME": {"min": 0, "max": 0, "std": 0, "mean": 0, "median": 0, "iqr": 0}
}
