# Imports: standard library
from typing import Any, Dict

EDW_PREFIX = "edw"
EDW_EXT = ".csv"

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
        "source": "static",
        "columns": ["TransferInDTS", "DepartmentID", "DepartmentDSC", "BedLabelNM"],
    },
    "adm_file": {
        "name": "admission-vitals.csv",
        "source": "static",
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
        "source": "static",
        "columns": ["BirthDTS", "PatientRaceDSC", "SexDSC", "DeathDTS"],
    },
    "med_file": {
        "name": "medications.csv",
        "source": "med",
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
        "source": "flowsheet",
        "columns": [
            "FlowsheetMeasureNM",  # signal_column
            "MeasureTXT",  # result_column
            "UnitsCD",  # units_column
            "RecordedDTS",  # time_column
            ["EntryTimeDTS"],  # additional_columns
        ],
    },
    "lab_file": {
        "name": "labs.csv",
        "source": "labs",
        "columns": [
            "ComponentCommonNM",  # signal_column
            "ResultTXT",  # result_column
            "ReferenceRangeUnitCD",  # units_column
            "SpecimenTakenTimeDTS",  # time_column
            [
                "OrderDTS",
                "StartDTS",
                "EndDTS",
                "ResultDTS",
                "ComponentObservedDTS",
                "SpecimenReceivedTimeDTS",
                "OrderDisplayNM",
                "ProcedureDSC",
            ],  # additional_columns
        ],
    },
    "surgery_file": {
        "name": "surgery.csv",
        "source": "surgery",
        "columns": [
            "ProcedureNM",  # signal_column
            "ProgressDSC",  # status_column
            "BeginDTS",  # start_column
            "EndDTS",  # end_column
        ],
    },
    "other_procedures_file": {
        "name": "procedures.csv",
        "source": "procedures",
        "columns": [
            "ProcedureDSC",  # signal_column
            "OrderStatusDSC",  # status_column
            "StartDTS",  # start_column
            "EndDTS",  # end_column
        ],
    },
    "transfusions_file": {
        "name": "transfusions.csv",
        "source": "transfusions",
        "columns": [
            "ProcedureDSC",  # signal_column
            "OrderStatusDSC",  # status_column
            "BloodStartInstantDTS",  # start_column
            "BloodEndInstantDTS",  # end_column
        ],
    },
    "events_file": {
        "name": "events.csv",
        "source": "events",
        "columns": ["EventNM", "EventDTS"],  # signal_column  # time_column
    },
    "medhist_file": {
        "name": "medical-history.csv",
        "source": "static",
        "columns": [
            "DiagnosisID",  # id_column
            "DiagnosisNM",  # name_column
            "CommentTXT",  # comment_column
            "MedicalHistoryDateTXT",  # date_column
        ],
    },
    "surghist_file": {
        "name": "surgical-history.csv",
        "source": "static",
        "columns": [
            "ProcedureID",  # id_column
            "ProcedureNM",  # name_column
            "CommentTXT",  # comment_column
            "SurgicalHistoryDateTXT",  # date_column
        ],
    },
    "socialhist_file": {
        "name": "social-history.csv",
        "source": "static",
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

STATIC_UNITS = {
    "height": "m",
    "weight": "lbs",
}
