# Imports: standard library
from typing import Any, Dict

DEFINED_TMAPS: Dict[str, Any] = {
    "waveform": [
        "i",
        "ii",
        "iii",
        "v",
        "spo2",
        "pa2",
        "art1",
        "bis",
        "co2",
        "cvp3",
        "fem4",
        "rr",
    ],
    "vitals": [
        "art1d",
        "art1m",
        "art1r",
        "art1s",
        "asbramp",
        "bis",
        "bt",
        "co",
        "co2rr",
        "cuff",
        "cvp3",
        "dynres",
        "emg",
        "fem4d",
        "fem4m",
        "fem4r",
        "fem4s",
        "flwr",
        "flwtrig",
        "hr",
        "i:e",
        "in_hld",
        "inspmeas",
        "insptm",
        "it",
        "mawp",
        "mv",
        "nbpd",
        "nbpm",
        "nbps",
        "pa2d",
        "pa2m",
        "pa2s",
        "peep",
        "pip",
        "pplat",
        "prssup",
        "ptrr",
        "pvc",
        "resp",
        "sens",
        "setfio2",
        "setie",
        "setpcp",
        "settv",
        "spo2%",
        "spo2r",
        "spontmv",
        "sqi",
        "sr",
        "statres",
        "tv",
        "vent rate",
    ],
    "ecg_features": [
        "r_peaks",
        "p_peaks",
        "q_peaks",
        "s_peaks",
        "t_peaks",
        "r_onsets",
        "r_offsets",
        "p_onsets",
        "p_offsets",
        "t_onsets",
        "t_offsets",
        "pr_interval",
        "pr_segment",
        "qt_interval",
        "st_segment",
        "tp_segment",
        "rr_interval",
        "qrs_interval",
        "st_height",
        "qrs_amplitude",
    ],
    "alarms": ["cpp_low", "v_tach", "apnea"],
    "events": ["code_start", "rapid_response_start"],
    "med": [
        "aspirin_325_mg_tablet",
        "cefazolin_2_gram|50_ml_in_dextrose_iso-osmotic_intravenous_piggyback",
        "lactated_ringers_iv_bolus",
        "norepinephrine_infusion_syringe_in_swfi_80_mcg|ml_cmpd_central_mgh",
        "sodium_chloride_0.9_%_intravenous_solution",
    ],
    "surgery": ["colonoscopy", "coronary_artery_bypass_graft"],
    "procedures": ["hemodialysis", "hemodialysis_|_ultrafiltration"],
    "transfusions": [
        "transfuse_red_blood_cells",
        "transfuse_platelets",
        "transfuse_cryoprecipitate",
        "transfuse_fresh_frozen_plasma",
        "massive_transfusion_protocol",
    ],
    "labs": [
        "creatinine",
        "magnesium",
        "potassium",
        "calcium",
        "sodium",
        "chloride",
        "glucose",
        "phosphorus",
        "bilirubin_direct",
        "bilirubin_total",
        "wbc",
        "troponin_t-hs_gen5",
        "hct",
        "hgb",
        "rbc",
        "bun",
        "ast",
        "base_excess_unspecified",
        "hco3_unspecified",
        "fio2",
        "ph_arterial",
        "ph_venous",
        "ph_unspecified",
        "po2_arterial",
        "po2_venous",
        "pco2_arterial",
        "pco2_unspecified",
        "pco2_uncorrected",
        "so2_unspecified",
        "lactic_acid-poc",
        "ptt",
        "o2_sat_so2_arterial",
        "alk_phos",
        "plt",
        "anion_gap",
        "total_protein",
        "albumin",
        "platelets",
        "saturated_o2",
        "hco3_unspecified",
    ],
    "flowsheet": [
        "pulse",
        "r_phs_ob_bp_systolic_outgoing",
        "blood_pressure",
        "ppi",
        "r_heart_rate_source",
        "r_saturated_venous_o2",
        "r_phs_ip_pulse_spo2",
        "r_map",
        "r_cpn_glasgow_coma_scale_score",
        "r_level_of_consciousness",
        "temperature",
        "respirations",
        "urine_output",
    ],
    "static_language": {
        "admin_diag",
        "admin_type",
        "alcohol_hist",
        "local_time",
        "race",
        "source",
        "tobacco_hist",
    },
    "static_continuous": {"height", "weight"},
    "static_event": {"admin_date", "birth_date", "end_date"},
    "static_timeseries": {
        "department_id",
        "department_nm",
        "medical_hist",
        "move_time",
        "room_bed",
        "surgical_hist",
    },
    "static_categorical": {
        "sex": {"male": 0, "female": 1},
        "end_stay_type": {"alive": 0, "deceased": 1},
    },
    "other": ["mrn", "visits", "length_of_stay"],
    "bedmaster_signals": [
        "waveform",
        "vitals",
        "ecg_features",
        "alarms",
    ],
    "edw_signals": [
        "events",
        "med",
        "surgery",
        "procedures",
        "transfusions",
        "labs",
        "flowsheet",
        "static",
    ],
}


ICU_TMAPS_METADATA: Dict[str, Dict[str, Any]] = {
    #    "TMAP_NAME": {"min": 0, "max": 0, "std": 0, "mean": 0, "median": 0, "iqr": 0}
}