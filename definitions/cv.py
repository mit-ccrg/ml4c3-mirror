# Imports: standard library
from typing import Dict

OBSERVATIONS_MAPPING: Dict[str, int] = {
    "v_lv": 0,
    "v_cas": 1,
    "v_cvs": 2,
    "v_rv": 3,
    "v_cap": 4,
    "v_cvp": 5,
    "p_lv": 6,
    "p_cas": 7,
    "p_cvs": 8,
    "p_rv": 9,
    "p_cap": 10,
    "p_cvp": 11,
    "c_out": 12,
}

OBSERVATIONS_NAME: Dict[str, str] = {
    "v_lv": "Left Ventricle Volume",
    "v_cas": "Systemic Arterial Volume",
    "v_cvs": "Systemic Venous Volume",
    "v_rv": "Right Ventricle Volume",
    "v_cap": "Pulmonic Arterial Volume",
    "v_cvp": "Pulmonic Venous Volume",
    "p_lv": "Left Ventricle Pressure",
    "p_cas": "Systemic Arterial Pressure",
    "p_cvs": "Systemic Venous Pressure",
    "p_rv": "Right Ventricle Pressure",
    "p_cap": "Pulmonic Arterial Pressure",
    "p_cvp": "Pulmonic Venous Pressure",
    "c_out": "Cardiac Output",
}

OBSERVATIONS_TO_UNITS: Dict[str, str] = {
    "v_lv": "mL",
    "v_cas": "mL",
    "v_cvs": "mL",
    "v_rv": "mL",
    "v_cap": "mL",
    "v_cvp": "mL",
    "p_lv": "mmHg",
    "p_cas": "mmHg",
    "p_cvs": "mmHg",
    "p_rv": "mmHg",
    "p_cap": "mmHg",
    "p_cvp": "mmHg",
    "c_out": "L/min",
}

PARAMETER_NAMES: Dict[str, str] = {
    # Heart parameters
    "Ees_rv": "Right Ventricle End-Systolic Elastance",
    "Ees_lv": "Left Ventricle End-Systolic Elastance",
    "Vo_rv": "Right Ventricle Unstressed Volume",
    "Vo_lv": "Left Ventricle Unstressed Volume",
    "Tes_rv": "Right Ventricle Time To End Systole",
    "Tes_lv": "Left Ventricle Time To End Systole",
    "tau_rv": "Right Ventricle Time Constant Of Relaxation",
    "tau_lv": "Left Ventricle Time Constant Of Relaxation",
    "A_rv": "Right Ventricle Scaling Factor",
    "A_lv": "Left Ventricle Scaling Factor",
    "B_rv": "Right Ventricle Exponent",
    "B_lv": "Left Ventricle Exponent",
    # Circulation parameters
    "Ra_pul": "Pulmonic Arterial Resistance",
    "Ra_sys": "Systemic Arterial Resistance",
    "Rc_pul": "Pulmonic Characteristic Resistance",
    "Rc_sys": "Systemic Characteristic Resistance",
    "Rv_pul": "Pulmonic Venous Resistance",
    "Rv_sys": "Systemic Venous Resistance",
    "Rt_pul": "Pulmonic Total Resistance",
    "Rt_sys": "Systemic Total Resistance",
    "Ca_pul": "Pulmonic Arterial Capacitance",
    "Ca_sys": "Systemic Arterial Capacitance",
    "Cv_pul": "Pulmonic Venous Capacitance",
    "Cv_sys": "Systemic Venous Capacitance",
    # Common parameters
    "HR": "Heart Rate",
    "Vt": "Total Blood Volume",
    "Vs": "Total Stressed Blood Volume",
    "Vu": "Total Unstressed Blood Volume",
}

PARAMETER_UNITS: Dict[str, str] = {
    # Heart parameters
    "Ees_rv": "mmHg/mL",
    "Ees_lv": "mmHg/mL",
    "Vo_rv": "mL",
    "Vo_lv": "mL",
    "Tes_rv": "s",
    "Tes_lv": "s",
    "tau_rv": "s",
    "tau_lv": "s",
    "A_rv": "mmHg",
    "A_lv": "mmHg",
    "B_rv": "1/mL",
    "B_lv": "1/mL",
    # Circulation parameters
    "Ra_pul": "mmHg*s/mL",
    "Ra_sys": "mmHg*s/mL",
    "Rc_pul": "mmHg*s/mL",
    "Rc_sys": "mmHg*s/mL",
    "Rv_pul": "mmHg*s/mL",
    "Rv_sys": "mmHg*s/mL",
    "Rt_pul": "mmHg*s/mL",
    "Rt_sys": "mmHg*s/mL",
    "Ca_pul": "mL/mmHg",
    "Ca_sys": "mL/mmHg",
    "Cv_pul": "mL/mmHg",
    "Cv_sys": "mL/mmHg",
    # Common parameters
    "HR": "bpm",
    "Vt": "mL",
    "Vs": "mL",
    "Vu": "mL",
}
