# Imports: standard library
from typing import Dict

# Imports: first party
from tensormap.ecg import (
    make_ecg_label_from_read_tff,
    make_binary_ecg_label_from_any_read_tff,
)
from ml4c3.TensorMap import TensorMap, Interpretation
from ml4c3.definitions import ECG_PREFIX
from tensormap.validators import validator_not_all_zero

tmaps: Dict[str, TensorMap] = {}
tmaps["asystole"] = TensorMap(
    "asystole",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_asystole": 0, "asystole": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"asystole": {"asystole"}},
        not_found_channel="no_asystole",
    ),
)


tmaps["asystole_any"] = TensorMap(
    "asystole_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_asystole": 0, "asystole": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"asystole": {"asystole"}},
        not_found_channel="no_asystole",
    ),
)


tmaps["atrial_fibrillation"] = TensorMap(
    "atrial_fibrillation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_atrial_fibrillation": 0, "atrial_fibrillation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_fibrillation": {
                "atrial fibrillation",
                "atrial fibrillation with moderate ventricular response",
                "atrialfibrillation",
                "atrial fib",
                "afib",
                "afibrillation",
                "atrial fibrillation with rapid ventricular response",
                "atrial fibrillation with controlled ventricular response",
                "fibrillation/flutter",
            },
        },
        not_found_channel="no_atrial_fibrillation",
    ),
)


tmaps["atrial_fibrillation_any"] = TensorMap(
    "atrial_fibrillation_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_atrial_fibrillation": 0, "atrial_fibrillation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_fibrillation": {
                "atrial fibrillation",
                "atrial fibrillation with moderate ventricular response",
                "atrialfibrillation",
                "atrial fib",
                "afib",
                "afibrillation",
                "atrial fibrillation with rapid ventricular response",
                "atrial fibrillation with controlled ventricular response",
                "fibrillation/flutter",
            },
        },
        not_found_channel="no_atrial_fibrillation",
    ),
)


tmaps["atrial_flutter"] = TensorMap(
    "atrial_flutter",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_atrial_flutter": 0, "atrial_flutter": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_flutter": {
                "tachycardia possibly flutter",
                "atrial flutter",
                "atrial flutter variable block",
                "probable flutter",
                "atrial flutter unspecified block",
                "flutter",
                "atrial flutter fixed block",
                "fibrillation/flutter",
                "aflutter",
            },
        },
        not_found_channel="no_atrial_flutter",
    ),
)


tmaps["atrial_flutter_any"] = TensorMap(
    "atrial_flutter_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_atrial_flutter": 0, "atrial_flutter": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_flutter": {
                "tachycardia possibly flutter",
                "atrial flutter",
                "atrial flutter variable block",
                "probable flutter",
                "atrial flutter unspecified block",
                "flutter",
                "atrial flutter fixed block",
                "fibrillation/flutter",
                "aflutter",
            },
        },
        not_found_channel="no_atrial_flutter",
    ),
)


tmaps["atrial_paced_rhythm"] = TensorMap(
    "atrial_paced_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_atrial_paced_rhythm": 0, "atrial_paced_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"atrial_paced_rhythm": {"atrial pacing", "atrial paced rhythm"}},
        not_found_channel="no_atrial_paced_rhythm",
    ),
)


tmaps["atrial_paced_rhythm_any"] = TensorMap(
    "atrial_paced_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_atrial_paced_rhythm": 0, "atrial_paced_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"atrial_paced_rhythm": {"atrial pacing", "atrial paced rhythm"}},
        not_found_channel="no_atrial_paced_rhythm",
    ),
)


tmaps["ectopic_atrial_bradycardia"] = TensorMap(
    "ectopic_atrial_bradycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ectopic_atrial_bradycardia": 0, "ectopic_atrial_bradycardia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_bradycardia": {
                "low atrial bradycardia",
                "ectopic atrial bradycardia",
            },
        },
        not_found_channel="no_ectopic_atrial_bradycardia",
    ),
)


tmaps["ectopic_atrial_bradycardia_any"] = TensorMap(
    "ectopic_atrial_bradycardia_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_ectopic_atrial_bradycardia": 0, "ectopic_atrial_bradycardia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_bradycardia": {
                "low atrial bradycardia",
                "ectopic atrial bradycardia",
            },
        },
        not_found_channel="no_ectopic_atrial_bradycardia",
    ),
)


tmaps["ectopic_atrial_rhythm"] = TensorMap(
    "ectopic_atrial_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ectopic_atrial_rhythm": 0, "ectopic_atrial_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_rhythm": {
                "p wave axis suggests atrial rather than sinus mechanism",
                "ectopicsupraventricular rhythm",
                "nonsinus atrial mechanism",
                "ectopic atrial rhythm",
                "abnormal p vector",
                "ectopic atrial rhythm ",
                "multifocal ectopic atrial rhythm",
                "low atrial pacer",
                "atrial rhythm",
                "multifocal atrial rhythm",
                "multifocal ear",
                "wandering ear",
                "atrial arrhythmia",
                "unifocal ear",
                "wandering ectopic atrial rhythm",
                "unifocal ectopic atrial rhythm",
                "unusual p wave axis",
                "dual atrial foci ",
                "multiple atrial foci",
                "wandering atrial pacemaker",
                "multifocal atrialrhythm",
            },
        },
        not_found_channel="no_ectopic_atrial_rhythm",
    ),
)


tmaps["ectopic_atrial_rhythm_any"] = TensorMap(
    "ectopic_atrial_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_ectopic_atrial_rhythm": 0, "ectopic_atrial_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_rhythm": {
                "p wave axis suggests atrial rather than sinus mechanism",
                "ectopicsupraventricular rhythm",
                "nonsinus atrial mechanism",
                "ectopic atrial rhythm",
                "abnormal p vector",
                "ectopic atrial rhythm ",
                "multifocal ectopic atrial rhythm",
                "low atrial pacer",
                "atrial rhythm",
                "multifocal atrial rhythm",
                "multifocal ear",
                "wandering ear",
                "atrial arrhythmia",
                "unifocal ear",
                "wandering ectopic atrial rhythm",
                "unifocal ectopic atrial rhythm",
                "unusual p wave axis",
                "dual atrial foci ",
                "multiple atrial foci",
                "wandering atrial pacemaker",
                "multifocal atrialrhythm",
            },
        },
        not_found_channel="no_ectopic_atrial_rhythm",
    ),
)


tmaps["ectopic_atrial_tachycardia"] = TensorMap(
    "ectopic_atrial_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ectopic_atrial_tachycardia": 0, "ectopic_atrial_tachycardia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_tachycardia": {
                "ectopic atrial tachycardia",
                "unifocal atrial tachycardia",
                "multifocal atrial tachycardia",
                "ectopic atrial tachycardia, unspecified",
                "wandering atrial tachycardia",
                "unspecified ectopic atrial tachycardia",
                "unifocal ectopic atrial tachycardia",
                "ectopic atrial tachycardia, multifocal",
                "ectopic atrial tachycardia, unifocal",
                "multifocal ectopic atrial tachycardia",
            },
        },
        not_found_channel="no_ectopic_atrial_tachycardia",
    ),
)


tmaps["ectopic_atrial_tachycardia_any"] = TensorMap(
    "ectopic_atrial_tachycardia_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_ectopic_atrial_tachycardia": 0, "ectopic_atrial_tachycardia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_tachycardia": {
                "ectopic atrial tachycardia",
                "unifocal atrial tachycardia",
                "multifocal atrial tachycardia",
                "ectopic atrial tachycardia, unspecified",
                "wandering atrial tachycardia",
                "unspecified ectopic atrial tachycardia",
                "unifocal ectopic atrial tachycardia",
                "ectopic atrial tachycardia, multifocal",
                "ectopic atrial tachycardia, unifocal",
                "multifocal ectopic atrial tachycardia",
            },
        },
        not_found_channel="no_ectopic_atrial_tachycardia",
    ),
)


tmaps["narrow_qrs_tachycardia"] = TensorMap(
    "narrow_qrs_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_narrow_qrs_tachycardia": 0, "narrow_qrs_tachycardia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "narrow_qrs_tachycardia": {
                "tachycardia narrow qrs",
                "narrow complex tachycardia",
                "narrow qrs tachycardia",
            },
        },
        not_found_channel="no_narrow_qrs_tachycardia",
    ),
)


tmaps["narrow_qrs_tachycardia_any"] = TensorMap(
    "narrow_qrs_tachycardia_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_narrow_qrs_tachycardia": 0, "narrow_qrs_tachycardia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "narrow_qrs_tachycardia": {
                "tachycardia narrow qrs",
                "narrow complex tachycardia",
                "narrow qrs tachycardia",
            },
        },
        not_found_channel="no_narrow_qrs_tachycardia",
    ),
)


tmaps["pulseless_electrical_activity"] = TensorMap(
    "pulseless_electrical_activity",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_pulseless_electrical_activity": 0,
        "pulseless_electrical_activity": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "pulseless_electrical_activity": {
                "pulseless",
                "pulseless electrical activity",
            },
        },
        not_found_channel="no_pulseless_electrical_activity",
    ),
)


tmaps["pulseless_electrical_activity_any"] = TensorMap(
    "pulseless_electrical_activity_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_pulseless_electrical_activity": 0,
        "pulseless_electrical_activity": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "pulseless_electrical_activity": {
                "pulseless",
                "pulseless electrical activity",
            },
        },
        not_found_channel="no_pulseless_electrical_activity",
    ),
)


tmaps["retrograde_atrial_activation"] = TensorMap(
    "retrograde_atrial_activation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_retrograde_atrial_activation": 0,
        "retrograde_atrial_activation": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "retrograde_atrial_activation": {"retrograde atrial activation"},
        },
        not_found_channel="no_retrograde_atrial_activation",
    ),
)


tmaps["retrograde_atrial_activation_any"] = TensorMap(
    "retrograde_atrial_activation_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_retrograde_atrial_activation": 0,
        "retrograde_atrial_activation": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "retrograde_atrial_activation": {"retrograde atrial activation"},
        },
        not_found_channel="no_retrograde_atrial_activation",
    ),
)


tmaps["sinus_arrest"] = TensorMap(
    "sinus_arrest",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_sinus_arrest": 0, "sinus_arrest": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sinus_arrest": {"sinus arrest"}},
        not_found_channel="no_sinus_arrest",
    ),
)


tmaps["sinus_arrest_any"] = TensorMap(
    "sinus_arrest_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_sinus_arrest": 0, "sinus_arrest": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sinus_arrest": {"sinus arrest"}},
        not_found_channel="no_sinus_arrest",
    ),
)


tmaps["sinus_pause"] = TensorMap(
    "sinus_pause",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_sinus_pause": 0, "sinus_pause": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sinus_pause": {"sinus pause", "sinus pauses"}},
        not_found_channel="no_sinus_pause",
    ),
)


tmaps["sinus_pause_any"] = TensorMap(
    "sinus_pause_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_sinus_pause": 0, "sinus_pause": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sinus_pause": {"sinus pause", "sinus pauses"}},
        not_found_channel="no_sinus_pause",
    ),
)


tmaps["sinus_rhythm"] = TensorMap(
    "sinus_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_sinus_rhythm": 0, "sinus_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "sinus_rhythm": {
                "frequent native sinus beats",
                "atrial bigeminal rhythm",
                "tracing is within normal limits",
                "type i sinoatrial block",
                "sinus mechanism has replaced",
                "sa block",
                "tracing within normal limits",
                "rhythm is normal sinus",
                "atrial trigeminy",
                "sinus rhythm at a rate",
                "type i sa block",
                "sinus tachycardia",
                "sinus arrhythmia",
                "rhythm remains normal sinus",
                "sinus bradycardia",
                "rhythm is now clearly sinus",
                "2nd degree sa block",
                "type ii sa block",
                "sinoatrial block",
                "sinoatrial block, type ii",
                "sa exit block",
                "sinus slowing",
                "atrial bigeminy and ventricular bigeminy",
                "sinus rhythm",
                "marked sinus arrhythmia",
                "atrialbigeminy",
                "atrial bigeminal  rhythm",
                "conducted sinus impulses",
                "normal ecg",
                "rhythm has reverted to normal",
                "sa block, type i",
                "type ii sinoatrial block",
                "sinus exit block",
                "with occasional native sinus beats",
                "1st degree sa block",
                "normal when compared with ecg of",
                "normal sinus rhythm",
            },
        },
        not_found_channel="no_sinus_rhythm",
    ),
)


tmaps["sinus_rhythm_any"] = TensorMap(
    "sinus_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_sinus_rhythm": 0, "sinus_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "sinus_rhythm": {
                "frequent native sinus beats",
                "atrial bigeminal rhythm",
                "tracing is within normal limits",
                "type i sinoatrial block",
                "sinus mechanism has replaced",
                "sa block",
                "tracing within normal limits",
                "rhythm is normal sinus",
                "atrial trigeminy",
                "sinus rhythm at a rate",
                "type i sa block",
                "sinus tachycardia",
                "sinus arrhythmia",
                "rhythm remains normal sinus",
                "sinus bradycardia",
                "rhythm is now clearly sinus",
                "2nd degree sa block",
                "type ii sa block",
                "sinoatrial block",
                "sinoatrial block, type ii",
                "sa exit block",
                "sinus slowing",
                "atrial bigeminy and ventricular bigeminy",
                "sinus rhythm",
                "marked sinus arrhythmia",
                "atrialbigeminy",
                "atrial bigeminal  rhythm",
                "conducted sinus impulses",
                "normal ecg",
                "rhythm has reverted to normal",
                "sa block, type i",
                "type ii sinoatrial block",
                "sinus exit block",
                "with occasional native sinus beats",
                "1st degree sa block",
                "normal when compared with ecg of",
                "normal sinus rhythm",
            },
        },
        not_found_channel="no_sinus_rhythm",
    ),
)


tmaps["supraventricular_tachycardia"] = TensorMap(
    "supraventricular_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_supraventricular_tachycardia": 0,
        "supraventricular_tachycardia": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "supraventricular_tachycardia": {
                "junctional tachycardia",
                "av nodal reentry tachycardia",
                "supraventricular tachycardia",
                "accelerated atrioventricular nodal rhythm",
                "av nodal reentrant",
                "atrial tachycardia",
                "avnrt",
                "accelerated nodal rhythm",
                "av reentrant tachycardia ",
                "avrt",
                "atrioventricular nodal reentry tachycardia",
                "atrioventricular reentrant tachycardia ",
                "accelerated atrioventricular junctional rhythm",
            },
        },
        not_found_channel="no_supraventricular_tachycardia",
    ),
)


tmaps["supraventricular_tachycardia_any"] = TensorMap(
    "supraventricular_tachycardia_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_supraventricular_tachycardia": 0,
        "supraventricular_tachycardia": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "supraventricular_tachycardia": {
                "junctional tachycardia",
                "av nodal reentry tachycardia",
                "supraventricular tachycardia",
                "accelerated atrioventricular nodal rhythm",
                "av nodal reentrant",
                "atrial tachycardia",
                "avnrt",
                "accelerated nodal rhythm",
                "av reentrant tachycardia ",
                "avrt",
                "atrioventricular nodal reentry tachycardia",
                "atrioventricular reentrant tachycardia ",
                "accelerated atrioventricular junctional rhythm",
            },
        },
        not_found_channel="no_supraventricular_tachycardia",
    ),
)


tmaps["torsade_de_pointes"] = TensorMap(
    "torsade_de_pointes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_torsade_de_pointes": 0, "torsade_de_pointes": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"torsade_de_pointes": {"torsade"}},
        not_found_channel="no_torsade_de_pointes",
    ),
)


tmaps["torsade_de_pointes_any"] = TensorMap(
    "torsade_de_pointes_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_torsade_de_pointes": 0, "torsade_de_pointes": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"torsade_de_pointes": {"torsade"}},
        not_found_channel="no_torsade_de_pointes",
    ),
)


tmaps["unspecified"] = TensorMap(
    "unspecified",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_unspecified": 0, "unspecified": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified": {
                "uncertain rhythm",
                "rhythm unclear",
                "rhythm uncertain",
                "undetermined  rhythm",
            },
        },
        not_found_channel="no_unspecified",
    ),
)


tmaps["unspecified_any"] = TensorMap(
    "unspecified_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_unspecified": 0, "unspecified": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified": {
                "uncertain rhythm",
                "rhythm unclear",
                "rhythm uncertain",
                "undetermined  rhythm",
            },
        },
        not_found_channel="no_unspecified",
    ),
)


tmaps["ventricular_rhythm"] = TensorMap(
    "ventricular_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ventricular_rhythm": 0, "ventricular_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_rhythm": {"accelerated idioventricular rhythm"}},
        not_found_channel="no_ventricular_rhythm",
    ),
)


tmaps["ventricular_rhythm_any"] = TensorMap(
    "ventricular_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_ventricular_rhythm": 0, "ventricular_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_rhythm": {"accelerated idioventricular rhythm"}},
        not_found_channel="no_ventricular_rhythm",
    ),
)


tmaps["wpw"] = TensorMap(
    "wpw",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_wpw": 0, "wpw": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "wpw": {"wolff-parkinson-white pattern", "wpw", "wolffparkinsonwhite"},
        },
        not_found_channel="no_wpw",
    ),
)


tmaps["wpw_any"] = TensorMap(
    "wpw_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_wpw": 0, "wpw": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "wpw": {"wolff-parkinson-white pattern", "wpw", "wolffparkinsonwhite"},
        },
        not_found_channel="no_wpw",
    ),
)


tmaps["pacemaker"] = TensorMap(
    "pacemaker",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_pacemaker": 0, "pacemaker": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "pacemaker": {
                "failure to inhibit ventricular",
                "ventricular pacing",
                "ventricular-paced complexes",
                "dual chamber pacing",
                "ventricular pacing has replaced av pacing",
                "ventricular paced",
                "ventricular-paced rhythm",
                "atrial-paced rhythm",
                "atrial triggered ventricular pacing",
                "v-paced beats",
                "failure to inhibit atrial",
                "atrially triggered v paced",
                "biventricular-paced rhythm",
                "v-paced rhythm",
                "a triggered v-paced rhythm",
                "electronic pacemaker",
                "failure to pace atrial",
                "demand ventricular pacemaker",
                "sequential pacing",
                "shows dual chamber pacing",
                "biventricular-paced complexes",
                "atrial-sensed ventricular-paced complexes",
                "competitive av pacing",
                "demand v-pacing",
                "atrial-paced complexes ",
                "v-paced",
                "ventricular demand pacing",
                "failure to pace ventricular",
                "failure to capture ventricular",
                "atrial-sensed ventricular-paced rhythm",
                "failure to capture atrial",
                "av dual-paced complexes",
                "av dual-paced rhythm",
                "unipolar right ventricular  pacing",
            },
        },
        not_found_channel="no_pacemaker",
    ),
)


tmaps["pacemaker_any"] = TensorMap(
    "pacemaker_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_pacemaker": 0, "pacemaker": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "pacemaker": {
                "failure to inhibit ventricular",
                "ventricular pacing",
                "ventricular-paced complexes",
                "dual chamber pacing",
                "ventricular pacing has replaced av pacing",
                "ventricular paced",
                "ventricular-paced rhythm",
                "atrial-paced rhythm",
                "atrial triggered ventricular pacing",
                "v-paced beats",
                "failure to inhibit atrial",
                "atrially triggered v paced",
                "biventricular-paced rhythm",
                "v-paced rhythm",
                "a triggered v-paced rhythm",
                "electronic pacemaker",
                "failure to pace atrial",
                "demand ventricular pacemaker",
                "sequential pacing",
                "shows dual chamber pacing",
                "biventricular-paced complexes",
                "atrial-sensed ventricular-paced complexes",
                "competitive av pacing",
                "demand v-pacing",
                "atrial-paced complexes ",
                "v-paced",
                "ventricular demand pacing",
                "failure to pace ventricular",
                "failure to capture ventricular",
                "atrial-sensed ventricular-paced rhythm",
                "failure to capture atrial",
                "av dual-paced complexes",
                "av dual-paced rhythm",
                "unipolar right ventricular  pacing",
            },
        },
        not_found_channel="no_pacemaker",
    ),
)


tmaps["abnormal_ecg"] = TensorMap(
    "abnormal_ecg",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_abnormal_ecg": 0, "abnormal_ecg": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"abnormal_ecg": {"abnormal"}},
        not_found_channel="no_abnormal_ecg",
    ),
)


tmaps["abnormal_ecg_any"] = TensorMap(
    "abnormal_ecg_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_abnormal_ecg": 0, "abnormal_ecg": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"abnormal_ecg": {"abnormal"}},
        not_found_channel="no_abnormal_ecg",
    ),
)


tmaps["normal_sinus_rhythm"] = TensorMap(
    "normal_sinus_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_normal_sinus_rhythm": 0, "normal_sinus_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "normal_sinus_rhythm": {
                "normal ecg",
                "sinus tachycardia",
                "normal tracing",
                "tracing is within normal limits",
                "sinus rhythm",
                "normal sinus rhythm",
            },
        },
        not_found_channel="no_normal_sinus_rhythm",
    ),
)


tmaps["normal_sinus_rhythm_any"] = TensorMap(
    "normal_sinus_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_normal_sinus_rhythm": 0, "normal_sinus_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "normal_sinus_rhythm": {
                "normal ecg",
                "sinus tachycardia",
                "normal tracing",
                "tracing is within normal limits",
                "sinus rhythm",
                "normal sinus rhythm",
            },
        },
        not_found_channel="no_normal_sinus_rhythm",
    ),
)


tmaps["uninterpretable_ecg"] = TensorMap(
    "uninterpretable_ecg",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_uninterpretable_ecg": 0, "uninterpretable_ecg": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"uninterpretable_ecg": {"uninterpretable"}},
        not_found_channel="no_uninterpretable_ecg",
    ),
)


tmaps["uninterpretable_ecg_any"] = TensorMap(
    "uninterpretable_ecg_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_uninterpretable_ecg": 0, "uninterpretable_ecg": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"uninterpretable_ecg": {"uninterpretable"}},
        not_found_channel="no_uninterpretable_ecg",
    ),
)


tmaps["indeterminate_axis"] = TensorMap(
    "indeterminate_axis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_indeterminate_axis": 0, "indeterminate_axis": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "indeterminate_axis": {
                "northwest axis",
                "indeterminate qrs axis",
                "indeterminate axis",
            },
        },
        not_found_channel="no_indeterminate_axis",
    ),
)


tmaps["indeterminate_axis_any"] = TensorMap(
    "indeterminate_axis_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_indeterminate_axis": 0, "indeterminate_axis": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "indeterminate_axis": {
                "northwest axis",
                "indeterminate qrs axis",
                "indeterminate axis",
            },
        },
        not_found_channel="no_indeterminate_axis",
    ),
)


tmaps["left_axis_deviation"] = TensorMap(
    "left_axis_deviation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_left_axis_deviation": 0, "left_axis_deviation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_axis_deviation": {
                "leftward axis",
                "left axis deviation",
                "axis shifted left",
            },
        },
        not_found_channel="no_left_axis_deviation",
    ),
)


tmaps["left_axis_deviation_any"] = TensorMap(
    "left_axis_deviation_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_left_axis_deviation": 0, "left_axis_deviation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_axis_deviation": {
                "leftward axis",
                "left axis deviation",
                "axis shifted left",
            },
        },
        not_found_channel="no_left_axis_deviation",
    ),
)


tmaps["right_axis_deviation"] = TensorMap(
    "right_axis_deviation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_right_axis_deviation": 0, "right_axis_deviation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "right_axis_deviation": {
                "rightward axis",
                "right superior axis deviation",
                "axis shifted right",
                "right axis deviation",
            },
        },
        not_found_channel="no_right_axis_deviation",
    ),
)


tmaps["right_axis_deviation_any"] = TensorMap(
    "right_axis_deviation_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_right_axis_deviation": 0, "right_axis_deviation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "right_axis_deviation": {
                "rightward axis",
                "right superior axis deviation",
                "axis shifted right",
                "right axis deviation",
            },
        },
        not_found_channel="no_right_axis_deviation",
    ),
)


tmaps["right_superior_axis"] = TensorMap(
    "right_superior_axis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_right_superior_axis": 0, "right_superior_axis": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"right_superior_axis": {"right superior axis"}},
        not_found_channel="no_right_superior_axis",
    ),
)


tmaps["right_superior_axis_any"] = TensorMap(
    "right_superior_axis_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_right_superior_axis": 0, "right_superior_axis": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"right_superior_axis": {"right superior axis"}},
        not_found_channel="no_right_superior_axis",
    ),
)


tmaps["abnormal_p_wave_axis"] = TensorMap(
    "abnormal_p_wave_axis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_abnormal_p_wave_axis": 0, "abnormal_p_wave_axis": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"abnormal_p_wave_axis": {"abnormal p wave axis"}},
        not_found_channel="no_abnormal_p_wave_axis",
    ),
)


tmaps["abnormal_p_wave_axis_any"] = TensorMap(
    "abnormal_p_wave_axis_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_abnormal_p_wave_axis": 0, "abnormal_p_wave_axis": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"abnormal_p_wave_axis": {"abnormal p wave axis"}},
        not_found_channel="no_abnormal_p_wave_axis",
    ),
)


tmaps["electrical_alternans"] = TensorMap(
    "electrical_alternans",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_electrical_alternans": 0, "electrical_alternans": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"electrical_alternans": {"electrical alternans"}},
        not_found_channel="no_electrical_alternans",
    ),
)


tmaps["electrical_alternans_any"] = TensorMap(
    "electrical_alternans_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_electrical_alternans": 0, "electrical_alternans": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"electrical_alternans": {"electrical alternans"}},
        not_found_channel="no_electrical_alternans",
    ),
)


tmaps["low_voltage"] = TensorMap(
    "low_voltage",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_low_voltage": 0, "low_voltage": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"low_voltage": {"low voltage"}},
        not_found_channel="no_low_voltage",
    ),
)


tmaps["low_voltage_any"] = TensorMap(
    "low_voltage_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_low_voltage": 0, "low_voltage": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"low_voltage": {"low voltage"}},
        not_found_channel="no_low_voltage",
    ),
)


tmaps["poor_r_wave_progression"] = TensorMap(
    "poor_r_wave_progression",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_poor_r_wave_progression": 0, "poor_r_wave_progression": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "poor_r_wave_progression": {
                "abnormal precordial r wave progression or poor r wave progression",
                "unusual r wave progression",
                "early r wave progression",
                "abnormal precordial r wave progression",
                "slow precordial r wave progression",
                "poor precordial r wave progression",
                "poor r wave progression",
                "slowprecordial r wave progression",
            },
        },
        not_found_channel="no_poor_r_wave_progression",
    ),
)


tmaps["poor_r_wave_progression_any"] = TensorMap(
    "poor_r_wave_progression_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_poor_r_wave_progression": 0, "poor_r_wave_progression": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "poor_r_wave_progression": {
                "abnormal precordial r wave progression or poor r wave progression",
                "unusual r wave progression",
                "early r wave progression",
                "abnormal precordial r wave progression",
                "slow precordial r wave progression",
                "poor precordial r wave progression",
                "poor r wave progression",
                "slowprecordial r wave progression",
            },
        },
        not_found_channel="no_poor_r_wave_progression",
    ),
)


tmaps["reversed_r_wave_progression"] = TensorMap(
    "reversed_r_wave_progression",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_reversed_r_wave_progression": 0, "reversed_r_wave_progression": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "reversed_r_wave_progression": {
                "reverse r wave progression",
                "reversed r wave progression",
            },
        },
        not_found_channel="no_reversed_r_wave_progression",
    ),
)


tmaps["reversed_r_wave_progression_any"] = TensorMap(
    "reversed_r_wave_progression_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_reversed_r_wave_progression": 0, "reversed_r_wave_progression": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "reversed_r_wave_progression": {
                "reverse r wave progression",
                "reversed r wave progression",
            },
        },
        not_found_channel="no_reversed_r_wave_progression",
    ),
)


tmaps["mi"] = TensorMap(
    "mi",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_mi": 0, "mi": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mi": {
                "old anterolateral myocardial infarction",
                "myocardial infarction pattern",
                "possible anteroseptal myocardial infarction",
                "extensive anterolateral myocardial infarction",
                "old inferior and anterior myocardial infarctions",
                "septal infarct",
                "normal counterclockwise rotation versus old true posterior myocardial infarction",
                "old inferoapical myocardial infarction",
                "counterclockwise rotation versus old true posterior myocardial infarction",
                "antero-apical ischemia versus myocardial infarction",
                "possible myocardial infarction",
                "(possible old).*(true posterior).*(myocardial infarction)",
                "old anterolateral infarct",
                "counterclockwise rotation versus true posterior myocardial infarction",
                "acute anterior infarct",
                "old inferior anterior myocardial infarctions",
                "old infero-postero-lateral myocardial infarction",
                "rule out interim myocardial infarction",
                "old infero-posterior lateral myocardial infarction",
                "new incomplete posterior lateral myocardial infarction",
                "(true posterior).*(myocardial infarction)",
                "cannot rule out true posterior myocardial infarction",
                "suggestive of old true posterior myocardial infarction st abnormality",
                "inferior myocardial infarction",
                "old anterior infarct",
                "age indeterminate old inferior wall myocardial infarction",
                "possible true posterior myocardial infarction",
                "acute infarct",
                "possible anterolateral myocardial infarction",
                "old inferior myocardial infarction",
                "cannot rule out anteroseptal infarct",
                "old infero-posterior myocardial infarction",
                "acuteanterior myocardial infarction",
                "anterolateral myocardial infarction , possibly recent  ",
                "old posterolateral myocardial infarction",
                "lateral myocardial infarction of indeterminate age",
                "inferior myocardial infarction , age undetermined",
                "acute myocardial infarction",
                "apical myocardial infarction of indeterminate age",
                "lateral myocardial infarction - of indeterminate age",
                "anteroseptal myocardial infarction",
                "raises possibility of septal infarct",
                "possible inferior myocardial infarction",
                "old myocardial infarction",
                "old true posterior myocardial infarction",
                "infero-apical myocardial infarction",
                "old apicolateral myocardial infarction",
                "probable apicolateral myocardial infarction",
                "inferior myocardial infarction of indeterminate",
                "cannot rule out anterior wall ischemia myocardial infarction",
                "old anterior wall myocardial infarction",
                "possible old septal myocardial infarction",
                "apical myocardial infarction",
                "recent myocardial infarction",
                "acute anterior wall myocardial infarction",
                "lateral wall myocardial infarction",
                "possible old lateral myocardial infarction",
                "acute myocardial infarction of indeterminate age",
                "subendocardial ischemia subendocardial myocardial inf",
                "old inferolateral myocardial infarction",
                "inferior wall myocardial infarction of indeterminate age",
                "normal counterclockwise rotation versus true posterior myocardial infarction",
                "old infero-posterior and apical myocardial infarction",
                "counterclockwise rotation consistent with post myocardial infarction",
                "consistent with ischemia myocardial infarction",
                "myocardial infarction compared with the last previous ",
                "old anteroseptal myocardial infarction",
                "myocardial infarction nonspecific st segment",
                "cannot rule out inferoposterior myoca",
                "myocardial infarction cannot rule out",
                "old lateral myocardial infarction",
                "cannot rule out anterior infarct , age undetermined",
                "myocardial infarction when compared with ecg of",
                "transmural ischemia myocardial infarction",
                "inferior myocardial infarction of indeterminate age",
                "consistent with anteroseptal infarct",
                "myocardial infarction old high lateral",
                "anteroseptal infarct of indeterminate age",
                "normal left axis deviation counterclockwise rotation versus old true posterior myocardial infarction",
                "suggestive of old true posterior myocardial infarction",
                "cannot rule out true posterior myocardial infarction versus counterclockwise rotation",
                "subendocardial ischemia myocardial infarction",
                "myocardial infarction possible when compared",
                "normal counterclockwise rotation versusold true posterior myocardial infarction",
                "posterior wall myocardial infarction",
                "inferolateral myocardial infarction",
                "evolving inferior wall myocardial infarction",
                "inferior myocardial infarction , possibly acute",
                "true posterior myocardial infarction of indeterminate age",
                "posterior myocardial infarction",
                "anterolateral infarct of indeterminate age ",
                "old inferior wall myocardial infarction",
                "myocardial infarction",
                "myocardial infarction extension",
                "(counterclockwise rotation).*(true posterior)",
                "anterolateral myocardial infarction appears recent",
                "anterolateral myocardial infarction",
                "possible acute inferior myocardial infarction",
                "acute myocardial infarction in evolution",
                "old anterior myocardial infarction",
                "block inferior myocardial infarction",
                "ongoing ischemia versus myocardial infarction",
                "myocardial infarction versus pericarditis",
                "infero and apicolateral myocardial infarction",
                "possible septal myocardial infarction",
                "evolving anterior infarct",
                "possible acute myocardial infarction",
                "old inferoposterior myocardial infarction",
                "old high lateral myocardial infarction",
                "extensive myocardial infarction of indeterminate age ",
                "infero-apical myocardial infarction of indeterminate age",
                "old anteroseptal infarct",
                "myocardial infarction indeterminate",
                "anterior myocardial infarction of indeterminate age",
                "antero-apical and lateral myocardial infarction evolving",
                "anteroseptal and lateral myocardial infarction",
                "evolving counterclockwise rotation versus true posterior myocardial infarction",
                "subendocardial ischemia or myocardial infarction",
                "inferoapical myocardial infarction of indeterminate age",
                "concurrent ischemia myocardial infarction",
                "anterior myocardial infarction",
                "borderline anterolateral myocardial infarction",
                "old inferior posterolateral myocardial infarction",
                "anterior infarct of indeterminate age",
                "extensive anterior infarct",
                "known true posterior myocardial infarction",
                "myocardial infarction compared with the last previous tracing ",
                "subendocardial infarct",
                "consistent with anterior myocardial infarction of indeterminate age",
                "evolution of myocardial infarction",
                "evolving myocardial infarction",
                "myocardial infarction of indeterminate age",
                "(consistent with).*(true posterior).*(myocardial infarction)",
                "(myocardial infarction versus).*(counterclockwise rotation)",
                "subendocardial myocardial infarction",
                "possible anteroseptal myocardial infarction of uncertain age",
                "post myocardial infarction , of indeterminate age",
            },
        },
        not_found_channel="no_mi",
    ),
)


tmaps["mi_any"] = TensorMap(
    "mi_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_mi": 0, "mi": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mi": {
                "old anterolateral myocardial infarction",
                "myocardial infarction pattern",
                "possible anteroseptal myocardial infarction",
                "extensive anterolateral myocardial infarction",
                "old inferior and anterior myocardial infarctions",
                "septal infarct",
                "normal counterclockwise rotation versus old true posterior myocardial infarction",
                "old inferoapical myocardial infarction",
                "counterclockwise rotation versus old true posterior myocardial infarction",
                "antero-apical ischemia versus myocardial infarction",
                "possible myocardial infarction",
                "(possible old).*(true posterior).*(myocardial infarction)",
                "old anterolateral infarct",
                "counterclockwise rotation versus true posterior myocardial infarction",
                "acute anterior infarct",
                "old inferior anterior myocardial infarctions",
                "old infero-postero-lateral myocardial infarction",
                "rule out interim myocardial infarction",
                "old infero-posterior lateral myocardial infarction",
                "new incomplete posterior lateral myocardial infarction",
                "(true posterior).*(myocardial infarction)",
                "cannot rule out true posterior myocardial infarction",
                "suggestive of old true posterior myocardial infarction st abnormality",
                "inferior myocardial infarction",
                "old anterior infarct",
                "age indeterminate old inferior wall myocardial infarction",
                "possible true posterior myocardial infarction",
                "acute infarct",
                "possible anterolateral myocardial infarction",
                "old inferior myocardial infarction",
                "cannot rule out anteroseptal infarct",
                "old infero-posterior myocardial infarction",
                "acuteanterior myocardial infarction",
                "anterolateral myocardial infarction , possibly recent  ",
                "old posterolateral myocardial infarction",
                "lateral myocardial infarction of indeterminate age",
                "inferior myocardial infarction , age undetermined",
                "acute myocardial infarction",
                "apical myocardial infarction of indeterminate age",
                "lateral myocardial infarction - of indeterminate age",
                "anteroseptal myocardial infarction",
                "raises possibility of septal infarct",
                "possible inferior myocardial infarction",
                "old myocardial infarction",
                "old true posterior myocardial infarction",
                "infero-apical myocardial infarction",
                "old apicolateral myocardial infarction",
                "probable apicolateral myocardial infarction",
                "inferior myocardial infarction of indeterminate",
                "cannot rule out anterior wall ischemia myocardial infarction",
                "old anterior wall myocardial infarction",
                "possible old septal myocardial infarction",
                "apical myocardial infarction",
                "recent myocardial infarction",
                "acute anterior wall myocardial infarction",
                "lateral wall myocardial infarction",
                "possible old lateral myocardial infarction",
                "acute myocardial infarction of indeterminate age",
                "subendocardial ischemia subendocardial myocardial inf",
                "old inferolateral myocardial infarction",
                "inferior wall myocardial infarction of indeterminate age",
                "normal counterclockwise rotation versus true posterior myocardial infarction",
                "old infero-posterior and apical myocardial infarction",
                "counterclockwise rotation consistent with post myocardial infarction",
                "consistent with ischemia myocardial infarction",
                "myocardial infarction compared with the last previous ",
                "old anteroseptal myocardial infarction",
                "myocardial infarction nonspecific st segment",
                "cannot rule out inferoposterior myoca",
                "myocardial infarction cannot rule out",
                "old lateral myocardial infarction",
                "cannot rule out anterior infarct , age undetermined",
                "myocardial infarction when compared with ecg of",
                "transmural ischemia myocardial infarction",
                "inferior myocardial infarction of indeterminate age",
                "consistent with anteroseptal infarct",
                "myocardial infarction old high lateral",
                "anteroseptal infarct of indeterminate age",
                "normal left axis deviation counterclockwise rotation versus old true posterior myocardial infarction",
                "suggestive of old true posterior myocardial infarction",
                "cannot rule out true posterior myocardial infarction versus counterclockwise rotation",
                "subendocardial ischemia myocardial infarction",
                "myocardial infarction possible when compared",
                "normal counterclockwise rotation versusold true posterior myocardial infarction",
                "posterior wall myocardial infarction",
                "inferolateral myocardial infarction",
                "evolving inferior wall myocardial infarction",
                "inferior myocardial infarction , possibly acute",
                "true posterior myocardial infarction of indeterminate age",
                "posterior myocardial infarction",
                "anterolateral infarct of indeterminate age ",
                "old inferior wall myocardial infarction",
                "myocardial infarction",
                "myocardial infarction extension",
                "(counterclockwise rotation).*(true posterior)",
                "anterolateral myocardial infarction appears recent",
                "anterolateral myocardial infarction",
                "possible acute inferior myocardial infarction",
                "acute myocardial infarction in evolution",
                "old anterior myocardial infarction",
                "block inferior myocardial infarction",
                "ongoing ischemia versus myocardial infarction",
                "myocardial infarction versus pericarditis",
                "infero and apicolateral myocardial infarction",
                "possible septal myocardial infarction",
                "evolving anterior infarct",
                "possible acute myocardial infarction",
                "old inferoposterior myocardial infarction",
                "old high lateral myocardial infarction",
                "extensive myocardial infarction of indeterminate age ",
                "infero-apical myocardial infarction of indeterminate age",
                "old anteroseptal infarct",
                "myocardial infarction indeterminate",
                "anterior myocardial infarction of indeterminate age",
                "antero-apical and lateral myocardial infarction evolving",
                "anteroseptal and lateral myocardial infarction",
                "evolving counterclockwise rotation versus true posterior myocardial infarction",
                "subendocardial ischemia or myocardial infarction",
                "inferoapical myocardial infarction of indeterminate age",
                "concurrent ischemia myocardial infarction",
                "anterior myocardial infarction",
                "borderline anterolateral myocardial infarction",
                "old inferior posterolateral myocardial infarction",
                "anterior infarct of indeterminate age",
                "extensive anterior infarct",
                "known true posterior myocardial infarction",
                "myocardial infarction compared with the last previous tracing ",
                "subendocardial infarct",
                "consistent with anterior myocardial infarction of indeterminate age",
                "evolution of myocardial infarction",
                "evolving myocardial infarction",
                "myocardial infarction of indeterminate age",
                "(consistent with).*(true posterior).*(myocardial infarction)",
                "(myocardial infarction versus).*(counterclockwise rotation)",
                "subendocardial myocardial infarction",
                "possible anteroseptal myocardial infarction of uncertain age",
                "post myocardial infarction , of indeterminate age",
            },
        },
        not_found_channel="no_mi",
    ),
)


tmaps["aberrant_conduction_of_supraventricular_beats"] = TensorMap(
    "aberrant_conduction_of_supraventricular_beats",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_aberrant_conduction_of_supraventricular_beats": 0,
        "aberrant_conduction_of_supraventricular_beats": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "aberrant_conduction_of_supraventricular_beats": {
                "aberrant conduction of supraventricular beats",
                "aberrant conduction",
            },
        },
        not_found_channel="no_aberrant_conduction_of_supraventricular_beats",
    ),
)


tmaps["aberrant_conduction_of_supraventricular_beats_any"] = TensorMap(
    "aberrant_conduction_of_supraventricular_beats_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_aberrant_conduction_of_supraventricular_beats": 0,
        "aberrant_conduction_of_supraventricular_beats": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "aberrant_conduction_of_supraventricular_beats": {
                "aberrant conduction of supraventricular beats",
                "aberrant conduction",
            },
        },
        not_found_channel="no_aberrant_conduction_of_supraventricular_beats",
    ),
)


tmaps["crista_pattern"] = TensorMap(
    "crista_pattern",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_crista_pattern": 0, "crista_pattern": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"crista_pattern": {"crista pattern"}},
        not_found_channel="no_crista_pattern",
    ),
)


tmaps["crista_pattern_any"] = TensorMap(
    "crista_pattern_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_crista_pattern": 0, "crista_pattern": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"crista_pattern": {"crista pattern"}},
        not_found_channel="no_crista_pattern",
    ),
)


tmaps["epsilon_wave"] = TensorMap(
    "epsilon_wave",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_epsilon_wave": 0, "epsilon_wave": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"epsilon_wave": {"epsilon wave"}},
        not_found_channel="no_epsilon_wave",
    ),
)


tmaps["epsilon_wave_any"] = TensorMap(
    "epsilon_wave_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_epsilon_wave": 0, "epsilon_wave": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"epsilon_wave": {"epsilon wave"}},
        not_found_channel="no_epsilon_wave",
    ),
)


tmaps["incomplete_right_bundle_branch_block"] = TensorMap(
    "incomplete_right_bundle_branch_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_incomplete_right_bundle_branch_block": 0,
        "incomplete_right_bundle_branch_block": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "incomplete_right_bundle_branch_block": {
                "incomplete right bundle branch block",
            },
        },
        not_found_channel="no_incomplete_right_bundle_branch_block",
    ),
)


tmaps["incomplete_right_bundle_branch_block_any"] = TensorMap(
    "incomplete_right_bundle_branch_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_incomplete_right_bundle_branch_block": 0,
        "incomplete_right_bundle_branch_block": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "incomplete_right_bundle_branch_block": {
                "incomplete right bundle branch block",
            },
        },
        not_found_channel="no_incomplete_right_bundle_branch_block",
    ),
)


tmaps["intraventricular_conduction_delay"] = TensorMap(
    "intraventricular_conduction_delay",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_intraventricular_conduction_delay": 0,
        "intraventricular_conduction_delay": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "intraventricular_conduction_delay": {
                "intraventricular conduction delay",
                "intraventricular conduction defect",
            },
        },
        not_found_channel="no_intraventricular_conduction_delay",
    ),
)


tmaps["intraventricular_conduction_delay_any"] = TensorMap(
    "intraventricular_conduction_delay_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_intraventricular_conduction_delay": 0,
        "intraventricular_conduction_delay": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "intraventricular_conduction_delay": {
                "intraventricular conduction delay",
                "intraventricular conduction defect",
            },
        },
        not_found_channel="no_intraventricular_conduction_delay",
    ),
)


tmaps["left_anterior_fascicular_block"] = TensorMap(
    "left_anterior_fascicular_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_left_anterior_fascicular_block": 0,
        "left_anterior_fascicular_block": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_anterior_fascicular_block": {
                "left anterior fascicular block",
                "left anterior hemiblock",
            },
        },
        not_found_channel="no_left_anterior_fascicular_block",
    ),
)


tmaps["left_anterior_fascicular_block_any"] = TensorMap(
    "left_anterior_fascicular_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_left_anterior_fascicular_block": 0,
        "left_anterior_fascicular_block": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_anterior_fascicular_block": {
                "left anterior fascicular block",
                "left anterior hemiblock",
            },
        },
        not_found_channel="no_left_anterior_fascicular_block",
    ),
)


tmaps["left_atrial_conduction_abnormality"] = TensorMap(
    "left_atrial_conduction_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_left_atrial_conduction_abnormality": 0,
        "left_atrial_conduction_abnormality": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_atrial_conduction_abnormality": {
                "left atrial conduction abnormality",
            },
        },
        not_found_channel="no_left_atrial_conduction_abnormality",
    ),
)


tmaps["left_atrial_conduction_abnormality_any"] = TensorMap(
    "left_atrial_conduction_abnormality_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_left_atrial_conduction_abnormality": 0,
        "left_atrial_conduction_abnormality": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_atrial_conduction_abnormality": {
                "left atrial conduction abnormality",
            },
        },
        not_found_channel="no_left_atrial_conduction_abnormality",
    ),
)


tmaps["left_bundle_branch_block"] = TensorMap(
    "left_bundle_branch_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_left_bundle_branch_block": 0, "left_bundle_branch_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_bundle_branch_block": {
                "bundle branch block",
                "left bundle branch block",
                "lbbb",
                "left bbb",
            },
        },
        not_found_channel="no_left_bundle_branch_block",
    ),
)


tmaps["left_bundle_branch_block_any"] = TensorMap(
    "left_bundle_branch_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_left_bundle_branch_block": 0, "left_bundle_branch_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_bundle_branch_block": {
                "bundle branch block",
                "left bundle branch block",
                "lbbb",
                "left bbb",
            },
        },
        not_found_channel="no_left_bundle_branch_block",
    ),
)


tmaps["left_posterior_fascicular_block"] = TensorMap(
    "left_posterior_fascicular_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_left_posterior_fascicular_block": 0,
        "left_posterior_fascicular_block": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_posterior_fascicular_block": {
                "left posterior fascicular block",
                "left posterior hemiblock",
            },
        },
        not_found_channel="no_left_posterior_fascicular_block",
    ),
)


tmaps["left_posterior_fascicular_block_any"] = TensorMap(
    "left_posterior_fascicular_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_left_posterior_fascicular_block": 0,
        "left_posterior_fascicular_block": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_posterior_fascicular_block": {
                "left posterior fascicular block",
                "left posterior hemiblock",
            },
        },
        not_found_channel="no_left_posterior_fascicular_block",
    ),
)


tmaps["nonspecific_ivcd"] = TensorMap(
    "nonspecific_ivcd",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_nonspecific_ivcd": 0, "nonspecific_ivcd": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"nonspecific_ivcd": {"nonspecific ivcd"}},
        not_found_channel="no_nonspecific_ivcd",
    ),
)


tmaps["nonspecific_ivcd_any"] = TensorMap(
    "nonspecific_ivcd_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_nonspecific_ivcd": 0, "nonspecific_ivcd": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"nonspecific_ivcd": {"nonspecific ivcd"}},
        not_found_channel="no_nonspecific_ivcd",
    ),
)


tmaps["right_atrial_conduction_abnormality"] = TensorMap(
    "right_atrial_conduction_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_right_atrial_conduction_abnormality": 0,
        "right_atrial_conduction_abnormality": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "right_atrial_conduction_abnormality": {
                "right atrial conduction abnormality",
            },
        },
        not_found_channel="no_right_atrial_conduction_abnormality",
    ),
)


tmaps["right_atrial_conduction_abnormality_any"] = TensorMap(
    "right_atrial_conduction_abnormality_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_right_atrial_conduction_abnormality": 0,
        "right_atrial_conduction_abnormality": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "right_atrial_conduction_abnormality": {
                "right atrial conduction abnormality",
            },
        },
        not_found_channel="no_right_atrial_conduction_abnormality",
    ),
)


tmaps["right_bundle_branch_block"] = TensorMap(
    "right_bundle_branch_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_right_bundle_branch_block": 0, "right_bundle_branch_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "right_bundle_branch_block": {
                "right bundle branch block",
                "rbbb",
                "left bbb",
                "bundle branch block",
            },
        },
        not_found_channel="no_right_bundle_branch_block",
    ),
)


tmaps["right_bundle_branch_block_any"] = TensorMap(
    "right_bundle_branch_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_right_bundle_branch_block": 0, "right_bundle_branch_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "right_bundle_branch_block": {
                "right bundle branch block",
                "rbbb",
                "left bbb",
                "bundle branch block",
            },
        },
        not_found_channel="no_right_bundle_branch_block",
    ),
)


tmaps["ventricular_preexcitation"] = TensorMap(
    "ventricular_preexcitation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ventricular_preexcitation": 0, "ventricular_preexcitation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_preexcitation": {"ventricular preexcitation"}},
        not_found_channel="no_ventricular_preexcitation",
    ),
)


tmaps["ventricular_preexcitation_any"] = TensorMap(
    "ventricular_preexcitation_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_ventricular_preexcitation": 0, "ventricular_preexcitation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_preexcitation": {"ventricular preexcitation"}},
        not_found_channel="no_ventricular_preexcitation",
    ),
)


tmaps["brugada_pattern"] = TensorMap(
    "brugada_pattern",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_brugada_pattern": 0, "brugada_pattern": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"brugada_pattern": {"brugada pattern"}},
        not_found_channel="no_brugada_pattern",
    ),
)


tmaps["brugada_pattern_any"] = TensorMap(
    "brugada_pattern_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_brugada_pattern": 0, "brugada_pattern": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"brugada_pattern": {"brugada pattern"}},
        not_found_channel="no_brugada_pattern",
    ),
)


tmaps["digitalis_effect"] = TensorMap(
    "digitalis_effect",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_digitalis_effect": 0, "digitalis_effect": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"digitalis_effect": {"digitalis effect"}},
        not_found_channel="no_digitalis_effect",
    ),
)


tmaps["digitalis_effect_any"] = TensorMap(
    "digitalis_effect_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_digitalis_effect": 0, "digitalis_effect": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"digitalis_effect": {"digitalis effect"}},
        not_found_channel="no_digitalis_effect",
    ),
)


tmaps["early_repolarization"] = TensorMap(
    "early_repolarization",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_early_repolarization": 0, "early_repolarization": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"early_repolarization": {"early repolarization"}},
        not_found_channel="no_early_repolarization",
    ),
)


tmaps["early_repolarization_any"] = TensorMap(
    "early_repolarization_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_early_repolarization": 0, "early_repolarization": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"early_repolarization": {"early repolarization"}},
        not_found_channel="no_early_repolarization",
    ),
)


tmaps["inverted_u_waves"] = TensorMap(
    "inverted_u_waves",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_inverted_u_waves": 0, "inverted_u_waves": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"inverted_u_waves": {"inverted u waves"}},
        not_found_channel="no_inverted_u_waves",
    ),
)


tmaps["inverted_u_waves_any"] = TensorMap(
    "inverted_u_waves_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_inverted_u_waves": 0, "inverted_u_waves": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"inverted_u_waves": {"inverted u waves"}},
        not_found_channel="no_inverted_u_waves",
    ),
)


tmaps["ischemia"] = TensorMap(
    "ischemia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ischemia": 0, "ischemia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ischemia": {
                "antero-apical ischemia",
                "consider anterior and lateral ischemia",
                "anterolateral st segment depression",
                "diffuse elevation of st segments",
                "inferoapical st segment depression",
                "st segment depressions more marked",
                "st segment elevation",
                "anterolateral ischemia",
                "st segment depression is more marked in leads",
                "apical st depression",
                "consistent with subendocardial ischemia",
                "st segment elevation consistent with acute injury",
                "st segment depression in leads v4-v6",
                "suggesting anterior ischemia",
                "subendocardial ischemia",
                "anterolateral subendocardial ischemia",
                "marked st segment depression in leads",
                "nonspecific st segment depression",
                "diffuse st segment depression",
                "consistent with lateral ischemia",
                "consider anterior ischemia",
                "marked st segment depression",
                "anterior infarct or transmural ischemia",
                "inferior subendocardial ischemia",
                "lateral ischemia",
                "st segment depression",
                "st segment elevation in leads",
                "st segment depression in leads",
                "infero- st segment depression",
                "st depression",
                "apical subendocardial ischemia",
                "septal ischemia",
                "possible anterior wall ischemia",
                "anterior st segment depression",
                "anterior subendocardial ischemia",
                "minor st segment depression",
                "suggests anterolateral ischemia",
                "consistent with ischemia",
                "st elevation",
                "suggest anterior ischemia",
                "widespread st segment depression",
                "inferior st segment depression",
                "st segment depression in anterolateral leads",
                "inferior st segment elevation and q waves",
                "diffuse st segment elevation",
                "diffuse scooped st segment depression",
            },
        },
        not_found_channel="no_ischemia",
    ),
)


tmaps["ischemia_any"] = TensorMap(
    "ischemia_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_ischemia": 0, "ischemia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ischemia": {
                "antero-apical ischemia",
                "consider anterior and lateral ischemia",
                "anterolateral st segment depression",
                "diffuse elevation of st segments",
                "inferoapical st segment depression",
                "st segment depressions more marked",
                "st segment elevation",
                "anterolateral ischemia",
                "st segment depression is more marked in leads",
                "apical st depression",
                "consistent with subendocardial ischemia",
                "st segment elevation consistent with acute injury",
                "st segment depression in leads v4-v6",
                "suggesting anterior ischemia",
                "subendocardial ischemia",
                "anterolateral subendocardial ischemia",
                "marked st segment depression in leads",
                "nonspecific st segment depression",
                "diffuse st segment depression",
                "consistent with lateral ischemia",
                "consider anterior ischemia",
                "marked st segment depression",
                "anterior infarct or transmural ischemia",
                "inferior subendocardial ischemia",
                "lateral ischemia",
                "st segment depression",
                "st segment elevation in leads",
                "st segment depression in leads",
                "infero- st segment depression",
                "st depression",
                "apical subendocardial ischemia",
                "septal ischemia",
                "possible anterior wall ischemia",
                "anterior st segment depression",
                "anterior subendocardial ischemia",
                "minor st segment depression",
                "suggests anterolateral ischemia",
                "consistent with ischemia",
                "st elevation",
                "suggest anterior ischemia",
                "widespread st segment depression",
                "inferior st segment depression",
                "st segment depression in anterolateral leads",
                "inferior st segment elevation and q waves",
                "diffuse st segment elevation",
                "diffuse scooped st segment depression",
            },
        },
        not_found_channel="no_ischemia",
    ),
)


tmaps["metabolic_or_drug_effect"] = TensorMap(
    "metabolic_or_drug_effect",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_metabolic_or_drug_effect": 0, "metabolic_or_drug_effect": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"metabolic_or_drug_effect": {"metabolic or drug effect"}},
        not_found_channel="no_metabolic_or_drug_effect",
    ),
)


tmaps["metabolic_or_drug_effect_any"] = TensorMap(
    "metabolic_or_drug_effect_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_metabolic_or_drug_effect": 0, "metabolic_or_drug_effect": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"metabolic_or_drug_effect": {"metabolic or drug effect"}},
        not_found_channel="no_metabolic_or_drug_effect",
    ),
)


tmaps["osborn_wave"] = TensorMap(
    "osborn_wave",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_osborn_wave": 0, "osborn_wave": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"osborn_wave": {"osborn wave"}},
        not_found_channel="no_osborn_wave",
    ),
)


tmaps["osborn_wave_any"] = TensorMap(
    "osborn_wave_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_osborn_wave": 0, "osborn_wave": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"osborn_wave": {"osborn wave"}},
        not_found_channel="no_osborn_wave",
    ),
)


tmaps["pericarditis"] = TensorMap(
    "pericarditis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_pericarditis": 0, "pericarditis": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"pericarditis": {"pericarditis"}},
        not_found_channel="no_pericarditis",
    ),
)


tmaps["pericarditis_any"] = TensorMap(
    "pericarditis_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_pericarditis": 0, "pericarditis": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"pericarditis": {"pericarditis"}},
        not_found_channel="no_pericarditis",
    ),
)


tmaps["prominent_u_waves"] = TensorMap(
    "prominent_u_waves",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_prominent_u_waves": 0, "prominent_u_waves": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"prominent_u_waves": {"prominent u waves"}},
        not_found_channel="no_prominent_u_waves",
    ),
)


tmaps["prominent_u_waves_any"] = TensorMap(
    "prominent_u_waves_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_prominent_u_waves": 0, "prominent_u_waves": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"prominent_u_waves": {"prominent u waves"}},
        not_found_channel="no_prominent_u_waves",
    ),
)


tmaps["st_abnormality"] = TensorMap(
    "st_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_st_abnormality": 0, "st_abnormality": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "st_abnormality": {
                "anterolateral st segment depression",
                "diffuse elevation of st segments",
                "inferoapical st segment depression",
                "st segment depressions more marked",
                "st segment elevation",
                "st segment depression is more marked in leads",
                "apical st depression",
                "st segment elevation consistent with acute injury",
                "st segment abnormality",
                "st segment depression in leads v4-v6",
                "abnormal st segment changes",
                "marked st segment depression in leads",
                "nonspecific st segment depression",
                "diffuse st segment depression",
                "marked st segment depression",
                "st segment depression",
                "st segment elevation in leads",
                "st segment depression in leads",
                "infero- st segment depression",
                "st depression",
                "anterior st segment depression",
                "nonspecific st segment",
                "minor st segment depression",
                "st elevation",
                "nonspecific st segment and t wave abnormalities",
                "widespread st segment depression",
                "st segment depression in anterolateral leads",
                "inferior st segment elevation and q waves",
                "st segment changes",
                "diffuse st segment elevation",
                "diffuse scooped st segment depression",
            },
        },
        not_found_channel="no_st_abnormality",
    ),
)


tmaps["st_abnormality_any"] = TensorMap(
    "st_abnormality_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_st_abnormality": 0, "st_abnormality": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "st_abnormality": {
                "anterolateral st segment depression",
                "diffuse elevation of st segments",
                "inferoapical st segment depression",
                "st segment depressions more marked",
                "st segment elevation",
                "st segment depression is more marked in leads",
                "apical st depression",
                "st segment elevation consistent with acute injury",
                "st segment abnormality",
                "st segment depression in leads v4-v6",
                "abnormal st segment changes",
                "marked st segment depression in leads",
                "nonspecific st segment depression",
                "diffuse st segment depression",
                "marked st segment depression",
                "st segment depression",
                "st segment elevation in leads",
                "st segment depression in leads",
                "infero- st segment depression",
                "st depression",
                "anterior st segment depression",
                "nonspecific st segment",
                "minor st segment depression",
                "st elevation",
                "nonspecific st segment and t wave abnormalities",
                "widespread st segment depression",
                "st segment depression in anterolateral leads",
                "inferior st segment elevation and q waves",
                "st segment changes",
                "diffuse st segment elevation",
                "diffuse scooped st segment depression",
            },
        },
        not_found_channel="no_st_abnormality",
    ),
)


tmaps["st_or_t_change_due_to_ventricular_hypertrophy"] = TensorMap(
    "st_or_t_change_due_to_ventricular_hypertrophy",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_st_or_t_change_due_to_ventricular_hypertrophy": 0,
        "st_or_t_change_due_to_ventricular_hypertrophy": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "st_or_t_change_due_to_ventricular_hypertrophy": {
                "st or t change due to ventricular hypertrophy",
            },
        },
        not_found_channel="no_st_or_t_change_due_to_ventricular_hypertrophy",
    ),
)


tmaps["st_or_t_change_due_to_ventricular_hypertrophy_any"] = TensorMap(
    "st_or_t_change_due_to_ventricular_hypertrophy_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_st_or_t_change_due_to_ventricular_hypertrophy": 0,
        "st_or_t_change_due_to_ventricular_hypertrophy": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "st_or_t_change_due_to_ventricular_hypertrophy": {
                "st or t change due to ventricular hypertrophy",
            },
        },
        not_found_channel="no_st_or_t_change_due_to_ventricular_hypertrophy",
    ),
)


tmaps["t_wave_abnormality"] = TensorMap(
    "t_wave_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_t_wave_abnormality": 0, "t_wave_abnormality": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "t_wave_abnormality": {
                "t waves are slightly more inverted in leads",
                "t wave inversion",
                "t wave inversions",
                "recent diffuse t wave flattening",
                "t wave flattening",
                "(nonspecific st segment).*(t wave abnormalities)",
                "t wave abnormalities",
                "t wave changes",
                "t wave inversion in leads",
                "t waves are inverted in leads",
                "upright t waves",
                "tall t waves in precordial leads",
                "possible st segment and t wave abn",
                "t wave inver",
                "nonspecific t wave abnormali",
                "t waves are upright in leads",
                "nonspecific st segment and t wave abnormalities",
                "t wave inveions",
                "diffuse nonspecific st segment and t wave abnormalities",
                "t waves are lower or inverted in leads",
            },
        },
        not_found_channel="no_t_wave_abnormality",
    ),
)


tmaps["t_wave_abnormality_any"] = TensorMap(
    "t_wave_abnormality_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_t_wave_abnormality": 0, "t_wave_abnormality": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "t_wave_abnormality": {
                "t waves are slightly more inverted in leads",
                "t wave inversion",
                "t wave inversions",
                "recent diffuse t wave flattening",
                "t wave flattening",
                "(nonspecific st segment).*(t wave abnormalities)",
                "t wave abnormalities",
                "t wave changes",
                "t wave inversion in leads",
                "t waves are inverted in leads",
                "upright t waves",
                "tall t waves in precordial leads",
                "possible st segment and t wave abn",
                "t wave inver",
                "nonspecific t wave abnormali",
                "t waves are upright in leads",
                "nonspecific st segment and t wave abnormalities",
                "t wave inveions",
                "diffuse nonspecific st segment and t wave abnormalities",
                "t waves are lower or inverted in leads",
            },
        },
        not_found_channel="no_t_wave_abnormality",
    ),
)


tmaps["tu_fusion"] = TensorMap(
    "tu_fusion",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_tu_fusion": 0, "tu_fusion": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"tu_fusion": {"tu fusion"}},
        not_found_channel="no_tu_fusion",
    ),
)


tmaps["tu_fusion_any"] = TensorMap(
    "tu_fusion_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_tu_fusion": 0, "tu_fusion": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"tu_fusion": {"tu fusion"}},
        not_found_channel="no_tu_fusion",
    ),
)


tmaps["lae"] = TensorMap(
    "lae",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_lae": 0, "lae": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lae": {
                "combined atrial enlargement",
                "left atrial enla",
                "biatrial enlargement",
                "biatrial hypertrophy",
            },
        },
        not_found_channel="no_lae",
    ),
)


tmaps["lae_any"] = TensorMap(
    "lae_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_lae": 0, "lae": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lae": {
                "combined atrial enlargement",
                "left atrial enla",
                "biatrial enlargement",
                "biatrial hypertrophy",
            },
        },
        not_found_channel="no_lae",
    ),
)


tmaps["lvh"] = TensorMap(
    "lvh",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_lvh": 0, "lvh": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lvh": {
                "biventriclar hypertrophy",
                "left ventricular hypertr",
                "leftventricular hypertrophy",
                "combined ventricular hypertrophy",
                "biventricular hypertrophy",
            },
        },
        not_found_channel="no_lvh",
    ),
)


tmaps["lvh_any"] = TensorMap(
    "lvh_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_lvh": 0, "lvh": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lvh": {
                "biventriclar hypertrophy",
                "left ventricular hypertr",
                "leftventricular hypertrophy",
                "combined ventricular hypertrophy",
                "biventricular hypertrophy",
            },
        },
        not_found_channel="no_lvh",
    ),
)


tmaps["rae"] = TensorMap(
    "rae",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_rae": 0, "rae": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "rae": {
                "combined atrial enlargement",
                "biatrial enlargement",
                "right atrial enla",
                "biatrial hypertrophy",
            },
        },
        not_found_channel="no_rae",
    ),
)


tmaps["rae_any"] = TensorMap(
    "rae_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_rae": 0, "rae": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "rae": {
                "combined atrial enlargement",
                "biatrial enlargement",
                "right atrial enla",
                "biatrial hypertrophy",
            },
        },
        not_found_channel="no_rae",
    ),
)


tmaps["rvh"] = TensorMap(
    "rvh",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_rvh": 0, "rvh": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "rvh": {
                "right ventricular hypertrophy",
                "biventriclar hypertrophy",
                "rightventricular hypertrophy",
                "right ventricular enlargement",
                "combined ventricular hypertrophy",
                "biventricular hypertrophy",
            },
        },
        not_found_channel="no_rvh",
    ),
)


tmaps["rvh_any"] = TensorMap(
    "rvh_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_rvh": 0, "rvh": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "rvh": {
                "right ventricular hypertrophy",
                "biventriclar hypertrophy",
                "rightventricular hypertrophy",
                "right ventricular enlargement",
                "combined ventricular hypertrophy",
                "biventricular hypertrophy",
            },
        },
        not_found_channel="no_rvh",
    ),
)


tmaps["sh"] = TensorMap(
    "sh",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_sh": 0, "sh": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sh": {"septal hypertrophy", "septal lipomatous hypertrophy"}},
        not_found_channel="no_sh",
    ),
)


tmaps["sh_any"] = TensorMap(
    "sh_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_sh": 0, "sh": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sh": {"septal hypertrophy", "septal lipomatous hypertrophy"}},
        not_found_channel="no_sh",
    ),
)


tmaps["atrial_premature_complexes"] = TensorMap(
    "atrial_premature_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_atrial_premature_complexes": 0, "atrial_premature_complexes": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_premature_complexes": {
                "atrial trigeminy",
                "atrial ectopy has decreased",
                "atrial ectopy",
                "atrial premature beat",
                "isolated premature atrial contractions",
                "ectopic atrial complexes",
                "premature atrial complexes",
                "atrial premature complexes",
                "atrial bigeminy",
                "premature atrial co",
            },
        },
        not_found_channel="no_atrial_premature_complexes",
    ),
)


tmaps["atrial_premature_complexes_any"] = TensorMap(
    "atrial_premature_complexes_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_atrial_premature_complexes": 0, "atrial_premature_complexes": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_premature_complexes": {
                "atrial trigeminy",
                "atrial ectopy has decreased",
                "atrial ectopy",
                "atrial premature beat",
                "isolated premature atrial contractions",
                "ectopic atrial complexes",
                "premature atrial complexes",
                "atrial premature complexes",
                "atrial bigeminy",
                "premature atrial co",
            },
        },
        not_found_channel="no_atrial_premature_complexes",
    ),
)


tmaps["ectopy"] = TensorMap(
    "ectopy",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ectopy": 0, "ectopy": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopy": {
                "new ectopy",
                "increased ectopy",
                "return of ectopy",
                "other than the ectopy",
                "ectopy more pronounced",
                "ectopy is new",
                "ectopy have increased",
                "ectopy has appeared",
                "ectopy has increased",
            },
        },
        not_found_channel="no_ectopy",
    ),
)


tmaps["ectopy_any"] = TensorMap(
    "ectopy_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_ectopy": 0, "ectopy": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopy": {
                "new ectopy",
                "increased ectopy",
                "return of ectopy",
                "other than the ectopy",
                "ectopy more pronounced",
                "ectopy is new",
                "ectopy have increased",
                "ectopy has appeared",
                "ectopy has increased",
            },
        },
        not_found_channel="no_ectopy",
    ),
)


tmaps["junctional_premature_complexes"] = TensorMap(
    "junctional_premature_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_junctional_premature_complexes": 0,
        "junctional_premature_complexes": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "junctional_premature_complexes": {
                "junctional premature complexes",
                "junctional premature beats",
            },
        },
        not_found_channel="no_junctional_premature_complexes",
    ),
)


tmaps["junctional_premature_complexes_any"] = TensorMap(
    "junctional_premature_complexes_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_junctional_premature_complexes": 0,
        "junctional_premature_complexes": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "junctional_premature_complexes": {
                "junctional premature complexes",
                "junctional premature beats",
            },
        },
        not_found_channel="no_junctional_premature_complexes",
    ),
)


tmaps["no_ectopy"] = TensorMap(
    "no_ectopy",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_no_ectopy": 0, "no_ectopy": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "no_ectopy": {
                "no longer any ectopy",
                "ectopy is gone",
                "no ectopy",
                "ectopy has resolved",
                "ectopy has disappear",
                "atrial ectopy gone",
                "ectopy is no longer seen",
            },
        },
        not_found_channel="no_no_ectopy",
    ),
)


tmaps["no_ectopy_any"] = TensorMap(
    "no_ectopy_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_no_ectopy": 0, "no_ectopy": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "no_ectopy": {
                "no longer any ectopy",
                "ectopy is gone",
                "no ectopy",
                "ectopy has resolved",
                "ectopy has disappear",
                "atrial ectopy gone",
                "ectopy is no longer seen",
            },
        },
        not_found_channel="no_no_ectopy",
    ),
)


tmaps["premature_supraventricular_complexes"] = TensorMap(
    "premature_supraventricular_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_premature_supraventricular_complexes": 0,
        "premature_supraventricular_complexes": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "premature_supraventricular_complexes": {
                "premature supraventricular complexes",
            },
        },
        not_found_channel="no_premature_supraventricular_complexes",
    ),
)


tmaps["premature_supraventricular_complexes_any"] = TensorMap(
    "premature_supraventricular_complexes_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_premature_supraventricular_complexes": 0,
        "premature_supraventricular_complexes": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "premature_supraventricular_complexes": {
                "premature supraventricular complexes",
            },
        },
        not_found_channel="no_premature_supraventricular_complexes",
    ),
)


tmaps["ventricular_premature_complexes"] = TensorMap(
    "ventricular_premature_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_ventricular_premature_complexes": 0,
        "ventricular_premature_complexes": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ventricular_premature_complexes": {
                "ventricular bigeminy",
                "occasional premature ventricular complexes ",
                "ventricular ectopy",
                "premature ventricular beat",
                "ventricular premature beat",
                "ventriculaar ectopy is now present",
                "ventricular trigeminy",
                "isolated premature ventricular contractions",
                "premature ventricular compl",
                "frequent premature ventricular or aberrantly conducted complexes ",
                "premature ventricular and fusion complexes",
                "one premature ventricularbeat",
                "ventricular premature complexes",
            },
        },
        not_found_channel="no_ventricular_premature_complexes",
    ),
)


tmaps["ventricular_premature_complexes_any"] = TensorMap(
    "ventricular_premature_complexes_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_ventricular_premature_complexes": 0,
        "ventricular_premature_complexes": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ventricular_premature_complexes": {
                "ventricular bigeminy",
                "occasional premature ventricular complexes ",
                "ventricular ectopy",
                "premature ventricular beat",
                "ventricular premature beat",
                "ventriculaar ectopy is now present",
                "ventricular trigeminy",
                "isolated premature ventricular contractions",
                "premature ventricular compl",
                "frequent premature ventricular or aberrantly conducted complexes ",
                "premature ventricular and fusion complexes",
                "one premature ventricularbeat",
                "ventricular premature complexes",
            },
        },
        not_found_channel="no_ventricular_premature_complexes",
    ),
)


tmaps["_2_to_1_av_block"] = TensorMap(
    "_2_to_1_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no__2_to_1_av_block": 0, "_2_to_1_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "_2_to_1_av_block": {
                "2:1 av block",
                "2:1 atrioventricular block",
                "2 to 1 atrioventricular block",
                "2 to 1 av block",
                "2:1 block",
            },
        },
        not_found_channel="no__2_to_1_av_block",
    ),
)


tmaps["_2_to_1_av_block_any"] = TensorMap(
    "_2_to_1_av_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no__2_to_1_av_block": 0, "_2_to_1_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "_2_to_1_av_block": {
                "2:1 av block",
                "2:1 atrioventricular block",
                "2 to 1 atrioventricular block",
                "2 to 1 av block",
                "2:1 block",
            },
        },
        not_found_channel="no__2_to_1_av_block",
    ),
)


tmaps["_4_to_1_av_block"] = TensorMap(
    "_4_to_1_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no__4_to_1_av_block": 0, "_4_to_1_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"_4_to_1_av_block": {"4:1atrioventricular conduction"}},
        not_found_channel="no__4_to_1_av_block",
    ),
)


tmaps["_4_to_1_av_block_any"] = TensorMap(
    "_4_to_1_av_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no__4_to_1_av_block": 0, "_4_to_1_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"_4_to_1_av_block": {"4:1atrioventricular conduction"}},
        not_found_channel="no__4_to_1_av_block",
    ),
)


tmaps["av_dissociation"] = TensorMap(
    "av_dissociation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_av_dissociation": 0, "av_dissociation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "av_dissociation": {"atrioventricular dissociation", "av dissociation"},
        },
        not_found_channel="no_av_dissociation",
    ),
)


tmaps["av_dissociation_any"] = TensorMap(
    "av_dissociation_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_av_dissociation": 0, "av_dissociation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "av_dissociation": {"atrioventricular dissociation", "av dissociation"},
        },
        not_found_channel="no_av_dissociation",
    ),
)


tmaps["mobitz_type_i_second_degree_av_block_"] = TensorMap(
    "mobitz_type_i_second_degree_av_block_",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_mobitz_type_i_second_degree_av_block_": 0,
        "mobitz_type_i_second_degree_av_block_": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_i_second_degree_av_block_": {
                "mobitz i",
                "fixed block",
                "mobitz type 1",
                "second degree type 1",
                "wenckebach",
                "second degree ",
                "mobitz 1 block",
            },
        },
        not_found_channel="no_mobitz_type_i_second_degree_av_block_",
    ),
)


tmaps["mobitz_type_i_second_degree_av_block__any"] = TensorMap(
    "mobitz_type_i_second_degree_av_block__any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_mobitz_type_i_second_degree_av_block_": 0,
        "mobitz_type_i_second_degree_av_block_": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_i_second_degree_av_block_": {
                "mobitz i",
                "fixed block",
                "mobitz type 1",
                "second degree type 1",
                "wenckebach",
                "second degree ",
                "mobitz 1 block",
            },
        },
        not_found_channel="no_mobitz_type_i_second_degree_av_block_",
    ),
)


tmaps["mobitz_type_ii_second_degree_av_block"] = TensorMap(
    "mobitz_type_ii_second_degree_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_mobitz_type_ii_second_degree_av_block": 0,
        "mobitz_type_ii_second_degree_av_block": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_ii_second_degree_av_block": {
                "hay block",
                "2nd degree sa block",
                "mobitz ii",
                "second degree type 2",
            },
        },
        not_found_channel="no_mobitz_type_ii_second_degree_av_block",
    ),
)


tmaps["mobitz_type_ii_second_degree_av_block_any"] = TensorMap(
    "mobitz_type_ii_second_degree_av_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={
        "no_mobitz_type_ii_second_degree_av_block": 0,
        "mobitz_type_ii_second_degree_av_block": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_ii_second_degree_av_block": {
                "hay block",
                "2nd degree sa block",
                "mobitz ii",
                "second degree type 2",
            },
        },
        not_found_channel="no_mobitz_type_ii_second_degree_av_block",
    ),
)


tmaps["third_degree_av_block"] = TensorMap(
    "third_degree_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_third_degree_av_block": 0, "third_degree_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "third_degree_av_block": {
                "3rd degree av block",
                "third degree av block",
                "complete heart block",
                "third degree atrioventricular block",
                "3rd degree atrioventricular block",
            },
        },
        not_found_channel="no_third_degree_av_block",
    ),
)


tmaps["third_degree_av_block_any"] = TensorMap(
    "third_degree_av_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_third_degree_av_block": 0, "third_degree_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "third_degree_av_block": {
                "3rd degree av block",
                "third degree av block",
                "complete heart block",
                "third degree atrioventricular block",
                "3rd degree atrioventricular block",
            },
        },
        not_found_channel="no_third_degree_av_block",
    ),
)


tmaps["unspecified_av_block"] = TensorMap(
    "unspecified_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_unspecified_av_block": 0, "unspecified_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified_av_block": {
                "high degree of block",
                "atrioventricular block",
                "heart block",
                "heartblock",
                "high grade atrioventricular block",
                "av block",
            },
        },
        not_found_channel="no_unspecified_av_block",
    ),
)


tmaps["unspecified_av_block_any"] = TensorMap(
    "unspecified_av_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_unspecified_av_block": 0, "unspecified_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified_av_block": {
                "high degree of block",
                "atrioventricular block",
                "heart block",
                "heartblock",
                "high grade atrioventricular block",
                "av block",
            },
        },
        not_found_channel="no_unspecified_av_block",
    ),
)


tmaps["variable_av_block"] = TensorMap(
    "variable_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_variable_av_block": 0, "variable_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "variable_av_block": {"variable block", "varying degree of block"},
        },
        not_found_channel="no_variable_av_block",
    ),
)


tmaps["variable_av_block_any"] = TensorMap(
    "variable_av_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix="ECG_PREFIX",
    channel_map={"no_variable_av_block": 0, "variable_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "variable_av_block": {"variable block", "varying degree of block"},
        },
        not_found_channel="no_variable_av_block",
    ),
)
