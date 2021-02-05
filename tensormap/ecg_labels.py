# Imports: standard library
from typing import Dict

# Imports: first party
from tensormap.ecg import (
    make_ecg_label_from_read_tff,
    make_binary_ecg_label_from_any_read_tff,
)
from definitions.ecg import ECG_PREFIX
from tensormap.TensorMap import TensorMap, Interpretation
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
    path_prefix=ECG_PREFIX,
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
                "afib",
                "afibrillation",
                "fibrillation/flutter",
                "atrial fibrillation",
                "atrial fibrillation with moderate ventricular response",
                "atrial fib",
                "atrial fibrillation with rapid ventricular response",
                "atrialfibrillation",
                "atrial fibrillation with controlled ventricular response",
            },
        },
        not_found_channel="no_atrial_fibrillation",
    ),
)


tmaps["atrial_fibrillation_any"] = TensorMap(
    "atrial_fibrillation_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_atrial_fibrillation": 0, "atrial_fibrillation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_fibrillation": {
                "afib",
                "afibrillation",
                "fibrillation/flutter",
                "atrial fibrillation",
                "atrial fibrillation with moderate ventricular response",
                "atrial fib",
                "atrial fibrillation with rapid ventricular response",
                "atrialfibrillation",
                "atrial fibrillation with controlled ventricular response",
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
                "flutter",
                "aflutter",
                "fibrillation/flutter",
                "probable flutter",
                "atrial flutter fixed block",
                "atrial flutter",
                "tachycardia possibly flutter",
                "atrial flutter variable block",
                "atrial flutter unspecified block",
            },
        },
        not_found_channel="no_atrial_flutter",
    ),
)


tmaps["atrial_flutter_any"] = TensorMap(
    "atrial_flutter_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_atrial_flutter": 0, "atrial_flutter": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_flutter": {
                "flutter",
                "aflutter",
                "fibrillation/flutter",
                "probable flutter",
                "atrial flutter fixed block",
                "atrial flutter",
                "tachycardia possibly flutter",
                "atrial flutter variable block",
                "atrial flutter unspecified block",
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
        channel_terms={"atrial_paced_rhythm": {"atrial paced rhythm", "atrial pacing"}},
        not_found_channel="no_atrial_paced_rhythm",
    ),
)


tmaps["atrial_paced_rhythm_any"] = TensorMap(
    "atrial_paced_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_atrial_paced_rhythm": 0, "atrial_paced_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"atrial_paced_rhythm": {"atrial paced rhythm", "atrial pacing"}},
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
                "ectopic atrial bradycardia",
                "low atrial bradycardia",
            },
        },
        not_found_channel="no_ectopic_atrial_bradycardia",
    ),
)


tmaps["ectopic_atrial_bradycardia_any"] = TensorMap(
    "ectopic_atrial_bradycardia_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ectopic_atrial_bradycardia": 0, "ectopic_atrial_bradycardia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_bradycardia": {
                "ectopic atrial bradycardia",
                "low atrial bradycardia",
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
                "wandering ectopic atrial rhythm",
                "wandering atrial pacemaker",
                "unusual p wave axis",
                "multifocal ear",
                "low atrial pacer",
                "multifocal atrialrhythm",
                "unifocal ectopic atrial rhythm",
                "dual atrial foci ",
                "p wave axis suggests atrial rather than sinus mechanism",
                "ectopicsupraventricular rhythm",
                "atrial rhythm",
                "multifocal ectopic atrial rhythm",
                "atrial arrhythmia",
                "ectopic atrial rhythm",
                "multiple atrial foci",
                "multifocal atrial rhythm",
                "ectopic atrial rhythm ",
                "abnormal p vector",
                "unifocal ear",
                "wandering ear",
                "nonsinus atrial mechanism",
            },
        },
        not_found_channel="no_ectopic_atrial_rhythm",
    ),
)


tmaps["ectopic_atrial_rhythm_any"] = TensorMap(
    "ectopic_atrial_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ectopic_atrial_rhythm": 0, "ectopic_atrial_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_rhythm": {
                "wandering ectopic atrial rhythm",
                "wandering atrial pacemaker",
                "unusual p wave axis",
                "multifocal ear",
                "low atrial pacer",
                "multifocal atrialrhythm",
                "unifocal ectopic atrial rhythm",
                "dual atrial foci ",
                "p wave axis suggests atrial rather than sinus mechanism",
                "ectopicsupraventricular rhythm",
                "atrial rhythm",
                "multifocal ectopic atrial rhythm",
                "atrial arrhythmia",
                "ectopic atrial rhythm",
                "multiple atrial foci",
                "multifocal atrial rhythm",
                "ectopic atrial rhythm ",
                "abnormal p vector",
                "unifocal ear",
                "wandering ear",
                "nonsinus atrial mechanism",
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
                "ectopic atrial tachycardia, unifocal",
                "wandering atrial tachycardia",
                "multifocal ectopic atrial tachycardia",
                "unspecified ectopic atrial tachycardia",
                "multifocal atrial tachycardia",
                "unifocal atrial tachycardia",
                "ectopic atrial tachycardia, multifocal",
                "ectopic atrial tachycardia",
                "ectopic atrial tachycardia, unspecified",
                "unifocal ectopic atrial tachycardia",
            },
        },
        not_found_channel="no_ectopic_atrial_tachycardia",
    ),
)


tmaps["ectopic_atrial_tachycardia_any"] = TensorMap(
    "ectopic_atrial_tachycardia_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ectopic_atrial_tachycardia": 0, "ectopic_atrial_tachycardia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_tachycardia": {
                "ectopic atrial tachycardia, unifocal",
                "wandering atrial tachycardia",
                "multifocal ectopic atrial tachycardia",
                "unspecified ectopic atrial tachycardia",
                "multifocal atrial tachycardia",
                "unifocal atrial tachycardia",
                "ectopic atrial tachycardia, multifocal",
                "ectopic atrial tachycardia",
                "ectopic atrial tachycardia, unspecified",
                "unifocal ectopic atrial tachycardia",
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
    path_prefix=ECG_PREFIX,
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
                "pulseless electrical activity",
                "pulseless",
            },
        },
        not_found_channel="no_pulseless_electrical_activity",
    ),
)


tmaps["pulseless_electrical_activity_any"] = TensorMap(
    "pulseless_electrical_activity_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_pulseless_electrical_activity": 0,
        "pulseless_electrical_activity": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "pulseless_electrical_activity": {
                "pulseless electrical activity",
                "pulseless",
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
                "sinus bradycardia",
                "normal sinus rhythm",
                "marked sinus arrhythmia",
                "sinus rhythm",
                "atrialbigeminy",
                "sa exit block",
                "frequent native sinus beats",
                "normal ecg",
                "atrial bigeminal  rhythm",
                "type i sa block",
                "type ii sa block",
                "2nd degree sa block",
                "sinus slowing",
                "conducted sinus impulses",
                "type ii sinoatrial block",
                "sinus arrhythmia",
                "tracing within normal limits",
                "atrial bigeminal rhythm",
                "sinoatrial block, type ii",
                "sa block, type i",
                "atrial trigeminy",
                "rhythm is now clearly sinus",
                "sinus exit block",
                "type i sinoatrial block",
                "with occasional native sinus beats",
                "sinus rhythm at a rate",
                "rhythm is normal sinus",
                "sinus tachycardia",
                "sinus mechanism has replaced",
                "rhythm remains normal sinus",
                "sa block",
                "1st degree sa block",
                "tracing is within normal limits",
                "atrial bigeminy and ventricular bigeminy",
                "rhythm has reverted to normal",
                "sinoatrial block",
                "normal when compared with ecg of",
            },
        },
        not_found_channel="no_sinus_rhythm",
    ),
)


tmaps["sinus_rhythm_any"] = TensorMap(
    "sinus_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_sinus_rhythm": 0, "sinus_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "sinus_rhythm": {
                "sinus bradycardia",
                "normal sinus rhythm",
                "marked sinus arrhythmia",
                "sinus rhythm",
                "atrialbigeminy",
                "sa exit block",
                "frequent native sinus beats",
                "normal ecg",
                "atrial bigeminal  rhythm",
                "type i sa block",
                "type ii sa block",
                "2nd degree sa block",
                "sinus slowing",
                "conducted sinus impulses",
                "type ii sinoatrial block",
                "sinus arrhythmia",
                "tracing within normal limits",
                "atrial bigeminal rhythm",
                "sinoatrial block, type ii",
                "sa block, type i",
                "atrial trigeminy",
                "rhythm is now clearly sinus",
                "sinus exit block",
                "type i sinoatrial block",
                "with occasional native sinus beats",
                "sinus rhythm at a rate",
                "rhythm is normal sinus",
                "sinus tachycardia",
                "sinus mechanism has replaced",
                "rhythm remains normal sinus",
                "sa block",
                "1st degree sa block",
                "tracing is within normal limits",
                "atrial bigeminy and ventricular bigeminy",
                "rhythm has reverted to normal",
                "sinoatrial block",
                "normal when compared with ecg of",
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
                "atrial tachycardia",
                "atrioventricular reentrant tachycardia ",
                "avnrt",
                "accelerated atrioventricular nodal rhythm",
                "av reentrant tachycardia ",
                "av nodal reentrant",
                "supraventricular tachycardia",
                "atrioventricular nodal reentry tachycardia",
                "accelerated nodal rhythm",
                "avrt",
                "av nodal reentry tachycardia",
                "junctional tachycardia",
                "accelerated atrioventricular junctional rhythm",
            },
        },
        not_found_channel="no_supraventricular_tachycardia",
    ),
)


tmaps["supraventricular_tachycardia_any"] = TensorMap(
    "supraventricular_tachycardia_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_supraventricular_tachycardia": 0,
        "supraventricular_tachycardia": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "supraventricular_tachycardia": {
                "atrial tachycardia",
                "atrioventricular reentrant tachycardia ",
                "avnrt",
                "accelerated atrioventricular nodal rhythm",
                "av reentrant tachycardia ",
                "av nodal reentrant",
                "supraventricular tachycardia",
                "atrioventricular nodal reentry tachycardia",
                "accelerated nodal rhythm",
                "avrt",
                "av nodal reentry tachycardia",
                "junctional tachycardia",
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
    path_prefix=ECG_PREFIX,
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
                "undetermined  rhythm",
                "rhythm uncertain",
                "rhythm unclear",
                "uncertain rhythm",
            },
        },
        not_found_channel="no_unspecified",
    ),
)


tmaps["unspecified_any"] = TensorMap(
    "unspecified_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_unspecified": 0, "unspecified": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified": {
                "undetermined  rhythm",
                "rhythm uncertain",
                "rhythm unclear",
                "uncertain rhythm",
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
    path_prefix=ECG_PREFIX,
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
            "wpw": {"wolffparkinsonwhite", "wpw", "wolff-parkinson-white pattern"},
        },
        not_found_channel="no_wpw",
    ),
)


tmaps["wpw_any"] = TensorMap(
    "wpw_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_wpw": 0, "wpw": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "wpw": {"wolffparkinsonwhite", "wpw", "wolff-parkinson-white pattern"},
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
                "av dual-paced rhythm",
                "biventricular-paced complexes",
                "v-paced",
                "failure to inhibit ventricular",
                "demand ventricular pacemaker",
                "atrial-sensed ventricular-paced complexes",
                "failure to pace atrial",
                "sequential pacing",
                "atrial-paced rhythm",
                "ventricular demand pacing",
                "dual chamber pacing",
                "atrial triggered ventricular pacing",
                "ventricular pacing has replaced av pacing",
                "ventricular pacing",
                "ventricular paced",
                "atrial-paced complexes ",
                "failure to capture atrial",
                "biventricular-paced rhythm",
                "demand v-pacing",
                "shows dual chamber pacing",
                "atrially triggered v paced",
                "av dual-paced complexes",
                "failure to capture ventricular",
                "failure to pace ventricular",
                "atrial-sensed ventricular-paced rhythm",
                "competitive av pacing",
                "failure to inhibit atrial",
                "ventricular-paced rhythm",
                "ventricular-paced complexes",
                "unipolar right ventricular  pacing",
                "v-paced beats",
                "v-paced rhythm",
                "a triggered v-paced rhythm",
                "electronic pacemaker",
            },
        },
        not_found_channel="no_pacemaker",
    ),
)


tmaps["pacemaker_any"] = TensorMap(
    "pacemaker_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_pacemaker": 0, "pacemaker": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "pacemaker": {
                "av dual-paced rhythm",
                "biventricular-paced complexes",
                "v-paced",
                "failure to inhibit ventricular",
                "demand ventricular pacemaker",
                "atrial-sensed ventricular-paced complexes",
                "failure to pace atrial",
                "sequential pacing",
                "atrial-paced rhythm",
                "ventricular demand pacing",
                "dual chamber pacing",
                "atrial triggered ventricular pacing",
                "ventricular pacing has replaced av pacing",
                "ventricular pacing",
                "ventricular paced",
                "atrial-paced complexes ",
                "failure to capture atrial",
                "biventricular-paced rhythm",
                "demand v-pacing",
                "shows dual chamber pacing",
                "atrially triggered v paced",
                "av dual-paced complexes",
                "failure to capture ventricular",
                "failure to pace ventricular",
                "atrial-sensed ventricular-paced rhythm",
                "competitive av pacing",
                "failure to inhibit atrial",
                "ventricular-paced rhythm",
                "ventricular-paced complexes",
                "unipolar right ventricular  pacing",
                "v-paced beats",
                "v-paced rhythm",
                "a triggered v-paced rhythm",
                "electronic pacemaker",
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
    path_prefix=ECG_PREFIX,
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
                "normal sinus rhythm",
                "sinus rhythm",
                "tracing is within normal limits",
                "sinus tachycardia",
                "normal tracing",
                "normal ecg",
            },
        },
        not_found_channel="no_normal_sinus_rhythm",
    ),
)


tmaps["normal_sinus_rhythm_any"] = TensorMap(
    "normal_sinus_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_normal_sinus_rhythm": 0, "normal_sinus_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "normal_sinus_rhythm": {
                "normal sinus rhythm",
                "sinus rhythm",
                "tracing is within normal limits",
                "sinus tachycardia",
                "normal tracing",
                "normal ecg",
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
    path_prefix=ECG_PREFIX,
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
                "indeterminate qrs axis",
                "indeterminate axis",
                "northwest axis",
            },
        },
        not_found_channel="no_indeterminate_axis",
    ),
)


tmaps["indeterminate_axis_any"] = TensorMap(
    "indeterminate_axis_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_indeterminate_axis": 0, "indeterminate_axis": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "indeterminate_axis": {
                "indeterminate qrs axis",
                "indeterminate axis",
                "northwest axis",
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
                "left axis deviation",
                "leftward axis",
                "axis shifted left",
            },
        },
        not_found_channel="no_left_axis_deviation",
    ),
)


tmaps["left_axis_deviation_any"] = TensorMap(
    "left_axis_deviation_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_left_axis_deviation": 0, "left_axis_deviation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_axis_deviation": {
                "left axis deviation",
                "leftward axis",
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
                "slow precordial r wave progression",
                "poor r wave progression",
                "abnormal precordial r wave progression",
                "early r wave progression",
                "unusual r wave progression",
                "slowprecordial r wave progression",
                "poor precordial r wave progression",
                "abnormal precordial r wave progression or poor r wave progression",
            },
        },
        not_found_channel="no_poor_r_wave_progression",
    ),
)


tmaps["poor_r_wave_progression_any"] = TensorMap(
    "poor_r_wave_progression_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_poor_r_wave_progression": 0, "poor_r_wave_progression": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "poor_r_wave_progression": {
                "slow precordial r wave progression",
                "poor r wave progression",
                "abnormal precordial r wave progression",
                "early r wave progression",
                "unusual r wave progression",
                "slowprecordial r wave progression",
                "poor precordial r wave progression",
                "abnormal precordial r wave progression or poor r wave progression",
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
                "reversed r wave progression",
                "reverse r wave progression",
            },
        },
        not_found_channel="no_reversed_r_wave_progression",
    ),
)


tmaps["reversed_r_wave_progression_any"] = TensorMap(
    "reversed_r_wave_progression_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_reversed_r_wave_progression": 0, "reversed_r_wave_progression": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "reversed_r_wave_progression": {
                "reversed r wave progression",
                "reverse r wave progression",
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
                "acute anterior wall myocardial infarction",
                "possible acute myocardial infarction",
                "acute myocardial infarction in evolution",
                "extensive myocardial infarction of indeterminate age ",
                "acute anterior infarct",
                "cannot rule out inferoposterior myoca",
                "evolving inferior wall myocardial infarction",
                "subendocardial ischemia subendocardial myocardial inf",
                "anterolateral myocardial infarction",
                "anterior infarct of indeterminate age",
                "old inferior wall myocardial infarction",
                "myocardial infarction compared with the last previous tracing ",
                "lateral myocardial infarction of indeterminate age",
                "(counterclockwise rotation).*(true posterior)",
                "anteroseptal myocardial infarction",
                "cannot rule out anterior wall ischemia myocardial infarction",
                "inferoapical myocardial infarction of indeterminate age",
                "lateral myocardial infarction - of indeterminate age",
                "old infero-posterior lateral myocardial infarction",
                "acute myocardial infarction of indeterminate age",
                "possible septal myocardial infarction",
                "myocardial infarction possible when compared",
                "inferior myocardial infarction of indeterminate",
                "posterior myocardial infarction",
                "evolving anterior infarct",
                "anterolateral myocardial infarction , possibly recent  ",
                "anterior myocardial infarction of indeterminate age",
                "myocardial infarction extension",
                "inferolateral myocardial infarction",
                "myocardial infarction compared with the last previous ",
                "old posterolateral myocardial infarction",
                "apical myocardial infarction of indeterminate age",
                "old anterior myocardial infarction",
                "inferior myocardial infarction of indeterminate age",
                "cannot rule out true posterior myocardial infarction versus counterclockwise rotation",
                "normal left axis deviation counterclockwise rotation versus old true posterior myocardial infarction",
                "anteroseptal infarct of indeterminate age",
                "myocardial infarction cannot rule out",
                "possible inferior myocardial infarction",
                "normal counterclockwise rotation versus old true posterior myocardial infarction",
                "suggestive of old true posterior myocardial infarction st abnormality",
                "antero-apical and lateral myocardial infarction evolving",
                "old anterior infarct",
                "(consistent with).*(true posterior).*(myocardial infarction)",
                "possible true posterior myocardial infarction",
                "antero-apical ischemia versus myocardial infarction",
                "anterolateral myocardial infarction appears recent",
                "evolution of myocardial infarction",
                "anteroseptal and lateral myocardial infarction",
                "old infero-posterior and apical myocardial infarction",
                "old myocardial infarction",
                "infero-apical myocardial infarction",
                "counterclockwise rotation versus true posterior myocardial infarction",
                "rule out interim myocardial infarction",
                "normal counterclockwise rotation versusold true posterior myocardial infarction",
                "lateral wall myocardial infarction",
                "old inferoposterior myocardial infarction",
                "old inferolateral myocardial infarction",
                "concurrent ischemia myocardial infarction",
                "cannot rule out true posterior myocardial infarction",
                "normal counterclockwise rotation versus true posterior myocardial infarction",
                "(myocardial infarction versus).*(counterclockwise rotation)",
                "infero-apical myocardial infarction of indeterminate age",
                "consistent with ischemia myocardial infarction",
                "inferior myocardial infarction , possibly acute",
                "extensive anterior infarct",
                "block inferior myocardial infarction",
                "subendocardial ischemia myocardial infarction",
                "old anteroseptal myocardial infarction",
                "old lateral myocardial infarction",
                "old infero-postero-lateral myocardial infarction",
                "old infero-posterior myocardial infarction",
                "old anterolateral myocardial infarction",
                "anterior myocardial infarction",
                "inferior myocardial infarction",
                "inferior wall myocardial infarction of indeterminate age",
                "old apicolateral myocardial infarction",
                "myocardial infarction nonspecific st segment",
                "possible anteroseptal myocardial infarction",
                "age indeterminate old inferior wall myocardial infarction",
                "cannot rule out anterior infarct , age undetermined",
                "possible myocardial infarction",
                "myocardial infarction when compared with ecg of",
                "old anterior wall myocardial infarction",
                "new incomplete posterior lateral myocardial infarction",
                "counterclockwise rotation versus old true posterior myocardial infarction",
                "inferior myocardial infarction , age undetermined",
                "probable apicolateral myocardial infarction",
                "old true posterior myocardial infarction",
                "borderline anterolateral myocardial infarction",
                "posterior wall myocardial infarction",
                "possible acute inferior myocardial infarction",
                "known true posterior myocardial infarction",
                "ongoing ischemia versus myocardial infarction",
                "extensive anterolateral myocardial infarction",
                "subendocardial ischemia or myocardial infarction",
                "(true posterior).*(myocardial infarction)",
                "possible anterolateral myocardial infarction",
                "myocardial infarction versus pericarditis",
                "old inferior myocardial infarction",
                "acute infarct",
                "old inferoapical myocardial infarction",
                "subendocardial myocardial infarction",
                "(possible old).*(true posterior).*(myocardial infarction)",
                "cannot rule out anteroseptal infarct",
                "old anterolateral infarct",
                "infero and apicolateral myocardial infarction",
                "consistent with anteroseptal infarct",
                "acuteanterior myocardial infarction",
                "myocardial infarction old high lateral",
                "subendocardial infarct",
                "anterolateral infarct of indeterminate age ",
                "evolving myocardial infarction",
                "evolving counterclockwise rotation versus true posterior myocardial infarction",
                "myocardial infarction pattern",
                "old high lateral myocardial infarction",
                "raises possibility of septal infarct",
                "post myocardial infarction , of indeterminate age",
                "possible old lateral myocardial infarction",
                "transmural ischemia myocardial infarction",
                "old inferior anterior myocardial infarctions",
                "old anteroseptal infarct",
                "recent myocardial infarction",
                "consistent with anterior myocardial infarction of indeterminate age",
                "suggestive of old true posterior myocardial infarction",
                "old inferior and anterior myocardial infarctions",
                "septal infarct",
                "acute myocardial infarction",
                "myocardial infarction",
                "possible old septal myocardial infarction",
                "myocardial infarction of indeterminate age",
                "apical myocardial infarction",
                "counterclockwise rotation consistent with post myocardial infarction",
                "possible anteroseptal myocardial infarction of uncertain age",
                "true posterior myocardial infarction of indeterminate age",
                "myocardial infarction indeterminate",
                "old inferior posterolateral myocardial infarction",
            },
        },
        not_found_channel="no_mi",
    ),
)


tmaps["mi_any"] = TensorMap(
    "mi_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_mi": 0, "mi": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mi": {
                "acute anterior wall myocardial infarction",
                "possible acute myocardial infarction",
                "acute myocardial infarction in evolution",
                "extensive myocardial infarction of indeterminate age ",
                "acute anterior infarct",
                "cannot rule out inferoposterior myoca",
                "evolving inferior wall myocardial infarction",
                "subendocardial ischemia subendocardial myocardial inf",
                "anterolateral myocardial infarction",
                "anterior infarct of indeterminate age",
                "old inferior wall myocardial infarction",
                "myocardial infarction compared with the last previous tracing ",
                "lateral myocardial infarction of indeterminate age",
                "(counterclockwise rotation).*(true posterior)",
                "anteroseptal myocardial infarction",
                "cannot rule out anterior wall ischemia myocardial infarction",
                "inferoapical myocardial infarction of indeterminate age",
                "lateral myocardial infarction - of indeterminate age",
                "old infero-posterior lateral myocardial infarction",
                "acute myocardial infarction of indeterminate age",
                "possible septal myocardial infarction",
                "myocardial infarction possible when compared",
                "inferior myocardial infarction of indeterminate",
                "posterior myocardial infarction",
                "evolving anterior infarct",
                "anterolateral myocardial infarction , possibly recent  ",
                "anterior myocardial infarction of indeterminate age",
                "myocardial infarction extension",
                "inferolateral myocardial infarction",
                "myocardial infarction compared with the last previous ",
                "old posterolateral myocardial infarction",
                "apical myocardial infarction of indeterminate age",
                "old anterior myocardial infarction",
                "inferior myocardial infarction of indeterminate age",
                "cannot rule out true posterior myocardial infarction versus counterclockwise rotation",
                "normal left axis deviation counterclockwise rotation versus old true posterior myocardial infarction",
                "anteroseptal infarct of indeterminate age",
                "myocardial infarction cannot rule out",
                "possible inferior myocardial infarction",
                "normal counterclockwise rotation versus old true posterior myocardial infarction",
                "suggestive of old true posterior myocardial infarction st abnormality",
                "antero-apical and lateral myocardial infarction evolving",
                "old anterior infarct",
                "(consistent with).*(true posterior).*(myocardial infarction)",
                "possible true posterior myocardial infarction",
                "antero-apical ischemia versus myocardial infarction",
                "anterolateral myocardial infarction appears recent",
                "evolution of myocardial infarction",
                "anteroseptal and lateral myocardial infarction",
                "old infero-posterior and apical myocardial infarction",
                "old myocardial infarction",
                "infero-apical myocardial infarction",
                "counterclockwise rotation versus true posterior myocardial infarction",
                "rule out interim myocardial infarction",
                "normal counterclockwise rotation versusold true posterior myocardial infarction",
                "lateral wall myocardial infarction",
                "old inferoposterior myocardial infarction",
                "old inferolateral myocardial infarction",
                "concurrent ischemia myocardial infarction",
                "cannot rule out true posterior myocardial infarction",
                "normal counterclockwise rotation versus true posterior myocardial infarction",
                "(myocardial infarction versus).*(counterclockwise rotation)",
                "infero-apical myocardial infarction of indeterminate age",
                "consistent with ischemia myocardial infarction",
                "inferior myocardial infarction , possibly acute",
                "extensive anterior infarct",
                "block inferior myocardial infarction",
                "subendocardial ischemia myocardial infarction",
                "old anteroseptal myocardial infarction",
                "old lateral myocardial infarction",
                "old infero-postero-lateral myocardial infarction",
                "old infero-posterior myocardial infarction",
                "old anterolateral myocardial infarction",
                "anterior myocardial infarction",
                "inferior myocardial infarction",
                "inferior wall myocardial infarction of indeterminate age",
                "old apicolateral myocardial infarction",
                "myocardial infarction nonspecific st segment",
                "possible anteroseptal myocardial infarction",
                "age indeterminate old inferior wall myocardial infarction",
                "cannot rule out anterior infarct , age undetermined",
                "possible myocardial infarction",
                "myocardial infarction when compared with ecg of",
                "old anterior wall myocardial infarction",
                "new incomplete posterior lateral myocardial infarction",
                "counterclockwise rotation versus old true posterior myocardial infarction",
                "inferior myocardial infarction , age undetermined",
                "probable apicolateral myocardial infarction",
                "old true posterior myocardial infarction",
                "borderline anterolateral myocardial infarction",
                "posterior wall myocardial infarction",
                "possible acute inferior myocardial infarction",
                "known true posterior myocardial infarction",
                "ongoing ischemia versus myocardial infarction",
                "extensive anterolateral myocardial infarction",
                "subendocardial ischemia or myocardial infarction",
                "(true posterior).*(myocardial infarction)",
                "possible anterolateral myocardial infarction",
                "myocardial infarction versus pericarditis",
                "old inferior myocardial infarction",
                "acute infarct",
                "old inferoapical myocardial infarction",
                "subendocardial myocardial infarction",
                "(possible old).*(true posterior).*(myocardial infarction)",
                "cannot rule out anteroseptal infarct",
                "old anterolateral infarct",
                "infero and apicolateral myocardial infarction",
                "consistent with anteroseptal infarct",
                "acuteanterior myocardial infarction",
                "myocardial infarction old high lateral",
                "subendocardial infarct",
                "anterolateral infarct of indeterminate age ",
                "evolving myocardial infarction",
                "evolving counterclockwise rotation versus true posterior myocardial infarction",
                "myocardial infarction pattern",
                "old high lateral myocardial infarction",
                "raises possibility of septal infarct",
                "post myocardial infarction , of indeterminate age",
                "possible old lateral myocardial infarction",
                "transmural ischemia myocardial infarction",
                "old inferior anterior myocardial infarctions",
                "old anteroseptal infarct",
                "recent myocardial infarction",
                "consistent with anterior myocardial infarction of indeterminate age",
                "suggestive of old true posterior myocardial infarction",
                "old inferior and anterior myocardial infarctions",
                "septal infarct",
                "acute myocardial infarction",
                "myocardial infarction",
                "possible old septal myocardial infarction",
                "myocardial infarction of indeterminate age",
                "apical myocardial infarction",
                "counterclockwise rotation consistent with post myocardial infarction",
                "possible anteroseptal myocardial infarction of uncertain age",
                "true posterior myocardial infarction of indeterminate age",
                "myocardial infarction indeterminate",
                "old inferior posterolateral myocardial infarction",
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
                "intraventricular conduction defect",
                "intraventricular conduction delay",
            },
        },
        not_found_channel="no_intraventricular_conduction_delay",
    ),
)


tmaps["intraventricular_conduction_delay_any"] = TensorMap(
    "intraventricular_conduction_delay_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_intraventricular_conduction_delay": 0,
        "intraventricular_conduction_delay": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "intraventricular_conduction_delay": {
                "intraventricular conduction defect",
                "intraventricular conduction delay",
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
                "left anterior hemiblock",
                "left anterior fascicular block",
            },
        },
        not_found_channel="no_left_anterior_fascicular_block",
    ),
)


tmaps["left_anterior_fascicular_block_any"] = TensorMap(
    "left_anterior_fascicular_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_left_anterior_fascicular_block": 0,
        "left_anterior_fascicular_block": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_anterior_fascicular_block": {
                "left anterior hemiblock",
                "left anterior fascicular block",
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
    path_prefix=ECG_PREFIX,
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
                "left bbb",
                "lbbb",
                "left bundle branch block",
            },
        },
        not_found_channel="no_left_bundle_branch_block",
    ),
)


tmaps["left_bundle_branch_block_any"] = TensorMap(
    "left_bundle_branch_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_left_bundle_branch_block": 0, "left_bundle_branch_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_bundle_branch_block": {
                "bundle branch block",
                "left bbb",
                "lbbb",
                "left bundle branch block",
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
                "left posterior hemiblock",
                "left posterior fascicular block",
            },
        },
        not_found_channel="no_left_posterior_fascicular_block",
    ),
)


tmaps["left_posterior_fascicular_block_any"] = TensorMap(
    "left_posterior_fascicular_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_left_posterior_fascicular_block": 0,
        "left_posterior_fascicular_block": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_posterior_fascicular_block": {
                "left posterior hemiblock",
                "left posterior fascicular block",
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
                "bundle branch block",
                "left bbb",
                "rbbb",
                "right bundle branch block",
            },
        },
        not_found_channel="no_right_bundle_branch_block",
    ),
)


tmaps["right_bundle_branch_block_any"] = TensorMap(
    "right_bundle_branch_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_right_bundle_branch_block": 0, "right_bundle_branch_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "right_bundle_branch_block": {
                "bundle branch block",
                "left bbb",
                "rbbb",
                "right bundle branch block",
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
                "diffuse st segment elevation",
                "consistent with lateral ischemia",
                "subendocardial ischemia",
                "apical subendocardial ischemia",
                "inferior subendocardial ischemia",
                "anterolateral ischemia",
                "antero-apical ischemia",
                "consider anterior and lateral ischemia",
                "st segment depression",
                "minor st segment depression",
                "st segment depression in leads v4-v6",
                "anterolateral st segment depression",
                "infero- st segment depression",
                "st depression",
                "suggest anterior ischemia",
                "st segment depression is more marked in leads",
                "possible anterior wall ischemia",
                "consistent with ischemia",
                "diffuse scooped st segment depression",
                "anterolateral subendocardial ischemia",
                "diffuse st segment depression",
                "st segment elevation consistent with acute injury",
                "inferior st segment elevation and q waves",
                "st segment depression in anterolateral leads",
                "widespread st segment depression",
                "consider anterior ischemia",
                "suggesting anterior ischemia",
                "consistent with subendocardial ischemia",
                "marked st segment depression in leads",
                "inferior st segment depression",
                "st segment elevation in leads",
                "st segment elevation",
                "st segment depressions more marked",
                "anterior st segment depression",
                "apical st depression",
                "septal ischemia",
                "st segment depression in leads",
                "suggests anterolateral ischemia",
                "st elevation",
                "diffuse elevation of st segments",
                "marked st segment depression",
                "anterior infarct or transmural ischemia",
                "inferoapical st segment depression",
                "lateral ischemia",
                "nonspecific st segment depression",
                "anterior subendocardial ischemia",
            },
        },
        not_found_channel="no_ischemia",
    ),
)


tmaps["ischemia_any"] = TensorMap(
    "ischemia_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ischemia": 0, "ischemia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ischemia": {
                "diffuse st segment elevation",
                "consistent with lateral ischemia",
                "subendocardial ischemia",
                "apical subendocardial ischemia",
                "inferior subendocardial ischemia",
                "anterolateral ischemia",
                "antero-apical ischemia",
                "consider anterior and lateral ischemia",
                "st segment depression",
                "minor st segment depression",
                "st segment depression in leads v4-v6",
                "anterolateral st segment depression",
                "infero- st segment depression",
                "st depression",
                "suggest anterior ischemia",
                "st segment depression is more marked in leads",
                "possible anterior wall ischemia",
                "consistent with ischemia",
                "diffuse scooped st segment depression",
                "anterolateral subendocardial ischemia",
                "diffuse st segment depression",
                "st segment elevation consistent with acute injury",
                "inferior st segment elevation and q waves",
                "st segment depression in anterolateral leads",
                "widespread st segment depression",
                "consider anterior ischemia",
                "suggesting anterior ischemia",
                "consistent with subendocardial ischemia",
                "marked st segment depression in leads",
                "inferior st segment depression",
                "st segment elevation in leads",
                "st segment elevation",
                "st segment depressions more marked",
                "anterior st segment depression",
                "apical st depression",
                "septal ischemia",
                "st segment depression in leads",
                "suggests anterolateral ischemia",
                "st elevation",
                "diffuse elevation of st segments",
                "marked st segment depression",
                "anterior infarct or transmural ischemia",
                "inferoapical st segment depression",
                "lateral ischemia",
                "nonspecific st segment depression",
                "anterior subendocardial ischemia",
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
    path_prefix=ECG_PREFIX,
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
                "diffuse st segment elevation",
                "st segment abnormality",
                "abnormal st segment changes",
                "st segment depression",
                "minor st segment depression",
                "st segment depression in leads v4-v6",
                "anterolateral st segment depression",
                "infero- st segment depression",
                "st depression",
                "st segment depression is more marked in leads",
                "diffuse scooped st segment depression",
                "diffuse st segment depression",
                "st segment elevation consistent with acute injury",
                "inferior st segment elevation and q waves",
                "st segment depression in anterolateral leads",
                "widespread st segment depression",
                "nonspecific st segment and t wave abnormalities",
                "marked st segment depression in leads",
                "nonspecific st segment",
                "st segment elevation in leads",
                "st segment changes",
                "st segment elevation",
                "st segment depressions more marked",
                "anterior st segment depression",
                "apical st depression",
                "st segment depression in leads",
                "st elevation",
                "diffuse elevation of st segments",
                "marked st segment depression",
                "inferoapical st segment depression",
                "nonspecific st segment depression",
            },
        },
        not_found_channel="no_st_abnormality",
    ),
)


tmaps["st_abnormality_any"] = TensorMap(
    "st_abnormality_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_st_abnormality": 0, "st_abnormality": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "st_abnormality": {
                "diffuse st segment elevation",
                "st segment abnormality",
                "abnormal st segment changes",
                "st segment depression",
                "minor st segment depression",
                "st segment depression in leads v4-v6",
                "anterolateral st segment depression",
                "infero- st segment depression",
                "st depression",
                "st segment depression is more marked in leads",
                "diffuse scooped st segment depression",
                "diffuse st segment depression",
                "st segment elevation consistent with acute injury",
                "inferior st segment elevation and q waves",
                "st segment depression in anterolateral leads",
                "widespread st segment depression",
                "nonspecific st segment and t wave abnormalities",
                "marked st segment depression in leads",
                "nonspecific st segment",
                "st segment elevation in leads",
                "st segment changes",
                "st segment elevation",
                "st segment depressions more marked",
                "anterior st segment depression",
                "apical st depression",
                "st segment depression in leads",
                "st elevation",
                "diffuse elevation of st segments",
                "marked st segment depression",
                "inferoapical st segment depression",
                "nonspecific st segment depression",
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
    path_prefix=ECG_PREFIX,
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
                "t wave inveions",
                "t wave inversions",
                "t waves are upright in leads",
                "t wave abnormalities",
                "t wave changes",
                "t waves are inverted in leads",
                "(nonspecific st segment).*(t wave abnormalities)",
                "possible st segment and t wave abn",
                "t waves are lower or inverted in leads",
                "t wave inversion",
                "tall t waves in precordial leads",
                "nonspecific st segment and t wave abnormalities",
                "t wave inver",
                "t waves are slightly more inverted in leads",
                "t wave flattening",
                "t wave inversion in leads",
                "diffuse nonspecific st segment and t wave abnormalities",
                "upright t waves",
                "recent diffuse t wave flattening",
                "nonspecific t wave abnormali",
            },
        },
        not_found_channel="no_t_wave_abnormality",
    ),
)


tmaps["t_wave_abnormality_any"] = TensorMap(
    "t_wave_abnormality_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_t_wave_abnormality": 0, "t_wave_abnormality": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "t_wave_abnormality": {
                "t wave inveions",
                "t wave inversions",
                "t waves are upright in leads",
                "t wave abnormalities",
                "t wave changes",
                "t waves are inverted in leads",
                "(nonspecific st segment).*(t wave abnormalities)",
                "possible st segment and t wave abn",
                "t waves are lower or inverted in leads",
                "t wave inversion",
                "tall t waves in precordial leads",
                "nonspecific st segment and t wave abnormalities",
                "t wave inver",
                "t waves are slightly more inverted in leads",
                "t wave flattening",
                "t wave inversion in leads",
                "diffuse nonspecific st segment and t wave abnormalities",
                "upright t waves",
                "recent diffuse t wave flattening",
                "nonspecific t wave abnormali",
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
    path_prefix=ECG_PREFIX,
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
                "biatrial hypertrophy",
                "biatrial enlargement",
                "left atrial enla",
            },
        },
        not_found_channel="no_lae",
    ),
)


tmaps["lae_any"] = TensorMap(
    "lae_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_lae": 0, "lae": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lae": {
                "combined atrial enlargement",
                "biatrial hypertrophy",
                "biatrial enlargement",
                "left atrial enla",
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
                "biventricular hypertrophy",
                "biventriclar hypertrophy",
                "leftventricular hypertrophy",
                "combined ventricular hypertrophy",
                "left ventricular hypertr",
            },
        },
        not_found_channel="no_lvh",
    ),
)


tmaps["lvh_any"] = TensorMap(
    "lvh_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_lvh": 0, "lvh": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lvh": {
                "biventricular hypertrophy",
                "biventriclar hypertrophy",
                "leftventricular hypertrophy",
                "combined ventricular hypertrophy",
                "left ventricular hypertr",
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
                "biatrial hypertrophy",
                "biatrial enlargement",
                "right atrial enla",
            },
        },
        not_found_channel="no_rae",
    ),
)


tmaps["rae_any"] = TensorMap(
    "rae_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_rae": 0, "rae": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "rae": {
                "combined atrial enlargement",
                "biatrial hypertrophy",
                "biatrial enlargement",
                "right atrial enla",
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
                "biventricular hypertrophy",
                "biventriclar hypertrophy",
                "right ventricular enlargement",
                "right ventricular hypertrophy",
                "combined ventricular hypertrophy",
                "rightventricular hypertrophy",
            },
        },
        not_found_channel="no_rvh",
    ),
)


tmaps["rvh_any"] = TensorMap(
    "rvh_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_rvh": 0, "rvh": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "rvh": {
                "biventricular hypertrophy",
                "biventriclar hypertrophy",
                "right ventricular enlargement",
                "right ventricular hypertrophy",
                "combined ventricular hypertrophy",
                "rightventricular hypertrophy",
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
        channel_terms={"sh": {"septal lipomatous hypertrophy", "septal hypertrophy"}},
        not_found_channel="no_sh",
    ),
)


tmaps["sh_any"] = TensorMap(
    "sh_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_sh": 0, "sh": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sh": {"septal lipomatous hypertrophy", "septal hypertrophy"}},
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
                "ectopic atrial complexes",
                "premature atrial complexes",
                "atrial premature complexes",
                "premature atrial co",
                "atrial bigeminy",
                "isolated premature atrial contractions",
                "atrial premature beat",
                "atrial ectopy has decreased",
                "atrial ectopy",
            },
        },
        not_found_channel="no_atrial_premature_complexes",
    ),
)


tmaps["atrial_premature_complexes_any"] = TensorMap(
    "atrial_premature_complexes_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_atrial_premature_complexes": 0, "atrial_premature_complexes": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_premature_complexes": {
                "atrial trigeminy",
                "ectopic atrial complexes",
                "premature atrial complexes",
                "atrial premature complexes",
                "premature atrial co",
                "atrial bigeminy",
                "isolated premature atrial contractions",
                "atrial premature beat",
                "atrial ectopy has decreased",
                "atrial ectopy",
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
                "ectopy has appeared",
                "ectopy more pronounced",
                "ectopy is new",
                "other than the ectopy",
                "new ectopy",
                "return of ectopy",
                "increased ectopy",
                "ectopy have increased",
                "ectopy has increased",
            },
        },
        not_found_channel="no_ectopy",
    ),
)


tmaps["ectopy_any"] = TensorMap(
    "ectopy_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ectopy": 0, "ectopy": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopy": {
                "ectopy has appeared",
                "ectopy more pronounced",
                "ectopy is new",
                "other than the ectopy",
                "new ectopy",
                "return of ectopy",
                "increased ectopy",
                "ectopy have increased",
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
                "junctional premature beats",
                "junctional premature complexes",
            },
        },
        not_found_channel="no_junctional_premature_complexes",
    ),
)


tmaps["junctional_premature_complexes_any"] = TensorMap(
    "junctional_premature_complexes_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_junctional_premature_complexes": 0,
        "junctional_premature_complexes": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "junctional_premature_complexes": {
                "junctional premature beats",
                "junctional premature complexes",
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
                "ectopy has resolved",
                "no ectopy",
                "ectopy is gone",
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
    path_prefix=ECG_PREFIX,
    channel_map={"no_no_ectopy": 0, "no_ectopy": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "no_ectopy": {
                "no longer any ectopy",
                "ectopy has resolved",
                "no ectopy",
                "ectopy is gone",
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
    path_prefix=ECG_PREFIX,
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
                "premature ventricular and fusion complexes",
                "ventriculaar ectopy is now present",
                "ventricular bigeminy",
                "one premature ventricularbeat",
                "isolated premature ventricular contractions",
                "ventricular premature complexes",
                "ventricular ectopy",
                "ventricular premature beat",
                "premature ventricular compl",
                "premature ventricular beat",
                "occasional premature ventricular complexes ",
                "frequent premature ventricular or aberrantly conducted complexes ",
                "ventricular trigeminy",
            },
        },
        not_found_channel="no_ventricular_premature_complexes",
    ),
)


tmaps["ventricular_premature_complexes_any"] = TensorMap(
    "ventricular_premature_complexes_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_ventricular_premature_complexes": 0,
        "ventricular_premature_complexes": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ventricular_premature_complexes": {
                "premature ventricular and fusion complexes",
                "ventriculaar ectopy is now present",
                "ventricular bigeminy",
                "one premature ventricularbeat",
                "isolated premature ventricular contractions",
                "ventricular premature complexes",
                "ventricular ectopy",
                "ventricular premature beat",
                "premature ventricular compl",
                "premature ventricular beat",
                "occasional premature ventricular complexes ",
                "frequent premature ventricular or aberrantly conducted complexes ",
                "ventricular trigeminy",
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
                "2:1 atrioventricular block",
                "2 to 1 av block",
                "2:1 block",
                "2:1 av block",
                "2 to 1 atrioventricular block",
            },
        },
        not_found_channel="no__2_to_1_av_block",
    ),
)


tmaps["_2_to_1_av_block_any"] = TensorMap(
    "_2_to_1_av_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no__2_to_1_av_block": 0, "_2_to_1_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "_2_to_1_av_block": {
                "2:1 atrioventricular block",
                "2 to 1 av block",
                "2:1 block",
                "2:1 av block",
                "2 to 1 atrioventricular block",
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
    path_prefix=ECG_PREFIX,
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
            "av_dissociation": {"av dissociation", "atrioventricular dissociation"},
        },
        not_found_channel="no_av_dissociation",
    ),
)


tmaps["av_dissociation_any"] = TensorMap(
    "av_dissociation_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_av_dissociation": 0, "av_dissociation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "av_dissociation": {"av dissociation", "atrioventricular dissociation"},
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
                "wenckebach",
                "mobitz 1 block",
                "fixed block",
                "second degree type 1",
                "mobitz type 1",
                "mobitz i",
                "second degree ",
            },
        },
        not_found_channel="no_mobitz_type_i_second_degree_av_block_",
    ),
)


tmaps["mobitz_type_i_second_degree_av_block__any"] = TensorMap(
    "mobitz_type_i_second_degree_av_block__any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_mobitz_type_i_second_degree_av_block_": 0,
        "mobitz_type_i_second_degree_av_block_": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_i_second_degree_av_block_": {
                "wenckebach",
                "mobitz 1 block",
                "fixed block",
                "second degree type 1",
                "mobitz type 1",
                "mobitz i",
                "second degree ",
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
                "second degree type 2",
                "mobitz ii",
                "2nd degree sa block",
                "hay block",
            },
        },
        not_found_channel="no_mobitz_type_ii_second_degree_av_block",
    ),
)


tmaps["mobitz_type_ii_second_degree_av_block_any"] = TensorMap(
    "mobitz_type_ii_second_degree_av_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={
        "no_mobitz_type_ii_second_degree_av_block": 0,
        "mobitz_type_ii_second_degree_av_block": 1,
    },
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_ii_second_degree_av_block": {
                "second degree type 2",
                "mobitz ii",
                "2nd degree sa block",
                "hay block",
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
                "complete heart block",
                "3rd degree atrioventricular block",
                "3rd degree av block",
                "third degree av block",
                "third degree atrioventricular block",
            },
        },
        not_found_channel="no_third_degree_av_block",
    ),
)


tmaps["third_degree_av_block_any"] = TensorMap(
    "third_degree_av_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_third_degree_av_block": 0, "third_degree_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "third_degree_av_block": {
                "complete heart block",
                "3rd degree atrioventricular block",
                "3rd degree av block",
                "third degree av block",
                "third degree atrioventricular block",
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
                "atrioventricular block",
                "high degree of block",
                "heartblock",
                "av block",
                "heart block",
                "high grade atrioventricular block",
            },
        },
        not_found_channel="no_unspecified_av_block",
    ),
)


tmaps["unspecified_av_block_any"] = TensorMap(
    "unspecified_av_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_unspecified_av_block": 0, "unspecified_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified_av_block": {
                "atrioventricular block",
                "high degree of block",
                "heartblock",
                "av block",
                "heart block",
                "high grade atrioventricular block",
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
    path_prefix=ECG_PREFIX,
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
