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
                "atrial fibrillation with controlled ventricular response",
                "atrial fibrillation with moderate ventricular response",
                "afibrillation",
                "atrial fibrillation with rapid ventricular response",
                "atrialfibrillation",
                "atrial fib",
                "fibrillation/flutter",
                "afib",
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
                "atrial fibrillation with controlled ventricular response",
                "atrial fibrillation with moderate ventricular response",
                "afibrillation",
                "atrial fibrillation with rapid ventricular response",
                "atrialfibrillation",
                "atrial fib",
                "fibrillation/flutter",
                "afib",
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
                "probable flutter",
                "tachycardia possibly flutter",
                "atrial flutter",
                "atrial flutter fixed block",
                "atrial flutter variable block",
                "flutter",
                "atrial flutter unspecified block",
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
                "probable flutter",
                "tachycardia possibly flutter",
                "atrial flutter",
                "atrial flutter fixed block",
                "atrial flutter variable block",
                "flutter",
                "atrial flutter unspecified block",
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
        channel_terms={"atrial_paced_rhythm": {"atrial paced rhythm", "atrial pacing"}},
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
                "low atrial pacer",
                "ectopicsupraventricular rhythm",
                "atrial rhythm",
                "p wave axis suggests atrial rather than sinus mechanism",
                "unifocal ear",
                "atrial arrhythmia",
                "wandering ectopic atrial rhythm",
                "unusual p wave axis",
                "wandering atrial pacemaker",
                "multifocal ear",
                "multiple atrial foci",
                "abnormal p vector",
                "multifocal atrial rhythm",
                "dual atrial foci ",
                "multifocal atrialrhythm",
                "multifocal ectopic atrial rhythm",
                "nonsinus atrial mechanism",
                "ectopic atrial rhythm ",
                "ectopic atrial rhythm",
                "unifocal ectopic atrial rhythm",
                "wandering ear",
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
                "low atrial pacer",
                "ectopicsupraventricular rhythm",
                "atrial rhythm",
                "p wave axis suggests atrial rather than sinus mechanism",
                "unifocal ear",
                "atrial arrhythmia",
                "wandering ectopic atrial rhythm",
                "unusual p wave axis",
                "wandering atrial pacemaker",
                "multifocal ear",
                "multiple atrial foci",
                "abnormal p vector",
                "multifocal atrial rhythm",
                "dual atrial foci ",
                "multifocal atrialrhythm",
                "multifocal ectopic atrial rhythm",
                "nonsinus atrial mechanism",
                "ectopic atrial rhythm ",
                "ectopic atrial rhythm",
                "unifocal ectopic atrial rhythm",
                "wandering ear",
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
                "unspecified ectopic atrial tachycardia",
                "ectopic atrial tachycardia",
                "ectopic atrial tachycardia, unspecified",
                "multifocal ectopic atrial tachycardia",
                "ectopic atrial tachycardia, multifocal",
                "multifocal atrial tachycardia",
                "unifocal ectopic atrial tachycardia",
                "unifocal atrial tachycardia",
                "ectopic atrial tachycardia, unifocal",
                "wandering atrial tachycardia",
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
                "unspecified ectopic atrial tachycardia",
                "ectopic atrial tachycardia",
                "ectopic atrial tachycardia, unspecified",
                "multifocal ectopic atrial tachycardia",
                "ectopic atrial tachycardia, multifocal",
                "multifocal atrial tachycardia",
                "unifocal ectopic atrial tachycardia",
                "unifocal atrial tachycardia",
                "ectopic atrial tachycardia, unifocal",
                "wandering atrial tachycardia",
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
                "narrow complex tachycardia",
                "narrow qrs tachycardia",
                "tachycardia narrow qrs",
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
                "narrow complex tachycardia",
                "narrow qrs tachycardia",
                "tachycardia narrow qrs",
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
                "sinus bradycardia",
                "sinus arrhythmia",
                "rhythm remains normal sinus",
                "sinus exit block",
                "sinus rhythm at a rate",
                "rhythm is now clearly sinus",
                "rhythm is normal sinus",
                "sa exit block",
                "marked sinus arrhythmia",
                "rhythm has reverted to normal",
                "atrial bigeminal  rhythm",
                "tracing is within normal limits",
                "atrialbigeminy",
                "type ii sinoatrial block",
                "atrial bigeminal rhythm",
                "atrial trigeminy",
                "tracing within normal limits",
                "conducted sinus impulses",
                "1st degree sa block",
                "sa block, type i",
                "sinus rhythm",
                "normal when compared with ecg of",
                "atrial bigeminy and ventricular bigeminy",
                "type ii sa block",
                "normal ecg",
                "frequent native sinus beats",
                "sinus mechanism has replaced",
                "type i sinoatrial block",
                "type i sa block",
                "sinoatrial block, type ii",
                "sinus slowing",
                "sinoatrial block",
                "2nd degree sa block",
                "sa block",
                "with occasional native sinus beats",
                "sinus tachycardia",
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
                "sinus bradycardia",
                "sinus arrhythmia",
                "rhythm remains normal sinus",
                "sinus exit block",
                "sinus rhythm at a rate",
                "rhythm is now clearly sinus",
                "rhythm is normal sinus",
                "sa exit block",
                "marked sinus arrhythmia",
                "rhythm has reverted to normal",
                "atrial bigeminal  rhythm",
                "tracing is within normal limits",
                "atrialbigeminy",
                "type ii sinoatrial block",
                "atrial bigeminal rhythm",
                "atrial trigeminy",
                "tracing within normal limits",
                "conducted sinus impulses",
                "1st degree sa block",
                "sa block, type i",
                "sinus rhythm",
                "normal when compared with ecg of",
                "atrial bigeminy and ventricular bigeminy",
                "type ii sa block",
                "normal ecg",
                "frequent native sinus beats",
                "sinus mechanism has replaced",
                "type i sinoatrial block",
                "type i sa block",
                "sinoatrial block, type ii",
                "sinus slowing",
                "sinoatrial block",
                "2nd degree sa block",
                "sa block",
                "with occasional native sinus beats",
                "sinus tachycardia",
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
                "av reentrant tachycardia ",
                "accelerated atrioventricular nodal rhythm",
                "supraventricular tachycardia",
                "atrioventricular nodal reentry tachycardia",
                "av nodal reentrant",
                "accelerated atrioventricular junctional rhythm",
                "atrioventricular reentrant tachycardia ",
                "avrt",
                "junctional tachycardia",
                "av nodal reentry tachycardia",
                "accelerated nodal rhythm",
                "atrial tachycardia",
                "avnrt",
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
                "av reentrant tachycardia ",
                "accelerated atrioventricular nodal rhythm",
                "supraventricular tachycardia",
                "atrioventricular nodal reentry tachycardia",
                "av nodal reentrant",
                "accelerated atrioventricular junctional rhythm",
                "atrioventricular reentrant tachycardia ",
                "avrt",
                "junctional tachycardia",
                "av nodal reentry tachycardia",
                "accelerated nodal rhythm",
                "atrial tachycardia",
                "avnrt",
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
                "undetermined  rhythm",
                "uncertain rhythm",
                "rhythm uncertain",
                "rhythm unclear",
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
                "undetermined  rhythm",
                "uncertain rhythm",
                "rhythm uncertain",
                "rhythm unclear",
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
            "wpw": {"wolffparkinsonwhite", "wpw", "wolff-parkinson-white pattern"},
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
                "sequential pacing",
                "atrial-paced rhythm",
                "a triggered v-paced rhythm",
                "ventricular paced",
                "shows dual chamber pacing",
                "failure to pace ventricular",
                "ventricular-paced complexes",
                "demand v-pacing",
                "ventricular-paced rhythm",
                "failure to pace atrial",
                "biventricular-paced complexes",
                "atrial-sensed ventricular-paced rhythm",
                "dual chamber pacing",
                "v-paced rhythm",
                "av dual-paced rhythm",
                "failure to capture atrial",
                "electronic pacemaker",
                "biventricular-paced rhythm",
                "v-paced",
                "unipolar right ventricular  pacing",
                "failure to inhibit atrial",
                "failure to inhibit ventricular",
                "atrially triggered v paced",
                "failure to capture ventricular",
                "ventricular pacing has replaced av pacing",
                "atrial triggered ventricular pacing",
                "atrial-sensed ventricular-paced complexes",
                "competitive av pacing",
                "atrial-paced complexes ",
                "demand ventricular pacemaker",
                "ventricular pacing",
                "ventricular demand pacing",
                "v-paced beats",
                "av dual-paced complexes",
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
                "sequential pacing",
                "atrial-paced rhythm",
                "a triggered v-paced rhythm",
                "ventricular paced",
                "shows dual chamber pacing",
                "failure to pace ventricular",
                "ventricular-paced complexes",
                "demand v-pacing",
                "ventricular-paced rhythm",
                "failure to pace atrial",
                "biventricular-paced complexes",
                "atrial-sensed ventricular-paced rhythm",
                "dual chamber pacing",
                "v-paced rhythm",
                "av dual-paced rhythm",
                "failure to capture atrial",
                "electronic pacemaker",
                "biventricular-paced rhythm",
                "v-paced",
                "unipolar right ventricular  pacing",
                "failure to inhibit atrial",
                "failure to inhibit ventricular",
                "atrially triggered v paced",
                "failure to capture ventricular",
                "ventricular pacing has replaced av pacing",
                "atrial triggered ventricular pacing",
                "atrial-sensed ventricular-paced complexes",
                "competitive av pacing",
                "atrial-paced complexes ",
                "demand ventricular pacemaker",
                "ventricular pacing",
                "ventricular demand pacing",
                "v-paced beats",
                "av dual-paced complexes",
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
                "sinus rhythm",
                "normal tracing",
                "tracing is within normal limits",
                "normal ecg",
                "sinus tachycardia",
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
                "sinus rhythm",
                "normal tracing",
                "tracing is within normal limits",
                "normal ecg",
                "sinus tachycardia",
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
                "axis shifted left",
                "leftward axis",
                "left axis deviation",
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
                "axis shifted left",
                "leftward axis",
                "left axis deviation",
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
                "right superior axis deviation",
                "axis shifted right",
                "rightward axis",
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
                "right superior axis deviation",
                "axis shifted right",
                "rightward axis",
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
                "poor precordial r wave progression",
                "early r wave progression",
                "slowprecordial r wave progression",
                "unusual r wave progression",
                "poor r wave progression",
                "abnormal precordial r wave progression or poor r wave progression",
                "slow precordial r wave progression",
                "abnormal precordial r wave progression",
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
                "poor precordial r wave progression",
                "early r wave progression",
                "slowprecordial r wave progression",
                "unusual r wave progression",
                "poor r wave progression",
                "abnormal precordial r wave progression or poor r wave progression",
                "slow precordial r wave progression",
                "abnormal precordial r wave progression",
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
    path_prefix="ECG_PREFIX",
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
                "possible anteroseptal myocardial infarction of uncertain age",
                "inferolateral myocardial infarction",
                "probable apicolateral myocardial infarction",
                "myocardial infarction cannot rule out",
                "old true posterior myocardial infarction",
                "infero-apical myocardial infarction of indeterminate age",
                "myocardial infarction of indeterminate age",
                "suggestive of old true posterior myocardial infarction st abnormality",
                "possible true posterior myocardial infarction",
                "anterolateral myocardial infarction",
                "counterclockwise rotation versus true posterior myocardial infarction",
                "septal infarct",
                "subendocardial ischemia or myocardial infarction",
                "normal counterclockwise rotation versusold true posterior myocardial infarction",
                "myocardial infarction pattern",
                "inferior myocardial infarction of indeterminate age",
                "apical myocardial infarction of indeterminate age",
                "borderline anterolateral myocardial infarction",
                "acute myocardial infarction in evolution",
                "block inferior myocardial infarction",
                "possible old lateral myocardial infarction",
                "normal counterclockwise rotation versus true posterior myocardial infarction",
                "consistent with anteroseptal infarct",
                "cannot rule out anteroseptal infarct",
                "subendocardial myocardial infarction",
                "ongoing ischemia versus myocardial infarction",
                "extensive anterolateral myocardial infarction",
                "old anterior myocardial infarction",
                "possible inferior myocardial infarction",
                "age indeterminate old inferior wall myocardial infarction",
                "anterolateral infarct of indeterminate age ",
                "myocardial infarction",
                "posterior wall myocardial infarction",
                "antero-apical and lateral myocardial infarction evolving",
                "old inferior and anterior myocardial infarctions",
                "inferior wall myocardial infarction of indeterminate age",
                "anterior infarct of indeterminate age",
                "old anteroseptal myocardial infarction",
                "possible septal myocardial infarction",
                "post myocardial infarction , of indeterminate age",
                "raises possibility of septal infarct",
                "true posterior myocardial infarction of indeterminate age",
                "myocardial infarction nonspecific st segment",
                "old inferoapical myocardial infarction",
                "(possible old).*(true posterior).*(myocardial infarction)",
                "known true posterior myocardial infarction",
                "inferoapical myocardial infarction of indeterminate age",
                "infero-apical myocardial infarction",
                "inferior myocardial infarction of indeterminate",
                "lateral myocardial infarction - of indeterminate age",
                "possible myocardial infarction",
                "cannot rule out true posterior myocardial infarction",
                "myocardial infarction possible when compared",
                "old infero-posterior lateral myocardial infarction",
                "old anterior infarct",
                "acute anterior wall myocardial infarction",
                "old high lateral myocardial infarction",
                "acute myocardial infarction of indeterminate age",
                "possible anterolateral myocardial infarction",
                "old posterolateral myocardial infarction",
                "apical myocardial infarction",
                "old lateral myocardial infarction",
                "old inferior myocardial infarction",
                "lateral myocardial infarction of indeterminate age",
                "old inferolateral myocardial infarction",
                "myocardial infarction compared with the last previous tracing ",
                "evolution of myocardial infarction",
                "acuteanterior myocardial infarction",
                "inferior myocardial infarction , age undetermined",
                "myocardial infarction compared with the last previous ",
                "evolving counterclockwise rotation versus true posterior myocardial infarction",
                "possible anteroseptal myocardial infarction",
                "evolving myocardial infarction",
                "anteroseptal and lateral myocardial infarction",
                "subendocardial ischemia myocardial infarction",
                "transmural ischemia myocardial infarction",
                "(true posterior).*(myocardial infarction)",
                "cannot rule out anterior wall ischemia myocardial infarction",
                "subendocardial ischemia subendocardial myocardial inf",
                "myocardial infarction indeterminate",
                "old anterolateral myocardial infarction",
                "counterclockwise rotation versus old true posterior myocardial infarction",
                "anterolateral myocardial infarction appears recent",
                "old infero-posterior and apical myocardial infarction",
                "anteroseptal infarct of indeterminate age",
                "recent myocardial infarction",
                "cannot rule out inferoposterior myoca",
                "acute infarct",
                "cannot rule out anterior infarct , age undetermined",
                "old apicolateral myocardial infarction",
                "posterior myocardial infarction",
                "counterclockwise rotation consistent with post myocardial infarction",
                "old anteroseptal infarct",
                "(consistent with).*(true posterior).*(myocardial infarction)",
                "possible acute inferior myocardial infarction",
                "myocardial infarction versus pericarditis",
                "consistent with anterior myocardial infarction of indeterminate age",
                "old anterolateral infarct",
                "possible acute myocardial infarction",
                "rule out interim myocardial infarction",
                "normal left axis deviation counterclockwise rotation versus old true posterior myocardial infarction",
                "inferior myocardial infarction , possibly acute",
                "old inferior posterolateral myocardial infarction",
                "anterolateral myocardial infarction , possibly recent  ",
                "possible old septal myocardial infarction",
                "myocardial infarction when compared with ecg of",
                "evolving anterior infarct",
                "subendocardial infarct",
                "(counterclockwise rotation).*(true posterior)",
                "myocardial infarction extension",
                "lateral wall myocardial infarction",
                "new incomplete posterior lateral myocardial infarction",
                "acute anterior infarct",
                "(myocardial infarction versus).*(counterclockwise rotation)",
                "normal counterclockwise rotation versus old true posterior myocardial infarction",
                "old anterior wall myocardial infarction",
                "old inferoposterior myocardial infarction",
                "cannot rule out true posterior myocardial infarction versus counterclockwise rotation",
                "anteroseptal myocardial infarction",
                "antero-apical ischemia versus myocardial infarction",
                "old inferior anterior myocardial infarctions",
                "anterior myocardial infarction",
                "acute myocardial infarction",
                "anterior myocardial infarction of indeterminate age",
                "old infero-postero-lateral myocardial infarction",
                "myocardial infarction old high lateral",
                "extensive anterior infarct",
                "concurrent ischemia myocardial infarction",
                "old infero-posterior myocardial infarction",
                "evolving inferior wall myocardial infarction",
                "infero and apicolateral myocardial infarction",
                "extensive myocardial infarction of indeterminate age ",
                "consistent with ischemia myocardial infarction",
                "inferior myocardial infarction",
                "old inferior wall myocardial infarction",
                "old myocardial infarction",
                "suggestive of old true posterior myocardial infarction",
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
                "possible anteroseptal myocardial infarction of uncertain age",
                "inferolateral myocardial infarction",
                "probable apicolateral myocardial infarction",
                "myocardial infarction cannot rule out",
                "old true posterior myocardial infarction",
                "infero-apical myocardial infarction of indeterminate age",
                "myocardial infarction of indeterminate age",
                "suggestive of old true posterior myocardial infarction st abnormality",
                "possible true posterior myocardial infarction",
                "anterolateral myocardial infarction",
                "counterclockwise rotation versus true posterior myocardial infarction",
                "septal infarct",
                "subendocardial ischemia or myocardial infarction",
                "normal counterclockwise rotation versusold true posterior myocardial infarction",
                "myocardial infarction pattern",
                "inferior myocardial infarction of indeterminate age",
                "apical myocardial infarction of indeterminate age",
                "borderline anterolateral myocardial infarction",
                "acute myocardial infarction in evolution",
                "block inferior myocardial infarction",
                "possible old lateral myocardial infarction",
                "normal counterclockwise rotation versus true posterior myocardial infarction",
                "consistent with anteroseptal infarct",
                "cannot rule out anteroseptal infarct",
                "subendocardial myocardial infarction",
                "ongoing ischemia versus myocardial infarction",
                "extensive anterolateral myocardial infarction",
                "old anterior myocardial infarction",
                "possible inferior myocardial infarction",
                "age indeterminate old inferior wall myocardial infarction",
                "anterolateral infarct of indeterminate age ",
                "myocardial infarction",
                "posterior wall myocardial infarction",
                "antero-apical and lateral myocardial infarction evolving",
                "old inferior and anterior myocardial infarctions",
                "inferior wall myocardial infarction of indeterminate age",
                "anterior infarct of indeterminate age",
                "old anteroseptal myocardial infarction",
                "possible septal myocardial infarction",
                "post myocardial infarction , of indeterminate age",
                "raises possibility of septal infarct",
                "true posterior myocardial infarction of indeterminate age",
                "myocardial infarction nonspecific st segment",
                "old inferoapical myocardial infarction",
                "(possible old).*(true posterior).*(myocardial infarction)",
                "known true posterior myocardial infarction",
                "inferoapical myocardial infarction of indeterminate age",
                "infero-apical myocardial infarction",
                "inferior myocardial infarction of indeterminate",
                "lateral myocardial infarction - of indeterminate age",
                "possible myocardial infarction",
                "cannot rule out true posterior myocardial infarction",
                "myocardial infarction possible when compared",
                "old infero-posterior lateral myocardial infarction",
                "old anterior infarct",
                "acute anterior wall myocardial infarction",
                "old high lateral myocardial infarction",
                "acute myocardial infarction of indeterminate age",
                "possible anterolateral myocardial infarction",
                "old posterolateral myocardial infarction",
                "apical myocardial infarction",
                "old lateral myocardial infarction",
                "old inferior myocardial infarction",
                "lateral myocardial infarction of indeterminate age",
                "old inferolateral myocardial infarction",
                "myocardial infarction compared with the last previous tracing ",
                "evolution of myocardial infarction",
                "acuteanterior myocardial infarction",
                "inferior myocardial infarction , age undetermined",
                "myocardial infarction compared with the last previous ",
                "evolving counterclockwise rotation versus true posterior myocardial infarction",
                "possible anteroseptal myocardial infarction",
                "evolving myocardial infarction",
                "anteroseptal and lateral myocardial infarction",
                "subendocardial ischemia myocardial infarction",
                "transmural ischemia myocardial infarction",
                "(true posterior).*(myocardial infarction)",
                "cannot rule out anterior wall ischemia myocardial infarction",
                "subendocardial ischemia subendocardial myocardial inf",
                "myocardial infarction indeterminate",
                "old anterolateral myocardial infarction",
                "counterclockwise rotation versus old true posterior myocardial infarction",
                "anterolateral myocardial infarction appears recent",
                "old infero-posterior and apical myocardial infarction",
                "anteroseptal infarct of indeterminate age",
                "recent myocardial infarction",
                "cannot rule out inferoposterior myoca",
                "acute infarct",
                "cannot rule out anterior infarct , age undetermined",
                "old apicolateral myocardial infarction",
                "posterior myocardial infarction",
                "counterclockwise rotation consistent with post myocardial infarction",
                "old anteroseptal infarct",
                "(consistent with).*(true posterior).*(myocardial infarction)",
                "possible acute inferior myocardial infarction",
                "myocardial infarction versus pericarditis",
                "consistent with anterior myocardial infarction of indeterminate age",
                "old anterolateral infarct",
                "possible acute myocardial infarction",
                "rule out interim myocardial infarction",
                "normal left axis deviation counterclockwise rotation versus old true posterior myocardial infarction",
                "inferior myocardial infarction , possibly acute",
                "old inferior posterolateral myocardial infarction",
                "anterolateral myocardial infarction , possibly recent  ",
                "possible old septal myocardial infarction",
                "myocardial infarction when compared with ecg of",
                "evolving anterior infarct",
                "subendocardial infarct",
                "(counterclockwise rotation).*(true posterior)",
                "myocardial infarction extension",
                "lateral wall myocardial infarction",
                "new incomplete posterior lateral myocardial infarction",
                "acute anterior infarct",
                "(myocardial infarction versus).*(counterclockwise rotation)",
                "normal counterclockwise rotation versus old true posterior myocardial infarction",
                "old anterior wall myocardial infarction",
                "old inferoposterior myocardial infarction",
                "cannot rule out true posterior myocardial infarction versus counterclockwise rotation",
                "anteroseptal myocardial infarction",
                "antero-apical ischemia versus myocardial infarction",
                "old inferior anterior myocardial infarctions",
                "anterior myocardial infarction",
                "acute myocardial infarction",
                "anterior myocardial infarction of indeterminate age",
                "old infero-postero-lateral myocardial infarction",
                "myocardial infarction old high lateral",
                "extensive anterior infarct",
                "concurrent ischemia myocardial infarction",
                "old infero-posterior myocardial infarction",
                "evolving inferior wall myocardial infarction",
                "infero and apicolateral myocardial infarction",
                "extensive myocardial infarction of indeterminate age ",
                "consistent with ischemia myocardial infarction",
                "inferior myocardial infarction",
                "old inferior wall myocardial infarction",
                "old myocardial infarction",
                "suggestive of old true posterior myocardial infarction",
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
                "aberrant conduction",
                "aberrant conduction of supraventricular beats",
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
                "aberrant conduction",
                "aberrant conduction of supraventricular beats",
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
                "left bbb",
                "lbbb",
                "left bundle branch block",
                "bundle branch block",
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
                "left bbb",
                "lbbb",
                "left bundle branch block",
                "bundle branch block",
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
                "left bbb",
                "right bundle branch block",
                "rbbb",
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
                "left bbb",
                "right bundle branch block",
                "rbbb",
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
                "anterolateral ischemia",
                "subendocardial ischemia",
                "septal ischemia",
                "nonspecific st segment depression",
                "consider anterior ischemia",
                "st segment depression",
                "anterior subendocardial ischemia",
                "anterolateral st segment depression",
                "antero-apical ischemia",
                "diffuse st segment depression",
                "minor st segment depression",
                "apical st depression",
                "suggests anterolateral ischemia",
                "suggest anterior ischemia",
                "diffuse scooped st segment depression",
                "inferior st segment depression",
                "anterolateral subendocardial ischemia",
                "consider anterior and lateral ischemia",
                "anterior st segment depression",
                "possible anterior wall ischemia",
                "st segment depression in anterolateral leads",
                "consistent with ischemia",
                "consistent with subendocardial ischemia",
                "suggesting anterior ischemia",
                "st segment elevation",
                "apical subendocardial ischemia",
                "st segment depression is more marked in leads",
                "st segment depression in leads",
                "st segment depression in leads v4-v6",
                "st elevation",
                "inferoapical st segment depression",
                "inferior subendocardial ischemia",
                "lateral ischemia",
                "st segment elevation consistent with acute injury",
                "widespread st segment depression",
                "anterior infarct or transmural ischemia",
                "diffuse st segment elevation",
                "infero- st segment depression",
                "marked st segment depression",
                "diffuse elevation of st segments",
                "st segment elevation in leads",
                "marked st segment depression in leads",
                "st depression",
                "st segment depressions more marked",
                "inferior st segment elevation and q waves",
                "consistent with lateral ischemia",
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
                "anterolateral ischemia",
                "subendocardial ischemia",
                "septal ischemia",
                "nonspecific st segment depression",
                "consider anterior ischemia",
                "st segment depression",
                "anterior subendocardial ischemia",
                "anterolateral st segment depression",
                "antero-apical ischemia",
                "diffuse st segment depression",
                "minor st segment depression",
                "apical st depression",
                "suggests anterolateral ischemia",
                "suggest anterior ischemia",
                "diffuse scooped st segment depression",
                "inferior st segment depression",
                "anterolateral subendocardial ischemia",
                "consider anterior and lateral ischemia",
                "anterior st segment depression",
                "possible anterior wall ischemia",
                "st segment depression in anterolateral leads",
                "consistent with ischemia",
                "consistent with subendocardial ischemia",
                "suggesting anterior ischemia",
                "st segment elevation",
                "apical subendocardial ischemia",
                "st segment depression is more marked in leads",
                "st segment depression in leads",
                "st segment depression in leads v4-v6",
                "st elevation",
                "inferoapical st segment depression",
                "inferior subendocardial ischemia",
                "lateral ischemia",
                "st segment elevation consistent with acute injury",
                "widespread st segment depression",
                "anterior infarct or transmural ischemia",
                "diffuse st segment elevation",
                "infero- st segment depression",
                "marked st segment depression",
                "diffuse elevation of st segments",
                "st segment elevation in leads",
                "marked st segment depression in leads",
                "st depression",
                "st segment depressions more marked",
                "inferior st segment elevation and q waves",
                "consistent with lateral ischemia",
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
                "nonspecific st segment depression",
                "st segment depression",
                "anterolateral st segment depression",
                "st segment abnormality",
                "diffuse st segment depression",
                "minor st segment depression",
                "apical st depression",
                "diffuse scooped st segment depression",
                "anterior st segment depression",
                "st segment depression in anterolateral leads",
                "st segment elevation",
                "st segment depression is more marked in leads",
                "st segment depression in leads",
                "st segment depression in leads v4-v6",
                "abnormal st segment changes",
                "st elevation",
                "inferoapical st segment depression",
                "nonspecific st segment and t wave abnormalities",
                "st segment elevation consistent with acute injury",
                "widespread st segment depression",
                "diffuse st segment elevation",
                "infero- st segment depression",
                "marked st segment depression",
                "diffuse elevation of st segments",
                "nonspecific st segment",
                "st segment elevation in leads",
                "st segment changes",
                "marked st segment depression in leads",
                "st depression",
                "st segment depressions more marked",
                "inferior st segment elevation and q waves",
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
                "nonspecific st segment depression",
                "st segment depression",
                "anterolateral st segment depression",
                "st segment abnormality",
                "diffuse st segment depression",
                "minor st segment depression",
                "apical st depression",
                "diffuse scooped st segment depression",
                "anterior st segment depression",
                "st segment depression in anterolateral leads",
                "st segment elevation",
                "st segment depression is more marked in leads",
                "st segment depression in leads",
                "st segment depression in leads v4-v6",
                "abnormal st segment changes",
                "st elevation",
                "inferoapical st segment depression",
                "nonspecific st segment and t wave abnormalities",
                "st segment elevation consistent with acute injury",
                "widespread st segment depression",
                "diffuse st segment elevation",
                "infero- st segment depression",
                "marked st segment depression",
                "diffuse elevation of st segments",
                "nonspecific st segment",
                "st segment elevation in leads",
                "st segment changes",
                "marked st segment depression in leads",
                "st depression",
                "st segment depressions more marked",
                "inferior st segment elevation and q waves",
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
                "t wave inversion in leads",
                "t waves are slightly more inverted in leads",
                "(nonspecific st segment).*(t wave abnormalities)",
                "t wave abnormalities",
                "t wave inver",
                "t waves are upright in leads",
                "recent diffuse t wave flattening",
                "diffuse nonspecific st segment and t wave abnormalities",
                "tall t waves in precordial leads",
                "t wave inversions",
                "t wave changes",
                "possible st segment and t wave abn",
                "t wave inversion",
                "nonspecific st segment and t wave abnormalities",
                "t waves are lower or inverted in leads",
                "nonspecific t wave abnormali",
                "t wave flattening",
                "upright t waves",
                "t waves are inverted in leads",
                "t wave inveions",
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
                "t wave inversion in leads",
                "t waves are slightly more inverted in leads",
                "(nonspecific st segment).*(t wave abnormalities)",
                "t wave abnormalities",
                "t wave inver",
                "t waves are upright in leads",
                "recent diffuse t wave flattening",
                "diffuse nonspecific st segment and t wave abnormalities",
                "tall t waves in precordial leads",
                "t wave inversions",
                "t wave changes",
                "possible st segment and t wave abn",
                "t wave inversion",
                "nonspecific st segment and t wave abnormalities",
                "t waves are lower or inverted in leads",
                "nonspecific t wave abnormali",
                "t wave flattening",
                "upright t waves",
                "t waves are inverted in leads",
                "t wave inveions",
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
    path_prefix="ECG_PREFIX",
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
                "combined ventricular hypertrophy",
                "biventriclar hypertrophy",
                "left ventricular hypertr",
                "biventricular hypertrophy",
                "leftventricular hypertrophy",
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
                "combined ventricular hypertrophy",
                "biventriclar hypertrophy",
                "left ventricular hypertr",
                "biventricular hypertrophy",
                "leftventricular hypertrophy",
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
    path_prefix="ECG_PREFIX",
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
                "combined ventricular hypertrophy",
                "right ventricular hypertrophy",
                "biventriclar hypertrophy",
                "biventricular hypertrophy",
                "right ventricular enlargement",
                "rightventricular hypertrophy",
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
                "combined ventricular hypertrophy",
                "right ventricular hypertrophy",
                "biventriclar hypertrophy",
                "biventricular hypertrophy",
                "right ventricular enlargement",
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
                "atrial ectopy",
                "atrial premature beat",
                "atrial bigeminy",
                "isolated premature atrial contractions",
                "atrial trigeminy",
                "premature atrial complexes",
                "ectopic atrial complexes",
                "atrial premature complexes",
                "atrial ectopy has decreased",
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
                "atrial ectopy",
                "atrial premature beat",
                "atrial bigeminy",
                "isolated premature atrial contractions",
                "atrial trigeminy",
                "premature atrial complexes",
                "ectopic atrial complexes",
                "atrial premature complexes",
                "atrial ectopy has decreased",
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
                "ectopy more pronounced",
                "return of ectopy",
                "ectopy is new",
                "increased ectopy",
                "ectopy has increased",
                "ectopy have increased",
                "other than the ectopy",
                "ectopy has appeared",
                "new ectopy",
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
                "ectopy more pronounced",
                "return of ectopy",
                "ectopy is new",
                "increased ectopy",
                "ectopy has increased",
                "ectopy have increased",
                "other than the ectopy",
                "ectopy has appeared",
                "new ectopy",
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
                "ectopy has disappear",
                "ectopy has resolved",
                "no ectopy",
                "atrial ectopy gone",
                "ectopy is no longer seen",
                "ectopy is gone",
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
                "ectopy has disappear",
                "ectopy has resolved",
                "no ectopy",
                "atrial ectopy gone",
                "ectopy is no longer seen",
                "ectopy is gone",
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
                "premature ventricular and fusion complexes",
                "premature ventricular beat",
                "occasional premature ventricular complexes ",
                "ventricular ectopy",
                "ventricular trigeminy",
                "frequent premature ventricular or aberrantly conducted complexes ",
                "ventricular premature beat",
                "ventriculaar ectopy is now present",
                "ventricular premature complexes",
                "one premature ventricularbeat",
                "ventricular bigeminy",
                "premature ventricular compl",
                "isolated premature ventricular contractions",
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
                "premature ventricular and fusion complexes",
                "premature ventricular beat",
                "occasional premature ventricular complexes ",
                "ventricular ectopy",
                "ventricular trigeminy",
                "frequent premature ventricular or aberrantly conducted complexes ",
                "ventricular premature beat",
                "ventriculaar ectopy is now present",
                "ventricular premature complexes",
                "one premature ventricularbeat",
                "ventricular bigeminy",
                "premature ventricular compl",
                "isolated premature ventricular contractions",
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
                "2 to 1 av block",
                "2:1 block",
                "2:1 av block",
                "2 to 1 atrioventricular block",
                "2:1 atrioventricular block",
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
                "2 to 1 av block",
                "2:1 block",
                "2:1 av block",
                "2 to 1 atrioventricular block",
                "2:1 atrioventricular block",
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
            "av_dissociation": {"av dissociation", "atrioventricular dissociation"},
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
                "second degree type 1",
                "mobitz i",
                "second degree ",
                "mobitz type 1",
                "wenckebach",
                "fixed block",
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
                "second degree type 1",
                "mobitz i",
                "second degree ",
                "mobitz type 1",
                "wenckebach",
                "fixed block",
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
                "mobitz ii",
                "2nd degree sa block",
                "hay block",
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
                "mobitz ii",
                "2nd degree sa block",
                "hay block",
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
                "third degree atrioventricular block",
                "3rd degree atrioventricular block",
                "complete heart block",
                "third degree av block",
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
                "third degree atrioventricular block",
                "3rd degree atrioventricular block",
                "complete heart block",
                "third degree av block",
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
                "high grade atrioventricular block",
                "heart block",
                "av block",
                "high degree of block",
                "atrioventricular block",
                "heartblock",
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
                "high grade atrioventricular block",
                "heart block",
                "av block",
                "high degree of block",
                "atrioventricular block",
                "heartblock",
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
