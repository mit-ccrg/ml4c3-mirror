# Imports: standard library
from typing import Dict

# Imports: first party
from ml4c3.validators import validator_not_all_zero
from ml4c3.tensormap.ecg import (
    make_ecg_label_from_read_tff,
    make_binary_ecg_label_from_any_read_tff,
)
from ml4c3.definitions.ecg import ECG_PREFIX
from ml4c3.tensormap.TensorMap import TensorMap, Interpretation

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
                "atrial fibrillation with rapid ventricular response",
                "atrial fibrillation with moderate ventricular response",
                "fibrillation/flutter",
                "atrial fibrillation with controlled ventricular response",
                "afib",
                "atrial fib",
                "afibrillation",
                "atrial fibrillation",
                "atrialfibrillation",
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
                "atrial fibrillation with rapid ventricular response",
                "atrial fibrillation with moderate ventricular response",
                "fibrillation/flutter",
                "atrial fibrillation with controlled ventricular response",
                "afib",
                "atrial fib",
                "afibrillation",
                "atrial fibrillation",
                "atrialfibrillation",
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
                "fibrillation/flutter",
                "atrial flutter fixed block",
                "atrial flutter",
                "atrial flutter variable block",
                "probable flutter",
                "atrial flutter unspecified block",
                "tachycardia possibly flutter",
                "flutter",
                "aflutter",
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
                "fibrillation/flutter",
                "atrial flutter fixed block",
                "atrial flutter",
                "atrial flutter variable block",
                "probable flutter",
                "atrial flutter unspecified block",
                "tachycardia possibly flutter",
                "flutter",
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
    path_prefix=ECG_PREFIX,
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
                "wandering ear",
                "wandering ectopic atrial rhythm",
                "multifocal ectopic atrial rhythm",
                "dual atrial foci ",
                "unifocal ectopic atrial rhythm",
                "ectopicsupraventricular rhythm",
                "ectopic atrial rhythm",
                "low atrial pacer",
                "nonsinus atrial mechanism",
                "multiple atrial foci",
                "ectopic atrial rhythm ",
                "multifocal ear",
                "wandering atrial pacemaker",
                "unifocal ear",
                "unusual p wave axis",
                "multifocal atrialrhythm",
                "abnormal p vector",
                "atrial arrhythmia",
                "atrial rhythm",
                "multifocal atrial rhythm",
                "p wave axis suggests atrial rather than sinus mechanism",
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
                "wandering ear",
                "wandering ectopic atrial rhythm",
                "multifocal ectopic atrial rhythm",
                "dual atrial foci ",
                "unifocal ectopic atrial rhythm",
                "ectopicsupraventricular rhythm",
                "ectopic atrial rhythm",
                "low atrial pacer",
                "nonsinus atrial mechanism",
                "multiple atrial foci",
                "ectopic atrial rhythm ",
                "multifocal ear",
                "wandering atrial pacemaker",
                "unifocal ear",
                "unusual p wave axis",
                "multifocal atrialrhythm",
                "abnormal p vector",
                "atrial arrhythmia",
                "atrial rhythm",
                "multifocal atrial rhythm",
                "p wave axis suggests atrial rather than sinus mechanism",
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
                "unifocal ectopic atrial tachycardia",
                "ectopic atrial tachycardia",
                "multifocal atrial tachycardia",
                "wandering atrial tachycardia",
                "ectopic atrial tachycardia, unifocal",
                "unifocal atrial tachycardia",
                "ectopic atrial tachycardia, unspecified",
                "unspecified ectopic atrial tachycardia",
                "ectopic atrial tachycardia, multifocal",
                "multifocal ectopic atrial tachycardia",
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
                "unifocal ectopic atrial tachycardia",
                "ectopic atrial tachycardia",
                "multifocal atrial tachycardia",
                "wandering atrial tachycardia",
                "ectopic atrial tachycardia, unifocal",
                "unifocal atrial tachycardia",
                "ectopic atrial tachycardia, unspecified",
                "unspecified ectopic atrial tachycardia",
                "ectopic atrial tachycardia, multifocal",
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
    path_prefix=ECG_PREFIX,
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
        channel_terms={"sinus_pause": {"sinus pauses", "sinus pause"}},
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
        channel_terms={"sinus_pause": {"sinus pauses", "sinus pause"}},
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
                "type i sinoatrial block",
                "conducted sinus impulses",
                "marked sinus arrhythmia",
                "normal when compared with ecg of",
                "rhythm remains normal sinus",
                "normal sinus rhythm",
                "frequent native sinus beats",
                "normal ecg",
                "atrialbigeminy",
                "sinoatrial block, type ii",
                "type ii sinoatrial block",
                "type ii sa block",
                "type i sa block",
                "sa block",
                "atrial trigeminy",
                "rhythm has reverted to normal",
                "rhythm is now clearly sinus",
                "sinus exit block",
                "tracing is within normal limits",
                "1st degree sa block",
                "sinus arrhythmia",
                "2nd degree sa block",
                "sinus tachycardia",
                "sinus rhythm at a rate",
                "sinus rhythm",
                "tracing within normal limits",
                "sinus mechanism has replaced",
                "atrial bigeminal rhythm",
                "sa exit block",
                "sinoatrial block",
                "rhythm is normal sinus",
                "with occasional native sinus beats",
                "sa block, type i",
                "sinus slowing",
                "atrial bigeminal  rhythm",
                "atrial bigeminy and ventricular bigeminy",
                "sinus bradycardia",
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
                "type i sinoatrial block",
                "conducted sinus impulses",
                "marked sinus arrhythmia",
                "normal when compared with ecg of",
                "rhythm remains normal sinus",
                "normal sinus rhythm",
                "frequent native sinus beats",
                "normal ecg",
                "atrialbigeminy",
                "sinoatrial block, type ii",
                "type ii sinoatrial block",
                "type ii sa block",
                "type i sa block",
                "sa block",
                "atrial trigeminy",
                "rhythm has reverted to normal",
                "rhythm is now clearly sinus",
                "sinus exit block",
                "tracing is within normal limits",
                "1st degree sa block",
                "sinus arrhythmia",
                "2nd degree sa block",
                "sinus tachycardia",
                "sinus rhythm at a rate",
                "sinus rhythm",
                "tracing within normal limits",
                "sinus mechanism has replaced",
                "atrial bigeminal rhythm",
                "sa exit block",
                "sinoatrial block",
                "rhythm is normal sinus",
                "with occasional native sinus beats",
                "sa block, type i",
                "sinus slowing",
                "atrial bigeminal  rhythm",
                "atrial bigeminy and ventricular bigeminy",
                "sinus bradycardia",
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
                "atrioventricular reentrant tachycardia ",
                "junctional tachycardia",
                "av reentrant tachycardia ",
                "supraventricular tachycardia",
                "accelerated atrioventricular junctional rhythm",
                "atrial tachycardia",
                "av nodal reentry tachycardia",
                "av nodal reentrant",
                "accelerated nodal rhythm",
                "accelerated atrioventricular nodal rhythm",
                "avrt",
                "atrioventricular nodal reentry tachycardia",
                "avnrt",
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
                "atrioventricular reentrant tachycardia ",
                "junctional tachycardia",
                "av reentrant tachycardia ",
                "supraventricular tachycardia",
                "accelerated atrioventricular junctional rhythm",
                "atrial tachycardia",
                "av nodal reentry tachycardia",
                "av nodal reentrant",
                "accelerated nodal rhythm",
                "accelerated atrioventricular nodal rhythm",
                "avrt",
                "atrioventricular nodal reentry tachycardia",
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
                "rhythm unclear",
                "uncertain rhythm",
                "rhythm uncertain",
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
                "rhythm unclear",
                "uncertain rhythm",
                "rhythm uncertain",
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
                "apical st depression",
                "st segment depression is more marked in leads",
                "diffuse st segment elevation",
                "consistent with subendocardial ischemia",
                "anterolateral st segment depression",
                "possible anterior wall ischemia",
                "anterior subendocardial ischemia",
                "st segment depression in anterolateral leads",
                "inferior st segment elevation and q waves",
                "minor st segment depression",
                "consider anterior ischemia",
                "st elevation",
                "consider anterior and lateral ischemia",
                "lateral ischemia",
                "infero- st segment depression",
                "suggest anterior ischemia",
                "diffuse elevation of st segments",
                "st segment elevation",
                "inferoapical st segment depression",
                "consistent with lateral ischemia",
                "subendocardial ischemia",
                "nonspecific st segment depression",
                "diffuse scooped st segment depression",
                "apical subendocardial ischemia",
                "suggesting anterior ischemia",
                "suggests anterolateral ischemia",
                "anterolateral subendocardial ischemia",
                "st segment elevation consistent with acute injury",
                "septal ischemia",
                "st depression",
                "st segment depressions more marked",
                "widespread st segment depression",
                "anterior infarct or transmural ischemia",
                "anterolateral ischemia",
                "consistent with ischemia",
                "marked st segment depression in leads",
                "st segment depression in leads",
                "antero-apical ischemia",
                "marked st segment depression",
                "diffuse st segment depression",
                "st segment depression in leads v4-v6",
                "inferior subendocardial ischemia",
                "anterior st segment depression",
                "st segment depression",
                "inferior st segment depression",
                "st segment elevation in leads",
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
                "apical st depression",
                "st segment depression is more marked in leads",
                "diffuse st segment elevation",
                "consistent with subendocardial ischemia",
                "anterolateral st segment depression",
                "possible anterior wall ischemia",
                "anterior subendocardial ischemia",
                "st segment depression in anterolateral leads",
                "inferior st segment elevation and q waves",
                "minor st segment depression",
                "consider anterior ischemia",
                "st elevation",
                "consider anterior and lateral ischemia",
                "lateral ischemia",
                "infero- st segment depression",
                "suggest anterior ischemia",
                "diffuse elevation of st segments",
                "st segment elevation",
                "inferoapical st segment depression",
                "consistent with lateral ischemia",
                "subendocardial ischemia",
                "nonspecific st segment depression",
                "diffuse scooped st segment depression",
                "apical subendocardial ischemia",
                "suggesting anterior ischemia",
                "suggests anterolateral ischemia",
                "anterolateral subendocardial ischemia",
                "st segment elevation consistent with acute injury",
                "septal ischemia",
                "st depression",
                "st segment depressions more marked",
                "widespread st segment depression",
                "anterior infarct or transmural ischemia",
                "anterolateral ischemia",
                "consistent with ischemia",
                "marked st segment depression in leads",
                "st segment depression in leads",
                "antero-apical ischemia",
                "marked st segment depression",
                "diffuse st segment depression",
                "st segment depression in leads v4-v6",
                "inferior subendocardial ischemia",
                "anterior st segment depression",
                "st segment depression",
                "inferior st segment depression",
                "st segment elevation in leads",
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
                "abnormal st segment changes",
                "apical st depression",
                "st segment depression is more marked in leads",
                "diffuse st segment elevation",
                "nonspecific st segment and t wave abnormalities",
                "anterolateral st segment depression",
                "st segment depression in anterolateral leads",
                "inferior st segment elevation and q waves",
                "minor st segment depression",
                "st elevation",
                "infero- st segment depression",
                "st segment changes",
                "diffuse elevation of st segments",
                "st segment elevation",
                "inferoapical st segment depression",
                "nonspecific st segment depression",
                "diffuse scooped st segment depression",
                "st segment elevation consistent with acute injury",
                "st depression",
                "st segment depressions more marked",
                "widespread st segment depression",
                "st segment abnormality",
                "marked st segment depression in leads",
                "st segment depression in leads",
                "marked st segment depression",
                "nonspecific st segment",
                "diffuse st segment depression",
                "st segment depression in leads v4-v6",
                "anterior st segment depression",
                "st segment depression",
                "st segment elevation in leads",
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
                "abnormal st segment changes",
                "apical st depression",
                "st segment depression is more marked in leads",
                "diffuse st segment elevation",
                "nonspecific st segment and t wave abnormalities",
                "anterolateral st segment depression",
                "st segment depression in anterolateral leads",
                "inferior st segment elevation and q waves",
                "minor st segment depression",
                "st elevation",
                "infero- st segment depression",
                "st segment changes",
                "diffuse elevation of st segments",
                "st segment elevation",
                "inferoapical st segment depression",
                "nonspecific st segment depression",
                "diffuse scooped st segment depression",
                "st segment elevation consistent with acute injury",
                "st depression",
                "st segment depressions more marked",
                "widespread st segment depression",
                "st segment abnormality",
                "marked st segment depression in leads",
                "st segment depression in leads",
                "marked st segment depression",
                "nonspecific st segment",
                "diffuse st segment depression",
                "st segment depression in leads v4-v6",
                "anterior st segment depression",
                "st segment depression",
                "st segment elevation in leads",
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
                "diffuse nonspecific st segment and t wave abnormalities",
                "t waves are lower or inverted in leads",
                "t waves are upright in leads",
                "upright t waves",
                "recent diffuse t wave flattening",
                "nonspecific t wave abnormali",
                "tall t waves in precordial leads",
                "nonspecific st segment and t wave abnormalities",
                "(nonspecific st segment).*(t wave abnormalities)",
                "t wave inver",
                "t wave flattening",
                "t wave inversions",
                "t wave inversion in leads",
                "t wave inversion",
                "t wave changes",
                "possible st segment and t wave abn",
                "t wave abnormalities",
                "t waves are slightly more inverted in leads",
                "t waves are inverted in leads",
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
                "diffuse nonspecific st segment and t wave abnormalities",
                "t waves are lower or inverted in leads",
                "t waves are upright in leads",
                "upright t waves",
                "recent diffuse t wave flattening",
                "nonspecific t wave abnormali",
                "tall t waves in precordial leads",
                "nonspecific st segment and t wave abnormalities",
                "(nonspecific st segment).*(t wave abnormalities)",
                "t wave inver",
                "t wave flattening",
                "t wave inversions",
                "t wave inversion in leads",
                "t wave inversion",
                "t wave changes",
                "possible st segment and t wave abn",
                "t wave abnormalities",
                "t waves are slightly more inverted in leads",
                "t waves are inverted in leads",
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


tmaps["fascicular_rhythm"] = TensorMap(
    "fascicular_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_fascicular_rhythm": 0, "fascicular_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"fascicular_rhythm": {"fascicular rhythm"}},
        not_found_channel="no_fascicular_rhythm",
    ),
)


tmaps["fascicular_rhythm_any"] = TensorMap(
    "fascicular_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_fascicular_rhythm": 0, "fascicular_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"fascicular_rhythm": {"fascicular rhythm"}},
        not_found_channel="no_fascicular_rhythm",
    ),
)


tmaps["fusion_complexes"] = TensorMap(
    "fusion_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_fusion_complexes": 0, "fusion_complexes": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"fusion_complexes": {"fusion beats", "fusion complexes"}},
        not_found_channel="no_fusion_complexes",
    ),
)


tmaps["fusion_complexes_any"] = TensorMap(
    "fusion_complexes_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_fusion_complexes": 0, "fusion_complexes": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"fusion_complexes": {"fusion beats", "fusion complexes"}},
        not_found_channel="no_fusion_complexes",
    ),
)


tmaps["idioventricular_rhythm"] = TensorMap(
    "idioventricular_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_idioventricular_rhythm": 0, "idioventricular_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"idioventricular_rhythm": {"idioventricular rhythm"}},
        not_found_channel="no_idioventricular_rhythm",
    ),
)


tmaps["idioventricular_rhythm_any"] = TensorMap(
    "idioventricular_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_idioventricular_rhythm": 0, "idioventricular_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"idioventricular_rhythm": {"idioventricular rhythm"}},
        not_found_channel="no_idioventricular_rhythm",
    ),
)


tmaps["junctional_rhythm"] = TensorMap(
    "junctional_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_junctional_rhythm": 0, "junctional_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"junctional_rhythm": {"junctional rhythm"}},
        not_found_channel="no_junctional_rhythm",
    ),
)


tmaps["junctional_rhythm_any"] = TensorMap(
    "junctional_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_junctional_rhythm": 0, "junctional_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"junctional_rhythm": {"junctional rhythm"}},
        not_found_channel="no_junctional_rhythm",
    ),
)


tmaps["parasystole"] = TensorMap(
    "parasystole",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_parasystole": 0, "parasystole": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"parasystole": {"parasystole"}},
        not_found_channel="no_parasystole",
    ),
)


tmaps["parasystole_any"] = TensorMap(
    "parasystole_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_parasystole": 0, "parasystole": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"parasystole": {"parasystole"}},
        not_found_channel="no_parasystole",
    ),
)


tmaps["ventricular_fibrillation"] = TensorMap(
    "ventricular_fibrillation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ventricular_fibrillation": 0, "ventricular_fibrillation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_fibrillation": {"ventricular fibrillation"}},
        not_found_channel="no_ventricular_fibrillation",
    ),
)


tmaps["ventricular_fibrillation_any"] = TensorMap(
    "ventricular_fibrillation_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ventricular_fibrillation": 0, "ventricular_fibrillation": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_fibrillation": {"ventricular fibrillation"}},
        not_found_channel="no_ventricular_fibrillation",
    ),
)


tmaps["ventricular_tachycardia"] = TensorMap(
    "ventricular_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ventricular_tachycardia": 0, "ventricular_tachycardia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ventricular_tachycardia": {
                " ventricular tachy",
                "\\w*(?<!supra)(ventricular tachycardia)",
            },
        },
        not_found_channel="no_ventricular_tachycardia",
    ),
)


tmaps["ventricular_tachycardia_any"] = TensorMap(
    "ventricular_tachycardia_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_ventricular_tachycardia": 0, "ventricular_tachycardia": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ventricular_tachycardia": {
                " ventricular tachy",
                "\\w*(?<!supra)(ventricular tachycardia)",
            },
        },
        not_found_channel="no_ventricular_tachycardia",
    ),
)


tmaps["wide_qrs_rhythm"] = TensorMap(
    "wide_qrs_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_wide_qrs_rhythm": 0, "wide_qrs_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"wide_qrs_rhythm": {"wide qrs rhythm"}},
        not_found_channel="no_wide_qrs_rhythm",
    ),
)


tmaps["wide_qrs_rhythm_any"] = TensorMap(
    "wide_qrs_rhythm_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_wide_qrs_rhythm": 0, "wide_qrs_rhythm": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"wide_qrs_rhythm": {"wide qrs rhythm"}},
        not_found_channel="no_wide_qrs_rhythm",
    ),
)


tmaps["first_degree_av_block"] = TensorMap(
    "first_degree_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix=ECG_PREFIX,
    channel_map={"no_first_degree_av_block": 0, "first_degree_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "first_degree_av_block": {
                "first degree atrioventricular block ",
                "first degree avb",
                "first degree av block",
                "first degree atrioventricular block",
                "first degree atrioventricular  block",
                "1st degree atrioventricular  block",
                "first degree atrioventricular",
            },
        },
        not_found_channel="no_first_degree_av_block",
    ),
)


tmaps["first_degree_av_block_any"] = TensorMap(
    "first_degree_av_block_any",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"no_first_degree_av_block": 0, "first_degree_av_block": 1},
    validators=validator_not_all_zero,
    tensor_from_file=make_binary_ecg_label_from_any_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "first_degree_av_block": {
                "first degree atrioventricular block ",
                "first degree avb",
                "first degree av block",
                "first degree atrioventricular block",
                "first degree atrioventricular  block",
                "1st degree atrioventricular  block",
                "first degree atrioventricular",
            },
        },
        not_found_channel="no_first_degree_av_block",
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
                "lbbb",
                "left bundle branch block",
                "bundle branch block",
                "left bbb",
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
                "lbbb",
                "left bundle branch block",
                "bundle branch block",
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
                "rbbb",
                "bundle branch block",
                "right bundle branch block",
                "left bbb",
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
                "rbbb",
                "bundle branch block",
                "right bundle branch block",
                "left bbb",
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
                "2:1 block",
                "2:1 atrioventricular block",
                "2 to 1 av block",
                "2 to 1 atrioventricular block",
                "2:1 av block",
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
                "2:1 block",
                "2:1 atrioventricular block",
                "2 to 1 av block",
                "2 to 1 atrioventricular block",
                "2:1 av block",
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
                "mobitz type 1",
                "fixed block",
                "second degree ",
                "mobitz i",
                "mobitz 1 block",
                "second degree type 1",
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
                "mobitz type 1",
                "fixed block",
                "second degree ",
                "mobitz i",
                "mobitz 1 block",
                "second degree type 1",
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
                "2nd degree sa block",
                "hay block",
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
                "2nd degree sa block",
                "hay block",
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
                "third degree av block",
                "complete heart block",
                "third degree atrioventricular block",
                "3rd degree av block",
                "3rd degree atrioventricular block",
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
                "third degree av block",
                "complete heart block",
                "third degree atrioventricular block",
                "3rd degree av block",
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
                "high grade atrioventricular block",
                "heartblock",
                "high degree of block",
                "atrioventricular block",
                "av block",
                "heart block",
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
                "high grade atrioventricular block",
                "heartblock",
                "high degree of block",
                "atrioventricular block",
                "av block",
                "heart block",
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
                "atrial premature complexes",
                "atrial bigeminy",
                "isolated premature atrial contractions",
                "ectopic atrial complexes",
                "atrial premature beat",
                "atrial ectopy has decreased",
                "premature atrial complexes",
                "atrial ectopy",
                "atrial trigeminy",
                "premature atrial co",
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
                "atrial premature complexes",
                "atrial bigeminy",
                "isolated premature atrial contractions",
                "ectopic atrial complexes",
                "atrial premature beat",
                "atrial ectopy has decreased",
                "premature atrial complexes",
                "atrial ectopy",
                "atrial trigeminy",
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
                "ectopy has increased",
                "ectopy have increased",
                "return of ectopy",
                "ectopy is new",
                "new ectopy",
                "increased ectopy",
                "other than the ectopy",
                "ectopy more pronounced",
                "ectopy has appeared",
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
                "ectopy has increased",
                "ectopy have increased",
                "return of ectopy",
                "ectopy is new",
                "new ectopy",
                "increased ectopy",
                "other than the ectopy",
                "ectopy more pronounced",
                "ectopy has appeared",
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
                "ectopy has resolved",
                "ectopy is no longer seen",
                "no ectopy",
                "ectopy is gone",
                "no longer any ectopy",
                "atrial ectopy gone",
                "ectopy has disappear",
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
                "ectopy has resolved",
                "ectopy is no longer seen",
                "no ectopy",
                "ectopy is gone",
                "no longer any ectopy",
                "atrial ectopy gone",
                "ectopy has disappear",
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
                "ventricular premature beat",
                "ventricular bigeminy",
                "one premature ventricularbeat",
                "ventricular trigeminy",
                "isolated premature ventricular contractions",
                "occasional premature ventricular complexes ",
                "premature ventricular beat",
                "premature ventricular and fusion complexes",
                "frequent premature ventricular or aberrantly conducted complexes ",
                "ventricular premature complexes",
                "ventricular ectopy",
                "premature ventricular compl",
                "ventriculaar ectopy is now present",
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
                "ventricular premature beat",
                "ventricular bigeminy",
                "one premature ventricularbeat",
                "ventricular trigeminy",
                "isolated premature ventricular contractions",
                "occasional premature ventricular complexes ",
                "premature ventricular beat",
                "premature ventricular and fusion complexes",
                "frequent premature ventricular or aberrantly conducted complexes ",
                "ventricular premature complexes",
                "ventricular ectopy",
                "premature ventricular compl",
                "ventriculaar ectopy is now present",
            },
        },
        not_found_channel="no_ventricular_premature_complexes",
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
                "possible acute inferior myocardial infarction",
                "counterclockwise rotation consistent with post myocardial infarction",
                "subendocardial myocardial infarction",
                "old anterior myocardial infarction",
                "possible myocardial infarction",
                "myocardial infarction nonspecific st segment",
                "lateral myocardial infarction - of indeterminate age",
                "recent myocardial infarction",
                "myocardial infarction cannot rule out",
                "anterior infarct of indeterminate age",
                "cannot rule out true posterior myocardial infarction",
                "old inferolateral myocardial infarction",
                "cannot rule out true posterior myocardial infarction versus counterclockwise rotation",
                "possible acute myocardial infarction",
                "acute myocardial infarction of indeterminate age",
                "anterolateral myocardial infarction",
                "consistent with anterior myocardial infarction of indeterminate age",
                "apical myocardial infarction",
                "old anterolateral infarct",
                "possible true posterior myocardial infarction",
                "evolving myocardial infarction",
                "septal infarct",
                "possible anteroseptal myocardial infarction of uncertain age",
                "myocardial infarction possible when compared",
                "normal counterclockwise rotation versusold true posterior myocardial infarction",
                "anterolateral infarct of indeterminate age ",
                "inferolateral myocardial infarction",
                "anterior myocardial infarction",
                "old high lateral myocardial infarction",
                "anterior myocardial infarction of indeterminate age",
                "(myocardial infarction versus).*(counterclockwise rotation)",
                "old inferoapical myocardial infarction",
                "old infero-posterior myocardial infarction",
                "borderline anterolateral myocardial infarction",
                "normal counterclockwise rotation versus old true posterior myocardial infarction",
                "(true posterior).*(myocardial infarction)",
                "rule out interim myocardial infarction",
                "myocardial infarction versus pericarditis",
                "cannot rule out anteroseptal infarct",
                "consistent with anteroseptal infarct",
                "extensive anterior infarct",
                "acuteanterior myocardial infarction",
                "transmural ischemia myocardial infarction",
                "inferoapical myocardial infarction of indeterminate age",
                "lateral myocardial infarction of indeterminate age",
                "myocardial infarction old high lateral",
                "raises possibility of septal infarct",
                "apical myocardial infarction of indeterminate age",
                "inferior myocardial infarction of indeterminate age",
                "possible old septal myocardial infarction",
                "old infero-postero-lateral myocardial infarction",
                "posterior myocardial infarction",
                "posterior wall myocardial infarction",
                "age indeterminate old inferior wall myocardial infarction",
                "myocardial infarction pattern",
                "inferior wall myocardial infarction of indeterminate age",
                "subendocardial infarct",
                "old anterior infarct",
                "possible septal myocardial infarction",
                "suggestive of old true posterior myocardial infarction st abnormality",
                "myocardial infarction compared with the last previous ",
                "infero-apical myocardial infarction of indeterminate age",
                "subendocardial ischemia myocardial infarction",
                "post myocardial infarction , of indeterminate age",
                "myocardial infarction of indeterminate age",
                "normal left axis deviation counterclockwise rotation versus old true posterior myocardial infarction",
                "anterolateral myocardial infarction appears recent",
                "old inferoposterior myocardial infarction",
                "evolution of myocardial infarction",
                "myocardial infarction extension",
                "concurrent ischemia myocardial infarction",
                "old anterior wall myocardial infarction",
                "old myocardial infarction",
                "anteroseptal infarct of indeterminate age",
                "extensive anterolateral myocardial infarction",
                "cannot rule out anterior infarct , age undetermined",
                "anteroseptal and lateral myocardial infarction",
                "normal counterclockwise rotation versus true posterior myocardial infarction",
                "inferior myocardial infarction of indeterminate",
                "probable apicolateral myocardial infarction",
                "subendocardial ischemia or myocardial infarction",
                "acute myocardial infarction in evolution",
                "acute infarct",
                "(counterclockwise rotation).*(true posterior)",
                "evolving inferior wall myocardial infarction",
                "evolving counterclockwise rotation versus true posterior myocardial infarction",
                "myocardial infarction indeterminate",
                "old lateral myocardial infarction",
                "ongoing ischemia versus myocardial infarction",
                "inferior myocardial infarction",
                "acute myocardial infarction",
                "old infero-posterior lateral myocardial infarction",
                "block inferior myocardial infarction",
                "possible anterolateral myocardial infarction",
                "counterclockwise rotation versus true posterior myocardial infarction",
                "anteroseptal myocardial infarction",
                "old posterolateral myocardial infarction",
                "(possible old).*(true posterior).*(myocardial infarction)",
                "lateral wall myocardial infarction",
                "known true posterior myocardial infarction",
                "(consistent with).*(true posterior).*(myocardial infarction)",
                "possible old lateral myocardial infarction",
                "old inferior posterolateral myocardial infarction",
                "myocardial infarction",
                "consistent with ischemia myocardial infarction",
                "possible inferior myocardial infarction",
                "possible anteroseptal myocardial infarction",
                "extensive myocardial infarction of indeterminate age ",
                "infero and apicolateral myocardial infarction",
                "inferior myocardial infarction , age undetermined",
                "suggestive of old true posterior myocardial infarction",
                "old anteroseptal myocardial infarction",
                "inferior myocardial infarction , possibly acute",
                "antero-apical ischemia versus myocardial infarction",
                "subendocardial ischemia subendocardial myocardial inf",
                "old inferior and anterior myocardial infarctions",
                "infero-apical myocardial infarction",
                "anterolateral myocardial infarction , possibly recent  ",
                "old anteroseptal infarct",
                "myocardial infarction compared with the last previous tracing ",
                "old anterolateral myocardial infarction",
                "old inferior myocardial infarction",
                "acute anterior wall myocardial infarction",
                "cannot rule out anterior wall ischemia myocardial infarction",
                "old apicolateral myocardial infarction",
                "old true posterior myocardial infarction",
                "old infero-posterior and apical myocardial infarction",
                "true posterior myocardial infarction of indeterminate age",
                "myocardial infarction when compared with ecg of",
                "antero-apical and lateral myocardial infarction evolving",
                "old inferior wall myocardial infarction",
                "new incomplete posterior lateral myocardial infarction",
                "cannot rule out inferoposterior myoca",
                "evolving anterior infarct",
                "acute anterior infarct",
                "counterclockwise rotation versus old true posterior myocardial infarction",
                "old inferior anterior myocardial infarctions",
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
                "possible acute inferior myocardial infarction",
                "counterclockwise rotation consistent with post myocardial infarction",
                "subendocardial myocardial infarction",
                "old anterior myocardial infarction",
                "possible myocardial infarction",
                "myocardial infarction nonspecific st segment",
                "lateral myocardial infarction - of indeterminate age",
                "recent myocardial infarction",
                "myocardial infarction cannot rule out",
                "anterior infarct of indeterminate age",
                "cannot rule out true posterior myocardial infarction",
                "old inferolateral myocardial infarction",
                "cannot rule out true posterior myocardial infarction versus counterclockwise rotation",
                "possible acute myocardial infarction",
                "acute myocardial infarction of indeterminate age",
                "anterolateral myocardial infarction",
                "consistent with anterior myocardial infarction of indeterminate age",
                "apical myocardial infarction",
                "old anterolateral infarct",
                "possible true posterior myocardial infarction",
                "evolving myocardial infarction",
                "septal infarct",
                "possible anteroseptal myocardial infarction of uncertain age",
                "myocardial infarction possible when compared",
                "normal counterclockwise rotation versusold true posterior myocardial infarction",
                "anterolateral infarct of indeterminate age ",
                "inferolateral myocardial infarction",
                "anterior myocardial infarction",
                "old high lateral myocardial infarction",
                "anterior myocardial infarction of indeterminate age",
                "(myocardial infarction versus).*(counterclockwise rotation)",
                "old inferoapical myocardial infarction",
                "old infero-posterior myocardial infarction",
                "borderline anterolateral myocardial infarction",
                "normal counterclockwise rotation versus old true posterior myocardial infarction",
                "(true posterior).*(myocardial infarction)",
                "rule out interim myocardial infarction",
                "myocardial infarction versus pericarditis",
                "cannot rule out anteroseptal infarct",
                "consistent with anteroseptal infarct",
                "extensive anterior infarct",
                "acuteanterior myocardial infarction",
                "transmural ischemia myocardial infarction",
                "inferoapical myocardial infarction of indeterminate age",
                "lateral myocardial infarction of indeterminate age",
                "myocardial infarction old high lateral",
                "raises possibility of septal infarct",
                "apical myocardial infarction of indeterminate age",
                "inferior myocardial infarction of indeterminate age",
                "possible old septal myocardial infarction",
                "old infero-postero-lateral myocardial infarction",
                "posterior myocardial infarction",
                "posterior wall myocardial infarction",
                "age indeterminate old inferior wall myocardial infarction",
                "myocardial infarction pattern",
                "inferior wall myocardial infarction of indeterminate age",
                "subendocardial infarct",
                "old anterior infarct",
                "possible septal myocardial infarction",
                "suggestive of old true posterior myocardial infarction st abnormality",
                "myocardial infarction compared with the last previous ",
                "infero-apical myocardial infarction of indeterminate age",
                "subendocardial ischemia myocardial infarction",
                "post myocardial infarction , of indeterminate age",
                "myocardial infarction of indeterminate age",
                "normal left axis deviation counterclockwise rotation versus old true posterior myocardial infarction",
                "anterolateral myocardial infarction appears recent",
                "old inferoposterior myocardial infarction",
                "evolution of myocardial infarction",
                "myocardial infarction extension",
                "concurrent ischemia myocardial infarction",
                "old anterior wall myocardial infarction",
                "old myocardial infarction",
                "anteroseptal infarct of indeterminate age",
                "extensive anterolateral myocardial infarction",
                "cannot rule out anterior infarct , age undetermined",
                "anteroseptal and lateral myocardial infarction",
                "normal counterclockwise rotation versus true posterior myocardial infarction",
                "inferior myocardial infarction of indeterminate",
                "probable apicolateral myocardial infarction",
                "subendocardial ischemia or myocardial infarction",
                "acute myocardial infarction in evolution",
                "acute infarct",
                "(counterclockwise rotation).*(true posterior)",
                "evolving inferior wall myocardial infarction",
                "evolving counterclockwise rotation versus true posterior myocardial infarction",
                "myocardial infarction indeterminate",
                "old lateral myocardial infarction",
                "ongoing ischemia versus myocardial infarction",
                "inferior myocardial infarction",
                "acute myocardial infarction",
                "old infero-posterior lateral myocardial infarction",
                "block inferior myocardial infarction",
                "possible anterolateral myocardial infarction",
                "counterclockwise rotation versus true posterior myocardial infarction",
                "anteroseptal myocardial infarction",
                "old posterolateral myocardial infarction",
                "(possible old).*(true posterior).*(myocardial infarction)",
                "lateral wall myocardial infarction",
                "known true posterior myocardial infarction",
                "(consistent with).*(true posterior).*(myocardial infarction)",
                "possible old lateral myocardial infarction",
                "old inferior posterolateral myocardial infarction",
                "myocardial infarction",
                "consistent with ischemia myocardial infarction",
                "possible inferior myocardial infarction",
                "possible anteroseptal myocardial infarction",
                "extensive myocardial infarction of indeterminate age ",
                "infero and apicolateral myocardial infarction",
                "inferior myocardial infarction , age undetermined",
                "suggestive of old true posterior myocardial infarction",
                "old anteroseptal myocardial infarction",
                "inferior myocardial infarction , possibly acute",
                "antero-apical ischemia versus myocardial infarction",
                "subendocardial ischemia subendocardial myocardial inf",
                "old inferior and anterior myocardial infarctions",
                "infero-apical myocardial infarction",
                "anterolateral myocardial infarction , possibly recent  ",
                "old anteroseptal infarct",
                "myocardial infarction compared with the last previous tracing ",
                "old anterolateral myocardial infarction",
                "old inferior myocardial infarction",
                "acute anterior wall myocardial infarction",
                "cannot rule out anterior wall ischemia myocardial infarction",
                "old apicolateral myocardial infarction",
                "old true posterior myocardial infarction",
                "old infero-posterior and apical myocardial infarction",
                "true posterior myocardial infarction of indeterminate age",
                "myocardial infarction when compared with ecg of",
                "antero-apical and lateral myocardial infarction evolving",
                "old inferior wall myocardial infarction",
                "new incomplete posterior lateral myocardial infarction",
                "cannot rule out inferoposterior myoca",
                "evolving anterior infarct",
                "acute anterior infarct",
                "counterclockwise rotation versus old true posterior myocardial infarction",
                "old inferior anterior myocardial infarctions",
            },
        },
        not_found_channel="no_mi",
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
                "poor r wave progression",
                "unusual r wave progression",
                "slow precordial r wave progression",
                "slowprecordial r wave progression",
                "abnormal precordial r wave progression",
                "abnormal precordial r wave progression or poor r wave progression",
                "early r wave progression",
                "poor precordial r wave progression",
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
                "poor r wave progression",
                "unusual r wave progression",
                "slow precordial r wave progression",
                "slowprecordial r wave progression",
                "abnormal precordial r wave progression",
                "abnormal precordial r wave progression or poor r wave progression",
                "early r wave progression",
                "poor precordial r wave progression",
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
    path_prefix=ECG_PREFIX,
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
                "biatrial hypertrophy",
                "biatrial enlargement",
                "left atrial enla",
                "combined atrial enlargement",
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
                "biatrial hypertrophy",
                "biatrial enlargement",
                "left atrial enla",
                "combined atrial enlargement",
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
                "leftventricular hypertrophy",
                "combined ventricular hypertrophy",
                "left ventricular hypertr",
                "biventriclar hypertrophy",
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
                "leftventricular hypertrophy",
                "combined ventricular hypertrophy",
                "left ventricular hypertr",
                "biventriclar hypertrophy",
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
                "biatrial hypertrophy",
                "right atrial enla",
                "biatrial enlargement",
                "combined atrial enlargement",
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
                "biatrial hypertrophy",
                "right atrial enla",
                "biatrial enlargement",
                "combined atrial enlargement",
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
                "biventricular hypertrophy",
                "combined ventricular hypertrophy",
                "right ventricular enlargement",
                "rightventricular hypertrophy",
                "biventriclar hypertrophy",
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
                "right ventricular hypertrophy",
                "biventricular hypertrophy",
                "combined ventricular hypertrophy",
                "right ventricular enlargement",
                "rightventricular hypertrophy",
                "biventriclar hypertrophy",
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
                "ventricular paced",
                "failure to inhibit ventricular",
                "atrial-sensed ventricular-paced rhythm",
                "dual chamber pacing",
                "ventricular pacing has replaced av pacing",
                "competitive av pacing",
                "biventricular-paced rhythm",
                "shows dual chamber pacing",
                "atrial-paced complexes ",
                "demand v-pacing",
                "demand ventricular pacemaker",
                "unipolar right ventricular  pacing",
                "atrial-paced rhythm",
                "ventricular-paced complexes",
                "av dual-paced complexes",
                "failure to capture ventricular",
                "failure to capture atrial",
                "atrial-sensed ventricular-paced complexes",
                "a triggered v-paced rhythm",
                "failure to inhibit atrial",
                "atrial triggered ventricular pacing",
                "failure to pace atrial",
                "electronic pacemaker",
                "ventricular-paced rhythm",
                "av dual-paced rhythm",
                "v-paced rhythm",
                "ventricular demand pacing",
                "v-paced",
                "atrially triggered v paced",
                "v-paced beats",
                "biventricular-paced complexes",
                "sequential pacing",
                "failure to pace ventricular",
                "ventricular pacing",
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
                "ventricular paced",
                "failure to inhibit ventricular",
                "atrial-sensed ventricular-paced rhythm",
                "dual chamber pacing",
                "ventricular pacing has replaced av pacing",
                "competitive av pacing",
                "biventricular-paced rhythm",
                "shows dual chamber pacing",
                "atrial-paced complexes ",
                "demand v-pacing",
                "demand ventricular pacemaker",
                "unipolar right ventricular  pacing",
                "atrial-paced rhythm",
                "ventricular-paced complexes",
                "av dual-paced complexes",
                "failure to capture ventricular",
                "failure to capture atrial",
                "atrial-sensed ventricular-paced complexes",
                "a triggered v-paced rhythm",
                "failure to inhibit atrial",
                "atrial triggered ventricular pacing",
                "failure to pace atrial",
                "electronic pacemaker",
                "ventricular-paced rhythm",
                "av dual-paced rhythm",
                "v-paced rhythm",
                "ventricular demand pacing",
                "v-paced",
                "atrially triggered v paced",
                "v-paced beats",
                "biventricular-paced complexes",
                "sequential pacing",
                "failure to pace ventricular",
                "ventricular pacing",
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
                "tracing is within normal limits",
                "normal ecg",
                "normal sinus rhythm",
                "sinus tachycardia",
                "normal tracing",
                "sinus rhythm",
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
                "tracing is within normal limits",
                "normal ecg",
                "normal sinus rhythm",
                "sinus tachycardia",
                "normal tracing",
                "sinus rhythm",
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
                "indeterminate axis",
                "indeterminate qrs axis",
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
                "indeterminate axis",
                "indeterminate qrs axis",
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
                "axis shifted left",
                "left axis deviation",
                "leftward axis",
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
                "axis shifted left",
                "left axis deviation",
                "leftward axis",
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
                "right axis deviation",
                "right superior axis deviation",
                "rightward axis",
                "axis shifted right",
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
                "right axis deviation",
                "right superior axis deviation",
                "rightward axis",
                "axis shifted right",
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
