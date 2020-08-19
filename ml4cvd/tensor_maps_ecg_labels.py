# Imports: standard library
from typing import Dict

# Imports: first party
from ml4cvd.TensorMap import TensorMap, Interpretation
from ml4cvd.tensor_maps_ecg import make_ecg_label_from_read_tff

tmaps: Dict[str, TensorMap] = {}
tmaps["asystole"] = TensorMap(
    "asystole",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_asystole": 0, "asystole": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"asystole": {"asystole"}},
        not_found_channel="no_asystole",
    ),
)


tmaps["atrial_fibrillation"] = TensorMap(
    "atrial_fibrillation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_atrial_fibrillation": 0, "atrial_fibrillation": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_fibrillation": {
                "atrial fibrillation with controlled ventricular response",
                "afib",
                "fibrillation/flutter",
                "atrial fibrillation with moderate ventricular response",
                "atrial fibrillation with rapid ventricular response",
                "atrialfibrillation",
                "atrial fibrillation",
                "afibrillation",
                "atrial fib",
            },
        },
        not_found_channel="no_atrial_fibrillation",
    ),
)


tmaps["atrial_flutter"] = TensorMap(
    "atrial_flutter",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_atrial_flutter": 0, "atrial_flutter": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_flutter": {
                "aflutter",
                "atrial flutter unspecified block",
                "atrial flutter fixed block",
                "tachycardia possibly flutter",
                "fibrillation/flutter",
                "probable flutter",
                "atrial flutter variable block",
                "flutter",
                "atrial flutter",
            },
        },
        not_found_channel="no_atrial_flutter",
    ),
)


tmaps["atrial_paced_rhythm"] = TensorMap(
    "atrial_paced_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_atrial_paced_rhythm": 0, "atrial_paced_rhythm": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"atrial_paced_rhythm": {"atrial pacing", "atrial paced rhythm"}},
        not_found_channel="no_atrial_paced_rhythm",
    ),
)


tmaps["ectopic_atrial_bradycardia"] = TensorMap(
    "ectopic_atrial_bradycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ectopic_atrial_bradycardia": 0, "ectopic_atrial_bradycardia": 1},
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


tmaps["ectopic_atrial_rhythm"] = TensorMap(
    "ectopic_atrial_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ectopic_atrial_rhythm": 0, "ectopic_atrial_rhythm": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_rhythm": {
                "atrial arrhythmia",
                "abnormal p vector",
                "ectopic atrial rhythm ",
                "low atrial pacer",
                "unifocal ear",
                "nonsinus atrial mechanism",
                "unusual p wave axis",
                "multifocal atrialrhythm",
                "unifocal ectopic atrial rhythm",
                "multifocal ectopic atrial rhythm",
                "atrial rhythm",
                "dual atrial foci ",
                "multifocal ear",
                "ectopic atrial rhythm",
                "multifocal atrial rhythm",
                "wandering atrial pacemaker",
                "p wave axis suggests atrial rather than sinus mechanism",
                "multiple atrial foci",
                "wandering ear",
                "wandering ectopic atrial rhythm",
                "ectopicsupraventricular rhythm",
            },
        },
        not_found_channel="no_ectopic_atrial_rhythm",
    ),
)


tmaps["ectopic_atrial_tachycardia"] = TensorMap(
    "ectopic_atrial_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ectopic_atrial_tachycardia": 0, "ectopic_atrial_tachycardia": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_tachycardia": {
                "unifocal ectopic atrial tachycardia",
                "ectopic atrial tachycardia, multifocal",
                "multifocal ectopic atrial tachycardia",
                "ectopic atrial tachycardia, unspecified",
                "unspecified ectopic atrial tachycardia",
                "ectopic atrial tachycardia, unifocal",
                "ectopic atrial tachycardia",
                "multifocal atrial tachycardia",
                "wandering atrial tachycardia",
                "unifocal atrial tachycardia",
            },
        },
        not_found_channel="no_ectopic_atrial_tachycardia",
    ),
)


tmaps["narrow_qrs_tachycardia"] = TensorMap(
    "narrow_qrs_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_narrow_qrs_tachycardia": 0, "narrow_qrs_tachycardia": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "narrow_qrs_tachycardia": {
                "narrow complex tachycardia",
                "tachycardia narrow qrs",
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
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_pulseless_electrical_activity": 0,
        "pulseless_electrical_activity": 1,
    },
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


tmaps["retrograde_atrial_activation"] = TensorMap(
    "retrograde_atrial_activation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_retrograde_atrial_activation": 0,
        "retrograde_atrial_activation": 1,
    },
    tensor_from_file=make_ecg_label_from_read_tff(
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_sinus_arrest": 0, "sinus_arrest": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sinus_arrest": {"sinus arrest"}},
        not_found_channel="no_sinus_arrest",
    ),
)


tmaps["sinus_pause"] = TensorMap(
    "sinus_pause",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_sinus_pause": 0, "sinus_pause": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sinus_pause": {"sinus pauses", "sinus pause"}},
        not_found_channel="no_sinus_pause",
    ),
)


tmaps["sinus_rhythm"] = TensorMap(
    "sinus_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_sinus_rhythm": 0, "sinus_rhythm": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "sinus_rhythm": {
                "normal sinus rhythm",
                "rhythm has reverted to normal",
                "atrial bigeminy and ventricular bigeminy",
                "sinoatrial block, type ii",
                "rhythm remains normal sinus",
                "atrialbigeminy",
                "frequent native sinus beats",
                "rhythm is normal sinus",
                "atrial bigeminal rhythm",
                "sinus arrhythmia",
                "atrial trigeminy",
                "sinus tachycardia",
                "sinus rhythm at a rate",
                "type ii sa block",
                "tracing is within normal limits",
                "marked sinus arrhythmia",
                "1st degree sa block",
                "2nd degree sa block",
                "sinus exit block",
                "sinus bradycardia",
                "sinoatrial block",
                "sinus slowing",
                "sinus rhythm",
                "sa exit block",
                "atrial bigeminal  rhythm",
                "sinus mechanism has replaced",
                "sa block, type i",
                "type i sa block",
                "rhythm is now clearly sinus",
                "sa block",
                "normal when compared with ecg of",
                "conducted sinus impulses",
                "with occasional native sinus beats",
                "type i sinoatrial block",
                "type ii sinoatrial block",
                "normal ecg",
                "tracing within normal limits",
            },
        },
        not_found_channel="no_sinus_rhythm",
    ),
)


tmaps["supraventricular_tachycardia"] = TensorMap(
    "supraventricular_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_supraventricular_tachycardia": 0,
        "supraventricular_tachycardia": 1,
    },
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "supraventricular_tachycardia": {
                "av nodal reentry tachycardia",
                "avnrt",
                "accelerated atrioventricular junctional rhythm",
                "accelerated atrioventricular nodal rhythm",
                "accelerated nodal rhythm",
                "av nodal reentrant",
                "av reentrant tachycardia ",
                "supraventricular tachycardia",
                "junctional tachycardia",
                "atrioventricular reentrant tachycardia ",
                "atrial tachycardia",
                "atrioventricular nodal reentry tachycardia",
                "avrt",
            },
        },
        not_found_channel="no_supraventricular_tachycardia",
    ),
)


tmaps["torsade_de_pointes"] = TensorMap(
    "torsade_de_pointes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_torsade_de_pointes": 0, "torsade_de_pointes": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"torsade_de_pointes": {"torsade"}},
        not_found_channel="no_torsade_de_pointes",
    ),
)


tmaps["unspecified"] = TensorMap(
    "unspecified",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_unspecified": 0, "unspecified": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified": {
                "uncertain rhythm",
                "undetermined  rhythm",
                "rhythm unclear",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_ventricular_rhythm": 0, "ventricular_rhythm": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_rhythm": {"accelerated idioventricular rhythm"}},
        not_found_channel="no_ventricular_rhythm",
    ),
)


tmaps["wpw"] = TensorMap(
    "wpw",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_wpw": 0, "wpw": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "wpw": {"wolff-parkinson-white pattern", "wpw", "wolffparkinsonwhite"},
        },
        not_found_channel="no_wpw",
    ),
)


tmaps["brugada_pattern"] = TensorMap(
    "brugada_pattern",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_brugada_pattern": 0, "brugada_pattern": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"brugada_pattern": {"brugada pattern"}},
        not_found_channel="no_brugada_pattern",
    ),
)


tmaps["digitalis_effect"] = TensorMap(
    "digitalis_effect",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_digitalis_effect": 0, "digitalis_effect": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"digitalis_effect": {"digitalis effect"}},
        not_found_channel="no_digitalis_effect",
    ),
)


tmaps["early_repolarization"] = TensorMap(
    "early_repolarization",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_early_repolarization": 0, "early_repolarization": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"early_repolarization": {"early repolarization"}},
        not_found_channel="no_early_repolarization",
    ),
)


tmaps["inverted_u_waves"] = TensorMap(
    "inverted_u_waves",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_inverted_u_waves": 0, "inverted_u_waves": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"inverted_u_waves": {"inverted u waves"}},
        not_found_channel="no_inverted_u_waves",
    ),
)


tmaps["ischemia"] = TensorMap(
    "ischemia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ischemia": 0, "ischemia": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ischemia": {
                "st elevation",
                "inferior st segment elevation and q waves",
                "consistent with subendocardial ischemia",
                "anterolateral subendocardial ischemia",
                "anterolateral ischemia",
                "diffuse st segment depression",
                "septal ischemia",
                "subendocardial ischemia",
                "widespread st segment depression",
                "diffuse elevation of st segments",
                "suggesting anterior ischemia",
                "diffuse scooped st segment depression",
                "st segment elevation consistent with acute injury",
                "st depression",
                "antero-apical ischemia",
                "st segment depression in anterolateral leads",
                "apical st depression",
                "suggest anterior ischemia",
                "anterior subendocardial ischemia",
                "anterior st segment depression",
                "minor st segment depression",
                "st segment depression in leads",
                "diffuse st segment elevation",
                "st segment depressions more marked",
                "infero- st segment depression",
                "consistent with ischemia",
                "consistent with lateral ischemia",
                "st segment elevation in leads",
                "st segment elevation",
                "suggests anterolateral ischemia",
                "anterior infarct or transmural ischemia",
                "marked st segment depression in leads",
                "st segment depression in leads v4-v6",
                "st segment depression",
                "consider anterior ischemia",
                "nonspecific st segment depression",
                "inferior st segment depression",
                "anterolateral st segment depression",
                "st segment depression is more marked in leads",
                "consider anterior and lateral ischemia",
                "inferior subendocardial ischemia",
                "lateral ischemia",
                "apical subendocardial ischemia",
                "possible anterior wall ischemia",
                "marked st segment depression",
                "inferoapical st segment depression",
            },
        },
        not_found_channel="no_ischemia",
    ),
)


tmaps["metabolic_or_drug_effect"] = TensorMap(
    "metabolic_or_drug_effect",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_metabolic_or_drug_effect": 0, "metabolic_or_drug_effect": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"metabolic_or_drug_effect": {"metabolic or drug effect"}},
        not_found_channel="no_metabolic_or_drug_effect",
    ),
)


tmaps["osborn_wave"] = TensorMap(
    "osborn_wave",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_osborn_wave": 0, "osborn_wave": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"osborn_wave": {"osborn wave"}},
        not_found_channel="no_osborn_wave",
    ),
)


tmaps["pericarditis"] = TensorMap(
    "pericarditis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_pericarditis": 0, "pericarditis": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"pericarditis": {"pericarditis"}},
        not_found_channel="no_pericarditis",
    ),
)


tmaps["prominent_u_waves"] = TensorMap(
    "prominent_u_waves",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_prominent_u_waves": 0, "prominent_u_waves": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"prominent_u_waves": {"prominent u waves"}},
        not_found_channel="no_prominent_u_waves",
    ),
)


tmaps["st_abnormality"] = TensorMap(
    "st_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_st_abnormality": 0, "st_abnormality": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "st_abnormality": {
                "abnormal st segment changes",
                "st elevation",
                "inferior st segment elevation and q waves",
                "diffuse st segment depression",
                "widespread st segment depression",
                "diffuse elevation of st segments",
                "diffuse scooped st segment depression",
                "st segment elevation consistent with acute injury",
                "st segment changes",
                "nonspecific st segment",
                "st depression",
                "st segment depression in anterolateral leads",
                "apical st depression",
                "st segment abnormality",
                "anterior st segment depression",
                "minor st segment depression",
                "st segment depression in leads",
                "diffuse st segment elevation",
                "nonspecific st segment and t wave abnormalities",
                "st segment depressions more marked",
                "infero- st segment depression",
                "st segment elevation in leads",
                "st segment elevation",
                "marked st segment depression in leads",
                "st segment depression in leads v4-v6",
                "st segment depression",
                "nonspecific st segment depression",
                "anterolateral st segment depression",
                "st segment depression is more marked in leads",
                "marked st segment depression",
                "inferoapical st segment depression",
            },
        },
        not_found_channel="no_st_abnormality",
    ),
)


tmaps["st_or_t_change_due_to_ventricular_hypertrophy"] = TensorMap(
    "st_or_t_change_due_to_ventricular_hypertrophy",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_st_or_t_change_due_to_ventricular_hypertrophy": 0,
        "st_or_t_change_due_to_ventricular_hypertrophy": 1,
    },
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


tmaps["t_wave_abnormality"] = TensorMap(
    "t_wave_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_t_wave_abnormality": 0, "t_wave_abnormality": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "t_wave_abnormality": {
                "possible st segment and t wave abn",
                "t wave inversions",
                "upright t waves",
                "t waves are slightly more inverted in leads",
                "t wave inversion in leads",
                "t wave inver",
                "tall t waves in precordial leads",
                "t wave flattening",
                "t wave abnormalities",
                "nonspecific t wave abnormali",
                "t waves are lower or inverted in leads",
                "diffuse nonspecific st segment and t wave abnormalities",
                "t waves are upright in leads",
                "t wave changes",
                "nonspecific st segment and t wave abnormalities",
                "(nonspecific st segment).*(t wave abnormalities)",
                "t waves are inverted in leads",
                "t wave inveions",
                "t wave inversion",
                "recent diffuse t wave flattening",
            },
        },
        not_found_channel="no_t_wave_abnormality",
    ),
)


tmaps["tu_fusion"] = TensorMap(
    "tu_fusion",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_tu_fusion": 0, "tu_fusion": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"tu_fusion": {"tu fusion"}},
        not_found_channel="no_tu_fusion",
    ),
)


tmaps["fascicular_rhythm"] = TensorMap(
    "fascicular_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_fascicular_rhythm": 0, "fascicular_rhythm": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"fascicular_rhythm": {"fascicular rhythm"}},
        not_found_channel="no_fascicular_rhythm",
    ),
)


tmaps["fusion_complexes"] = TensorMap(
    "fusion_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_fusion_complexes": 0, "fusion_complexes": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"fusion_complexes": {"fusion complexes", "fusion beats"}},
        not_found_channel="no_fusion_complexes",
    ),
)


tmaps["idioventricular_rhythm"] = TensorMap(
    "idioventricular_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_idioventricular_rhythm": 0, "idioventricular_rhythm": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"idioventricular_rhythm": {"idioventricular rhythm"}},
        not_found_channel="no_idioventricular_rhythm",
    ),
)


tmaps["junctional_rhythm"] = TensorMap(
    "junctional_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_junctional_rhythm": 0, "junctional_rhythm": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"junctional_rhythm": {"junctional rhythm"}},
        not_found_channel="no_junctional_rhythm",
    ),
)


tmaps["parasystole"] = TensorMap(
    "parasystole",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_parasystole": 0, "parasystole": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"parasystole": {"parasystole"}},
        not_found_channel="no_parasystole",
    ),
)


tmaps["ventricular_fibrillation"] = TensorMap(
    "ventricular_fibrillation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ventricular_fibrillation": 0, "ventricular_fibrillation": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_fibrillation": {"ventricular fibrillation"}},
        not_found_channel="no_ventricular_fibrillation",
    ),
)


tmaps["ventricular_tachycardia"] = TensorMap(
    "ventricular_tachycardia",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ventricular_tachycardia": 0, "ventricular_tachycardia": 1},
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


tmaps["wide_qrs_rhythm"] = TensorMap(
    "wide_qrs_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_wide_qrs_rhythm": 0, "wide_qrs_rhythm": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"wide_qrs_rhythm": {"wide qrs rhythm"}},
        not_found_channel="no_wide_qrs_rhythm",
    ),
)


tmaps["first_degree_av_block"] = TensorMap(
    "first_degree_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_first_degree_av_block": 0, "first_degree_av_block": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "first_degree_av_block": {
                "first degree avb",
                "first degree atrioventricular block",
                "first degree atrioventricular  block",
                "first degree atrioventricular",
                "1st degree atrioventricular  block",
                "first degree atrioventricular block ",
                "first degree av block",
            },
        },
        not_found_channel="no_first_degree_av_block",
    ),
)


tmaps["aberrant_conduction_of_supraventricular_beats"] = TensorMap(
    "aberrant_conduction_of_supraventricular_beats",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_aberrant_conduction_of_supraventricular_beats": 0,
        "aberrant_conduction_of_supraventricular_beats": 1,
    },
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


tmaps["crista_pattern"] = TensorMap(
    "crista_pattern",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_crista_pattern": 0, "crista_pattern": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"crista_pattern": {"crista pattern"}},
        not_found_channel="no_crista_pattern",
    ),
)


tmaps["epsilon_wave"] = TensorMap(
    "epsilon_wave",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_epsilon_wave": 0, "epsilon_wave": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"epsilon_wave": {"epsilon wave"}},
        not_found_channel="no_epsilon_wave",
    ),
)


tmaps["incomplete_right_bundle_branch_block"] = TensorMap(
    "incomplete_right_bundle_branch_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_incomplete_right_bundle_branch_block": 0,
        "incomplete_right_bundle_branch_block": 1,
    },
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


tmaps["intraventricular_conduction_delay"] = TensorMap(
    "intraventricular_conduction_delay",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_intraventricular_conduction_delay": 0,
        "intraventricular_conduction_delay": 1,
    },
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


tmaps["left_anterior_fascicular_block"] = TensorMap(
    "left_anterior_fascicular_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_left_anterior_fascicular_block": 0,
        "left_anterior_fascicular_block": 1,
    },
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


tmaps["left_atrial_conduction_abnormality"] = TensorMap(
    "left_atrial_conduction_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_left_atrial_conduction_abnormality": 0,
        "left_atrial_conduction_abnormality": 1,
    },
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


tmaps["left_bundle_branch_block"] = TensorMap(
    "left_bundle_branch_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_left_bundle_branch_block": 0, "left_bundle_branch_block": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_bundle_branch_block": {
                "left bbb",
                "lbbb",
                "bundle branch block",
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
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_left_posterior_fascicular_block": 0,
        "left_posterior_fascicular_block": 1,
    },
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


tmaps["nonspecific_ivcd"] = TensorMap(
    "nonspecific_ivcd",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_nonspecific_ivcd": 0, "nonspecific_ivcd": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"nonspecific_ivcd": {"nonspecific ivcd"}},
        not_found_channel="no_nonspecific_ivcd",
    ),
)


tmaps["right_atrial_conduction_abnormality"] = TensorMap(
    "right_atrial_conduction_abnormality",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_right_atrial_conduction_abnormality": 0,
        "right_atrial_conduction_abnormality": 1,
    },
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


tmaps["right_bundle_branch_block"] = TensorMap(
    "right_bundle_branch_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_right_bundle_branch_block": 0, "right_bundle_branch_block": 1},
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


tmaps["ventricular_preexcitation"] = TensorMap(
    "ventricular_preexcitation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_ventricular_preexcitation": 0, "ventricular_preexcitation": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"ventricular_preexcitation": {"ventricular preexcitation"}},
        not_found_channel="no_ventricular_preexcitation",
    ),
)


tmaps["av_dissociation"] = TensorMap(
    "av_dissociation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_av_dissociation": 0, "av_dissociation": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "av_dissociation": {"atrioventricular dissociation", "av dissociation"},
        },
        not_found_channel="no_av_dissociation",
    ),
)


tmaps["_2_to_1_av_block"] = TensorMap(
    "_2_to_1_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no__2_to_1_av_block": 0, "_2_to_1_av_block": 1},
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


tmaps["_4_to_1_av_block"] = TensorMap(
    "_4_to_1_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no__4_to_1_av_block": 0, "_4_to_1_av_block": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"_4_to_1_av_block": {"4:1atrioventricular conduction"}},
        not_found_channel="no__4_to_1_av_block",
    ),
)


tmaps["av_dissociation"] = TensorMap(
    "av_dissociation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_av_dissociation": 0, "av_dissociation": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
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
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_mobitz_type_i_second_degree_av_block_": 0,
        "mobitz_type_i_second_degree_av_block_": 1,
    },
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_i_second_degree_av_block_": {
                "second degree ",
                "second degree type 1",
                "fixed block",
                "mobitz i",
                "mobitz type 1",
                "mobitz 1 block",
                "wenckebach",
            },
        },
        not_found_channel="no_mobitz_type_i_second_degree_av_block_",
    ),
)


tmaps["mobitz_type_ii_second_degree_av_block"] = TensorMap(
    "mobitz_type_ii_second_degree_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_mobitz_type_ii_second_degree_av_block": 0,
        "mobitz_type_ii_second_degree_av_block": 1,
    },
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_ii_second_degree_av_block": {
                "2nd degree sa block",
                "mobitz ii",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_third_degree_av_block": 0, "third_degree_av_block": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "third_degree_av_block": {
                "complete heart block",
                "3rd degree atrioventricular block",
                "third degree av block",
                "third degree atrioventricular block",
                "3rd degree av block",
            },
        },
        not_found_channel="no_third_degree_av_block",
    ),
)


tmaps["unspecified_av_block"] = TensorMap(
    "unspecified_av_block",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_unspecified_av_block": 0, "unspecified_av_block": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified_av_block": {
                "high grade atrioventricular block",
                "high degree of block",
                "av block",
                "heart block",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_variable_av_block": 0, "variable_av_block": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "variable_av_block": {"varying degree of block", "variable block"},
        },
        not_found_channel="no_variable_av_block",
    ),
)


tmaps["atrial_premature_complexes"] = TensorMap(
    "atrial_premature_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_atrial_premature_complexes": 0, "atrial_premature_complexes": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_premature_complexes": {
                "atrial premature complexes",
                "atrial trigeminy",
                "atrial premature beat",
                "premature atrial complexes",
                "atrial bigeminy",
                "ectopic atrial complexes",
                "isolated premature atrial contractions",
                "atrial ectopy has decreased",
                "premature atrial co",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_ectopy": 0, "ectopy": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopy": {
                "other than the ectopy",
                "ectopy is new",
                "ectopy has increased",
                "return of ectopy",
                "increased ectopy",
                "ectopy have increased",
                "ectopy has appeared",
                "new ectopy",
                "ectopy more pronounced",
            },
        },
        not_found_channel="no_ectopy",
    ),
)


tmaps["junctional_premature_complexes"] = TensorMap(
    "junctional_premature_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_junctional_premature_complexes": 0,
        "junctional_premature_complexes": 1,
    },
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


tmaps["no_ectopy"] = TensorMap(
    "no_ectopy",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_no_ectopy": 0, "no_ectopy": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "no_ectopy": {
                "atrial ectopy gone",
                "ectopy has disappear",
                "ectopy has resolved",
                "ectopy is no longer seen",
                "no longer any ectopy",
                "ectopy is gone",
                "no ectopy",
            },
        },
        not_found_channel="no_no_ectopy",
    ),
)


tmaps["premature_supraventricular_complexes"] = TensorMap(
    "premature_supraventricular_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_premature_supraventricular_complexes": 0,
        "premature_supraventricular_complexes": 1,
    },
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


tmaps["ventricular_premature_complexes"] = TensorMap(
    "ventricular_premature_complexes",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_ventricular_premature_complexes": 0,
        "ventricular_premature_complexes": 1,
    },
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ventricular_premature_complexes": {
                "premature ventricular and fusion complexes",
                "frequent premature ventricular or aberrantly conducted complexes ",
                "ventricular trigeminy",
                "ventricular premature complexes",
                "one premature ventricularbeat",
                "ventricular bigeminy",
                "premature ventricular beat",
                "ventricular premature beat",
                "ventricular ectopy",
                "premature ventricular compl",
                "isolated premature ventricular contractions",
                "occasional premature ventricular complexes ",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_mi": 0, "mi": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mi": {
                "concurrent ischemia myocardial infarction",
                "possible anteroseptal myocardial infarction",
                "counterclockwise rotation versus true posterior myocardial infarction",
                "normal counterclockwise rotation versus true posterior myocardial infarction",
                "lateral myocardial infarction of indeterminate age",
                "inferior myocardial infarction",
                "transmural ischemia myocardial infarction",
                "old apicolateral myocardial infarction",
                "(myocardial infarction versus).*(counterclockwise rotation)",
                "cannot rule out anteroseptal infarct",
                "inferior wall myocardial infarction of indeterminate age",
                "block inferior myocardial infarction",
                "myocardial infarction when compared with ecg of",
                "old inferior myocardial infarction",
                "myocardial infarction possible when compared",
                "possible old lateral myocardial infarction",
                "possible acute myocardial infarction",
                "anterolateral myocardial infarction appears recent",
                "lateral wall myocardial infarction",
                "myocardial infarction old high lateral",
                "myocardial infarction pattern",
                "consistent with ischemia myocardial infarction",
                "probable apicolateral myocardial infarction",
                "consistent with anterior myocardial infarction of indeterminate age",
                "acuteanterior myocardial infarction",
                "inferior myocardial infarction of indeterminate age",
                "anterior myocardial infarction",
                "infero-apical myocardial infarction",
                "old infero-posterior and apical myocardial infarction",
                "myocardial infarction compared with the last previous ",
                "rule out interim myocardial infarction",
                "inferoapical myocardial infarction of indeterminate age",
                "cannot rule out true posterior myocardial infarction",
                "evolving inferior wall myocardial infarction",
                "infero and apicolateral myocardial infarction",
                "apical myocardial infarction of indeterminate age",
                "extensive myocardial infarction of indeterminate age ",
                "myocardial infarction",
                "anterolateral myocardial infarction , possibly recent  ",
                "normal counterclockwise rotation versus old true posterior myocardial infarction",
                "old infero-posterior myocardial infarction",
                "anterolateral infarct of indeterminate age ",
                "evolving myocardial infarction",
                "anteroseptal myocardial infarction",
                "acute myocardial infarction of indeterminate age",
                "old infero-posterior lateral myocardial infarction",
                "age indeterminate old inferior wall myocardial infarction",
                "true posterior myocardial infarction of indeterminate age",
                "subendocardial infarct",
                "anteroseptal and lateral myocardial infarction",
                "anterior myocardial infarction of indeterminate age",
                "old inferior and anterior myocardial infarctions",
                "old inferoposterior myocardial infarction",
                "possible anteroseptal myocardial infarction of uncertain age",
                "acute anterior infarct",
                "known true posterior myocardial infarction",
                "old inferior wall myocardial infarction",
                "old inferolateral myocardial infarction",
                "myocardial infarction compared with the last previous tracing ",
                "subendocardial ischemia or myocardial infarction",
                "anteroseptal infarct of indeterminate age",
                "old anteroseptal myocardial infarction",
                "possible septal myocardial infarction",
                "old high lateral myocardial infarction",
                "old anterior myocardial infarction",
                "old anterolateral infarct",
                "apical myocardial infarction",
                "normal left axis deviation counterclockwise rotation versus old true posterior myocardial infarction",
                "(possible old).*(true posterior).*(myocardial infarction)",
                "old anteroseptal infarct",
                "counterclockwise rotation versus old true posterior myocardial infarction",
                "possible old septal myocardial infarction",
                "myocardial infarction nonspecific st segment",
                "myocardial infarction extension",
                "antero-apical and lateral myocardial infarction evolving",
                "extensive anterolateral myocardial infarction",
                "cannot rule out true posterior myocardial infarction versus counterclockwise rotation",
                "evolution of myocardial infarction",
                "inferior myocardial infarction , possibly acute",
                "old anterolateral myocardial infarction",
                "new incomplete posterior lateral myocardial infarction",
                "lateral myocardial infarction - of indeterminate age",
                "suggestive of old true posterior myocardial infarction",
                "possible myocardial infarction",
                "suggestive of old true posterior myocardial infarction st abnormality",
                "posterior myocardial infarction",
                "acute myocardial infarction in evolution",
                "consistent with anteroseptal infarct",
                "(counterclockwise rotation).*(true posterior)",
                "possible inferior myocardial infarction",
                "possible acute inferior myocardial infarction",
                "old myocardial infarction",
                "cannot rule out inferoposterior myoca",
                "old true posterior myocardial infarction",
                "acute anterior wall myocardial infarction",
                "post myocardial infarction , of indeterminate age",
                "raises possibility of septal infarct",
                "anterolateral myocardial infarction",
                "myocardial infarction of indeterminate age",
                "old inferior posterolateral myocardial infarction",
                "(consistent with).*(true posterior).*(myocardial infarction)",
                "old posterolateral myocardial infarction",
                "old anterior wall myocardial infarction",
                "inferior myocardial infarction of indeterminate",
                "old inferior anterior myocardial infarctions",
                "normal counterclockwise rotation versusold true posterior myocardial infarction",
                "myocardial infarction indeterminate",
                "evolving anterior infarct",
                "antero-apical ischemia versus myocardial infarction",
                "inferior myocardial infarction , age undetermined",
                "subendocardial myocardial infarction",
                "counterclockwise rotation consistent with post myocardial infarction",
                "myocardial infarction versus pericarditis",
                "recent myocardial infarction",
                "(true posterior).*(myocardial infarction)",
                "anterior infarct of indeterminate age",
                "possible anterolateral myocardial infarction",
                "posterior wall myocardial infarction",
                "myocardial infarction cannot rule out",
                "infero-apical myocardial infarction of indeterminate age",
                "ongoing ischemia versus myocardial infarction",
                "subendocardial ischemia myocardial infarction",
                "inferolateral myocardial infarction",
                "acute myocardial infarction",
                "septal infarct",
                "cannot rule out anterior infarct , age undetermined",
                "acute infarct",
                "old infero-postero-lateral myocardial infarction",
                "cannot rule out anterior wall ischemia myocardial infarction",
                "extensive anterior infarct",
                "subendocardial ischemia subendocardial myocardial inf",
                "old inferoapical myocardial infarction",
                "borderline anterolateral myocardial infarction",
                "old anterior infarct",
                "evolving counterclockwise rotation versus true posterior myocardial infarction",
                "old lateral myocardial infarction",
                "possible true posterior myocardial infarction",
            },
        },
        not_found_channel="no_mi",
    ),
)


tmaps["abnormal_p_wave_axis"] = TensorMap(
    "abnormal_p_wave_axis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_abnormal_p_wave_axis": 0, "abnormal_p_wave_axis": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"abnormal_p_wave_axis": {"abnormal p wave axis"}},
        not_found_channel="no_abnormal_p_wave_axis",
    ),
)


tmaps["electrical_alternans"] = TensorMap(
    "electrical_alternans",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_electrical_alternans": 0, "electrical_alternans": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"electrical_alternans": {"electrical alternans"}},
        not_found_channel="no_electrical_alternans",
    ),
)


tmaps["low_voltage"] = TensorMap(
    "low_voltage",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_low_voltage": 0, "low_voltage": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"low_voltage": {"low voltage"}},
        not_found_channel="no_low_voltage",
    ),
)


tmaps["poor_r_wave_progression"] = TensorMap(
    "poor_r_wave_progression",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_poor_r_wave_progression": 0, "poor_r_wave_progression": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "poor_r_wave_progression": {
                "unusual r wave progression",
                "abnormal precordial r wave progression",
                "abnormal precordial r wave progression or poor r wave progression",
                "slowprecordial r wave progression",
                "early r wave progression",
                "poor r wave progression",
                "slow precordial r wave progression",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_reversed_r_wave_progression": 0, "reversed_r_wave_progression": 1},
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


tmaps["lae"] = TensorMap(
    "lae",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_lae": 0, "lae": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lae": {
                "biatrial hypertrophy",
                "left atrial enla",
                "biatrial enlargement",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_lvh": 0, "lvh": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lvh": {
                "left ventricular hypertr",
                "biventricular hypertrophy",
                "biventriclar hypertrophy",
                "combined ventricular hypertrophy",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_rae": 0, "rae": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "rae": {
                "right atrial enla",
                "biatrial hypertrophy",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_rvh": 0, "rvh": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "rvh": {
                "rightventricular hypertrophy",
                "biventricular hypertrophy",
                "right ventricular enlargement",
                "right ventricular hypertrophy",
                "biventriclar hypertrophy",
                "combined ventricular hypertrophy",
            },
        },
        not_found_channel="no_rvh",
    ),
)


tmaps["sh"] = TensorMap(
    "sh",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_sh": 0, "sh": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sh": {"septal lipomatous hypertrophy", "septal hypertrophy"}},
        not_found_channel="no_sh",
    ),
)


tmaps["pacemaker"] = TensorMap(
    "pacemaker",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_pacemaker": 0, "pacemaker": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "pacemaker": {
                "failure to pace ventricular",
                "demand ventricular pacemaker",
                "atrial-sensed ventricular-paced complexes",
                "ventricular paced",
                "failure to inhibit ventricular",
                "unipolar right ventricular  pacing",
                "v-paced",
                "failure to pace atrial",
                "ventricular-paced rhythm",
                "sequential pacing",
                "atrially triggered v paced",
                "biventricular-paced complexes",
                "competitive av pacing",
                "ventricular pacing",
                "demand v-pacing",
                "biventricular-paced rhythm",
                "atrial-paced rhythm",
                "av dual-paced rhythm",
                "a triggered v-paced rhythm",
                "dual chamber pacing",
                "ventricular pacing has replaced av pacing",
                "av dual-paced complexes",
                "atrial triggered ventricular pacing",
                "v-paced rhythm",
                "v-paced beats",
                "atrial-sensed ventricular-paced rhythm",
                "failure to capture atrial",
                "shows dual chamber pacing",
                "electronic pacemaker",
                "failure to capture ventricular",
                "atrial-paced complexes ",
                "ventricular demand pacing",
                "ventricular-paced complexes",
                "failure to inhibit atrial",
            },
        },
        not_found_channel="no_pacemaker",
    ),
)


tmaps["abnormal_ecg"] = TensorMap(
    "abnormal_ecg",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_abnormal_ecg": 0, "abnormal_ecg": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"abnormal_ecg": {"abnormal"}},
        not_found_channel="no_abnormal_ecg",
    ),
)


tmaps["normal_sinus_rhythm"] = TensorMap(
    "normal_sinus_rhythm",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_normal_sinus_rhythm": 0, "normal_sinus_rhythm": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "normal_sinus_rhythm": {
                "normal sinus rhythm",
                "sinus tachycardia",
                "sinus rhythm",
                "tracing is within normal limits",
                "normal ecg",
                "normal tracing",
            },
        },
        not_found_channel="no_normal_sinus_rhythm",
    ),
)


tmaps["uninterpretable_ecg"] = TensorMap(
    "uninterpretable_ecg",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_uninterpretable_ecg": 0, "uninterpretable_ecg": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"uninterpretable_ecg": {"uninterpretable"}},
        not_found_channel="no_uninterpretable_ecg",
    ),
)


tmaps["indeterminate_axis"] = TensorMap(
    "indeterminate_axis",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_indeterminate_axis": 0, "indeterminate_axis": 1},
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


tmaps["left_axis_deviation"] = TensorMap(
    "left_axis_deviation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_left_axis_deviation": 0, "left_axis_deviation": 1},
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


tmaps["right_axis_deviation"] = TensorMap(
    "right_axis_deviation",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_right_axis_deviation": 0, "right_axis_deviation": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "right_axis_deviation": {
                "right superior axis deviation",
                "right axis deviation",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_right_superior_axis": 0, "right_superior_axis": 1},
    tensor_from_file=make_ecg_label_from_read_tff(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"right_superior_axis": {"right superior axis"}},
        not_found_channel="no_right_superior_axis",
    ),
)
