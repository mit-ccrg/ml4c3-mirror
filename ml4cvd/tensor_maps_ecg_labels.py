# Imports: standard library
from typing import Dict

# Imports: first party
from ml4cvd.TensorMap import TensorMap, Interpretation
from ml4cvd.tensor_maps_ecg import make_ecg_label

tmaps: Dict[str, TensorMap] = {}
tmaps["asystole"] = TensorMap(
    "asystole",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_asystole": 0, "asystole": 1},
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_fibrillation": {
                "atrial fibrillation with rapid ventricular response",
                "fibrillation/flutter",
                "atrial fibrillation",
                "atrialfibrillation",
                "afibrillation",
                "atrial fibrillation with controlled ventricular response",
                "afib",
                "atrial fibrillation with moderate ventricular response",
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_flutter": {
                "aflutter",
                "tachycardia possibly flutter",
                "atrial flutter",
                "atrial flutter variable block",
                "fibrillation/flutter",
                "probable flutter",
                "flutter",
                "atrial flutter unspecified block",
                "atrial flutter fixed block",
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_rhythm": {
                "multiple atrial foci",
                "unusual p wave axis",
                "atrial rhythm",
                "atrial arrhythmia",
                "low atrial pacer",
                "p wave axis suggests atrial rather than sinus mechanism",
                "nonsinus atrial mechanism",
                "wandering atrial pacemaker",
                "wandering ectopic atrial rhythm",
                "unifocal ectopic atrial rhythm",
                "multifocal ear",
                "abnormal p vector",
                "multifocal atrial rhythm",
                "unifocal ear",
                "wandering ear",
                "ectopic atrial rhythm ",
                "dual atrial foci ",
                "multifocal atrialrhythm",
                "multifocal ectopic atrial rhythm",
                "ectopicsupraventricular rhythm",
                "ectopic atrial rhythm",
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopic_atrial_tachycardia": {
                "ectopic atrial tachycardia, unifocal",
                "multifocal ectopic atrial tachycardia",
                "ectopic atrial tachycardia, multifocal",
                "ectopic atrial tachycardia, unspecified",
                "ectopic atrial tachycardia",
                "unifocal atrial tachycardia",
                "multifocal atrial tachycardia",
                "wandering atrial tachycardia",
                "unspecified ectopic atrial tachycardia",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_narrow_qrs_tachycardia": 0, "narrow_qrs_tachycardia": 1},
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_retrograde_atrial_activation": 0,
        "retrograde_atrial_activation": 1,
    },
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "sinus_rhythm": {
                "normal when compared with ecg of",
                "type i sa block",
                "sa block",
                "atrial bigeminal rhythm",
                "atrial bigeminal  rhythm",
                "atrial trigeminy",
                "marked sinus arrhythmia",
                "sinus exit block",
                "tracing within normal limits",
                "sinus bradycardia",
                "type ii sa block",
                "sinoatrial block",
                "sa block, type i",
                "sinus rhythm at a rate",
                "sinus rhythm",
                "rhythm is normal sinus",
                "sinus mechanism has replaced",
                "sa exit block",
                "normal sinus rhythm",
                "rhythm is now clearly sinus",
                "atrialbigeminy",
                "2nd degree sa block",
                "rhythm remains normal sinus",
                "rhythm has reverted to normal",
                "sinus tachycardia",
                "sinus slowing",
                "with occasional native sinus beats",
                "1st degree sa block",
                "frequent native sinus beats",
                "conducted sinus impulses",
                "type ii sinoatrial block",
                "sinoatrial block, type ii",
                "type i sinoatrial block",
                "atrial bigeminy and ventricular bigeminy",
                "sinus arrhythmia",
                "tracing is within normal limits",
                "normal ecg",
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "supraventricular_tachycardia": {
                "avrt",
                "supraventricular tachycardia",
                "avnrt",
                "accelerated atrioventricular junctional rhythm",
                "av nodal reentrant",
                "accelerated atrioventricular nodal rhythm",
                "accelerated nodal rhythm",
                "av reentrant tachycardia ",
                "av nodal reentry tachycardia",
                "junctional tachycardia",
                "atrioventricular reentrant tachycardia ",
                "atrial tachycardia",
                "atrioventricular nodal reentry tachycardia",
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified": {
                "rhythm uncertain",
                "rhythm unclear",
                "uncertain rhythm",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_ventricular_rhythm": 0, "ventricular_rhythm": 1},
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "wpw": {"wolffparkinsonwhite", "wolff-parkinson-white pattern", "wpw"},
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ischemia": {
                "minor st segment depression",
                "st segment depression in leads",
                "anterolateral ischemia",
                "widespread st segment depression",
                "suggests anterolateral ischemia",
                "possible anterior wall ischemia",
                "marked st segment depression",
                "diffuse st segment depression",
                "lateral ischemia",
                "inferior st segment depression",
                "st segment depression in anterolateral leads",
                "st elevation",
                "inferior st segment elevation and q waves",
                "inferoapical st segment depression",
                "nonspecific st segment depression",
                "consistent with lateral ischemia",
                "st segment depression in leads v4-v6",
                "st segment depression is more marked in leads",
                "anterior st segment depression",
                "septal ischemia",
                "marked st segment depression in leads",
                "diffuse scooped st segment depression",
                "suggest anterior ischemia",
                "consistent with subendocardial ischemia",
                "anterior infarct or transmural ischemia",
                "suggesting anterior ischemia",
                "infero- st segment depression",
                "diffuse elevation of st segments",
                "st segment depression",
                "anterolateral subendocardial ischemia",
                "apical subendocardial ischemia",
                "consistent with ischemia",
                "st segment depressions more marked",
                "st segment elevation",
                "st depression",
                "consider anterior ischemia",
                "antero-apical ischemia",
                "st segment elevation consistent with acute injury",
                "anterior subendocardial ischemia",
                "anterolateral st segment depression",
                "inferior subendocardial ischemia",
                "subendocardial ischemia",
                "diffuse st segment elevation",
                "apical st depression",
                "st segment elevation in leads",
                "consider anterior and lateral ischemia",
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "st_abnormality": {
                "minor st segment depression",
                "st segment depression in leads",
                "nonspecific st segment",
                "widespread st segment depression",
                "marked st segment depression",
                "diffuse st segment depression",
                "st segment changes",
                "st segment depression in anterolateral leads",
                "st elevation",
                "inferior st segment elevation and q waves",
                "inferoapical st segment depression",
                "nonspecific st segment depression",
                "st segment depression in leads v4-v6",
                "st segment depression is more marked in leads",
                "anterior st segment depression",
                "abnormal st segment changes",
                "nonspecific st segment and t wave abnormalities",
                "marked st segment depression in leads",
                "diffuse scooped st segment depression",
                "infero- st segment depression",
                "diffuse elevation of st segments",
                "st segment depression",
                "st segment depressions more marked",
                "st segment elevation",
                "st depression",
                "st segment elevation consistent with acute injury",
                "st segment abnormality",
                "anterolateral st segment depression",
                "diffuse st segment elevation",
                "apical st depression",
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
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_st_or_t_change_due_to_ventricular_hypertrophy": 0,
        "st_or_t_change_due_to_ventricular_hypertrophy": 1,
    },
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "t_wave_abnormality": {
                "t wave inver",
                "t wave changes",
                "tall t waves in precordial leads",
                "t waves are slightly more inverted in leads",
                "t wave inveions",
                "recent diffuse t wave flattening",
                "diffuse nonspecific st segment and t wave abnormalities",
                "upright t waves",
                "t waves are lower or inverted in leads",
                "t wave abnormalities",
                "t wave inversions",
                "t wave flattening",
                "nonspecific t wave abnormali",
                "nonspecific st segment and t wave abnormalities",
                "(nonspecific st segment).*(t wave abnormalities)",
                "t wave inversion in leads",
                "t wave inversion",
                "possible st segment and t wave abn",
                "t waves are inverted in leads",
                "t waves are upright in leads",
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ventricular_tachycardia": {
                "\\w*(?<!supra)(ventricular tachycardia)",
                " ventricular tachy",
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "first_degree_av_block": {
                "1st degree atrioventricular  block",
                "first degree atrioventricular block",
                "first degree atrioventricular",
                "first degree avb",
                "first degree atrioventricular block ",
                "first degree av block",
                "first degree atrioventricular  block",
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
    tensor_from_file=make_ecg_label(
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_crista_pattern": 0, "crista_pattern": 1},
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_left_atrial_conduction_abnormality": 0,
        "left_atrial_conduction_abnormality": 1,
    },
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_bundle_branch_block": {
                "bundle branch block",
                "lbbb",
                "left bundle branch block",
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
    path_prefix="partners_ecg_rest",
    channel_map={
        "no_left_posterior_fascicular_block": 0,
        "left_posterior_fascicular_block": 1,
    },
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_ventricular_preexcitation": 0, "ventricular_preexcitation": 1},
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "_2_to_1_av_block": {
                "2 to 1 av block",
                "2:1 block",
                "2 to 1 atrioventricular block",
                "2:1 atrioventricular block",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no__4_to_1_av_block": 0, "_4_to_1_av_block": 1},
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_i_second_degree_av_block_": {
                "mobitz i",
                "second degree ",
                "mobitz type 1",
                "wenckebach",
                "second degree type 1",
                "mobitz 1 block",
                "fixed block",
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mobitz_type_ii_second_degree_av_block": {
                "mobitz ii",
                "2nd degree sa block",
                "second degree type 2",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_third_degree_av_block": 0, "third_degree_av_block": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "third_degree_av_block": {
                "third degree av block",
                "3rd degree atrioventricular block",
                "complete heart block",
                "3rd degree av block",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_unspecified_av_block": 0, "unspecified_av_block": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "unspecified_av_block": {
                "high degree of block",
                "heartblock",
                "heart block",
                "high grade atrioventricular block",
                "av block",
                "atrioventricular block",
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "atrial_premature_complexes": {
                "atrial ectopy",
                "atrial bigeminy",
                "atrial premature complexes",
                "premature atrial complexes",
                "atrial trigeminy",
                "isolated premature atrial contractions",
                "ectopic atrial complexes",
                "premature atrial co",
                "atrial premature beat",
                "atrial ectopy has decreased",
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ectopy": {
                "increased ectopy",
                "other than the ectopy",
                "ectopy has appeared",
                "ectopy have increased",
                "ectopy has increased",
                "ectopy is new",
                "new ectopy",
                "ectopy more pronounced",
                "return of ectopy",
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
    tensor_from_file=make_ecg_label(
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_no_ectopy": 0, "no_ectopy": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "no_ectopy": {
                "ectopy has resolved",
                "atrial ectopy gone",
                "ectopy is gone",
                "ectopy is no longer seen",
                "no longer any ectopy",
                "ectopy has disappear",
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "ventricular_premature_complexes": {
                "isolated premature ventricular contractions",
                "ventriculaar ectopy is now present",
                "ventricular ectopy",
                "ventricular bigeminy",
                "one premature ventricularbeat",
                "occasional premature ventricular complexes ",
                "premature ventricular compl",
                "ventricular premature beat",
                "frequent premature ventricular or aberrantly conducted complexes ",
                "ventricular premature complexes",
                "ventricular trigeminy",
                "premature ventricular and fusion complexes",
                "premature ventricular beat",
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "mi": {
                "possible anteroseptal myocardial infarction",
                "myocardial infarction extension",
                "old inferior wall myocardial infarction",
                "old anterolateral myocardial infarction",
                "myocardial infarction versus pericarditis",
                "counterclockwise rotation versus old true posterior myocardial infarction",
                "subendocardial ischemia subendocardial myocardial inf",
                "known true posterior myocardial infarction",
                "myocardial infarction pattern",
                "normal counterclockwise rotation versus true posterior myocardial infarction",
                "apical myocardial infarction",
                "anteroseptal and lateral myocardial infarction",
                "myocardial infarction",
                "anterolateral myocardial infarction",
                "counterclockwise rotation versus true posterior myocardial infarction",
                "lateral wall myocardial infarction",
                "old lateral myocardial infarction",
                "acute myocardial infarction",
                "old inferior posterolateral myocardial infarction",
                "anteroseptal myocardial infarction",
                "old infero-posterior myocardial infarction",
                "anterior infarct of indeterminate age",
                "subendocardial ischemia or myocardial infarction",
                "possible old septal myocardial infarction",
                "inferior myocardial infarction of indeterminate",
                "evolving anterior infarct",
                "possible old lateral myocardial infarction",
                "(consistent with).*(true posterior).*(myocardial infarction)",
                "myocardial infarction compared with the last previous ",
                "age indeterminate old inferior wall myocardial infarction",
                "(possible old).*(true posterior).*(myocardial infarction)",
                "myocardial infarction of indeterminate age",
                "myocardial infarction possible when compared",
                "possible septal myocardial infarction",
                "subendocardial myocardial infarction",
                "concurrent ischemia myocardial infarction",
                "infero and apicolateral myocardial infarction",
                "acute infarct",
                "old inferior myocardial infarction",
                "possible acute inferior myocardial infarction",
                "old anterior myocardial infarction",
                "myocardial infarction nonspecific st segment",
                "possible inferior myocardial infarction",
                "posterior wall myocardial infarction",
                "old inferoposterior myocardial infarction",
                "inferior myocardial infarction , possibly acute",
                "old high lateral myocardial infarction",
                "infero-apical myocardial infarction",
                "myocardial infarction when compared with ecg of",
                "(true posterior).*(myocardial infarction)",
                "lateral myocardial infarction of indeterminate age",
                "anterior myocardial infarction",
                "inferior myocardial infarction of indeterminate age",
                "cannot rule out anterior wall ischemia myocardial infarction",
                "(myocardial infarction versus).*(counterclockwise rotation)",
                "old infero-posterior lateral myocardial infarction",
                "old inferoapical myocardial infarction",
                "posterior myocardial infarction",
                "acute myocardial infarction of indeterminate age",
                "evolving inferior wall myocardial infarction",
                "myocardial infarction compared with the last previous tracing ",
                "(counterclockwise rotation).*(true posterior)",
                "post myocardial infarction , of indeterminate age",
                "consistent with ischemia myocardial infarction",
                "cannot rule out true posterior myocardial infarction",
                "myocardial infarction indeterminate",
                "subendocardial infarct",
                "old anterolateral infarct",
                "extensive anterior infarct",
                "normal counterclockwise rotation versusold true posterior myocardial infarction",
                "rule out interim myocardial infarction",
                "old apicolateral myocardial infarction",
                "old anteroseptal myocardial infarction",
                "true posterior myocardial infarction of indeterminate age",
                "transmural ischemia myocardial infarction",
                "evolution of myocardial infarction",
                "recent myocardial infarction",
                "ongoing ischemia versus myocardial infarction",
                "suggestive of old true posterior myocardial infarction",
                "cannot rule out inferoposterior myoca",
                "anterior myocardial infarction of indeterminate age",
                "counterclockwise rotation consistent with post myocardial infarction",
                "lateral myocardial infarction - of indeterminate age",
                "cannot rule out true posterior myocardial infarction versus counterclockwise rotation",
                "inferoapical myocardial infarction of indeterminate age",
                "acute anterior wall myocardial infarction",
                "infero-apical myocardial infarction of indeterminate age",
                "anterolateral myocardial infarction , possibly recent  ",
                "suggestive of old true posterior myocardial infarction st abnormality",
                "possible anterolateral myocardial infarction",
                "old anteroseptal infarct",
                "old inferior and anterior myocardial infarctions",
                "antero-apical ischemia versus myocardial infarction",
                "acuteanterior myocardial infarction",
                "old anterior wall myocardial infarction",
                "inferior myocardial infarction",
                "acute myocardial infarction in evolution",
                "extensive anterolateral myocardial infarction",
                "inferolateral myocardial infarction",
                "evolving myocardial infarction",
                "septal infarct",
                "myocardial infarction old high lateral",
                "old posterolateral myocardial infarction",
                "subendocardial ischemia myocardial infarction",
                "cannot rule out anteroseptal infarct",
                "anterolateral myocardial infarction appears recent",
                "possible acute myocardial infarction",
                "consistent with anterior myocardial infarction of indeterminate age",
                "old myocardial infarction",
                "anteroseptal infarct of indeterminate age",
                "old infero-posterior and apical myocardial infarction",
                "extensive myocardial infarction of indeterminate age ",
                "old inferolateral myocardial infarction",
                "possible myocardial infarction",
                "myocardial infarction cannot rule out",
                "old true posterior myocardial infarction",
                "consistent with anteroseptal infarct",
                "evolving counterclockwise rotation versus true posterior myocardial infarction",
                "possible anteroseptal myocardial infarction of uncertain age",
                "old infero-postero-lateral myocardial infarction",
                "block inferior myocardial infarction",
                "possible true posterior myocardial infarction",
                "antero-apical and lateral myocardial infarction evolving",
                "normal counterclockwise rotation versus old true posterior myocardial infarction",
                "probable apicolateral myocardial infarction",
                "anterolateral infarct of indeterminate age ",
                "old anterior infarct",
                "raises possibility of septal infarct",
                "inferior myocardial infarction , age undetermined",
                "borderline anterolateral myocardial infarction",
                "old inferior anterior myocardial infarctions",
                "cannot rule out anterior infarct , age undetermined",
                "acute anterior infarct",
                "new incomplete posterior lateral myocardial infarction",
                "apical myocardial infarction of indeterminate age",
                "inferior wall myocardial infarction of indeterminate age",
                "normal left axis deviation counterclockwise rotation versus old true posterior myocardial infarction",
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "poor_r_wave_progression": {
                "abnormal precordial r wave progression",
                "early r wave progression",
                "unusual r wave progression",
                "slow precordial r wave progression",
                "slowprecordial r wave progression",
                "poor r wave progression",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_reversed_r_wave_progression": 0, "reversed_r_wave_progression": 1},
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lae": {
                "left atrial enla",
                "biatrial hypertrophy",
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "lvh": {
                "combined ventricular hypertrophy",
                "biventriclar hypertrophy",
                "leftventricular hypertrophy",
                "biventricular hypertrophy",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_rae": 0, "rae": 1},
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "rvh": {
                "rightventricular hypertrophy",
                "combined ventricular hypertrophy",
                "biventriclar hypertrophy",
                "biventricular hypertrophy",
                "right ventricular enlargement",
                "right ventricular hypertrophy",
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"sh": {"septal hypertrophy", "septal lipomatous hypertrophy"}},
        not_found_channel="no_sh",
    ),
)


tmaps["pacemaker"] = TensorMap(
    "pacemaker",
    interpretation=Interpretation.CATEGORICAL,
    time_series_limit=0,
    path_prefix="partners_ecg_rest",
    channel_map={"no_pacemaker": 0, "pacemaker": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "pacemaker": {
                "atrial triggered ventricular pacing",
                "ventricular paced",
                "v-paced rhythm",
                "atrial-paced complexes ",
                "failure to inhibit ventricular",
                "failure to pace ventricular",
                "dual chamber pacing",
                "biventricular-paced rhythm",
                "v-paced beats",
                "ventricular pacing has replaced av pacing",
                "electronic pacemaker",
                "demand v-pacing",
                "a triggered v-paced rhythm",
                "av dual-paced complexes",
                "shows dual chamber pacing",
                "competitive av pacing",
                "unipolar right ventricular  pacing",
                "failure to inhibit atrial",
                "failure to pace atrial",
                "biventricular-paced complexes",
                "atrial-sensed ventricular-paced rhythm",
                "sequential pacing",
                "atrially triggered v paced",
                "atrial-paced rhythm",
                "ventricular-paced rhythm",
                "atrial-sensed ventricular-paced complexes",
                "failure to capture ventricular",
                "av dual-paced rhythm",
                "ventricular-paced complexes",
                "v-paced",
                "failure to capture atrial",
                "demand ventricular pacemaker",
                "ventricular pacing",
                "ventricular demand pacing",
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
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "normal_sinus_rhythm": {
                "sinus rhythm",
                "normal sinus rhythm",
                "normal tracing",
                "sinus tachycardia",
                "tracing is within normal limits",
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_uninterpretable_ecg": 0, "uninterpretable_ecg": 1},
    tensor_from_file=make_ecg_label(
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
    tensor_from_file=make_ecg_label(
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
    path_prefix="partners_ecg_rest",
    channel_map={"no_left_axis_deviation": 0, "left_axis_deviation": 1},
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "left_axis_deviation": {
                "left axis deviation",
                "axis shifted left",
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={
            "right_axis_deviation": {
                "right axis deviation",
                "rightward axis",
                "right superior axis deviation",
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
    tensor_from_file=make_ecg_label(
        keys=["read_md_clean", "read_pc_clean"],
        channel_terms={"right_superior_axis": {"right superior axis"}},
        not_found_channel="no_right_superior_axis",
    ),
)
