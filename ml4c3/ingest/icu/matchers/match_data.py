from .match_patient_bm import PatientBMMatcher


def match_data(args):
    bm_matcher = PatientBMMatcher(
        args.lm4, args.path_bedmaster, args.path_edw, args.desired_depts,
    )
    bm_matcher.match_files(args.path_xref)
