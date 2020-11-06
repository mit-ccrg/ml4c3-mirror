from .match_patient_bedmaster import PatientBedmasterMatcher


def match_data(args):
    bedmaster_matcher = PatientBedmasterMatcher(
        args.lm4,
        args.path_bedmaster,
        args.path_edw,
        args.desired_depts,
    )
    bedmaster_matcher.match_files(args.path_xref)
