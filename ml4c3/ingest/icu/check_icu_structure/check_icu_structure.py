from .check_bm_structure import BMChecker
from .check_edw_structure import EDWChecker


def check_icu_structure(args):
    if args.check_edw:
        edw_checker = EDWChecker(args.path_edw)
        edw_checker.check_structure()
    if args.check_bm:
        bm_checker = BMChecker(args.path_bedmaster, args.path_alarms)
        bm_checker.check_mat_files_structure()
        bm_checker.check_alarms_files_structure()
