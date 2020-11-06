from .check_edw_structure import EDWChecker
from .check_bedmaster_structure import BedmasterChecker


def check_icu_structure(args):
    if args.check_edw:
        edw_checker = EDWChecker(args.path_edw)
        edw_checker.check_structure()
    if args.check_bedmaster:
        bedmaster_checker = BedmasterChecker(args.path_bedmaster, args.path_alarms)
        bedmaster_checker.check_mat_files_structure()
        bedmaster_checker.check_alarms_files_structure()
