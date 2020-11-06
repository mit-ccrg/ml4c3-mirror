# Imports: standard library
import os
import argparse

# Imports: third party
import h5py


def remove_unfinished_hd5(folder, no_bedmaster):
    hd5_files_path = [
        os.path.join(folder, hd5_file)
        for hd5_file in os.listdir(folder)
        if hd5_file.endswith(".hd5")
    ]
    print(f"Number of files found: {len(hd5_files_path)}")
    for hd5_file_path in hd5_files_path:
        finished = False
        try:
            with h5py.File(hd5_file_path, "r") as hd5_file:
                finished = _check_if_finished(hd5_file, no_bedmaster)
            if not finished:
                os.remove(hd5_file_path)
        except OSError:
            os.remove(hd5_file_path)

    current_files_path = [
        os.path.join(folder, hd5_file)
        for hd5_file in os.listdir(folder)
        if hd5_file.endswith(".hd5")
    ]
    print(f"Number of files removed: {len(hd5_files_path)-len(current_files_path)}")


def _check_if_finished(hd5_file, no_bedmaster):
    try:
        if no_bedmaster:
            for bedmaster_visit in hd5_file["bedmaster"].keys():
                if len(hd5_file[f"bedmaster/{bedmaster_visit}"].keys()) == 0:
                    return False
        return (
            hd5_file["bedmaster"].attrs["completed"]
            and hd5_file["edw"].attrs["completed"]
        )
    except KeyError:
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(description="Remove unfinished hd5 files.")
    parser.add_argument(
        "--tensors",
        type=str,
        default="/media/mad3/hd5_arrests",
        help="Directory where the completness check of hd5 files is performed.",
    )
    parser.add_argument(
        "--no_bedmaster",
        action="store_true",
        help="If parameter is set, files with no Bedmaster data will be deleted too..",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    remove_unfinished_hd5(args.tensors, args.no_bedmaster)
