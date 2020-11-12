# Imports: standard library
import os
import argparse

# Imports: third party
import pandas as pd

# Imports: first party
from ml4c3.definitions.icu import ALARMS_FILES

COLUMNS = [
    "UnitBedUID",
    "AlarmStartTime",
    "AlarmLevel",
    "AlarmMessage",
    "AlarmDuration",
    "PatientID",
]

BEDS_COLUMNS = [
    "UnitBedUID",
    "Bed",
    "Unit",
]


class PreProcessBedmasterAlarms:
    """
    Reduce size bedmaster alarms files and save them in .csv files by departments.
    """

    def __init__(self, input_dir: str, output_dir: str = "./"):
        """
        Init Pre-Process Bedmaster Alarms.

        :param input_dir: <str> Directory where alarms files (.csv) are saved.
        :param output_dir: <str> Full path where the resulting data frames are saved.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

    def pre_process_alarms(self):
        """
        Extract columns of .csv files in self.input_directory and save all in
        files (.csv) by departments in self.output_directory.
        """
        alarms_files = [
            alarm_file
            for alarm_file in os.listdir(self.input_dir)
            if "AlarmTable" in alarm_file and alarm_file.endswith(".csv")
        ]
        beds_files = [
            beds_file
            for beds_file in os.listdir(self.input_dir)
            if "UnitBedTable" in beds_file and beds_file.endswith(".csv")
        ]
        k = 1
        print(f"Extracting alarms from file {k}/{len(alarms_files)}")
        data = self._read_files(alarms_files[0], beds_files)
        for alarm_file in alarms_files[1:]:
            k += 1
            print(f"Extracting alarms from file {k}/{len(alarms_files)}")
            new_data = self._read_files(alarm_file, beds_files)
            data = pd.concat([data, new_data], ignore_index=True)
        data.to_csv(os.path.join(self.output_dir, "bedmaster_alarms.csv"))

        print("Saving data by department...")
        for department in data["Unit"].unique():
            dept_data = data[data["Unit"] == department]
            output_file = os.path.join(
                self.output_dir,
                f"bedmaster_alarms_{department}.csv",
            )
            dept_data.to_csv(output_file)

    def _read_files(self, alarms_file, beds_files):
        """
        Join alarms_file with beds_file.
        """
        alarm_columns = ALARMS_FILES["columns"][:-2]
        alarms_df = pd.read_csv(os.path.join(self.input_dir, alarms_file))
        alarms_df = alarms_df[alarm_columns].set_index("UnitBedUID")
        key = alarms_file.split("_")[0] + "_"
        beds_file = [file_name for file_name in beds_files if key in file_name]
        if len(beds_file) > 1:
            print(
                "Something went wrong. It has been found more than one file "
                "mapping the Bedmaster beds IDs with ADT beds. It will be "
                "used just the first.",
            )
            beds_file = [beds_file[0]]
        elif len(beds_file) == 0:
            print(
                "Something went wrong. It has not been found any file "
                "mapping the Bedmaster beds IDs with ADT beds. The mapping "
                "won't be performed.",
            )
            return alarms_df
        bed_columns = ALARMS_FILES["columns"][:1] + ALARMS_FILES["columns"][-2:]
        beds_df = pd.read_csv(os.path.join(self.input_dir, beds_file[0]))
        beds_df = beds_df[bed_columns].set_index("UnitBedUID")
        return alarms_df.join(beds_df)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pre process Bedmaster alarms.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/media/lm4-alarms/2020-09-03",
        help="Directory where the Bedmaster alarms in raw format are saved. "
        "Remember to introduce the most recent date.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/ml4c3/bedmaster_alarms",
        help="Directory where the results will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    pre_processer = PreProcessBedmasterAlarms(args.input_dir, args.output_dir)
    pre_processer.pre_process_alarms()
