# Imports: standard library
import os
import random
import shutil
import string
from timeit import default_timer as timer

# Imports: third party
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Settings
root_dir = os.path.expanduser("~/benchmark")
csv_dir = os.path.join(root_dir, "csv")
hd5_dir = os.path.join(root_dir, "hd5")

# Define fake data to tensorize
rows = 100
num_mrns = 100
num_csns_per_mrn = 3
mrn_first = 2345
csn_first = 76543
data_categories = [
    "medications",
    "surgeries",
    "labs",
    "flowsheets",
    "demographics",
    "diagnoses",
]

# Set ratio of how many keys to read
# 1.0: read every key
# 0.5: read half the keys
# etc.
read_ratio = 0.9

num_random_string_fields = 100
num_string_fields_to_read = int(num_random_string_fields * read_ratio)
len_random_string_fields = 50
num_random_float_fields = 100
num_float_fields_to_read = int(num_random_float_fields * read_ratio)
max_float_val = 99
num_read_runs = 10

data = {}
letters = string.ascii_lowercase


def make_data() -> pd.DataFrame:
    dfs = []
    for row in range(rows):
        data = {}
        for i in range(num_random_string_fields):
            field_name = f"str_{i}"
            value = "".join(
                random.choice(letters) for i in range(len_random_string_fields)
            )
            data[field_name] = value

        for i in range(num_random_float_fields):
            field_name = f"float_{i}"
            value = random.uniform(0, max_float_val)
            data[field_name] = value

        df = pd.DataFrame(data, index=[row])
        dfs.append(df)
    del data
    data = pd.concat(dfs)
    return data


def write_data_to_hd5(path_hd5, csn, category, data):
    grp_name = f"{csn}/{category}"
    with h5py.File(path_hd5, "a") as hf:
        for col in data.columns:
            if "str" in col:
                tensor = data[col].to_numpy().astype("S")
            else:
                tensor = data[col].to_numpy()

            ds_name = f"{csn}/{category}/{col}"
            hf.create_dataset(ds_name, data=tensor, compression=32015)


if not os.path.isdir(root_dir):
    print(f"\nMaking data for {num_mrns} MRNs * {num_csns_per_mrn} CSNs")

    for mrn in tqdm(range(num_mrns)):
        for csn in range(num_csns_per_mrn):
            for category in data_categories:
                data = make_data()

                dir = os.path.join(csv_dir, str(mrn + mrn_first), str(csn + csn_first))
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                path_csv = os.path.join(dir, f"{category}.csv")
                data.to_csv(path_csv, index=False)

                if not os.path.isdir(hd5_dir):
                    os.makedirs(hd5_dir)
                path_hd5 = os.path.join(hd5_dir, f"{str(mrn + mrn_first)}.hd5")
                write_data_to_hd5(
                    path_hd5,
                    csn=csn + csn_first,
                    category=category,
                    data=data,
                )


def read_csv(csv_dir: str):
    for root, dirs, files in os.walk(csv_dir):
        for f in files:
            if f.endswith(".csv"):
                path = os.path.join(root, f)
                df = pd.read_csv(path)
                for i in range(num_random_string_fields):
                    key = f"str_{i}"
                    tensor = df[key].to_numpy()
                for i in range(num_random_float_fields):
                    key = f"float_{i}"
                    tensor = df[key].to_numpy()


# Iterate over CSVs
times = []
print(
    f"\nReading {num_string_fields_to_read} string fields and "
    f"{num_float_fields_to_read} float fields from CSVs {num_read_runs} times",
)
for i in tqdm(range(num_read_runs)):
    start_time = timer()
    read_csv(csv_dir)
    end_time = timer()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)

mean = np.mean(times)
std = np.std(times)
print(f"\nSubset of data from all CSV files took {mean:.2f} +/- {std:.2f} sec to read")


def read_hd5(hd5_dir):
    paths = []
    for root, dirs, files in os.walk(hd5_dir):
        for f in files:
            if f.endswith(".hd5"):
                path = os.path.join(root, f)
                paths.append(path)

    for path in paths:
        with h5py.File(path, "r") as hf:
            for csn in hf.keys():
                for category in hf[csn].keys():
                    for i in range(num_random_string_fields):
                        loc = f"{csn}/{category}/str_{i}"
                        tensor = hf[loc][:]
                    for i in range(num_random_float_fields):
                        loc = f"{csn}/{category}/float_{i}"
                        tensor = hf[loc][:]


# Iterate over HD5s
times = []
print(
    f"\nReading {num_string_fields_to_read} string fields and "
    f"{num_float_fields_to_read} float fields from HD5s {num_read_runs} times",
)
for i in tqdm(range(num_read_runs)):
    start_time = timer()
    read_hd5(hd5_dir)
    end_time = timer()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)

mean = np.mean(times)
std = np.std(times)
print(
    f"\nSubset of data from all HD5 files took {mean:.2f} +/- {std:.2f} sec to read\n",
)

# Clean up temp data
shutil.rmtree(csv_dir)
print(f"Deleted {csv_dir}")
shutil.rmtree(hd5_dir)
print(f"Deleted {hd5_dir}")
