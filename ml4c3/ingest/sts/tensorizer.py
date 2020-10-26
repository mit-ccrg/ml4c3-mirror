# Imports: standard library
import os
import argparse

# Imports: third party
import h5py
import pandas as pd

# Imports: first party
from ml4c3.definitions.sts import STS_PREFIX, STS_MRN_COLUMN, STS_SURGERY_DATE_COLUMN
from ml4c3.definitions.globals import TENSOR_EXT


def tensorize(args: argparse.Namespace):
    df = pd.read_csv(args.sts_csv)
    os.makedirs(args.tensors, exist_ok=True)
    for idx, row in df.iterrows():
        mrn = int(row[STS_MRN_COLUMN])
        surgery_date = row[STS_SURGERY_DATE_COLUMN]
        hd5_path = os.path.join(args.tensors, f"{mrn}{TENSOR_EXT}")
        with h5py.File(hd5_path, "a") as hd5:
            for col in df.columns:
                if col in {STS_MRN_COLUMN, STS_SURGERY_DATE_COLUMN}:
                    continue
                hd5.create_dataset(f"{STS_PREFIX}/{surgery_date}/{col}", data=row[col])
