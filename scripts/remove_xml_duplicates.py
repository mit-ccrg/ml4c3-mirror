# Imports: standard library
import os
import hashlib
import logging
import argparse
import datetime
from timeit import default_timer as timer


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src", default="", help="Path to directory containing organized source XMLs",
    )

    args = parser.parse_args()

    # This is temporary becuase I can't figure out how to import load_config from ml4c3.logger
    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{now_str}_remove_xml_duplicates_log.txt"),
            logging.StreamHandler(),
        ],
    )

    return args


def _sha256sum_of_xml(fname: str) -> str:
    with open(fname, "rb", buffering=0) as f:
        h = hashlib.sha256()
        b = bytearray(128 * 1024)
        mv = memoryview(b)
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def run(args):

    start = timer()
    fpath_xml = args.src

    # Identify list of date directories within xml/
    fpath_dirs = [f.path for f in os.scandir(fpath_xml) if f.is_dir()]
    fpath_dirs.sort()
    xml_fpaths_hashes = []

    # Iterate through all XML files in directory, hash contents, append to list of tuples
    logging.info(
        f"Iterating through alL XML files in directory, hashing contents, and appending original fpath and hash to list of tuples.",
    )
    for dirpath, subdirs, fnames in os.walk(fpath_xml):
        for fname in fnames:
            if fname.endswith(".xml"):
                fpath_xml = os.path.join(dirpath, fname)
                xml_hash = _sha256sum_of_xml(fpath_xml)
                xml_fpaths_hashes.append((fpath_xml, xml_hash))
    end = timer()
    logging.info(f"Hashing {len(xml_fpaths_hashes)} XML files took {end-start:.2f} sec")

    # Sort list of tuples by the hash
    start = timer()
    xml_fpaths_hashes = sorted(xml_fpaths_hashes, key=lambda x: x[1])
    end = timer()
    logging.info(
        f"Sorting list of (fpath_xml, xml_hash) by hash took {(end-start):.2f} sec",
    )

    # Find all duplicate hashes

    # Initialize first hash
    prev_xml = xml_fpaths_hashes[0][0]
    prev_hash = xml_fpaths_hashes[0][1]

    dup_count = 0

    start = timer()

    # Loop through all hashes, starting at the second entry
    for xml_and_hash in xml_fpaths_hashes[1:]:

        # If the hash matches the previous, it is a duplicate
        if xml_and_hash[1] == prev_hash:

            # Increment counter
            dup_count += 1

            # Delete duplicate XMLs
            os.remove(xml_and_hash[0])
            logging.info(f"{xml_and_hash[0]} is a duplicate")

        # If not, update previous hash
        else:
            prev_xml = xml_and_hash[0]
            prev_hash = xml_and_hash[1]

    end = timer()
    logging.info(
        f"Removing {dup_count} duplicates / {len(xml_fpaths_hashes)} ECGs"
        f" ({dup_count / len(xml_fpaths_hashes) * 100:.2f}%) took"
        f" {end-start:.2f} sec",
    )


if __name__ == "__main__":
    args = _parse_args()
    run(args)
