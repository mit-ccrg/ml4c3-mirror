#! /usr/bin/env python

# Imports: standard library
import re
import sys
import argparse
from typing import Optional, Sequence


def get_mrns_and_csns(argv: Optional[Sequence[str]] = None):
    """
    Detects all numbers of 4 or more digits on the committed files.
    Numbers preceded with 4 or more consecutive digits are ignored
    Numbers on the `--ignore_numbers` param are ignored too
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore_numbers", nargs="+", help="Ignore certain numbers.")
    parser.add_argument("filenames", nargs="*", help="Filenames to check.")
    args = parser.parse_args(argv)

    retval = 0
    for filename in args.filenames:
        with open(filename, "r") as f:
            for line_num, line in enumerate(f):
                if "sdc: nsd-stop" in line:
                    break
                if "sdc: nsd" in line:
                    continue
                sensitive_data = re.finditer(
                    r"\b(\d)(?!\1{3,})(\d{3,})",
                    line,
                    re.MULTILINE,
                )
                for data in sensitive_data:
                    if data.group() not in args.ignore_numbers:
                        print(
                            f"{filename} - line {line_num+1}: "
                            f"Possible sensitive data: {data.group()}",
                        )
                        retval = 1
    if retval:
        print(
            "If you have checked that this is not sensitive data, "
            "you can skip a line using '# sdc: nsd', the rest of a document "
            "using '#sdc: nsd-stop' or skip this hook by running "
            "'SKIP=sdc git commit -m <msg>' ",
        )
    return sys.exit(retval)


if __name__ == "__main__":
    get_mrns_and_csns()
