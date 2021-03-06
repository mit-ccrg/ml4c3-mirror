# Imports: standard library
import os
import re
import base64
import shutil
import struct
import logging
import argparse
import multiprocessing
from typing import Dict, List, Tuple, Union
from datetime import datetime
from collections import defaultdict

# Imports: third party
import bs4
import h5py
import numpy as np

# Imports: first party
from definitions.ecg import MUSE_ECG_XML_MRN_COLUMN
from definitions.globals import XML_EXT, TENSOR_EXT

excluded_keys = {
    "acquisitiondate",
    "acquisitiontime",
    "ageunits",
}


def tensorize(args: argparse.Namespace):
    """Convert data from GE Muse XMLs into HD5 files
    One HD5 is generated per patient. One HD5 may contain multiple ECGs.

    :param xml_folder: Path to folder containing ECG XML files organized in
                       subfolders by date
    :param tensors: Folder to populate with HD5 tensors

    :return: None
    """

    if not os.path.exists(args.bad_xml_dir):
        os.mkdir(args.bad_xml_dir)
        logging.info(f"Created {args.bad_xml_dir}")

    if not os.path.exists(args.bad_hd5_dir):
        os.mkdir(args.bad_hd5_dir)
        logging.info(f"Created {args.bad_hd5_dir}")

    logging.info("Mapping XMLs to MRNs")
    mrn_xmls_map = _get_mrn_xmls_map(
        xml_folder=args.xml_folder,
        num_workers=args.num_workers,
        mrn_key_in_xml=MUSE_ECG_XML_MRN_COLUMN,
    )

    logging.info("Converting XMLs into HD5s")
    _convert_mrn_xmls_to_hd5_wrapper(
        mrn_xmls_map=mrn_xmls_map,
        dir_hd5=args.tensors,
        num_workers=args.num_workers,
        bad_xml_dir=args.bad_xml_dir,
        bad_hd5_dir=args.bad_hd5_dir,
    )


def _get_mrn_from_xml(
    fpath_xml: str,
    mrn_key_in_xml: str,
) -> Union[Tuple[str, str], None]:
    with open(fpath_xml, "r") as f:
        for line in f:
            match = re.match(rf".*<{mrn_key_in_xml}>(.*)</{mrn_key_in_xml}>.*", line)
            if match:
                mrn = _clean_mrn(match.group(1), fallback="bad_mrn")
                return (mrn, fpath_xml)
    logging.warning(f"No PatientID found at {fpath_xml}")
    return None


def _get_mrn_xmls_map(
    xml_folder: str,
    num_workers: int,
    mrn_key_in_xml: str,
) -> Dict[str, List[str]]:
    fpath_xmls = []
    for root, dirs, files in os.walk(xml_folder):
        for file in files:
            if os.path.splitext(file)[-1].lower() != XML_EXT:
                continue
            fpath_xmls.append(os.path.join(root, file))
    logging.info(f"Found {len(fpath_xmls)} XMLs at {xml_folder}")

    mrn_xml_list = []

    if num_workers == 1:
        for fpath_xml in fpath_xmls:
            mrn_xml_list.append(
                _get_mrn_from_xml(fpath_xml=fpath_xml, mrn_key_in_xml=mrn_key_in_xml),
            )
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            mrn_xml_list = pool.starmap(
                _get_mrn_from_xml,
                [(fpath_xml, mrn_key_in_xml) for fpath_xml in fpath_xmls],
            )

    # Build dict of MRN to XML files with that MRN
    mrn_xml_dict = defaultdict(list)
    for mrn_xml in mrn_xml_list:
        if mrn_xml:
            mrn_xml_dict[mrn_xml[0]].append(mrn_xml[1])
    logging.info(f"Found {len(mrn_xml_dict)} distinct MRNs")

    return mrn_xml_dict


def _clean_mrn(mrn: str, fallback: str) -> str:
    # TODO additional cleaning like o->0, |->1
    # TODO https://github.com/aguirre-lab/ml4c3/issues/577
    try:
        clean = re.sub(r"[^0-9]", "", mrn)
        clean = int(clean)
        if not clean:
            raise ValueError()
        return str(clean)
    except ValueError:
        logging.warning(
            f'Could not clean MRN "{mrn}" to an int. Falling back to "{fallback}".',
        )
        return fallback


def _clean_read_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()

    # Replace newline character with space
    text = re.sub(r"\n", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Replace two+ spaces with one space
    text = re.sub(r"  +", " ", text)

    # Remove all leading and trailing whitespace
    text = text.strip()

    return text


def _data_from_xml(fpath_xml: str) -> Dict[str, Union[str, Dict[str, np.ndarray]]]:
    ecg_data = dict()

    # define tags that we want to find and use SoupStrainer to speed up search
    tags = [
        "patientdemographics",
        "testdemographics",
        "order",
        "restingecgmeasurements",
        "originalrestingecgmeasurements",
        "diagnosis",
        "originaldiagnosis",
        "intervalmeasurementtimeresolution",
        "intervalmeasurementamplituderesolution",
        "intervalmeasurementfilter",
        "waveform",
    ]
    strainer = bs4.SoupStrainer(tags)

    # lxml parser makes all tags lower case
    with open(fpath_xml, "r") as f:
        soup = bs4.BeautifulSoup(f, "lxml", parse_only=strainer)

    for tag in tags:
        tag_suffix = ""
        if tag == "restingecgmeasurements":
            tag_suffix = "_md"
        elif tag == "originalrestingecgmeasurements":
            tag_suffix = "_pc"
        elif tag == "diagnosis":
            soup_tag = soup.find(tag)
            if soup_tag is not None:
                ecg_data["diagnosis_md"] = _parse_soup_diagnosis(soup_tag)
            continue
        elif tag == "originaldiagnosis":
            soup_tag = soup.find(tag)
            if soup_tag is not None:
                ecg_data["diagnosis_pc"] = _parse_soup_diagnosis(soup_tag)
            continue
        elif tag == "waveform":
            voltage_data = _get_voltage_from_waveform_tags(soup.find_all(tag))
            ecg_data.update(voltage_data)
            continue

        soup_tag = soup.find(tag)

        if soup_tag is not None:
            # find sub tags
            soup_sub_tags = soup_tag.find_all()

            # if there are no sub tags, use original tag
            if len(soup_sub_tags) == 0:
                soup_sub_tags = [soup_tag]

            ecg_data.update({st.name + tag_suffix: st.text for st in soup_sub_tags})

    return ecg_data


def _parse_soup_diagnosis(input_from_soup: bs4.Tag) -> str:

    parsed_text = ""

    parts = input_from_soup.find_all("diagnosisstatement")

    # Check for edge case where <diagnosis> </diagnosis> does not encompass
    # <DiagnosisStatement> sub-element, which results in parts being length 0
    if len(parts) > 0:
        for part in parts:

            # Create list of all <stmtflag> entries
            flags = part.find_all("stmtflag")

            # Isolate text from part
            text_to_append = part.find("stmttext").text

            # Initialize flag to ignore sentence, e.g. do not append it
            flag_ignore_sentence = False

            # If no reasons found, append
            if not flag_ignore_sentence:
                # Append diagnosis string with contents within <stmttext> tags
                parsed_text += text_to_append

                endline_flag = False

                # Loop through flags and if 'ENDSLINE' found anywhere, mark flag
                for flag in flags:
                    if flag.text == "ENDSLINE":
                        endline_flag = True

                # If 'ENDSLINE' was found anywhere, append newline
                if endline_flag:
                    parsed_text += "\n"

                # Else append space
                else:
                    parsed_text += " "

        # Remove final newline character in diagnosis
        if parsed_text[-1] == "\n":
            parsed_text = parsed_text[:-1]

    return parsed_text


def _get_voltage_from_waveform_tags(
    waveform_tags: bs4.ResultSet,
) -> Dict[str, Union[str, Dict[str, np.ndarray]]]:
    voltage_data = dict()
    metadata_tags = [
        "samplebase",
        "sampleexponent",
        "highpassfilter",
        "lowpassfilter",
        "acfilter",
    ]
    for waveform_tag in waveform_tags:
        # only use full rhythm waveforms, do not use median waveforms
        if waveform_tag.find("waveformtype").text != "Rhythm":
            continue

        # get voltage metadata
        for metadata_tag in metadata_tags:
            mt = waveform_tag.find(metadata_tag)
            if mt is not None:
                voltage_data[f"waveform_{metadata_tag}"] = mt.text

        # get voltage leads and lead metadata
        lead_data = _get_voltage_from_lead_tags(waveform_tag.find_all("leaddata"))
        voltage_data.update(lead_data)
        break
    return voltage_data


def _get_voltage_from_lead_tags(
    lead_tags: bs4.ResultSet,
) -> Dict[str, Union[str, Dict[str, np.ndarray]]]:
    lead_data = dict()
    voltage = dict()
    all_lead_lengths = []
    all_lead_units = []

    def _decode_waveform(waveform_raw: str, scale: float) -> np.ndarray:
        decoded = base64.b64decode(waveform_raw)
        waveform = [
            struct.unpack("h", bytes([decoded[t], decoded[t + 1]]))[0]
            for t in range(0, len(decoded), 2)
        ]
        return np.array(waveform) * scale

    try:
        for lead_tag in lead_tags:
            # for each lead, we make sure all leads use 2 bytes per sample,
            # the decoded lead length is the same as the lead length tag,
            # the lead lengths are all the same, and the units are all the same
            lead_sample_size = int(lead_tag.find("leadsamplesize").text)
            assert lead_sample_size == 2

            lead_id = lead_tag.find("leadid").text
            lead_scale = lead_tag.find("leadamplitudeunitsperbit").text
            lead_waveform_raw = lead_tag.find("waveformdata").text
            lead_waveform = _decode_waveform(lead_waveform_raw, float(lead_scale))

            lead_length = lead_tag.find("leadsamplecounttotal").text
            lead_units = lead_tag.find("leadamplitudeunits").text

            assert int(lead_length) == len(lead_waveform)
            all_lead_lengths.append(lead_length)
            all_lead_units.append(lead_units)

            voltage[lead_id] = lead_waveform

        # vector math to get remaining leads
        assert len(voltage) == 8
        voltage["III"] = voltage["II"] - voltage["I"]
        voltage["aVR"] = -1 * (voltage["I"] + voltage["II"]) / 2
        voltage["aVL"] = voltage["I"] - voltage["II"] / 2
        voltage["aVF"] = voltage["II"] - voltage["I"] / 2

        # add voltage length and units to metadata
        assert len(set(all_lead_lengths)) == 1
        assert len(set(all_lead_units)) == 1
        lead_data["voltagelength"] = all_lead_lengths[0]
        lead_data["voltageunits"] = all_lead_units[0]

        lead_data["voltage"] = voltage
        return lead_data
    except (AssertionError, AttributeError, ValueError) as e:
        logging.exception(e)
        return dict()


def _compress_and_save_data(
    hd5: h5py.Group,
    name: str,
    data: Union[str, np.ndarray],
) -> h5py.Dataset:
    compression = None
    if isinstance(data, np.ndarray):
        compression = 32015
    return hd5.create_dataset(name=name, data=data, compression=compression)


def _get_max_voltage(voltage: Dict[str, np.ndarray]) -> float:
    max_voltage = 0
    for lead in voltage:
        if max(voltage[lead]) > max_voltage:
            max_voltage = max(voltage[lead])
    return max_voltage


def _convert_xml_to_hd5(fpath_xml: str, fpath_hd5: str, hd5: h5py.Group) -> int:
    # Return 1 if converted, 0 if ecg was bad or -1 if ecg was a duplicate
    # Set flag to check if we should convert to hd5
    convert = 1

    # Extract data from XML into dict
    ecg_data = _data_from_xml(fpath_xml)
    dt = datetime.strptime(
        f"{ecg_data['acquisitiondate']} {ecg_data['acquisitiontime']}",
        "%m-%d-%Y %H:%M:%S",
    )
    ecg_dt = dt.isoformat()

    # If XML is empty, remove the XML file and do not convert
    if os.stat(fpath_xml).st_size == 0 or not ecg_data:
        os.remove(fpath_xml)
        convert = 0
        logging.warning(f"Conversion of {fpath_xml} failed! XML is empty.")

    # If patient already has an ECG at given date and time, skip duplicate
    elif ecg_dt in hd5.keys():
        logging.warning(
            f"Conversion of {fpath_xml} skipped. Converted XML already exists in HD5.",
        )
        convert = -1

    # If we could not get voltage, do not convert (see _get_voltage_from_lead_tags)
    elif "voltage" not in ecg_data:
        logging.warning(
            f"Conversion of {fpath_xml} failed! Voltage is empty or badly formatted.",
        )
        convert = 0
    # If the max voltage value is 0, do not convert
    elif _get_max_voltage(ecg_data["voltage"]) == 0:
        logging.warning(f"Conversion of {fpath_xml} failed! Maximum voltage is 0.")
        convert = 0

    # If all prior checks passed, write hd5 group for ECG
    if convert == 1:
        gp = hd5.create_group(ecg_dt)

        # Save voltage leads
        voltage = ecg_data.pop("voltage")
        for lead in voltage:
            _compress_and_save_data(
                hd5=gp,
                name=lead,
                data=voltage[lead].astype("int16"),
            )

        # Save everything else
        for key in ecg_data:
            if key in excluded_keys:
                continue
            _compress_and_save_data(hd5=gp, name=key, data=ecg_data[key])

        # Clean Patient MRN to only numbers
        key_mrn_clean = "patientid_clean"
        if "patientid" in ecg_data:
            mrn_clean = _clean_mrn(ecg_data["patientid"], fallback="")
            _compress_and_save_data(
                hd5=gp,
                name=key_mrn_clean,
                data=mrn_clean,
            )

        # Clean cardiologist read
        key_read_md = "diagnosis_md"
        key_read_md_clean = "read_md_clean"
        if key_read_md in ecg_data:
            read_md_clean = _clean_read_text(text=ecg_data[key_read_md])
            _compress_and_save_data(
                hd5=gp,
                name=key_read_md_clean,
                data=read_md_clean,
            )

        # Clean MUSE read
        key_read_pc = "diagnosis_pc"
        key_read_pc_clean = "read_pc_clean"
        if key_read_pc in ecg_data:
            read_pc_clean = _clean_read_text(text=ecg_data[key_read_pc])
            _compress_and_save_data(
                hd5=gp,
                name=key_read_pc_clean,
                data=read_pc_clean,
            )

        logging.info(f"Wrote {fpath_xml} to {fpath_hd5}")
    return convert


def _move_bad_files(
    fpath_xmls: List[str],
    fpath_hd5: str,
    bad_xml_dir: str,
    bad_hd5_dir: str,
):
    # Move XMLs
    for fpath_xml in fpath_xmls:
        basename = os.path.basename(fpath_xml)
        newpath = os.path.join(bad_xml_dir, "xml", basename)
        shutil.move(src=fpath_xml, dst=newpath)
        logging.info(f"Moved bad XML to {newpath}")

    # Move HD5
    basename = os.path.basename(fpath_hd5)
    newpath = os.path.join(bad_hd5_dir, "hd5", basename)
    shutil.move(src=fpath_hd5, dst=newpath)
    logging.info(f"Moved bad HD5 to {newpath}")


def _convert_mrn_xmls_to_hd5(
    mrn: str,
    fpath_xmls: List[str],
    dir_hd5: str,
    hd5_prefix: str,
    bad_xml_dir: str,
    bad_hd5_dir: str,
) -> Tuple[int, int, int]:
    fpath_hd5 = os.path.join(dir_hd5, f"{mrn}{TENSOR_EXT}")
    num_xml_converted = 0
    num_dupe_skipped = 0
    num_src_in_hd5 = 0
    num_ecg_in_hd5 = 0
    hd5_failure = 0

    try:
        with h5py.File(fpath_hd5, "a") as hd5:
            hd5_ecg = (
                hd5[hd5_prefix]
                if hd5_prefix in hd5.keys()
                else hd5.create_group(hd5_prefix)
            )
            for fpath_xml in fpath_xmls:
                converted = _convert_xml_to_hd5(fpath_xml, fpath_hd5, hd5_ecg)
                if converted == 1:
                    num_xml_converted += 1
                elif converted == -1:
                    num_dupe_skipped += 1
            num_ecg_in_hd5 = len(hd5_ecg.keys())

            # If there are no ECGs in HD5, delete ECG group
            # There may be prior ECGs in HD5
            # num_xml_converted != num_ecg_in_hd5
            if not num_ecg_in_hd5:
                del hd5[hd5_prefix]

            num_src_in_hd5 = len(hd5.keys())

        if not num_src_in_hd5:
            # If there is no other data in HD5, delete HD5
            try:
                os.remove(fpath_hd5)
            except:
                logging.warning(f"Could not delete empty HD5 at {fpath_hd5}")
        num_hd5_written = 1 if num_xml_converted else 0
    except Exception as e:
        logging.warning(e)
        _move_bad_files(
            fpath_xmls=fpath_xmls,
            fpath_hd5=fpath_hd5,
            bad_xml_dir=bad_xml_dir,
            bad_hd5_dir=bad_hd5_dir,
        )
        num_hd5_written = 0
        hd5_failure = 1

    return (num_hd5_written, num_xml_converted, num_dupe_skipped, hd5_failure)


def _convert_mrn_xmls_to_hd5_wrapper(
    mrn_xmls_map: Dict[str, List[str]],
    dir_hd5: str,
    num_workers: int,
    bad_xml_dir: str,
    bad_hd5_dir: str,
    hd5_prefix: str = "ecg",
):
    tot_xml = sum([len(v) for k, v in mrn_xmls_map.items()])
    os.makedirs(dir_hd5, exist_ok=True)

    with multiprocessing.Pool(processes=num_workers) as pool:
        counts = pool.starmap(
            _convert_mrn_xmls_to_hd5,
            [
                (mrn, fpath_xmls, dir_hd5, hd5_prefix, bad_xml_dir, bad_hd5_dir)
                for mrn, fpath_xmls in mrn_xmls_map.items()
            ],
        )
    num_hd5 = sum([x[0] for x in counts])
    num_xml = sum([x[1] for x in counts])
    num_dup = sum([x[2] for x in counts])
    num_hd5_failures = sum([x[3] for x in counts])

    logging.info(f"Converted {num_xml} XMLs to {num_hd5} HD5s at {dir_hd5}")
    logging.info(f"Skipped {num_dup} duplicate XMLs")
    logging.info(f"Skipped {tot_xml - num_dup - num_xml} malformed XMLs")
    logging.info(f"Moved {num_hd5_failures} bad HD5 files to {bad_hd5_dir}")
