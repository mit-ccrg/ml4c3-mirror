# Imports: standard library
import re
import logging
import datetime
from typing import Set, Dict, List

# Imports: third party
import numpy as np

# Imports: first party
from definitions.ecg import (
    ECG_PREFIX,
    ECG_DATE_FORMAT,
    ECG_REST_LEADS_ALL,
    ECG_REST_LEADS_INDEPENDENT,
)
from ml4c3.normalizer import ZScore, RobustScale, ZScorePopulation
from ml4c3.validators import (
    RangeValidator,
    validator_no_empty,
    validator_clean_mrn,
    validator_no_negative,
    validator_not_all_zero,
    validator_voltage_no_zero_padding,
)
from definitions.globals import YEAR_DAYS
from tensormap.TensorMap import (
    TensorMap,
    Interpretation,
    make_hd5_path,
    is_dynamic_shape,
)

tmaps: Dict[str, TensorMap] = {}


def _resample_voltage(voltage: np.array, desired_samples: int, fs: float) -> np.array:
    """Resample array of voltage amplitudes (voltage) given desired number of samples
    (length) and reported sampling frequency (fs) in Hz. Note the reported fs is solely
    for processing 240 Hz ECGs that have 2500 samples."""
    if fs == 240:
        # If voltage array has more than 10 sec of data, truncate to the first 10 sec
        if len(voltage) == 2500:
            voltage = voltage[:2400]
        else:
            raise ValueError(
                f"Sampling frequency of 240 Hz reported, but voltage array is not 2500 elements.",
            )

    if len(voltage) == desired_samples:
        return voltage

    x = np.arange(len(voltage))
    x_interp = np.linspace(0, len(voltage), desired_samples)
    return np.interp(x_interp, x, voltage)


def make_voltage_tff(exact_length=False, no_pacemaker=False, conv_2d=False):
    if no_pacemaker:
        from tensormap.ecg_labels import tmaps as label_tmaps  # isort: skip

        pacemaker_tm = label_tmaps["pacemaker"]

    def tensor_from_file(tm, data):
        ecg_dates = tm.time_series_filter(data)
        if no_pacemaker:
            pacemaker_tm.time_series_filter = tm.time_series_filter
            pacemaker_tensor = pacemaker_tm.tensor_from_file(pacemaker_tm, data)
        dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
        if conv_2d:
            shape = shape[:-1]
        voltage_length = shape[1] if dynamic else shape[0]
        tensor = np.zeros(shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            if no_pacemaker and pacemaker_tensor[i, 1] == 1:
                continue
            for cm in tm.channel_map:
                try:
                    path = make_hd5_path(tm=tm, date_key=ecg_date, value_key=cm)
                    voltage = data[path][()]
                    path_waveform_samplebase = make_hd5_path(
                        tm=tm,
                        date_key=ecg_date,
                        value_key="waveform_samplebase",
                    )
                    try:
                        fs = float(data[path_waveform_samplebase][()])
                    except:
                        fs = 250
                    if exact_length:
                        assert len(voltage) == voltage_length
                    voltage = _resample_voltage(
                        voltage=voltage,
                        desired_samples=voltage_length,
                        fs=fs,
                    )
                    slices = (
                        (i, ..., tm.channel_map[cm])
                        if dynamic
                        else (..., tm.channel_map[cm])
                    )
                    tensor[slices] = voltage
                except (KeyError, AssertionError, ValueError):
                    logging.debug(
                        f"Could not get voltage for lead {cm} with {voltage_length}"
                        f" samples in {data.id}",
                    )
        if conv_2d:
            tensor = tensor[..., None]
        return tensor

    return tensor_from_file


# ECG augmenters
def _crop_ecg(ecg: np.array):
    cropped_ecg = ecg.copy()
    for j in range(ecg.shape[1]):
        crop_len = np.random.randint(len(ecg)) // 3
        crop_start = max(0, np.random.randint(-crop_len, len(ecg)))
        cropped_ecg[:, j][crop_start : crop_start + crop_len] = np.random.randn()
    return cropped_ecg


def _noise_ecg(ecg: np.array):
    noise_frac = np.random.rand() * 0.1  # max of 10% noise
    return ecg + noise_frac * ecg.mean(axis=0) * np.random.randn(
        ecg.shape[0],
        ecg.shape[1],
    )


def _warp_ecg(ecg: np.array):
    warp_strength = 0.02
    i = np.linspace(0, 1, len(ecg))
    envelope = warp_strength * (0.5 - np.abs(0.5 - i))
    warped = i + envelope * (
        np.sin(np.random.rand() * 5 + np.random.randn() * 5)
        + np.cos(np.random.rand() * 5 + np.random.randn() * 5)
    )
    warped_ecg = np.zeros_like(ecg)
    for j in range(ecg.shape[1]):
        warped_ecg[:, j] = np.interp(i, warped, ecg[:, j])
    return warped_ecg


name2augmenters = {
    "crop": _crop_ecg,
    "noise": _noise_ecg,
    "warp": _warp_ecg,
}


def update_tmaps_ecg_voltage(
    tmap_name: str,
    tmaps: Dict[str, TensorMap],
) -> Dict[str, TensorMap]:
    """
    Generates ECG voltage TMaps that are given by the name format:
        [8_lead_]ecg_{length}[_exact][_scaling][_no_pacemaker][_2d][_augmentations]

    Required:
        length: The number of samples present in each lead.

    Optional:
        8_lead:        Use the 8 independent leads.
        exact:         Only return voltages when raw data has exactly {length} samples
                       in each lead. If not given, voltages are resampled to {length}.
        scaling:       Scaling can be one of zscore, robustscale, zscore_pop, or
                       robustscale_pop, where zscore and robustscale scale each voltage
                       independently and zscore_pop and robustscale_pop scale each
                       voltage using population level distributions.
        augmentations: Apply crop, noise, and warp transformations to voltages.
        no_pacemaker:  Excludes waveforms with a pacemaker
        2d:            Return tensor shaped for 2D convolutions

    Examples:
        valid: ecg_2500_zscore
        valid: 8_lead_ecg_625_crop_warp
        valid: ecg_2500_robustscale_pop_no_pacemaker_2d
        invalid: ecg_2500_noise_exact
    """
    voltage_tm_pattern = re.compile(
        r"^(8_lead_)?ecg_(\d+)(_exact)?"
        r"(_zscore_pop|_robustscale_pop|_zscore|_robustscale)?"
        r"(_no_pacemaker)?(_2d)?(_warp|_crop|_noise)*",
    )
    match = voltage_tm_pattern.match(tmap_name)
    if match is None:
        return tmaps

    # Isolate matching components of tmap name and build it

    # fmt: off
    # 8_lead_ecg_2500_exact_zscore_no_pacemaker_2d
    new_tmap_name = match[0]       # 8_lead_ecg_2500_exact_zscore_no_pacemaker_2d
    use_8_lead = bool(match[1])    # 8_lead
    length = int(match[2])         # 2500
    exact = bool(match[3])         # exact
    scaling = match[4]             # zscore
    no_pacemaker = bool(match[5])  # no_pacemaker
    conv_2d = bool(match[6])       # 2d
    # fmt: on

    leads = ECG_REST_LEADS_INDEPENDENT if use_8_lead else ECG_REST_LEADS_ALL
    if scaling == "_zscore":
        normalizer = ZScore()
    elif scaling == "_robustscale":
        normalizer = RobustScale()
    elif scaling == "_zscore_pop":
        normalizer = ZScorePopulation(mean=0, std=2000)
    elif scaling == "_robustscale_pop":
        raise ValueError(
            f"Cannot create {new_tmap_name}, population median and IQR undefined for "
            f"ECG voltage.",
        )
    else:
        normalizer = None
    augmenters = [
        augment_function
        for augment_option, augment_function in name2augmenters.items()
        if augment_option in tmap_name
    ]
    shape = (length, len(leads))
    if conv_2d:
        shape += (1,)
    tmap = TensorMap(
        name=new_tmap_name,
        shape=shape,
        path_prefix=ECG_PREFIX,
        tensor_from_file=make_voltage_tff(
            exact_length=exact,
            no_pacemaker=no_pacemaker,
            conv_2d=conv_2d,
        ),
        normalizers=normalizer,
        channel_map=leads,
        time_series_limit=0,
        validators=validator_voltage_no_zero_padding,
        augmenters=augmenters,
    )
    tmaps[new_tmap_name] = tmap
    return tmaps


def voltage_stat(tm, data):
    ecg_dates = tm.time_series_filter(data)
    dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=np.float32)
    for i, ecg_date in enumerate(ecg_dates):
        try:
            slices = (
                lambda stat: (i, tm.channel_map[stat])
                if dynamic
                else (tm.channel_map[stat],)
            )
            path = lambda lead: make_hd5_path(tm, ecg_date, lead)
            voltages = np.array([data[path(lead)][()] for lead in ECG_REST_LEADS_ALL])
            tensor[slices("mean")] = np.mean(voltages)
            tensor[slices("std")] = np.std(voltages)
            tensor[slices("min")] = np.min(voltages)
            tensor[slices("max")] = np.max(voltages)
            tensor[slices("median")] = np.median(voltages)
        except KeyError:
            logging.warning(f"Could not get voltage stats for ECG at {data.id}")
    return tensor


tmaps["ecg_voltage_stats"] = TensorMap(
    "ecg_voltage_stats",
    shape=(5,),
    path_prefix=ECG_PREFIX,
    tensor_from_file=voltage_stat,
    channel_map={"mean": 0, "std": 1, "min": 2, "max": 3, "median": 4},
    time_series_limit=0,
)


def make_voltage_attribute_tff(volt_attr: str = ""):
    def tensor_from_file(tm, data):
        ecg_dates = tm.time_series_filter(data)
        dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            for cm in tm.channel_map:
                try:
                    path = make_hd5_path(tm, ecg_date, cm)
                    slices = (
                        (i, tm.channel_map[cm]) if dynamic else (tm.channel_map[cm],)
                    )
                    tensor[slices] = data[path].attrs[volt_attr]
                except KeyError:
                    pass
        return tensor

    return tensor_from_file


tmaps["voltage_len"] = TensorMap(
    "voltage_len",
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_voltage_attribute_tff(volt_attr="len"),
    shape=(12,),
    channel_map=ECG_REST_LEADS_ALL,
    time_series_limit=0,
)


def make_binary_ecg_label_from_any_read_tff(
    keys: List[str],
    channel_terms: Dict[str, Set[str]],
    not_found_channel: str,
):
    def tensor_from_file(tm, data):
        # get all the ecgs in range (time series lookup is set)
        tm.time_series_limit = 0
        ecg_dates = tm.time_series_filter(data)
        tm.time_series_limit = None

        tensor = np.zeros(tm.shape, dtype=np.float32)

        read = ""
        for ecg_idx, ecg_date in enumerate(ecg_dates):
            for key in keys:
                path = make_hd5_path(tm, ecg_date, key)
                if path not in data:
                    continue
                read += data[path][()]
            read = read.lower()

        if read != "":
            found = False
            for channel, channel_idx in sorted(
                tm.channel_map.items(),
                key=lambda cm: cm[1],
            ):
                if channel not in channel_terms:
                    continue
                if any(
                    re.search(term.lower(), read) is not None
                    for term in channel_terms[channel]
                ):
                    tensor[channel_idx] = 1
                    found = True
                    break
            if not found:
                not_found_idx = tm.channel_map[not_found_channel]
                tensor[not_found_idx] = 1
        return tensor

    return tensor_from_file


def make_ecg_label_from_read_tff(
    keys: List[str],
    channel_terms: Dict[str, Set[str]],
    not_found_channel: str,
):
    def tensor_from_file(tm, data):
        ecg_dates = tm.time_series_filter(data)
        dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=np.float32)
        for ecg_idx, ecg_date in enumerate(ecg_dates):
            read = ""
            for key in keys:
                path = make_hd5_path(tm, ecg_date, key)
                if path not in data:
                    continue
                read += data[path][()]
            read = read.lower()

            found = False
            for channel, channel_idx in sorted(
                tm.channel_map.items(),
                key=lambda cm: cm[1],
            ):
                if channel not in channel_terms:
                    continue
                if any(
                    re.search(term.lower(), read) is not None
                    for term in channel_terms[channel]
                ):
                    slices = (ecg_idx, channel_idx) if dynamic else (channel_idx,)
                    tensor[slices] = 1
                    found = True
                    break
            if not found:
                not_found_idx = tm.channel_map[not_found_channel]
                slices = (ecg_idx, not_found_idx) if dynamic else (not_found_idx,)
                tensor[slices] = 1
        return tensor

    return tensor_from_file


def get_ecg_datetime(tm, data):
    ecg_dates = tm.time_series_filter(data)
    dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.full(shape, "", dtype=f"<U19")
    for i, ecg_date in enumerate(ecg_dates):
        tensor[i] = ecg_date
    return tensor


tmap_name = "ecg_datetime"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=get_ecg_datetime,
    shape=(1,),
    time_series_limit=0,
    validators=validator_no_empty,
)


def make_voltage_len_tff(
    lead,
    channel_prefix="_",
    channel_unknown="other",
):
    def tensor_from_file(tm, data):
        ecg_dates = tm.time_series_filter(data)
        dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=float)
        for i, ecg_date in enumerate(ecg_dates):
            path = make_hd5_path(tm, ecg_date, lead)
            try:
                lead_len = data[path].attrs["len"]
                lead_len = f"{channel_prefix}{lead_len}"
                matched = False
                for cm in tm.channel_map:
                    if lead_len.lower() == cm.lower():
                        slices = (
                            (i, tm.channel_map[cm])
                            if dynamic
                            else (tm.channel_map[cm],)
                        )
                        tensor[slices] = 1.0
                        matched = True
                        break
                if not matched:
                    slices = (
                        (i, tm.channel_map[channel_unknown])
                        if dynamic
                        else (tm.channel_map[channel_unknown],)
                    )
                    tensor[slices] = 1.0
            except KeyError:
                logging.debug(
                    f"Could not get voltage length for lead {lead} from ECG on"
                    f" {ecg_date} in {data.id}",
                )
        return tensor

    return tensor_from_file


for lead in ECG_REST_LEADS_ALL:
    tmap_name = f"lead_{lead}_len"
    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        interpretation=Interpretation.CATEGORICAL,
        path_prefix=ECG_PREFIX,
        tensor_from_file=make_voltage_len_tff(lead=lead),
        channel_map={"_2500": 0, "_5000": 1, "other": 2},
        time_series_limit=0,
        validators=validator_not_all_zero,
    )


def make_ecg_tensor(
    key: str,
    fill: float = 0,
    channel_prefix: str = "",
    channel_unknown: str = "other",
):
    def get_ecg_tensor(tm, data):
        ecg_dates = tm.time_series_filter(data)
        dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
        if tm.interpretation == Interpretation.LANGUAGE:
            tensor = np.full(shape, "", dtype=object)
        elif tm.interpretation == Interpretation.CONTINUOUS:
            tensor = (
                np.zeros(shape, dtype=float)
                if fill == 0
                else np.full(shape, fill, dtype=float)
            )
        elif tm.interpretation == Interpretation.CATEGORICAL:
            tensor = np.zeros(shape, dtype=float)
        else:
            raise NotImplementedError(
                f"unsupported interpretation for ecg tmaps: {tm.interpretation}",
            )

        for i, ecg_date in enumerate(ecg_dates):
            path = make_hd5_path(tm, ecg_date, key)
            try:
                value = data[path][()]
                if tm.interpretation == Interpretation.CATEGORICAL:
                    matched = False
                    value = f"{channel_prefix}{value}"
                    for cm in tm.channel_map:
                        if value.lower() == cm.lower():
                            slices = (
                                (i, tm.channel_map[cm])
                                if dynamic
                                else (tm.channel_map[cm],)
                            )
                            tensor[slices] = 1.0
                            matched = True
                            break
                    if not matched:
                        slices = (
                            (i, tm.channel_map[channel_unknown])
                            if dynamic
                            else (tm.channel_map[channel_unknown],)
                        )
                        tensor[slices] = 1.0
                else:
                    tensor[i] = value
            except (KeyError, ValueError):
                logging.debug(
                    f"Could not obtain tensor {tm.name} from ECG on {ecg_date} in"
                    f" {data.id}",
                )
        if tm.interpretation == Interpretation.LANGUAGE:
            tensor = tensor.astype(str)
        return tensor

    return get_ecg_tensor


tmap_name = "ecg_read_md"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="read_md_clean"),
    shape=(1,),
    time_series_limit=0,
    validators=validator_no_empty,
)


tmap_name = "ecg_read_pc"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="read_pc_clean"),
    shape=(1,),
    time_series_limit=0,
    validators=validator_no_empty,
)


tmap_name = "ecg_patientid"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientid"),
    shape=(1,),
    time_series_limit=0,
    validators=validator_no_empty,
)


tmap_name = "ecg_patientid_clean"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientid_clean"),
    shape=(1,),
    time_series_limit=0,
    validators=validator_clean_mrn,
)


tmap_name = "ecg_firstname"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientfirstname"),
    shape=(1,),
    time_series_limit=0,
    validators=validator_no_empty,
)


tmap_name = "ecg_lastname"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientlastname"),
    shape=(1,),
    time_series_limit=0,
    validators=validator_no_empty,
)


tmap_name = "ecg_sex"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="gender"),
    channel_map={"female": 0, "male": 1},
    time_series_limit=0,
    validators=validator_not_all_zero,
)

tmap_name = "ecg_date"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="acquisitiondate"),
    shape=(1,),
    time_series_limit=0,
    validators=validator_no_empty,
)


tmap_name = "ecg_time"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="acquisitiontime"),
    shape=(1,),
    time_series_limit=0,
    validators=validator_no_empty,
)


tmap_name = "ecg_sitename"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="sitename"),
    shape=(1,),
    time_series_limit=0,
    validators=validator_no_empty,
)


tmap_name = "ecg_location"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="location"),
    shape=(1,),
    time_series_limit=0,
    validators=validator_no_empty,
)


tmap_name = "ecg_dob"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="dateofbirth"),
    shape=(1,),
    time_series_limit=0,
    validators=validator_no_empty,
)


def make_sampling_frequency_from_file(
    lead: str = "I",
    duration: int = 10,
    channel_prefix: str = "_",
    channel_unknown: str = "other",
    fill: int = -1,
):
    def sampling_frequency_from_file(tm, data):
        ecg_dates = tm.time_series_filter(data)
        dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
        if tm.interpretation == Interpretation.CATEGORICAL:
            tensor = np.zeros(shape, dtype=np.float32)
        else:
            tensor = np.full(shape, fill, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            path = make_hd5_path(tm, ecg_date, lead)
            lead_length = data[path].attrs["len"]
            sampling_frequency = lead_length / duration
            try:
                if tm.interpretation == Interpretation.CATEGORICAL:
                    matched = False
                    sampling_frequency = f"{channel_prefix}{sampling_frequency}"
                    for cm in tm.channel_map:
                        if sampling_frequency.lower() == cm.lower():
                            slices = (
                                (i, tm.channel_map[cm])
                                if dynamic
                                else (tm.channel_map[cm],)
                            )
                            tensor[slices] = 1.0
                            matched = True
                            break
                    if not matched:
                        slices = (
                            (i, tm.channel_map[channel_unknown])
                            if dynamic
                            else (tm.channel_map[channel_unknown],)
                        )
                        tensor[slices] = 1.0
                else:
                    tensor[i] = sampling_frequency
            except (KeyError, ValueError):
                logging.debug(
                    f"Could not calculate sampling frequency from ECG on {ecg_date} in"
                    f" {data.id}",
                )
        return tensor

    return sampling_frequency_from_file


# sampling frequency without any suffix calculates the sampling frequency directly from the voltage array
# other metadata that are reported by the muse system are unreliable
tmap_name = "ecg_sampling_frequency"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_sampling_frequency_from_file(),
    channel_map={"_250": 0, "_500": 1, "other": 2},
    time_series_limit=0,
    validators=validator_not_all_zero,
)


tmap_name = "ecg_sampling_frequency_pc"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_pc", channel_prefix="_"),
    channel_map={"_0": 0, "_250": 1, "_500": 2, "other": 3},
    time_series_limit=0,
    validators=validator_not_all_zero,
)


tmap_name = "ecg_sampling_frequency_md"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_md", channel_prefix="_"),
    channel_map={"_0": 0, "_250": 1, "_500": 2, "other": 3},
    time_series_limit=0,
    validators=validator_not_all_zero,
)


tmap_name = "ecg_sampling_frequency_lead"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_samplebase", channel_prefix="_"),
    channel_map={"_0": 0, "_240": 1, "_250": 2, "_500": 3, "other": 4},
    time_series_limit=0,
    validators=validator_not_all_zero,
)


tmap_name = "ecg_sampling_frequency_continuous"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_sampling_frequency_from_file(),
    time_series_limit=0,
    shape=(1,),
    validators=validator_no_negative,
)


tmap_name = "ecg_sampling_frequency_pc_continuous"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_pc", fill=-1),
    time_series_limit=0,
    shape=(1,),
    validators=validator_no_negative,
)


tmap_name = "ecg_sampling_frequency_md_continuous"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_md", fill=-1),
    time_series_limit=0,
    shape=(1,),
    validators=validator_no_negative,
)


tmap_name = "ecg_sampling_frequency_lead_continuous"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_samplebase", fill=-1),
    time_series_limit=0,
    shape=(1,),
    validators=validator_no_negative,
)


tmap_name = "ecg_time_resolution"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(
        key="intervalmeasurementtimeresolution",
        channel_prefix="_",
    ),
    channel_map={"_25": 0, "_50": 1, "_100": 2, "other": 3},
    time_series_limit=0,
    validators=validator_not_all_zero,
)


tmap_name = "ecg_amplitude_resolution"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(
        key="intervalmeasurementamplituderesolution",
        channel_prefix="_",
    ),
    channel_map={"_10": 0, "_20": 1, "_40": 2, "other": 3},
    time_series_limit=0,
    validators=validator_not_all_zero,
)


tmap_name = "ecg_measurement_filter"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(
        key="intervalmeasurementfilter",
        channel_prefix="_",
    ),
    time_series_limit=0,
    channel_map={"_None": 0, "_40": 1, "_80": 2, "other": 3},
    validators=validator_not_all_zero,
)


tmap_name = "ecg_high_pass_filter"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_highpassfilter", fill=-1),
    time_series_limit=0,
    shape=(1,),
    validators=validator_no_negative,
)


tmap_name = "ecg_low_pass_filter"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_lowpassfilter", fill=-1),
    time_series_limit=0,
    shape=(1,),
    validators=validator_no_negative,
)


tmap_name = "ecg_ac_filter"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_acfilter", channel_prefix="_"),
    time_series_limit=0,
    channel_map={"_None": 0, "_50": 1, "_60": 2, "other": 3},
    validators=validator_not_all_zero,
)

# Creates TMaps for interval measurements.
# Creates TMaps with _md and _pc suffix.
# Examples:
#     ecg_rate_md
#     ecg_rate_std_md
#     ecg_rate_pc
#     ecg_rate_std_pc

# fmt: off
# TMap name      ->      (hd5 key,          fill, validator,                 normalizer)
interval_key_map = {
    "ecg_rate":          ("ventricularrate", 0,   RangeValidator(10, 200),   None),
    "ecg_rate_std":      ("ventricularrate", 0,   RangeValidator(10, 200),   ZScorePopulation(mean=70, std=16)),
    "ecg_pr":            ("printerval",      0,   RangeValidator(50, 500),   None),
    "ecg_pr_std":        ("printerval",      0,   RangeValidator(50, 500),   ZScorePopulation(mean=175, std=36)),
    "ecg_qrs":           ("qrsduration",     0,   RangeValidator(20, 400),   None),
    "ecg_qrs_std":       ("qrsduration",     0,   RangeValidator(20, 400),   ZScorePopulation(mean=104, std=26)),
    "ecg_qt":            ("qtinterval",      0,   RangeValidator(100, 800),  None),
    "ecg_qt_std":        ("qtinterval",      0,   RangeValidator(100, 800),  ZScorePopulation(mean=411, std=45)),
    "ecg_qtc":           ("qtcorrected",     0,   RangeValidator(100, 800),  None),
    "ecg_qtc_std":       ("qtcorrected",     0,   RangeValidator(100, 800),  ZScorePopulation(mean=440, std=39)),
    "ecg_paxis":         ("paxis",           999, RangeValidator(-90, 360),  None),
    "ecg_paxis_std":     ("paxis",           999, RangeValidator(-90, 360),  ZScorePopulation(mean=47, std=30)),
    "ecg_raxis":         ("raxis",           999, RangeValidator(-90, 360),  None),
    "ecg_raxis_std":     ("raxis",           999, RangeValidator(-90, 360),  ZScorePopulation(mean=18, std=53)),
    "ecg_taxis":         ("taxis",           999, RangeValidator(-90, 360),  None),
    "ecg_taxis_std":     ("taxis",           999, RangeValidator(-90, 360),  ZScorePopulation(mean=58, std=63)),
    "ecg_qrs_count":     ("qrscount",        -1,  RangeValidator(0, 100),    None),
    "ecg_qrs_count_std": ("qrscount",        -1,  RangeValidator(0, 100),    ZScorePopulation(mean=12, std=3)),
}
# fmt: on

for interval, (key, fill, validator, normalizer) in interval_key_map.items():
    for suffix in ["_md", "_pc"]:
        name = f"{interval}{suffix}"
        _key = f"{key}{suffix}"
        tmaps[name] = TensorMap(
            name,
            interpretation=Interpretation.CONTINUOUS,
            path_prefix=ECG_PREFIX,
            loss="logcosh",
            tensor_from_file=make_ecg_tensor(key=_key, fill=fill),
            shape=(1,),
            time_series_limit=0,
            validators=validator,
            normalizers=normalizer,
        )


def get_ecg_age_from_hd5(tm, data):
    ecg_dates = tm.time_series_filter(data)
    dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.full(shape, fill_value=-1, dtype=float)
    for i, ecg_date in enumerate(ecg_dates):
        if i >= shape[0]:
            break
        path = lambda key: make_hd5_path(tm, ecg_date, key)
        try:
            birthday = data[path("dateofbirth")][()]
            acquisition = data[path("acquisitiondate")][()]
            delta = _ecg_str2date(acquisition) - _ecg_str2date(birthday)
            years = delta.days / YEAR_DAYS
            tensor[i] = years
        except KeyError:
            try:
                tensor[i] = data[path("patientage")][()]
            except KeyError:
                logging.debug(
                    f"Could not get patient date of birth or age from ECG on {ecg_date}"
                    f" in {data.id}",
                )
    return tensor


tmap_name = "ecg_age"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    path_prefix=ECG_PREFIX,
    loss="logcosh",
    tensor_from_file=get_ecg_age_from_hd5,
    shape=(1,),
    time_series_limit=0,
    validators=RangeValidator(0, 120),
)

tmap_name = "ecg_age_std"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    path_prefix=ECG_PREFIX,
    loss="logcosh",
    tensor_from_file=get_ecg_age_from_hd5,
    shape=(1,),
    time_series_limit=0,
    validators=RangeValidator(0, 120),
    normalizers=ZScorePopulation(mean=65, std=16),
)


def ecg_acquisition_year(tm, data):
    ecg_dates = tm.time_series_filter(data)
    dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=int)
    for i, ecg_date in enumerate(ecg_dates):
        path = make_hd5_path(tm, ecg_date, "acquisitiondate")
        try:
            acquisition = data[path][()]
            tensor[i] = _ecg_str2date(acquisition).year
        except KeyError:
            pass
    return tensor


tmaps["ecg_acquisition_year"] = TensorMap(
    "ecg_acquisition_year",
    path_prefix=ECG_PREFIX,
    loss="logcosh",
    tensor_from_file=ecg_acquisition_year,
    shape=(1,),
    time_series_limit=0,
)


MIN_HEIGHT_IN = 48
MAX_HEIGHT_IN = 96
MIN_WEIGHT_LBS = 60
MAX_WEIGHT_LBS = 800


def ecg_bmi(tm, data):
    ecg_dates = tm.time_series_filter(data)
    dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=float)
    for i, ecg_date in enumerate(ecg_dates):
        path = lambda key: make_hd5_path(tm, ecg_date, key)
        try:
            weight_lbs = float(data[path("weightlbs")][()])
            height_in = float(data[path("heightin")][()])
            if (
                height_in < MIN_HEIGHT_IN
                or height_in > MAX_HEIGHT_IN
                or weight_lbs < MIN_WEIGHT_LBS
                or weight_lbs > MAX_WEIGHT_LBS
            ):
                raise ValueError(f"Height/Weight outside valid range")

            weight_kg = 0.454 * weight_lbs
            height_m = 0.0254 * height_in
            bmi = weight_kg / (height_m ** 2)
            tensor[i] = bmi
        except (KeyError, ZeroDivisionError, ValueError):
            pass
    return tensor


tmaps["ecg_bmi"] = TensorMap(
    "ecg_bmi",
    path_prefix=ECG_PREFIX,
    shape=(1,),
    time_series_limit=0,
    tensor_from_file=ecg_bmi,
    validators=validator_not_all_zero,
)


tmaps["ecg_bmi_std"] = TensorMap(
    "ecg_bmi_std",
    path_prefix=ECG_PREFIX,
    shape=(1,),
    time_series_limit=0,
    tensor_from_file=ecg_bmi,
    normalizers=ZScorePopulation(mean=29, std=6),
    validators=validator_not_all_zero,
)


tmap_name = "ecg_weight_lbs"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="weightlbs"),
    shape=(1,),
    time_series_limit=0,
    validators=RangeValidator(MIN_WEIGHT_LBS, MAX_WEIGHT_LBS),
)


tmap_name = "ecg_weight_lbs_std"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="weightlbs"),
    shape=(1,),
    time_series_limit=0,
    validators=RangeValidator(MIN_WEIGHT_LBS, MAX_WEIGHT_LBS),
    normalizers=ZScorePopulation(mean=182, std=45),
)


tmap_name = "ecg_height_in"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="heightin"),
    shape=(1,),
    time_series_limit=0,
    validators=RangeValidator(MIN_HEIGHT_IN, MAX_HEIGHT_IN),
)


tmap_name = "ecg_height_in_std"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="heightin"),
    shape=(1,),
    time_series_limit=0,
    validators=RangeValidator(MIN_HEIGHT_IN, MAX_HEIGHT_IN),
    normalizers=ZScorePopulation(mean=66, std=4),
)


def voltage_zeros(tm, data):
    ecg_dates = tm.time_series_filter(data)
    dynamic, shape = is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=np.float32)
    for i, ecg_date in enumerate(ecg_dates):
        for cm in tm.channel_map:
            path = make_hd5_path(tm, ecg_date, cm)
            voltage = data[path][()]
            slices = (i, tm.channel_map[cm]) if dynamic else (tm.channel_map[cm],)
            tensor[slices] = np.count_nonzero(voltage == 0)
    return tensor


tmaps["voltage_zeros"] = TensorMap(
    "voltage_zeros",
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=voltage_zeros,
    shape=(12,),
    channel_map=ECG_REST_LEADS_ALL,
    time_series_limit=0,
)


def _ecg_str2date(d) -> datetime.date:
    return datetime.datetime.strptime(d, ECG_DATE_FORMAT).date()
