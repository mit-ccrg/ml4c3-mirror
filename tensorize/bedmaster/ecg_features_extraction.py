# Imports: standard library
import os
import re
import logging
import multiprocessing
from typing import Any, Dict, List

# Imports: third party
import h5py
import numpy as np
import neurokit2 as nk

# Imports: first party
from tensormap.TensorMap import TensorMap
from tensormap.icu_signals import get_tmap as GET_SIGNAL_TMAP
from tensormap.icu_list_signals import get_tmap as GET_LIST_TMAP
from tensormap.icu_first_visit_with_signal import get_tmap as get_visit_tmap

LEADS = ["i", "ii", "iii", "v"]
ECG_TMAPS = {
    NAME: GET_SIGNAL_TMAP(NAME)
    for LEAD in LEADS
    for NAME in (f"{LEAD}_value", f"{LEAD}_sample_freq")
}

# pylint: disable=comparison-with-itself, too-many-branches
# pylint: disable=attribute-defined-outside-init


class ECGFeatureFileExtractor(h5py.File):
    """
    Class that calculates the P, Q, R, S and T peaks, P, QRS, T waves onsets
    and offsets, PR, QT, RR and QRS intervals duration and QRS amplitude from
    all the ecg signals of the specified hd5 file.

    The neurokit2 package is used.
    """

    def __init__(self, hd5_path: str):
        """
        Inherits from h5py.File to profit its features.

        :param hd5_path: <str> Full path of the hd5 file.
        """
        super().__init__(hd5_path, "r+")
        self.r_peaks: Dict[str, Any] = {}
        self.waves_peaks: Dict[str, Any] = {}
        self.other_features: Dict[str, List[np.int]] = {
            "ECG_PR_Interval": [],
            "ECG_QT_Interval": [],
            "ECG_RR_Interval": [],
            "ECG_QRS_Interval": [],
            "ECG_QRS_Amplitude": [],
            "ECG_PR_Segment": [],
            "ECG_TP_Segment": [],
            "ECG_ST_Segment": [],
            "ECG_ST_Height": [],
        }
        try:
            self.visits = GET_SIGNAL_TMAP("visits").tensor_from_file(
                GET_SIGNAL_TMAP("visits"),
                self,
            )
        except KeyError:
            self.visits = np.array([[None]])
        self.visit = self.visits[0][0]
        self.lead = self.list_available_leads()[0]
        self.change_lead(self.list_available_leads()[0])

    def change_visit(self, visit: str) -> bool:
        """
        Function to select the desired visit, read the ecg signal and clean it.
        The result is saved internally.

        :param visit: <str> Desired visit.
        :return: <bool> True, if the visit is one of the available visits IDs and it is
                 been able to do the computations, False in the contrary case.
        """
        if visit not in self.visits:
            return False
        self.visit = visit
        change = self.change_lead(self.lead)
        if not change:
            self.change_lead(self.list_available_leads()[0])
        return True

    def change_lead(self, lead: str) -> bool:
        """
        Function to select the desired lead, read the ecg signal and clean it.
        The result is saved internally.

        :param lead: <str> Desired lead.
        :return: <bool> True, if the lead is one of the available leads and it is been
                 able to do the computations, False in the contrary case.
        """
        if lead not in self.list_available_leads() or not lead:
            return False
        self.lead = lead
        self.sampling_rate = ECG_TMAPS[f"{lead}_sample_freq"].tensor_from_file(
            ECG_TMAPS[f"{lead}_sample_freq"],
            self,
            visit=self.visit,
        )[0]
        self.update_sampling_rate()
        return True

    def list_available_leads(self) -> List[Any]:
        """
        List the available leads for the current visit.

        :return: <List[str]> List with the name of the available leads.
        """
        try:
            signals = GET_LIST_TMAP("bedmaster_waveform_signals").tensor_from_file(
                GET_LIST_TMAP("bedmaster_waveform_signals"),
                self,
                visits=self.visit,
            )[0]
        except KeyError:
            return [None]
        leads = [lead for lead in signals if lead in LEADS]
        return leads

    def set_tmap_params(self, signal_tm) -> bool:
        pattern = re.compile(
            r"^(i|ii|iii|v)_value_(\d+)_to_(\d+)_hrs_(pre|post)_(.*)$",
        )
        match = pattern.findall(signal_tm.name)
        if not match:
            return False
        lead, time_1, time_2, period, event_proc_tm = match[0]
        self.lead = lead
        visit_tm = get_visit_tmap(
            re.sub(r"(end_date|start_date)", "first_visit", event_proc_tm),
        )
        self.visit = visit_tm.tensor_from_file(visit_tm, self)
        event_tm = GET_SIGNAL_TMAP(event_proc_tm)
        event_time = event_tm.tensor_from_file(event_tm, self, visits=self.visit)[0][0]
        if period == "pre":
            offset = event_time - time_1 * 60 * 60
        else:
            offset = event_time + time_1 * 60 * 60
        signal_size = (time_2 - time_1) * 60 * 60

        self.sampling_rate = ECG_TMAPS[f"{lead}_sample_freq"].tensor_from_file(
            ECG_TMAPS[f"{lead}_sample_freq"],
            self,
            visit=self.visit,
        )[0]
        if self.sampling_rate.size == 1:
            return True

        last = -1
        for k, _ in enumerate(self.sampling_rate):
            self.sampling_rate[k][1] -= offset
            if self.sampling_rate[k][1] < 0:
                last_negative = k
            if self.sampling_rate[k][1] > signal_size:
                last = k
        self.sampling_rate = self.sampling_rate[last_negative:last]
        self.sampling_rate[0][1] = 0
        self.update_sampling_rate()
        return True

    def update_sampling_rate(self):
        new_sampling_rate = np.array([[np.nan, np.nan]] * self.sampling_rate.size)
        for k, _ in enumerate(self.sampling_rate):
            new_sampling_rate[k] = [self.sampling_rate[k][0], np.nan]
            if k == 0:
                new_sampling_rate[k][1] = self.sampling_rate[k][1]
                continue
            new_sampling_rate[k][1] = (
                new_sampling_rate[k - 1][1]
                + (self.sampling_rate[k][1] - self.sampling_rate[k - 1][1])
                * self.sampling_rate[k - 1][0]
            )
        self.sampling_rate = new_sampling_rate.astype(int)
        return True

    def extract_features(
        self,
        clean_method: str = "neurokit",
        r_method: str = "neurokit",
        wave_method: str = "dwt",
        min_peaks: int = 200,
        size: int = 200000,
    ):
        """
        Function to extract the ecg features using the neurokit2 package. That
        is the P, Q, R, S and T peaks and the P, QRS and T waves onsets and
        offsets. The result is saved internally.

        :param clean_method: <str> The processing pipeline to apply. Can be one of
                             ‘neurokit’ (default), ‘biosppy’, ‘pantompkins1985’,
                             ‘hamilton2002’, ‘elgendi2010’, ‘engzeemod2012’.
        :param r_method: <str> The algorithm to be used for R-peak detection. Can be one
                         of ‘neurokit’ (default), ‘pantompkins1985’, ‘hamilton2002’,
                         ‘christov2004’, ‘gamboa2008’, ‘elgendi2010’, ‘engzeemod2012’
                         or ‘kalidas2017’.
        :param wave_method: <str> Can be one of ‘dwt’ (default) for discrete
                            wavelet transform or ‘cwt’ for continuous wavelet transform.
        :param min_peaks: <int> Minimum R peaks to be detected to proceed with
                          further calculations.
        :param size: <int> ECG sample size to analyze per loop.
        """
        if not self.lead:
            return

        for i, _ in enumerate(self.sampling_rate):
            sampling_rate = self.sampling_rate[i][0]
            init = self.sampling_rate[i][1]
            if i == len(self.sampling_rate) - 1:
                ecg_signal_size = (
                    ECG_TMAPS[f"{self.lead}_value"]
                    .tensor_from_file(
                        ECG_TMAPS[f"{self.lead}_value"],
                        self,
                        visit=self.visit,
                    )[0][init:]
                    .shape[0]
                )
            else:
                ecg_signal_size = self.sampling_rate[i + 1][1] - init
            if size < ecg_signal_size:
                end = init + size
            else:
                end = init + ecg_signal_size
            while init < ecg_signal_size + self.sampling_rate[i][1]:
                ecg_signal = ECG_TMAPS[f"{self.lead}_value"].tensor_from_file(
                    ECG_TMAPS[f"{self.lead}_value"],
                    self,
                    visit=self.visit,
                )[0][init:end]
                ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate, clean_method)
                try:
                    _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate, r_method)
                except IndexError:
                    init = end
                    end = init + size
                    if end > ecg_signal_size + self.sampling_rate[i][1]:
                        end = ecg_signal_size + self.sampling_rate[i][1]
                    continue

                if len(r_peaks["ECG_R_Peaks"]) < min_peaks:
                    init = end
                    end = init + size
                    if end > ecg_signal_size + self.sampling_rate[i][1]:
                        end = ecg_signal_size + self.sampling_rate[i][1]
                    continue
                _, waves_peaks = nk.ecg_delineate(ecg_signal, r_peaks, sampling_rate)
                _, waves_peaks_2 = nk.ecg_delineate(
                    ecg_signal,
                    r_peaks,
                    sampling_rate,
                    wave_method,
                )
                waves_peaks.update(waves_peaks_2)
                for peak_type in r_peaks:
                    if peak_type not in self.r_peaks:
                        self.r_peaks[peak_type] = r_peaks[peak_type]
                    else:
                        self.r_peaks[peak_type] = np.append(
                            self.r_peaks[peak_type],
                            r_peaks[peak_type],
                        )
                for peak_type in waves_peaks:
                    if peak_type not in self.waves_peaks:
                        self.waves_peaks[peak_type] = waves_peaks[peak_type]
                    else:
                        self.waves_peaks[peak_type] = np.append(
                            self.waves_peaks[peak_type],
                            waves_peaks[peak_type],
                        )
                init = end
                end = init + size
                if end > ecg_signal_size + self.sampling_rate[i][1]:
                    end = ecg_signal_size + self.sampling_rate[i][1]

        for peak_type in self.r_peaks:
            self.r_peaks[peak_type] = list(self.r_peaks[peak_type])
        for peak_type in self.waves_peaks:
            self.waves_peaks[peak_type] = list(self.waves_peaks[peak_type])

    def extract_features_tmaps(
        self,
        signal_tm: TensorMap,
        clean_method: str = "neurokit",
        r_method: str = "neurokit",
        wave_method: str = "dwt",
        min_peaks: int = 200,
    ):
        """
        Function to extract the ecg features using the neurokit2 package. That
        is the P, Q, R, S and T peaks and the P, QRS and T waves onsets and
        offsets. The result is saved internally.

        :param signal_tm: <TensorMap>
        :param clean_method: <str> The processing pipeline to apply. Can be one of
                             ‘neurokit’ (default), ‘biosppy’, ‘pantompkins1985’,
                             ‘hamilton2002’, ‘elgendi2010’, ‘engzeemod2012’.
        :param r_method: <str> The algorithm to be used for R-peak detection. Can be one
                         of ‘neurokit’ (default), ‘pantompkins1985’, ‘hamilton2002’,
                         ‘christov2004’, ‘gamboa2008’, ‘elgendi2010’, ‘engzeemod2012’
                         or ‘kalidas2017’.
        :param wave_method: <str> Can be one of ‘dwt’ (default) for discrete
                            wavelet transform or ‘cwt’ for continuous wavelet transform.
        :param min_peaks: <int> Minimum R peaks to be detected to proceed with
                          further calculations.
        """
        for i, _ in enumerate(self.sampling_rate):
            sampling_rate = self.sampling_rate[i][0]
            init = self.sampling_rate[i][1]
            if i == len(self.sampling_rate) - 1:
                end = -1
            else:
                end = self.sampling_rate[i + 1][1]
            ecg_signal = signal_tm.tensor_from_file(signal_tm, self)[0][init:end]
            ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate, clean_method)

            try:
                _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate, r_method)
            except IndexError:
                continue
            if len(r_peaks["ECG_R_Peaks"]) < min_peaks:
                continue
            _, waves_peaks = nk.ecg_delineate(ecg_signal, r_peaks, sampling_rate)
            _, waves_peaks_2 = nk.ecg_delineate(
                ecg_signal,
                r_peaks,
                sampling_rate,
                wave_method,
            )
            waves_peaks.update(waves_peaks_2)
            for peak_type in r_peaks:
                if peak_type not in self.r_peaks:
                    self.r_peaks[peak_type] = r_peaks[peak_type]
                else:
                    self.r_peaks[peak_type] = np.append(
                        self.r_peaks[peak_type],
                        r_peaks[peak_type],
                    )
            for peak_type in waves_peaks:
                if peak_type not in self.waves_peaks:
                    self.waves_peaks[peak_type] = waves_peaks[peak_type]
                else:
                    self.waves_peaks[peak_type] = np.append(
                        self.waves_peaks[peak_type],
                        waves_peaks[peak_type],
                    )

        for peak_type in self.r_peaks:
            self.r_peaks[peak_type] = list(self.r_peaks[peak_type])
        for peak_type in self.waves_peaks:
            self.waves_peaks[peak_type] = list(self.waves_peaks[peak_type])

    def compute_additional_features(self):
        """
        Calculates additional features from the ECG using the results from
        extract_features().

        That is, the PR, QT, RR and QRS intervals duration and the QRS
        amplitude. The result is saved internally.
        """
        if not self.lead:
            return
        no_nans = False
        i = -1
        while no_nans:
            i += 1
            if i >= len(self.wave_peaks["ECG_P_Peaks"]) - 1:
                no_nans = True
            features = np.arra(
                [
                    self.r_peaks["ECG_R_Peaks"][i],
                    self.waves_peaks["ECG_P_Peaks"][i],
                    self.waves_peaks["ECG_T_Peaks"][i],
                ],
            )
            if not np.isnan(features).any():
                no_nans = True

        sampling_rate = np.array([])
        for i, k in enumerate(self.sampling_rate):
            if i == len(self.sampling_rate) - 1:
                length = len(self.r_peaks["ECG_R_Peaks"]) - len(sampling_rate)
            else:
                length = len(
                    np.array(self.r_peaks["ECG_R_Peaks"])[
                        (k[1] <= self.r_peaks["ECG_R_Peaks"])
                        & (self.r_peaks["ECG_R_Peaks"] < self.sampling_rate[i + 1][1])
                    ],
                )
            sampling_rate = np.append(sampling_rate, np.array([k[0]] * length))

        # Compute PR interval duration
        # Compute PR segment duration
        if self.waves_peaks["ECG_P_Peaks"][i] < self.r_peaks["ECG_R_Peaks"][i]:
            pr_interval = (
                np.array(self.waves_peaks["ECG_P_Onsets"])
                - np.array(self.waves_peaks["ECG_R_Onsets"])
            ) / sampling_rate
            pr_segment = (
                np.array(self.waves_peaks["ECG_P_Offsets"])
                - np.array(self.waves_peaks["ECG_R_Onsets"])
            ) / sampling_rate
        else:
            self.other_features["ECG_PR_Interval"].append(np.nan)
            self.other_features["ECG_PR_Segment"].append(np.nan)
            pr_interval = (
                np.array(self.waves_peaks["ECG_P_Onsets"][:-1])
                - np.array(self.waves_peaks["ECG_R_Onsets"][1:])
            ) / sampling_rate[1:]
            pr_segment = (
                np.array(self.waves_peaks["ECG_P_Offsets"][:-1])
                - np.array(self.waves_peaks["ECG_R_Onsets"][1:])
            ) / sampling_rate[1:]
        self.other_features["ECG_PR_Interval"].extend(pr_interval)
        self.other_features["ECG_PR_Segment"].extend(pr_segment)

        # Compute QT interval duration
        # Compute ST segment duration
        # Compute ST segment height
        r_offsets = np.array(self.waves_peaks["ECG_R_Offsets"])
        is_nan_r_offsets = np.isnan(r_offsets)
        no_nan_r_offsets = r_offsets[~is_nan_r_offsets]
        t_onsets = np.array(self.waves_peaks["ECG_T_Onsets"])
        is_nan_t_onsets = np.isnan(t_onsets)
        no_nan_t_onsets = t_onsets[~is_nan_t_onsets]

        ecg_values = ECG_TMAPS[f"{self.lead}_value"].tensor_from_file(
            ECG_TMAPS[f"{self.lead}_value"],
            self,
            visit=self.visit,
        )[0][np.append(no_nan_r_offsets, no_nan_t_onsets).astype(int)]

        ecg_r_offsets = ecg_values[: no_nan_r_offsets.shape[0]]
        index = np.where(is_nan_r_offsets)[0]
        index -= np.array(range(index.shape[0])).astype(int)
        ecg_r_offsets = np.append(ecg_r_offsets, np.array([np.nan] * index.shape[0]))
        ecg_r_offsets = np.insert(ecg_r_offsets, index, np.nan)[: r_offsets.shape[0]]

        ecg_t_onsets = ecg_values[no_nan_r_offsets.shape[0] :]
        index = np.where(is_nan_t_onsets)[0]
        index -= np.array(range(index.shape[0])).astype(int)
        ecg_t_onsets = np.append(ecg_t_onsets, np.array([np.nan] * index.shape[0]))
        ecg_t_onsets = np.insert(ecg_t_onsets, index, np.nan)[: t_onsets.shape[0]]

        if self.r_peaks["ECG_R_Peaks"][i] < self.waves_peaks["ECG_T_Peaks"][i]:
            qt_interval = (
                np.array(self.waves_peaks["ECG_R_Onsets"])
                - np.array(self.waves_peaks["ECG_T_Offsets"])
            ) / sampling_rate
            st_segment = (
                np.array(self.waves_peaks["ECG_R_Offsets"])
                - np.array(self.waves_peaks["ECG_T_Onsets"])
            ) / sampling_rate
            st_height = ecg_r_offsets - ecg_t_onsets
        else:
            self.other_features["ECG_QT_Interval"].append(np.nan)
            self.other_features["ECG_ST_Segment"].append(np.nan)
            self.other_features["ECG_ST_Height"].append(np.nan)
            qt_interval = (
                np.array(self.waves_peaks["ECG_R_Onsets"][:-1])
                - np.array(self.waves_peaks["ECG_T_Offsets"][1:])
            ) / sampling_rate[:-1]
            st_segment = (
                np.array(self.waves_peaks["ECG_R_Offsets"][:-1])
                - np.array(self.waves_peaks["ECG_T_Onsets"][1:])
            ) / sampling_rate[:-1]
            st_height = ecg_r_offsets[:-1] - ecg_t_onsets[1:]
        self.other_features["ECG_QT_Interval"].extend(qt_interval)
        self.other_features["ECG_ST_Segment"].extend(st_segment)
        self.other_features["ECG_ST_Height"].extend(list(st_height))

        # Compute TP segment duration
        if self.waves_peaks["ECG_P_Peaks"][i] < self.waves_peaks["ECG_T_Peaks"][i]:
            self.other_features["ECG_TP_Segment"].append(np.nan)
            tp_segment = (
                np.array(self.waves_peaks["ECG_T_Offsets"][1:])
                - np.array(self.waves_peaks["ECG_P_Onsets"][:-1])
            ) / sampling_rate[1:]
        else:
            tp_segment = (
                np.array(self.waves_peaks["ECG_T_Offsets"])
                - np.array(self.waves_peaks["ECG_P_Onsets"])
            ) / sampling_rate
        self.other_features["ECG_TP_Segment"].extend(tp_segment)

        # Compute RR interval duration
        rr_interval = (
            np.array(self.r_peaks["ECG_R_Peaks"][1:])
            - np.array(self.r_peaks["ECG_R_Peaks"][:-1])
        ) / sampling_rate[1:]
        self.other_features["ECG_RR_Interval"].extend(rr_interval)

        # Compute QRS interval duration
        qrs_interval = (
            np.array(self.waves_peaks["ECG_R_Offsets"])
            - np.array(self.waves_peaks["ECG_R_Onsets"])
        ) / sampling_rate
        self.other_features["ECG_QRS_Interval"].extend(qrs_interval)

        # Compute QRS amplitude
        r_peaks = np.array(self.r_peaks["ECG_R_Peaks"])
        is_nan_r_peaks = np.isnan(r_peaks)
        no_nan_r_peaks = r_peaks[~is_nan_r_peaks]
        q_peaks = np.array(self.waves_peaks["ECG_Q_Peaks"])
        is_nan_q_peaks = np.isnan(q_peaks)
        no_nan_q_peaks = q_peaks[~is_nan_q_peaks]
        s_peaks = np.array(self.waves_peaks["ECG_S_Peaks"])
        is_nan_s_peaks = np.isnan(s_peaks)
        no_nan_s_peaks = s_peaks[~is_nan_s_peaks]

        ecg_values = ECG_TMAPS[f"{self.lead}_value"].tensor_from_file(
            ECG_TMAPS[f"{self.lead}_value"],
            self,
            visit=self.visit,
        )[0][
            np.append(no_nan_r_peaks, np.append(no_nan_q_peaks, no_nan_s_peaks)).astype(
                int,
            )
        ]

        ecg_r_peaks = ecg_values[: no_nan_r_peaks.shape[0]]
        index = np.where(is_nan_r_offsets)[0]
        index -= np.array(range(index.shape[0])).astype(int)
        ecg_r_peaks = np.append(ecg_r_peaks, np.array([np.nan] * index.shape[0]))
        ecg_r_peaks = np.insert(ecg_r_peaks, index, np.nan)[: r_peaks.shape[0]]

        ecg_q_peaks = ecg_values[
            no_nan_r_peaks.shape[0] : no_nan_r_peaks.shape[0] + no_nan_q_peaks.shape[0]
        ]
        index = np.where(is_nan_q_peaks)[0]
        index -= np.array(range(index.shape[0])).astype(int)
        ecg_q_peaks = np.append(ecg_q_peaks, np.array([np.nan] * index.shape[0]))
        ecg_q_peaks = np.insert(ecg_q_peaks, index, np.nan)[: q_peaks.shape[0]]

        ecg_s_peaks = ecg_values[no_nan_r_peaks.shape[0] + no_nan_q_peaks.shape[0] :]
        index = np.where(is_nan_s_peaks)[0]
        index -= np.array(range(index.shape[0])).astype(int)
        ecg_s_peaks = np.append(ecg_s_peaks, np.array([np.nan] * index.shape[0]))
        ecg_s_peaks = np.insert(ecg_s_peaks, index, np.nan)[: s_peaks.shape[0]]

        qrs_amplitude = ecg_r_peaks - np.min(
            np.array([ecg_q_peaks, ecg_s_peaks]),
            axis=0,
        )
        self.other_features["ECG_QRS_Amplitude"].extend(list(qrs_amplitude))

    def save_features(self):
        """
        Saves the information obtained by extract_features() and
        compute_additional_features() in the hd5 file.
        """
        if not self.lead:
            return
        base_dir = self[f"bedmaster/{self.visit}"]
        if "ecg_features" not in base_dir.keys():
            base_dir.create_group("ecg_features")
        if self.lead not in base_dir["ecg_features"].keys():
            base_dir["ecg_features"].create_group(self.lead)
        data = {}
        data.update(self.r_peaks)
        data.update(self.waves_peaks)
        data.update(self.other_features)
        for data_type in data:
            if data_type.lower() in base_dir[f"ecg_features/{self.lead}"]:
                del base_dir[f"ecg_features/{self.lead}/{data_type.lower()}"]
            base_dir[f"ecg_features/{self.lead}"].create_dataset(
                name=data_type.lower(),
                data=np.array(data[data_type], float),
                maxshape=(None,),
                compression=32015,
            )
        base_dir[f"ecg_features/{self.lead}"].attrs["completed"] = True


class ECGFeatureDirExtractor:
    """
    Class that calculates the P, Q, R, S and T peaks, P, QRS, T waves onsets
    and offsets, PR, QT, RR and QRS intervals duration and QRS amplitude from
    all the ecg signals of each hd5 file inside the specified directory.
    """

    def __init__(
        self,
        hd5_dir: str,
        r_method: str = "neurokit",
        wave_method: str = "dwt",
        tmaps: List[TensorMap] = None,
    ):
        """
        Init ECG Feature Dir Extractor.

        :param hd5_dir: <str> Full path of the directory containing the hd5 files to
                        extract the features.
        :param r_method: <str> The algorithm to be used for R-peak detection. Can be one
                         of neurokit (default), pantompkins1985, hamilton2002,
                         christov2004, gamboa2008, elgendi2010, engzeemod2012
                         or kalidas2017.
        :param wave_method: <str> Can be one of dwt (default) for discrete
                            wavelet transform or cwt for continuous wavelet transform.
        """
        self.hd5_dir = hd5_dir
        self.remaining_files = 0
        self.clean_method = "neurokit"
        self.r_method = r_method
        self.wave_method = wave_method
        self.tmaps = tmaps

    def extract_dir_features(self, n_workers: int = 1):
        """
        Function that iterates through hd5 file to extract its features using
        extract_file_features(). This process is parallelized.

        :param n_workers: <int> Integer indicating the number of cores used to
                          parallelize the extraction of features of the hd5 files.
                          It is parallelized by file.
        """
        hd5_files = [
            os.path.join(self.hd5_dir, hd5_file)
            for hd5_file in os.listdir(self.hd5_dir)
            if hd5_file.endswith(".hd5")
        ]
        # Check number of workers used is not higher than the number of cpus
        if os.cpu_count():
            if (n_workers or 0) > os.cpu_count():  # type: ignore
                n_workers = os.cpu_count()  # type: ignore
                logging.warning(
                    "Workers are higher than number of cpus. "
                    f"Number of workers is reduced to {os.cpu_count()}, "
                    "the number of cpus your computer have.",
                )
        else:
            logging.warning(
                "Couldn't determine the number of cpus. Blindly "
                "accepting the --n_workers option",
            )
        # Debugging purpose
        # self.extract_file_features(hd5_files[0],1,3)
        with multiprocessing.Pool(processes=n_workers) as pool:
            pool.starmap(
                self.extract_file_features,
                [
                    (hd5_file, k + 1, len(hd5_files))
                    for k, hd5_file in enumerate(hd5_files)
                ],
            )

    def extract_file_features(self, hd5_file: str, file_number: int, total_files: int):
        """
        Function that extracts all the ECG features from an hd5 file and save
        them in in the same file. The ECGFeatureFileExtractor instance is used.

        :param hd5_file: <str> Full path of the file to extract features.
        :param file_number: <int>
        :param total_files: <int>
        """
        extractor = ECGFeatureFileExtractor(hd5_file)
        if self.tmaps:
            for tmap in self.tmaps:
                if not extractor.set_tmap_params(tmap):
                    continue
                extractor.extract_features_tmaps(
                    clean_method=self.clean_method,
                    r_method=self.r_method,
                    wave_method=self.wave_method,
                    signal_tm=tmap,
                )
                extractor.compute_additional_features()
                extractor.save_features()
                logging.info(
                    f"Features extracted using tmap {tmap.name} from file "
                    f"{file_number}/{total_files}.",
                )
        else:
            for visit in extractor.visits:
                if not visit:
                    continue
                extractor.visit = visit[0]
                available_leads = extractor.list_available_leads()
                for lead in available_leads:
                    if not lead:
                        continue
                    extractor.change_lead(lead)
                    logging.info(
                        f"Lead changed {lead} from file {file_number}/{total_files}.",
                    )
                    extractor.extract_features(
                        clean_method=self.clean_method,
                        r_method=self.r_method,
                        wave_method=self.wave_method,
                    )
                    extractor.compute_additional_features()
                    extractor.save_features()
                    logging.info(
                        f"Features extracted from lead {lead} from file "
                        f"{file_number}/{total_files}.",
                    )
        logging.info(f"Features extracted from file {file_number}/{total_files}.")


def extract_ecg_features(args):
    extractor = ECGFeatureDirExtractor(
        args.tensors,
        args.r_method,
        args.wave_method,
        args.tensor_maps_in,
    )
    extractor.extract_dir_features(n_workers=args.num_workers)
