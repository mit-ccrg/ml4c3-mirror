# Imports: standard library
import os
import copy
import math
import logging
import warnings
import itertools

# Imports: third party
import numpy as np
import matplotlib as mpl
from scipy import stats, linalg
from matplotlib import pyplot as plt
from scipy.spatial import distance
from tqdm.notebook import tqdm as log_progress
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    KMeans,
    SpectralClustering,
    AgglomerativeClustering,
)
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Imports: first party
from clustering.utils import IncorrectSignalError
from clustering.globals import SIGNAL_LIMITS, PAIRWISE_DISTANCES


class _Curator:
    @classmethod
    def methods(cls):
        methods = {}
        for meth_name, meth in cls.__dict__.items():
            if meth_name.startswith("meth"):
                vars = meth.__func__.__code__.co_varnames[
                    : meth.__func__.__code__.co_argcount
                ]
                vars = [var for var in vars if not var.startswith("_") and var != "cls"]
                methods[meth_name] = vars

        return methods

    @classmethod
    def print_methods(cls):
        for meth, params in cls.methods().items():
            print(f"Method: {meth.replace('meth_', '')}; params: {params}")


class OutlierRemover(_Curator):
    @classmethod
    def remove_outliers(cls, struct, method, **kwargs):
        cls_method = f"meth_{method}"
        if cls_method not in cls.methods():
            raise ValueError(f"Method '{method}' is not valid!")

        return getattr(cls, cls_method)(struct, **kwargs)

    @classmethod
    def print_methods(cls):
        for meth, params in cls.methods().items():
            prefix = "Filter method" if "global" in meth else "Method"
            print(f"{prefix}: {meth.replace('meth_', '')}; params: {params}")

    @classmethod
    def meth_global_quantile_range(cls, _bundle, min_quantile, max_quantile, **kwargs):
        """ Removes all the points outside the set quantile range """
        signal_limits = {}
        signals = _bundle.signal_list()
        for signal_name in log_progress(
            signals,
            leave=False,
            desc="Calculating thresholds",
        ):
            conc_signal = np.array([])
            for patient in _bundle.patients.values():
                signal = patient.get_signal(signal_name)
                conc_signal = np.concatenate((conc_signal, signal.values))
            min_threshold = np.quantile(conc_signal, float(min_quantile))
            max_threshold = np.quantile(conc_signal, float(max_quantile))
            signal_limits[signal_name] = (min_threshold, max_threshold)

        return signal_limits

    @classmethod
    def meth_global_predefined_limits(cls, _bundle, **kwargs):
        """ Removes outliers based on predefined limits. """
        for signal in _bundle.signal_list():
            if signal not in SIGNAL_LIMITS:
                raise KeyError(f"Signal {signal} does not have a defined limit")

        return SIGNAL_LIMITS

    @classmethod
    def meth_min_max_values(
        cls, _signal, min_value, max_value, outliers_pc_th=20, **kwargs
    ):
        """ Removes points that are outside the min-max range """
        not_outliers = np.where(
            (_signal.values <= max_value) & (_signal.values >= min_value),
        )[0]

        outliers_removed = _signal.values.size - not_outliers.size
        outliers_removed_pc = outliers_removed / _signal.values.size * 100

        if outliers_removed_pc > outliers_pc_th:
            raise IncorrectSignalError(
                f"Signal {_signal} has less than {100-outliers_pc_th}% values "
                f"left after outlier removal with interval {min_value}-{max_value}",
            )
        _signal.values = _signal.values[not_outliers]
        _signal.time = _signal.time[not_outliers]

        return outliers_removed, outliers_removed_pc

    @classmethod
    def meth_local_quantile_range(cls, _signal, min_quantile, max_quantile, **kwargs):
        """ Removes all the points outside the set quantile range"""
        min_value = np.quantile(_signal.values, float(min_quantile))
        max_value = np.quantile(_signal.values, float(max_quantile))
        return min_value, max_value

    @classmethod
    def meth_zscore(cls, _signal, z_threshold, **kwargs):
        """
        Removes values if they are <z_threshold>-times the standard deviation
        away from the mean.
        """
        z_score = np.abs(stats.zscore(_signal.values))
        if np.isnan(z_score).all():
            z_score = np.ones(_signal.values.size)
        max_value = _signal.values[z_score < z_threshold].max()
        min_value = _signal.values[z_score < z_threshold].min()
        return min_value, max_value

    @classmethod
    def meth_remove_isolated(cls, _signal, find_method, jump_pc=0.01, **kwargs):
        """
        Removes values based on the quantile range only if they are isolated
        on a histogram.
        The isolation is considered if there is a jump higher than <jump_pc>% the range
        of values of the signal. E.g: for a signal that ranges between 20 and 80 and
        a jump_pc of 0.01, values are considered outliers if they are outside the
        quantile range and there are no other values closer than (80-20)*0.01 = 0.6
        to them.
        """
        bin_reps, bin_values = np.histogram(_signal.values, bins=int(1 / jump_pc))

        low_th, up_th = cls.remove_outliers(_signal, method=find_method, **kwargs)
        mask = np.concatenate(([False], bin_reps == 0, [False]))
        change_idx = np.nonzero(mask[1:] != mask[:-1])[0]

        min_value = 0
        max_value = _signal.values.max()
        for i in change_idx:
            if bin_reps[i] == 0:
                if all(bin_values[i:] > up_th):
                    max_value = bin_values[i]
                    break
            else:
                if all(bin_values[:i] < low_th):
                    min_value = bin_values[i]

        return min_value, max_value


class Padder:
    @classmethod
    def methods(cls):
        methods = ["zero", "edge", "mean", "median", "reflect", "symmetric", "wrap"]
        return {method: () for method in methods}

    @classmethod
    def print_methods(cls):
        for meth, params in cls.methods().items():
            print(f"Method: {meth}; params: {params}")

    @classmethod
    def pad(cls, _signal, method, min_time, max_time, **kwargs):
        """
        Pads the signal with the input padding strategy
        :param min_time: Start time of the new padded signal
        :param max_time: End time of the new padded signal
        :param filling: Options: (Ex: [1,2,3,4,5])
            - 'zero': pads with 0s. Ex: [0,0,1,2,3,4,5,0,0]
            - 'edge': pads with the first/last value. Ex: [1,1,1,2,3,4,5,5,5]
            - 'mean': pads with the mean value of the first/last 10% of values. Ex: [3,3,1,2,3,4,5,3,3]
            - 'median': pads with the median value. Ex: [3,3,1,2,3,4,5,3,3]
            - 'reflect': pads with the reflection of the vector mirrored on
                         the first and last values. Ex: [3,2,1,2,3,4,5,4,3]
            - 'symmetric': pads with the reflection of the vector mirrored along
                           the edge of the array. Ex: [2,1,1,2,3,4,5,5,4]
            - 'wrap': Pads with the wrap of the vector along the axis.
                      The first values are used to pad the end and the end values
                      are used to pad the beginning. Ex: [4,5,1,2,3,4,5,1,2]
        """
        if method not in cls.methods():
            raise ValueError(f"Method '{method}' not valid!")

        sample_freq = 0.5 if not _signal.sample_freq else _signal.sample_freq

        left_pad_size = (_signal.time[0] - min_time) * sample_freq
        left_pad_len = max(0, int(left_pad_size))
        right_pad_size = (max_time - _signal.time[-1]) * sample_freq
        right_pad_len = max(0, int(right_pad_size))
        padding = (left_pad_len, right_pad_len)
        _signal.time = np.pad(
            _signal.time,
            (left_pad_len, right_pad_len),
            "linear_ramp",
            end_values=(min_time, max_time),
        )
        if method == "zero":
            _signal.values = np.pad(
                _signal.values,
                (left_pad_len, right_pad_len),
                "constant",
                constant_values=(0, 0),
            )
        elif method == "mean":
            ten_percent = int(_signal.values.size / 10)
            left_mean, right_mean = (
                _signal.values[:ten_percent].mean(),
                _signal.values[-ten_percent:].mean(),
            )
            _signal.values = np.pad(
                _signal.values,
                (left_pad_len, right_pad_len),
                "constant",
                constant_values=(left_mean, right_mean),
            )
        else:
            _signal.values = np.pad(
                _signal.values,
                (left_pad_len, right_pad_len),
                method,
            )

        if left_pad_size % 1:
            _signal.values = np.insert(_signal.values, 0, _signal.values[0])
            _signal.time = np.insert(_signal.time, 0, min_time)
        if right_pad_size % 1:
            _signal.values = np.append(_signal.values, _signal.values[-1])
            _signal.time = np.append(_signal.time, max_time)

        return padding


class Downsampler(_Curator):
    @classmethod
    def downsample(cls, method, signal, new_rate):
        cls_method = f"meth_{method}"
        if cls_method not in cls.methods():
            raise ValueError(f"Method '{method}' not recognized")

        return getattr(cls, cls_method)(signal, float(new_rate))

    @classmethod
    def meth_linear_interpolation(cls, _signal, new_rate):
        """
        Downsamples time to <new_rate>. If there is a value associated with the
        timestamp, it takes the value. If the new timestamp falls between two values,
        it takes the linear interpolation between those two values.
        """
        new_values = []
        time = np.arange(_signal.time[0], _signal.time[-1] + 0.01, new_rate)

        for idx, t in enumerate(time):
            v = cls._interpolate(_signal, t)
            if np.isnan(v):
                print("**** Reason: nan value ****")
                # Imports: standard library
                import code

                code.interact(local=locals())
            new_values.append(v)
        ds_percent = ((_signal.values.size - time.size) / _signal.values.size) * 100

        values = np.array(new_values)
        if any(values < 0) and _signal.stype == "med":
            print("**** Reason: med with negative dose ****")
            # Imports: standard library
            import code

            code.interact(local=locals())

        _signal.time = time
        _signal.values = values
        _signal.sample_freq = 1 / new_rate

        return ds_percent

    @classmethod
    def meth_smooth_by_mean(cls, _signal, new_rate):
        """
        First, it linearly interpolates the signal to uniform it at 0.5Hz of frequency.
        Then it downsamples it to the desired new frequency smoothing by the mean
        of the discarted values.
        """
        if new_rate % 2:
            raise ValueError("New rate has to be even")

        initial_points = _signal.values.size
        if not all(np.diff(_signal.time) == 2):
            cls.meth_linear_interpolation(_signal, new_rate=2)

        new_time = np.arange(_signal.time[0], _signal.time[-1] + 0.01, new_rate)
        v_spacing = int(new_rate / 2)
        pad_size = (
            math.ceil(float(_signal.values.size) / v_spacing) * v_spacing
            - _signal.values.size
        )
        v_padded = np.append(_signal.values, np.zeros(pad_size) * np.NaN)

        new_values = np.nanmean(v_padded.reshape(-1, v_spacing), axis=1)

        if not all(np.diff(new_time) == new_rate):
            # Imports: standard library
            import code

            code.interact(local=locals())
        if new_time.size != new_values.size:
            # Imports: standard library
            import code

            code.interact(local=locals())

        _signal.values = new_values
        _signal.time = new_time

        ds_percent = (initial_points - new_values.size) / initial_points * 100
        return ds_percent

    @staticmethod
    def _interpolate(_signal, time):
        if time in _signal.time:
            idx = np.where(_signal.time == time)[0][0]
            return _signal.values[idx]
        else:
            idx = np.where(_signal.time < time)[0]
            if idx.size == 0:
                logging.info(
                    "Signal has a shorted length. Maybe the first points were outliers.",
                )
                idx = 0
            else:
                idx = idx[-1]
                if idx + 1 == _signal.time.size:
                    return _signal.values[idx]

            return _signal.values[idx] + (
                _signal.values[idx + 1] - _signal.values[idx]
            ) * (
                (time - _signal.time[idx]) / (_signal.time[idx + 1] - _signal.time[idx])
            )


class Normalizer(_Curator):
    @classmethod
    def normalize(cls, method, bundle, **kwargs):
        cls_method = f"meth_{method}"
        if cls_method not in cls.methods():
            raise ValueError(f"Method '{method}' is not valid!")

        getattr(cls, cls_method)(bundle, **kwargs)

    @classmethod
    def meth_min_max_values(cls, _bundle):
        signals = _bundle.signal_list()
        for signal_name in log_progress(signals, desc="Normalizing..."):
            signal_min, signal_max = _bundle._signal_min_max(signal_name)
            for patient in _bundle.patients.values():
                signal = patient.get_signal(signal_name)
                signal.values = (signal.values - signal_min) / (signal_max - signal_min)
                signal.scale_factor = (signal_min, signal_max)
                # signal.time = signal.time - signal.time[0]


class FeatureExtractor(_Curator):
    @classmethod
    def get_feature(cls, method, bundle, **kwargs):
        cls_method = f"meth_{method}"
        if cls_method not in cls.methods():
            raise ValueError(f"Method '{method}' is not valid!")

        return getattr(cls, cls_method)(bundle, **kwargs)

    @classmethod
    def meth_concatenation(cls, _bundle, order, verbose=False):
        if order == "xxyy":
            np_order = "F"
        elif order == "xyxy":
            np_order = "C"
        else:
            raise ValueError(f"Order {order} nor recognized")

        npatiens = len(_bundle.patients)
        ntimespans = len(_bundle.any_patient())
        nsignals = len(_bundle.signal_list())
        dist = np.zeros((npatiens, ntimespans, nsignals))

        for p_idx, (pname, patient) in log_progress(
            enumerate(sorted(_bundle.patients.items())),
            desc="concatenating...",
        ):
            # dim1: pacient
            for ts_idx in range(len(patient)):
                # dim2: timespan
                for s_idx in range(len(patient[ts_idx])):
                    # dim3: senyal
                    try:
                        dist[p_idx][ts_idx][s_idx] = patient[ts_idx][s_idx]
                    except:
                        # Imports: standard library
                        import code

                        code.interact(local=locals())
                    if np.isnan(patient[ts_idx][s_idx]):
                        # Imports: standard library
                        import code

                        code.interact(local=locals())
                    if verbose:
                        print(f"[{p_idx}][{ts_idx}][{s_idx}]: {patient[ts_idx][s_idx]}")

        dist = dist.reshape(
            (dist.shape[0], dist.shape[1] * dist.shape[2]),
            order=np_order,
        )
        return dist


class DistanceMetric(_Curator):
    @classmethod
    def get_distance(cls, method, patient1, patient2, verbose=False, **kwargs):
        cls_method = f"meth_{method}"
        if cls_method not in cls.methods():
            raise ValueError(f"Method '{method}' is not valid!")

        return getattr(cls, cls_method)(patient1, patient2, verbose, **kwargs)

    @classmethod
    def meth_euclidean(cls, _patient1, _patient2):
        if not sorted(_patient1.signals) == sorted(_patient2.signals):
            raise ValueError("Bundles don't contain the same signals!")

        dist = 0
        if not len(_patient1) == len(_patient2):
            raise ValueError(
                f"Something went wrong with patients "
                f"{_patient1} (len: {len(_patient1)}) and "
                f"{_patient2} (len: {len(_patient2)}). "
                f"They should have the same length. Have they been out-padded?",
            )

        length = len(_patient1)
        for i in range(length):
            dist += distance.euclidean(_patient1[i], _patient2[i])
        dist = dist / length

        return dist

    @classmethod
    def meth_lcss(cls, _patient1, _patient2, eps):
        if not sorted(_patient1.signals) == sorted(_patient2.signals):
            raise ValueError("Bundles don't contain the same signals!")

        eps = float(eps)
        L = np.zeros((len(_patient1) + 1, len(_patient2) + 1))

        def _points_equal(point1, point2, _eps):
            for idx, p in enumerate(point1):
                if p - point2[idx] > _eps:
                    return False
            return True

        for i in range(len(_patient1) + 1):
            for j in range(len(_patient2) + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif _points_equal(_patient1[i - 1], _patient2[j - 1], eps):
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        return L[len(_patient1), len(_patient2)]

    @classmethod
    def meth_dynamic_time_warping(
        cls,
        _patient1,
        _patient2,
        _distance_metric="euclidian",
    ):
        """
        Computes Dynamic time warping distance between two patients. Dyanmic time
        warping is robust to shifting but this implementation maximizes acceptable
        shifting to 20% of the length of the longest signal.

        The cost function is a distance between points. By default we use the
        euclidian distance. Implemented distances: ["euclidian"]
        """
        n, m = len(_patient1), len(_patient2)

        # No considerar un shifting major al 20% de l'array mes llarg
        window = int(max(n, m) * 0.2)
        w = np.max([window, abs(n - m)])

        # Initialize matrix full of inf
        dtw_matrix = np.zeros((n + 1, m + 1))
        for i in range(n + 1):
            for j in range(m + 1):
                dtw_matrix[i, j] = np.inf

        # Set the possible values to 0
        dtw_matrix[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
                dtw_matrix[i, j] = 0

        # Calculate distances
        for i in range(1, n + 1):
            for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
                cost = abs(distance.euclidean(_patient1[i - 1], _patient2[j - 1]))

                # take last min from a square box
                last_min = np.min(
                    [
                        dtw_matrix[i - 1, j],
                        dtw_matrix[i, j - 1],
                        dtw_matrix[i - 1, j - 1],
                    ],
                )
                dtw_matrix[i, j] = cost + last_min

        return dtw_matrix[n, m]


class DimensionsReducer(_Curator):
    @classmethod
    def reduce(cls, method, features):
        cls_method = f"meth_{method}"
        if cls_method not in cls.methods():
            raise ValueError(f"Method '{method}' is not valid!")

        return getattr(cls, cls_method)(features)

    @classmethod
    def meth_pca(cls, features):
        pca = PCA(n_components=4)
        projection = pca.fit_transform(features)
        return projection, sum(pca.explained_variance_ratio_) * 100

    @classmethod
    def meth_tsne(cls, features):
        tsne = TSNE(n_components=2)
        projection = tsne.fit_transform(features)
        return projection, tsne.kl_divergence_


class Cluster(_Curator):

    ACCEPTED_DISTANCES = {
        "meth_dbscan": {
            "distance_algo": {**PAIRWISE_DISTANCES, **{"precomputed": ["distance"]}},
            "cluster_algo": {"auto": []},
        },
        "meth_kmeans": {
            "distance_algo": {"euclidian": []},
            "cluster_algo": {"full": [], "elkan": []},
        },
        "meth_spectral_clustering": {
            "distance_algo": {
                "precomputed": ["distance"],
                "rbf": [],
                "nearest_neighbors": ["n_neighbors"],
            },
            "cluster_algo": {"kmeans": [], "discretize": []},
        },
        "meth_optics": {
            "distance_algo": {**PAIRWISE_DISTANCES, **{"precomputed": ["distance"]}},
            "cluster_algo": {"dbscan": [], "discretize": []},
        },
        "meth_agglomerative_clustering": {
            "distance_algo": {
                "euclidean": [],
                "l1": [],
                "l2": [],
                "manhattan": [],
                "cosine": [],
                "precomputed": ["distance"],
            },
            "cluster_algo": {"ward": [], "complete": [], "average": [], "single": []},
        },
        "meth_gmm": {
            "distance_algo": {"full": [], "tied": [], "diag": [], "spherical": []},
            "cluster_algo": {"gmm": []},
        },
    }

    @classmethod
    def cluster(
        cls, method, distances, distance_algo, cluster_algo, optimize, **kwargs
    ):
        cls_method = f"meth_{method}"
        if cls_method not in cls.methods():
            raise ValueError(f"Method '{method}' is not valid!")

        return getattr(cls, cls_method)(
            distances, distance_algo, cluster_algo, _optimize=optimize, **kwargs
        )

    @classmethod
    def meth_dbscan(
        cls,
        _distances,
        _distance_algo,
        _cluster_algo,
        max_eps,
        min_samples,
        _optimize=False,
        **kwargs,
    ):
        min_samples = float(min_samples)
        max_eps = float(max_eps)
        analyzer = ClusterAnalyzer(_distances)
        dbscan = lambda x: DBSCAN(metric=_distance_algo, min_samples=min_samples, eps=x)
        clusters = analyzer.optimize_hyperparams(
            min_eps=0.01,
            max_eps=max_eps,
            step=0.01,
            cluster_func=dbscan,
        )
        return clusters

    @classmethod
    def meth_agglomerative_clustering(
        cls, _X, _distance_algo, _cluster_algo, n_clusters, _optimize, **kwargs
    ):
        n_clusters = int(n_clusters)
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity=_distance_algo,
            linkage=_cluster_algo,
        )
        clusters = clusterer.fit(_X)
        ClusterAnalyzer.report(clusters)
        return clusters

    @classmethod
    def meth_spectral_clustering(
        cls, _X, _affinity_algo, _cluster_algo, n_clusters, _optimize=False, **kwargs
    ):
        if "n_neighbors" in kwargs:
            kwargs["n_neighbors"] = int(kwargs["n_neighbors"])

        n_clusters = int(n_clusters) if n_clusters != -1 else None

        _X = np.exp(-_X / _X.std())

        clusterer = SpectralClustering(
            affinity=_affinity_algo,
            assign_labels=_cluster_algo,
            n_clusters=n_clusters,
        )
        clusters = clusterer.fit(_X)
        ClusterAnalyzer.report(clusters)
        return clusters

    @classmethod
    def meth_gmm(
        cls, _X, _covariace_algo, _cluster_algo, n_clusters, _optimize=False, **kwargs
    ):
        n_clusters = int(n_clusters) if n_clusters != -1 else None

        if not _optimize:
            clusterer = GaussianMixture(
                n_components=n_clusters,
                covariance_type=_covariace_algo,
                n_init=10,
                **kwargs,
            )
            clusters = clusterer.fit(_X)
            clusters.labels_ = clusters.predict(_X)
            ClusterAnalyzer.report(clusters)
        else:
            bic = []
            cv_types = ["spherical", "tied", "diag", "full"]
            n_components_range = range(1, n_clusters + 1)
            lowest_bic = np.infty
            for cv_type in cv_types:
                for n_clust in n_components_range:
                    gmm = GaussianMixture(
                        n_components=n_clust,
                        covariance_type=cv_type,
                        n_init=10,
                        **kwargs,
                    )
                    gmm.fit(_X)
                    bic.append(gmm.bic(_X))
                    if bic[-1] < lowest_bic:
                        lowest_bic = bic[-1]
                        best_gmm = gmm

            bic = np.array(bic)
            color_iter = itertools.cycle(
                ["navy", "turquoise", "cornflowerblue", "darkorange"],
            )
            clf = best_gmm
            bars = []

            # Plot the BIC scores
            plt.figure(figsize=(8, 6))
            spl = plt.subplot(2, 1, 1)
            for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
                xpos = np.array(n_components_range) + 0.2 * (i - 2)
                bars.append(
                    plt.bar(
                        xpos,
                        bic[
                            i
                            * len(n_components_range) : (i + 1)
                            * len(n_components_range)
                        ],
                        width=0.2,
                        color=color,
                    ),
                )
            plt.xticks(n_components_range)
            plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
            plt.title("BIC score per model")
            xpos = (
                np.mod(bic.argmin(), len(n_components_range))
                + 0.65
                + 0.2 * np.floor(bic.argmin() / len(n_components_range))
            )
            plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
            spl.set_xlabel("Number of components")
            spl.legend([b[0] for b in bars], cv_types)

            # Plot the winner
            splot = plt.subplot(2, 1, 2)
            Y_ = clf.predict(_X)
            for i, (mean, cov, color) in enumerate(
                zip(
                    clf.means_,
                    clf.covariances_,
                    color_iter,
                ),
            ):
                # Imports: standard library
                import code

                code.interact(local=locals())
                v, w = linalg.eigh(cov)
                if not np.any(Y_ == i):
                    continue
                plt.scatter(_X[Y_ == i, 0], _X[Y_ == i, 1], 0.8, color=color)

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan2(w[0][1], w[0][0])
                angle = 180.0 * angle / np.pi  # convert to degrees
                v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.5)
                splot.add_artist(ell)

            plt.xticks(())
            plt.yticks(())
            plt.title("Selected GMM: full model, 2 components")
            plt.subplots_adjust(hspace=0.35, bottom=0.02)
            plt.savefig("optimize.png")

        return clusters

    @classmethod
    def meth_optics(
        cls, _X, _distance_algo, _cluster_algo, min_samples, _optimize=False, **kwargs
    ):
        """
        Clusters using OPTICS algorithm. Its closely related to DBSCAN but better
        suited for large databases and with less reliance on 'eps' hyperparam.

        Hyperparameters:
        :param min_samples: min samples on a cluster to consider it as such.
        Can be an absolute value (int>1) or a fraction of the total number of points (float [0,1])
        :param cluster_method: either DBSCAN or xi

        *Internal params*
        :param _X: either distance matrix (if metric=precomputed) or sample-features matrix
        :param _distance_algo: metric used to compute distance. Can be precomputed
        :param _optimize: optimize hyperparams instead of cluster
        :param kwargs: any other additional parameter needed

        """

        min_samples = int(min_samples) if min_samples > 1 else min_samples

        clusterer = OPTICS(
            min_samples=min_samples,
            metric=_distance_algo,
            cluster_method=_cluster_algo,
        )
        clusters = clusterer.fit(_X)
        ClusterAnalyzer.report(clusters)
        return clusters

    @classmethod
    def meth_kmeans(
        cls, _X, _distance_algo, _cluster_algo, n_clusters, _optimize=False, **kwargs
    ):
        n_clusters = int(n_clusters)
        if not _optimize:
            kmeans = KMeans(
                n_clusters=n_clusters, algorithm=_cluster_algo, n_jobs=4, **kwargs
            )
            clusters = kmeans.fit(_X)
            # ClusterAnalyzer.report(clusters)

        else:
            distorsions = []
            inertias = []
            silhouettes = []
            gaps = []

            k_values = range(2, n_clusters + 1)
            for k in k_values:
                print(f"Trying with {k} cluster...")
                kmeans = KMeans(n_clusters=k, algorithm=_cluster_algo)
                preds = kmeans.fit_predict(_X)

                # Calculate distortion
                distorsion = (
                    sum(
                        np.min(
                            distance.cdist(_X, kmeans.cluster_centers_, "euclidean"),
                            axis=1,
                        ),
                    )
                    / _X.shape[0]
                )

                # Calculate inertia
                inertia = kmeans.inertia_

                # Calculate silhouette
                silhouette = silhouette_score(_X, preds)

                # Calculate Gap statistic
                reference_inertias = []
                for i in range(50):
                    random_data = np.random.random_sample(size=_X.shape)
                    km = KMeans(n_clusters=k, algorithm=_cluster_algo).fit(random_data)
                    reference_inertias.append(km.inertia_)
                reference_inertia = np.mean(np.array(reference_inertias))
                gap = np.log(reference_inertia) - np.log(inertia)

                distorsions.append(distorsion)
                inertias.append(inertia)
                silhouettes.append(silhouette)
                gaps.append(gap)

            fig, axes = plt.subplots(2, 2)
            axes[0, 0].plot(k_values, distorsions)
            axes[0, 0].set_title("Elbow method - Distorsions")
            axes[0, 1].plot(k_values, inertias)
            axes[0, 1].set_title("Elbow method - Inertias")
            axes[1, 0].plot(k_values, silhouettes)
            axes[1, 0].set_title("Silhouette method")
            axes[1, 1].plot(k_values, gaps)
            axes[1, 1].set_title("Gap method")
            fig.xlabel = "k values"
            fig.xticks = k_values
            plt.savefig("optimization.png")

            clusters = cls.meth_kmeans(
                _X,
                _distance_algo,
                _cluster_algo,
                3,
                _optimize=False,
            )

        return clusters


class ClusterAnalyzer:
    def __init__(self, distances):
        self.distances = distances

    def optimize_hyperparams(self, min_eps, max_eps, step, cluster_func):
        max_clusters = 0
        max_noise = 10000
        max_clusters_eps = 0
        max_clusters_obj = None

        print("***********************")
        print("Dist matrix")
        print("Max: ", self.distances.flatten().max())
        print("Mean: ", self.distances.flatten().mean())
        print("Min: ", self.distances.flatten().min())
        print("***********************")

        for eps in np.arange(min_eps, max_eps + step, step):
            clusters = cluster_func(eps).fit(self.distances)
            n_clusters, n_noise = self.n_clusters(clusters)
            print(f"Trying with eps={eps} I got {n_clusters} clusters")
            if n_clusters > max_clusters:
                max_clusters = n_clusters
                max_clusters_eps = eps
                max_clusters_obj = clusters
            elif n_clusters == max_clusters:
                if n_noise < max_noise:
                    max_clusters = n_clusters
                    max_clusters_eps = eps
                    max_clusters_obj = clusters

        print("***************************")
        print("Max clusters: ", max_clusters)
        print("Max clusters eps: ", max_clusters_eps)
        print("***************************")
        print(self.report(max_clusters_obj))
        return max_clusters_obj

    @staticmethod
    def n_clusters(clusters):
        labels = clusters.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        return n_clusters, n_noise

    @classmethod
    def report(cls, clusters):
        labels = clusters.labels_
        n_clusters, n_noise = cls.n_clusters(clusters)
        print("Estimated number of clusters: %d" % n_clusters)
        print("Estimated number of noise points: %d" % n_noise)
        print("Points in each cluster: ")
        for clust_n in set(labels):
            print(f"\tCluster {clust_n}: {labels[labels == clust_n].size}")
