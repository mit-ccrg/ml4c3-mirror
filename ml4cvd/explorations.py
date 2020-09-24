# Imports: standard library
import os
import csv
import copy
import logging
import argparse
import datetime
import multiprocessing as mp
from typing import List, Tuple
from functools import reduce
from collections import OrderedDict, defaultdict

# Imports: third party
import h5py
import numpy as np
import pandas as pd
import seaborn as sns

# Imports: first party
from ml4cvd.plots import SUBPLOT_SIZE
from ml4cvd.definitions import IMAGE_EXT
from ml4cvd.tensor_generators import train_valid_test_tensor_generators
from ml4cvd.tensormap.TensorMap import (
    TensorMap,
    Interpretation,
    update_tmaps,
    binary_channel_map,
    find_negative_label_and_channel,
)

# fmt: off
# need matplotlib -> Agg -> pyplot
import matplotlib                       # isort:skip
matplotlib.use("Agg")                   # isort:skip
from matplotlib import pyplot as plt    # isort:skip
# fmt: on


def explore(args: argparse.Namespace, save_output: bool = True) -> pd.DataFrame:
    cohort_counts = OrderedDict()

    src_path = args.tensors
    src_name = args.source_name
    src_join = args.join_tensors
    src_cols = None if args.join_tensors is None else list(src_join)
    src_time = args.time_tensor

    ref_path = args.reference_tensors
    ref_name = args.reference_name
    ref_join = args.reference_join_tensors
    ref_cols = None if args.reference_join_tensors is None else list(ref_join)
    ref_start = args.reference_start_time_tensor
    ref_end = args.reference_end_time_tensor

    number_per_window = args.number_per_window
    order_in_window = args.order_in_window
    windows = args.window_name

    match_exact_window = order_in_window is not None
    match_min_window = not match_exact_window
    match_any_window = args.match_any_window
    match_every_window = not match_any_window

    if (args.reference_join_tensors is not None) and (
        args.explore_stratify_label is not None
    ):
        ref_cols.append(args.explore_stratify_label)

    # Ensure all required tmaps are parsed and modified as needed
    tmap_names_to_get = []
    tmap_names_to_get += [] if src_join is None else src_join
    tmap_names_to_get += [] if src_time is None else [src_time]

    tmaps = {}
    for tmap_name in tmap_names_to_get:
        tmaps = update_tmaps(tmap_name=tmap_name, tmaps=tmaps)
    tmaps = args.tensor_maps_in + [tmaps[tmap_name] for tmap_name in tmap_names_to_get]
    required_tmaps = []
    for tmap in tmaps:
        if _tmap_requires_modification_for_explore(tmap=tmap):
            tmap = _modify_tmap_to_return_mean(tmap=tmap)
        required_tmaps.append(tmap)

    df = _tensors_to_df(
        tensor_maps_in=required_tmaps,
        tensor_maps_out=[],
        tensors=args.tensors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        training_steps=args.training_steps,
        validation_steps=args.validation_steps,
        cache_size=args.cache_size,
        balance_csvs=args.balance_csvs,
        mixup_alpha=args.mixup_alpha,
        sample_csv=args.sample_csv,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        output_folder=args.output_folder,
        run_id=args.id,
        export_error=args.explore_export_error,
        export_fpath=args.explore_export_fpath,
        export_generator=args.explore_export_generator,
    )

    # Remove redundant columns for binary labels from df
    df = _remove_redundant_cols(tmaps=required_tmaps, df=df)

    # If time windows are specified, extend reference columns
    use_time = not any(arg is None for arg in [src_time, ref_start, ref_end])

    if use_time:
        if len(ref_start) != len(ref_end):
            raise ValueError(
                f"Invalid time windows, got {len(ref_start)} starts and {len(ref_end)}"
                " ends",
            )
        if order_in_window is None:
            # If not matching exactly N in time window, order_in_window is None
            # make array of blanks so zip doesnt break later
            order_in_window = [""] * len(ref_start)
        elif len(order_in_window) != len(ref_start):
            raise ValueError(
                f"Ambiguous time selection in time windows, got {len(order_in_window)}"
                f" order_in_window for {len(ref_start)} windows",
            )
        if windows is None:
            windows = [str(i) for i in range(len(ref_start))]
        elif len(windows) != len(ref_start):
            raise ValueError(
                f"Ambiguous time window names, got {len(windows)} names for"
                f" {len(ref_start)} windows",
            )
        # Ref start and end are lists of lists, defining time windows
        time_windows = list(zip(ref_start, ref_end))

        # Each time window is defined by a tuples of two lists,
        # where the first list of each tuple defines the start point of the time window
        # and the second list of each tuple defines the end point of the time window
        for start, end in time_windows:
            # Each start/end point list is two elements,
            # where the first element in the list is the name of the time tensor
            # and the second element is the offset to the value of the time tensor

            # Add day offset of 0
            start.append(0)
            end.append(0)

            # parse day offset as int
            start[1] = int(start[1])
            end[1] = int(end[1])

        # Add unique column names to ref_cols
        ref_cols.extend(_cols_from_time_windows(time_windows))

    # If reference_tensors are given, perform cross-reference functionality
    if args.reference_tensors is not None:

        # If path to reference tensors is dir, parse HD5 files
        if os.path.isdir(args.reference_tensors):
            ref_tmaps = {}
            for tm_name in ref_cols:
                ref_tmaps = update_tmaps(tm_name, ref_tmaps)
            ref_tmaps = [ref_tmaps[tm_name] for tm_name in ref_cols]
            df_ref = _tensors_to_df(
                tensor_maps_in=ref_tmaps,
                tensor_maps_out=[],
                tensors=args.reference_tensors,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                training_steps=args.training_steps,
                validation_steps=args.validation_steps,
                cache_size=args.cache_size,
                balance_csvs=args.balance_csvs,
                mixup_alpha=args.mixup_alpha,
                sample_csv=args.sample_csv,
                valid_ratio=args.valid_ratio,
                test_ratio=args.test_ratio,
                train_csv=args.train_csv,
                valid_csv=args.valid_csv,
                test_csv=args.test_csv,
                output_folder=args.output_folder,
                run_id=args.id,
                export_error=args.explore_export_error,
                export_fpath=args.explore_export_fpath,
                export_generator=args.explore_export_generator,
            )
        # Else, path to reference tensors is file (assume CSV)
        else:
            df_ref = pd.read_csv(
                filepath_or_buffer=args.reference_tensors, usecols=ref_cols,
            )

        # Remove rows in df with NaNs for src_join, or type casting fails
        df.dropna(subset=src_join, inplace=True)
        df_ref.dropna(subset=ref_join, inplace=True)

        # Cast source column to ref column type
        df[src_join] = df[src_join].astype(df_ref[ref_join].dtypes[0])

        if use_time:
            src_cols.append(src_time)

        # Remove duplicates defined by join tensors (and time tensor if exists)
        df.drop_duplicates(subset=src_cols, inplace=True, ignore_index=True)
        df_ref.drop_duplicates(subset=ref_cols, inplace=True, ignore_index=True)
        logging.info("Removed duplicates based on join and time columns")

        # Count total and unique entries in df: source
        cohort_counts = _update_cohort_counts_len_and_unique(
            cohort_counts=cohort_counts, df=df, name=src_name, join_col=src_join,
        )

        # Count total and unique entries in df: reference
        cohort_counts = _update_cohort_counts_len_and_unique(
            cohort_counts=cohort_counts, df=df_ref, name=ref_name, join_col=ref_join,
        )

        # Format datetime cols and remove nan rows
        if use_time:
            df[src_time] = pd.to_datetime(
                df[src_time], errors="coerce", infer_datetime_format=True,
            )
            df.dropna(subset=[src_time], inplace=True)

            for ref_time in _cols_from_time_windows(time_windows):
                df_ref[ref_time] = pd.to_datetime(
                    df_ref[ref_time], errors="coerce", infer_datetime_format=True,
                )
            df_ref.dropna(subset=_cols_from_time_windows(time_windows), inplace=True)

            # Iterate through time_windows, add day offsets to start and end strings,
            # update reference columns, and create new date columns with offsets in
            # reference dataframe
            time_windows_parsed = []
            for start, end in time_windows:

                # Start time
                name, days = start[0], start[1]
                start_name_offset = _offset_ref_name(name=name, days=days)
                ref_cols = _offset_ref_cols(name=name, days=days, ref_cols=ref_cols)
                df_ref = _offset_ref_df(
                    name=name, days=days, name_offset=start_name_offset, df=df_ref,
                )

                # End time
                name, days = end[0], end[1]
                end_name_offset = _offset_ref_name(name=name, days=days)
                ref_cols = _offset_ref_cols(name=name, days=days, ref_cols=ref_cols)
                df_ref = _offset_ref_df(
                    name=name, days=days, name_offset=end_name_offset, df=df_ref,
                )

                # Append list with tuple of offset start and end names
                time_windows_parsed.append((start_name_offset, end_name_offset))

            logging.info("Formatted datetime columns and removed unparsable rows")

        # Intersect with input tensors df on specified keys
        df_cross = df.merge(
            df_ref, how="inner", left_on=src_join, right_on=ref_join,
        ).sort_values(src_cols)
        logging.info("Cross-referenced using src and ref join tensors")

        # Calculate cohort counts on crossed dataframe
        cohort_counts = _update_cohort_counts_crossed_dataframe(
            cohort_counts=cohort_counts,
            df=df_cross,
            src_name=src_name,
            ref_name=ref_name,
            src_cols=src_cols,
            ref_cols=ref_cols,
            src_join=src_join,
            ref_join=ref_join,
        )

    # If reference_tensor given, we have cross-referenced df
    # if not, just have source df

    # Select time subsets and generate subset dfs
    # (info either in source tensors, or reference_tensors)
    if use_time:

        # Get list of dfs with >=1 occurrence per time window
        dfs_cross_window = _get_df_per_window(
            df=df_cross, src_time=src_time, windows=time_windows_parsed,
        )

        # Get list of dfs with >=N hits in any time window
        dfs_n_or_more_hits_any_window = _get_df_n_or_more_hits_any_window(
            dfs=dfs_cross_window,
            windows=time_windows_parsed,
            src_join=src_join,
            num_per_window=args.number_per_window,
        )

        # --------------------------------------------------------------------------- #
        # Scenario 1: match_min_window and match_any_window
        # --------------------------------------------------------------------------- #
        if match_min_window and match_any_window:
            df_aggregated = _aggregate_time_windows(
                dfs=dfs_n_or_more_hits_any_window, windows=windows, src_cols=src_cols,
            )
            logging.info(
                f"Cross-referenced so event occurs {number_per_window}+ times in"
                " any time window",
            )
            title = f"{number_per_window}+ in any window"

        # --------------------------------------------------------------------------- #
        # Scenario 2: match_min_window and match_every_window
        # --------------------------------------------------------------------------- #
        # Given list of dfs with >=N occurrences in any time window,
        # isolate rows that have join_tensors across all windows
        if match_min_window and match_every_window:
            dfs = _intersect_time_windows(
                dfs=dfs_n_or_more_hits_any_window, src_join=src_join,
            )
            df_aggregated = _aggregate_time_windows(
                dfs=dfs, windows=windows, src_cols=src_cols,
            )
            logging.info(
                f"Cross-referenced so unique event occurs {number_per_window}+ times in"
                " all windows",
            )
            title = f"{number_per_window}+ in every window"

        # --------------------------------------------------------------------------- #
        # Scenario 3: match_exact_window
        # --------------------------------------------------------------------------- #
        # Get exactly N occurrences in any time window
        if match_exact_window:
            dfs = [
                _get_df_exactly_n_any_window(
                    df=df,
                    order=order,
                    start=start,
                    end=end,
                    src_join=src_join,
                    number_per_window=number_per_window,
                )
                for df, order, (start, end) in zip(
                    dfs_n_or_more_hits_any_window, order_in_window, time_windows_parsed,
                )
            ]
            # What do we do if match_exact_window, but not match_any_window?

        # ?
        if match_exact_window and match_any_window:
            df_aggregated = _aggregate_time_windows(
                dfs=dfs, windows=windows, src_cols=src_cols,
            )
            logging.info(
                f"Cross-referenced so unique event occurs exactly {number_per_window} times in any window",
            )
            title = f"{number_per_window} in any window"

    # Add aggregated df to list of window dfs and
    # adjust title depending on cross-reference and time windowing
    if args.reference_tensors == None:
        title = "all"
        dfs = [df]
        windows = ["all"]
    else:
        if use_time:
            title = title.replace(" ", "_")
            dfs.append(df_aggregated)
            windows.append("all_windows")
        else:
            title = "crossref"
            dfs = [df_cross]
            windows = ["all"]

    # Iterate through time window names and dataframes
    for window, df_window in zip(windows, dfs):

        # If stratified, save label distribution for this window
        if args.explore_stratify_label is not None:
            if save_output:
                _save_label_distribution(
                    df=df_window,
                    title=title,
                    window=window,
                    stratify_label=args.explore_stratify_label,
                    output_folder=args.output_folder,
                    output_id=args.id,
                )

        # Calculate cross-referenced cohort counts
        cohort_counts = _update_cohort_counts(
            cohort_counts=cohort_counts,
            df=df_window,
            src_name=src_name,
            src_cols=src_cols,
            src_join=src_join,
            window=window,
            title=title,
        )

        # Iterate over union and intersect of df and calculate summary statistics
        for df, union_or_intersect in zip(
            [df_window, df_window.dropna()], ["union", "intersect"],
        ):
            # Get list of unique labels from df
            labels = ["all"]
            if args.explore_stratify_label is not None:
                labels.extend(
                    [label for label in df[args.explore_stratify_label].unique()],
                )

            # Iterate through interpretations
            for interpretation in [
                Interpretation.CONTINUOUS,
                Interpretation.CATEGORICAL,
                Interpretation.LANGUAGE,
            ]:
                # Initialize list of summary stats dicts for this interpretation
                stats_all = []
                stats_keys = []

                # Iterate over required tmaps for that interpretation
                for tm in [
                    tm for tm in required_tmaps if tm.interpretation is interpretation
                ]:

                    # Plot continuous histograms of tensors for union
                    if (union_or_intersect == "union") and (
                        interpretation is Interpretation.CONTINUOUS
                    ):
                        if save_output:
                            _plot_histogram_continuous_tensor(
                                tmap_name=tm.name,
                                df=df,
                                output_folder=args.output_folder,
                                output_id=args.id,
                                window=window,
                                stratify_label=args.explore_stratify_label,
                            )

                    # Iterate over label and isolate those df rows if stratified
                    for label in labels:
                        if isinstance(label, float) and np.isnan(label):
                            continue
                        df_label = (
                            df
                            if label == "all"
                            else df[df[args.explore_stratify_label] == label]
                        )
                        key_suffix = (
                            ""
                            if label == "all"
                            else f"_{args.explore_stratify_label}={label}"
                        )
                        # Calculate summary statistics for that tmap
                        if tm.channel_map:
                            # If the channel map is binary, we only want to parse the
                            # positive label and ignore the redundant negative label
                            if binary_channel_map(tm=tm):
                                key = tm.name
                                stats = _calculate_summary_stats(
                                    df=df_label, key=key, interpretation=interpretation,
                                )
                                stats_all.append(stats)
                                stats_keys.append(f"{tm.name}{key_suffix}")
                            # If not binary, parse each channel map
                            else:
                                for cm in tm.channel_map:
                                    key = f"{tm.name}_{cm}"
                                    stats = _calculate_summary_stats(
                                        df=df_label,
                                        key=key,
                                        interpretation=interpretation,
                                    )
                                    stats_all.append(stats)
                                    stats_keys.append(f"{tm.name}_{cm}{key_suffix}")
                        else:
                            key = tm.name
                            stats = _calculate_summary_stats(
                                df=df_label, key=key, interpretation=interpretation,
                            )
                            stats_all.append(stats)
                            stats_keys.append(f"{tm.name}{key_suffix}")

                # Convert summary statistics into dataframe and save to CSV
                if save_output and len(stats_all) > 0:
                    df_stats = pd.DataFrame(data=stats_all, index=[stats_keys])
                    fpath = os.path.join(
                        args.output_folder,
                        args.id,
                        f"stats_{interpretation}_{window}_{union_or_intersect}.csv",
                    )
                    df_stats.round(3).to_csv(fpath)
                    logging.info(
                        f"{window} / {union_or_intersect} / {interpretation} tmaps: saved summary stats to {fpath}",
                    )

    # Save tensors, including column with window name
    fpath = os.path.join(args.output_folder, args.id, f"tensors_union.csv")

    # Time-windowed
    if use_time:
        if save_output:
            df_aggregated.set_index(src_join, drop=True).to_csv(fpath)
        return_df = df_aggregated
    else:
        # No cross-reference
        if args.reference_tensors is None:
            if save_output:
                df.to_csv(fpath, index=False)
            return_df = df
        # Cross-reference
        else:
            if save_output:
                df_cross.set_index(src_join, drop=True).to_csv(fpath)
            return_df = df_cross
    if save_output:
        logging.info(f"Saved tensors to {fpath}")

    # Save cohort counts to CSV
    fpath = os.path.join(args.output_folder, args.id, "cohort_counts.csv")
    df_cohort_counts = pd.DataFrame.from_dict(
        cohort_counts, orient="index", columns=["count"],
    )
    df_cohort_counts = df_cohort_counts.rename_axis("description")
    if save_output:
        df_cohort_counts.to_csv(fpath)
        logging.info(f"Saved cohort counts to {fpath}")
    return return_df


def _remove_redundant_cols(tmaps: List[TensorMap], df: pd.DataFrame) -> pd.DataFrame:
    """Given a list of tensor maps, and a dataframe with tensors, checks categorical
    # tensor maps, parses binary tmaps to find the negative label, removes it from the
    dataframe, and simplifies the remaining positive column name"""
    if Interpretation.CATEGORICAL in [tm.interpretation for tm in tmaps]:
        for tm in [
            tm for tm in tmaps if tm.interpretation is Interpretation.CATEGORICAL
        ]:
            # If two-element channel map, with "no_" in it, it is probably binary
            if binary_channel_map(tm=tm):
                # Find name of redundant label so we can drop it
                negative_label, _ = find_negative_label_and_channel(tm.channel_map)
                df.drop(f"{tm.name}_{negative_label}", axis=1, inplace=True)

                # Find name of non-redundant label
                positive_label = [
                    cm for cm in tm.channel_map if cm is not negative_label
                ][0]

                # Simplify name of positive label column
                df.rename(
                    columns={f"{tm.name}_{positive_label}": f"{tm.name}"}, inplace=True,
                )
    return df


def _plot_histogram_continuous_tensor(
    tmap_name: str,
    df: pd.DataFrame,
    output_folder: str,
    output_id: str,
    window: str,
    stratify_label: str,
):
    sns.set_context("talk")
    plot_width = SUBPLOT_SIZE * 1.3 if stratify_label is not None else SUBPLOT_SIZE
    plot_height = SUBPLOT_SIZE * 0.6
    fig = plt.figure(figsize=(plot_width, plot_height))
    ax = plt.gca()
    plt.title(f"{tmap_name}: n={len(df)}")
    legend_labels = []

    # Iterate through unique values of stratify label
    if stratify_label is not None:
        for stratify_label_value in df[stratify_label].unique():
            n = sum(df[stratify_label] == stratify_label_value)
            legend_str = f"{stratify_label}={stratify_label_value} (n={n})"
            data = df[df[stratify_label] == stratify_label_value][tmap_name]
            kde = not np.isclose(data.var(), 0)
            sns.distplot(
                a=data, label=legend_str, kde=kde,
            )
            plt.xlabel("Value")
            plt.ylabel("Probability")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        data = df[tmap_name].to_numpy()
        kde = not np.isclose(data.var(), 0)
        sns.distplot(data, kde=kde)
        plt.xlabel("Value")
        plt.ylabel("Probability")
    fpath = os.path.join(
        output_folder, output_id, f"histogram_{tmap_name}_{window}{IMAGE_EXT}",
    )
    plt.savefig(fpath, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved histogram of {tmap_name} to {fpath}")


def _calculate_summary_stats(df: pd.DataFrame, key: str, interpretation) -> dict:
    stats = dict()
    if interpretation is Interpretation.CONTINUOUS:
        stats["min"] = df[key].min()
        stats["max"] = df[key].max()
        stats["mean"] = df[key].mean()
        stats["median"] = df[key].median()
        mode = df[key].mode()
        stats["mode"] = mode[0] if len(mode) != 0 else np.nan
        stats["variance"] = df[key].var()
        stats["stdev"] = df[key].std()
        stats["count"] = df[key].count()
        stats["missing"] = df[key].isna().sum()
        stats["total"] = len(df[key])
        stats["missing_fraction"] = stats["missing"] / stats["total"]
    elif interpretation is Interpretation.CATEGORICAL:
        num_key = np.count_nonzero(df[key] == 1)
        stats[f"count"] = num_key
        stats[f"fraction"] = num_key / len(df[key])
        stats["total"] = len(df[key])
    elif interpretation is Interpretation.LANGUAGE:
        stats["count"] = df[key].count()
        stats["count_fraction"] = stats["count"] / df[key].shape[0]
        if stats["count"] == 0:
            stats["count_unique"] = 0
        else:
            stats["count_unique"] = len(df[key].value_counts())
        stats["missing"] = df[key].isna().sum()
        stats["missing_fraction"] = stats["missing"] / len(df[key])
        stats["total"] = len(df[key])
    else:
        raise ValueError(f"Invalid interpretation: {interpretation}")
    return stats


def _get_df_exactly_n_any_window(
    df: pd.DataFrame,
    order: str,
    start: str,
    end: str,
    src_join: List[str],
    number_per_window: int,
) -> pd.DataFrame:
    if order == "newest":
        df = df.groupby(src_join + [start, end]).tail(number_per_window)
    elif order == "oldest":
        df = df.groupby(src_join + [start, end]).head(number_per_window)
    elif order == "random":
        df = df.groupby(src_join + [start, end]).apply(
            lambda g: g.sample(number_per_window),
        )
    else:
        raise NotImplementedError(
            f"Ordering for which rows to use in time window unknown: '{order}'",
        )
    return df.reset_index(drop=True)


def _update_cohort_counts(
    cohort_counts: dict,
    df: pd.DataFrame,
    src_name: str,
    src_cols: List,
    src_join: str,
    window: str,
    title: str,
) -> dict:
    cohort_counts[f"{window}: {src_name}"] = len(df)

    cohort_counts[f"{window}: {src_name} (unique by {src_cols}) / {title}"] = len(
        df.drop_duplicates(subset=src_cols),
    )

    logging.info(
        f"Updated cross-reference counts for window ({window}) and title ({title})",
    )
    return cohort_counts


def _save_label_distribution(
    df: pd.DataFrame,
    title: str,
    window: str,
    stratify_label: str,
    output_folder: str,
    output_id: str,
):
    # Get counts for each value of stratify_label in df
    label_counts = df[stratify_label].value_counts(dropna=False).to_dict()
    labels = [f"{stratify_label}={label}" for label in label_counts.keys()]
    labels.append("all")

    counts = [count for count in label_counts.values()]
    counts.append(df.shape[0])

    fractions = [count / df.shape[0] for count in counts]

    # Combine lists into dataframe
    df_label_distribution = pd.DataFrame(
        data=[counts, fractions], index=["count", "fraction"], columns=labels,
    ).T
    fpath = os.path.join(
        output_folder, output_id, f"label_distribution_{title}_{window}.csv",
    )
    df_label_distribution.round(3).to_csv(fpath)
    logging.info(f"Saved {fpath}")


def _intersect_time_windows(
    dfs: List[pd.DataFrame], src_join: list,
) -> List[pd.DataFrame]:
    for i, df in enumerate(dfs):
        if i == 0:
            intersect = df[src_join].drop_duplicates()
        else:
            intersect = intersect.merge(df[src_join].drop_duplicates())
    join_tensor_intersect = reduce(
        lambda a, b: a.merge(b),
        [pd.DataFrame(df[src_join].drop_duplicates()) for df in dfs],
    )
    # Filter list of dfs to only rows with join_tensors across all windows
    dfs_intersect = [df.merge(join_tensor_intersect) for df in dfs]
    return dfs_intersect


def _aggregate_time_windows(
    dfs: List[pd.DataFrame], windows: list, src_cols: list,
) -> pd.DataFrame:
    """Aggregate list of dataframes (one per time window) back into one dataframe with column indicating the time window index"""
    # Add time window column and value to each df in list
    for df, window in zip(dfs, windows):
        if "time_window" not in df:
            df["time_window"] = window
    # Concatenate dfs back together
    df_together = pd.concat(dfs, ignore_index=True).sort_values(
        by=src_cols + ["time_window"], ignore_index=True,
    )
    return df_together


def _get_df_per_window(
    df: pd.DataFrame, windows: List[Tuple], src_time: str,
) -> List[pd.DataFrame]:
    return [
        df[(df[start] < df[src_time]) & (df[src_time] < df[end])]
        for start, end in windows
    ]


def _get_df_n_or_more_hits_any_window(
    dfs: List[pd.DataFrame],
    windows: List[Tuple[str, str]],
    src_join: str,
    num_per_window: int,
) -> List[pd.DataFrame]:
    return [
        df.groupby(src_join + [start, end]).filter(lambda g: len(g) >= num_per_window)
        for df, (start, end) in zip(dfs, windows)
    ]


def _update_cohort_counts_len_and_unique(
    cohort_counts: dict, df: pd.DataFrame, name: str, join_col: str,
) -> dict:
    cohort_counts[f"{name} (total rows)"] = len(df)
    cohort_counts[f'{name} (unique {" + ".join(join_col)})'] = len(
        df.drop_duplicates(subset=join_col),
    )
    return cohort_counts


def _update_cohort_counts_crossed_dataframe(
    cohort_counts: dict,
    df: pd.DataFrame,
    src_name: str,
    ref_name: str,
    src_cols: list,
    ref_cols: list,
    src_join: str,
    ref_join: str,
) -> dict:
    cohort_counts[f'{src_name} in {ref_name} (unique {" + ".join(src_cols)})'] = len(
        df.drop_duplicates(subset=src_cols),
    )
    cohort_counts[f'{src_name} in {ref_name} (unique {" + ".join(src_join)})'] = len(
        df.drop_duplicates(subset=src_join),
    )
    cohort_counts[f"{ref_name} in {src_name} (unique joins + times + labels)"] = len(
        df.drop_duplicates(subset=ref_cols),
    )
    cohort_counts[f'{ref_name} in {src_name} (unique {" + ".join(ref_join)})'] = len(
        df.drop_duplicates(subset=ref_join),
    )
    return cohort_counts


def _offset_ref_name(name: str, days: int) -> str:
    if days == 0:
        return name
    name = f"{name}_{days:+}_days"
    return name


def _offset_ref_cols(name: str, days: int, ref_cols: list) -> list:
    if days == 0:
        return ref_cols
    ref_cols.append(f"{name}_{days:+}_days")
    return ref_cols


def _offset_ref_df(
    name: str, days: int, name_offset: str, df: pd.DataFrame,
) -> pd.DataFrame:
    if name_offset not in df:
        df[name_offset] = df[name].apply(lambda x: x + datetime.timedelta(days=days))
    return df


class TensorsToDataFrameParallelWrapper:
    def __init__(
        self,
        tmaps,
        paths,
        num_workers,
        output_folder,
        run_id,
        export_error,
        export_fpath,
        export_generator,
    ):
        self.tmaps = tmaps
        self.paths = paths
        self.num_workers = num_workers
        self.total = len(paths)
        self.output_folder = output_folder
        self.run_id = run_id
        self.export_error = export_error
        self.export_fpath = export_fpath
        self.export_generator = export_generator
        self.chunksize = self.total // num_workers
        self.counter = mp.Value("l", 1)

    def _hd5_to_disk(self, path, gen_name):
        with self.counter.get_lock():
            i = self.counter.value
            if i % 1000 == 0:
                logging.info(f"Parsing {i}/{self.total} ({i/self.total*100:.1f}%) done")
            self.counter.value += 1

        # each worker should write to it's own file
        pid = mp.current_process().pid
        fpath = os.path.join(
            self.output_folder, self.run_id, f"tensors_all_union_{pid}.csv",
        )
        write_header = not os.path.isfile(fpath)

        try:
            with h5py.File(path, "r") as hd5:
                dict_of_tensor_dicts = defaultdict(dict)
                for tm in self.tmaps:
                    shape = tm.shape
                    try:
                        tensors = tm.tensor_from_file(tm, hd5)
                        if tm.time_series_limit is None:
                            tensors = np.array([tensors])
                        for i, tensor in enumerate(tensors):
                            if tensor is None:
                                break
                            try:
                                tensor = tm.postprocess_tensor(
                                    tensor, augment=False, hd5=hd5,
                                )
                                if tm.channel_map:
                                    for cm in tm.channel_map:
                                        dict_of_tensor_dicts[i][
                                            f"{tm.name}_{cm}"
                                        ] = tensor[tm.channel_map[cm]]
                                else:
                                    # If tensor is a scalar, isolate value in array;
                                    # otherwise, retain value as array
                                    if shape[0] == 1:
                                        if isinstance(tensor, np.ndarray):
                                            tensor = tensor.item()
                                    dict_of_tensor_dicts[i][tm.name] = tensor
                            except (
                                IndexError,
                                KeyError,
                                ValueError,
                                OSError,
                                RuntimeError,
                            ) as e:
                                if tm.channel_map:
                                    for cm in tm.channel_map:
                                        dict_of_tensor_dicts[i][
                                            f"{tm.name}_{cm}"
                                        ] = np.nan
                                else:
                                    dict_of_tensor_dicts[i][tm.name] = np.full(
                                        shape, np.nan,
                                    )[0]
                            if self.export_error:
                                dict_of_tensor_dicts[i][f"error_type_{tm.name}"] = type(
                                    e,
                                ).__name__
                    except (
                        IndexError,
                        KeyError,
                        ValueError,
                        OSError,
                        RuntimeError,
                    ) as e:
                        # Most likely error came from tensor_from_file and dict_of_tensor_dicts is empty
                        if tm.channel_map:
                            for cm in tm.channel_map:
                                dict_of_tensor_dicts[0][f"{tm.name}_{cm}"] = np.nan
                        else:
                            dict_of_tensor_dicts[0][tm.name] = np.full(shape, np.nan)[0]

                        if self.export_error:
                            dict_of_tensor_dicts[0][f"error_type_{tm.name}"] = type(
                                e,
                            ).__name__

                for i in dict_of_tensor_dicts:
                    if self.export_fpath:
                        dict_of_tensor_dicts[i]["fpath"] = path
                    if self.export_generator:
                        dict_of_tensor_dicts[i]["generator"] = gen_name

                # write tdicts to disk
                if len(dict_of_tensor_dicts) > 0:
                    keys = dict_of_tensor_dicts[0].keys()
                    with open(fpath, "a") as output_file:
                        dict_writer = csv.DictWriter(output_file, keys)
                        if write_header:
                            dict_writer.writeheader()
                        dict_writer.writerows(dict_of_tensor_dicts.values())
        except OSError as e:
            logging.info(f"OSError {e}")

    def mp_worker(self, worker_idx):
        start = worker_idx * self.chunksize
        end = start + self.chunksize
        if worker_idx == self.num_workers - 1:
            end = self.total
        for path, gen in self.paths[start:end]:
            self._hd5_to_disk(path, gen)

    def run(self):
        workers = []
        for i in range(self.num_workers):
            worker = mp.Process(target=self.mp_worker, args=(i,))
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()


def _tensors_to_df(
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    tensors: str,
    batch_size: int,
    num_workers: int,
    training_steps: int,
    validation_steps: int,
    cache_size: float,
    balance_csvs: List[str],
    mixup_alpha: float = -1.0,
    sample_csv: str = None,
    valid_ratio: float = None,
    test_ratio: float = None,
    train_csv: str = None,
    valid_csv: str = None,
    test_csv: str = None,
    output_folder: str = "",
    run_id: str = "",
    export_error: bool = False,
    export_fpath: bool = False,
    export_generator: bool = False,
) -> pd.DataFrame:
    """
    Create generators, load TMaps, call run method of class that parses tensors from
    HD5 files using TMaps and saves temporary CSVs, set dtypes, consolidate CSVs into
    single dataframe, and return dataframe.
    """
    logging.info("Building generators for specified tensors")
    generators = train_valid_test_tensor_generators(
        tensor_maps_in=tensor_maps_in,
        tensor_maps_out=tensor_maps_out,
        tensors=tensors,
        batch_size=batch_size,
        num_workers=num_workers,
        training_steps=training_steps,
        validation_steps=validation_steps,
        cache_size=cache_size,
        balance_csvs=balance_csvs,
        mixup_alpha=mixup_alpha,
        sample_csv=sample_csv,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
        output_folder=output_folder,
        id=run_id,
    )
    tmaps = [tm for tm in tensor_maps_in]
    paths = [
        (path, gen.name.replace("_worker", ""))
        for gen in generators
        for path in gen.paths
    ]

    TensorsToDataFrameParallelWrapper(
        tmaps=tmaps,
        paths=paths,
        num_workers=num_workers,
        output_folder=output_folder,
        run_id=run_id,
        export_error=export_error,
        export_fpath=export_fpath,
        export_generator=export_generator,
    ).run()

    # Get columns that should have dtype 'string' instead of dtype 'O'
    str_cols = []
    if export_fpath:
        str_cols.extend("fpath")
    if export_generator:
        str_cols.extend("generator")
    for tm in tmaps:
        if tm.interpretation == Interpretation.LANGUAGE:
            str_cols.extend(
                [f"{tm.name}_{cm}" for cm in tm.channel_map]
                if tm.channel_map
                else [tm.name],
            )
        str_cols.append(f"error_type_{tm.name}")
    str_cols = {key: "string" for key in str_cols}

    # Consolidate temporary CSV files into one dataframe
    base = os.path.join(output_folder, run_id)
    temp_files = []
    df_list = []
    for name in os.listdir(base):
        if "tensors_all_union_" in name:
            fpath = os.path.join(base, name)
            _df = pd.read_csv(fpath, dtype=str_cols)
            logging.debug(f"Loaded {fpath} into memory")
            df_list.append(_df)
            logging.debug(f"Appended {fpath} to list of dataframes")
            temp_files.append(fpath)
    df = pd.concat(df_list, ignore_index=True)

    logging.info(
        f"{len(df)} samples extracted from {len(paths)} hd5 files using {len(tmaps)}"
        " tmaps, and consolidated to one DataFrame",
    )

    # Delete temporary files
    for fpath in temp_files:
        os.remove(fpath)
    logging.debug(f"Deleted {len(temp_files)} temporary files")

    return df


def _modify_tmap_to_return_mean(tmap: TensorMap) -> TensorMap:
    """Modifies tm so it returns it's mean unless previous tensor from file fails"""
    new_tm = copy.deepcopy(tmap)
    new_tm.shape = (1,)
    new_tm.interpretation = Interpretation.CONTINUOUS
    new_tm.channel_map = None

    def tff(_: TensorMap, hd5: h5py.File):
        return tmap.tensor_from_file(tmap, hd5).mean()

    new_tm.tensor_from_file = tff
    return new_tm


def _tmap_requires_modification_for_explore(tmap: TensorMap) -> bool:
    """Whether a tmap has to be modified to be used in explore"""
    if tmap.is_continuous:
        return tmap.shape != (1,)
    if tmap.is_categorical:
        return tmap.axes > 1
    if tmap.is_language:
        return False
    return True


def _cols_from_time_windows(time_windows):
    return {time_point[0] for time_window in time_windows for time_point in time_window}


def continuous_explore_header(tm: TensorMap) -> str:
    return tm.name


def categorical_explore_header(tm: TensorMap, channel: str) -> str:
    return f"{tm.name}_{channel}"
