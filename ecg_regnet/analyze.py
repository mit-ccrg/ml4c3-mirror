# Imports: standard library
import os
import glob
from typing import Any, Dict, List

# Imports: third party
import ray
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from ray import tune
from data import DOWNSTREAM_SIZES
from matplotlib import pyplot as plt
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    mean_squared_error,
    explained_variance_score,
)
from fully_train_pretrained import _get_training_dfs


def analyze_results(results_folder: str, output_folder: str) -> pd.DataFrame:
    os.makedirs(output_folder, exist_ok=True)
    df = tune.Analysis(results_folder).dataframe()
    training_dfs = _get_training_dfs(df)
    best_val_loss = [tdf["val_loss"].min() for tdf in training_dfs]
    best_val_loss_col = "best validation loss"
    df[best_val_loss_col] = best_val_loss
    cleaned_df = df.copy()
    cutoff = 7  # there are 7 standardized objectives with MSE loss -> guessing 0 should give loss ~= 7
    cleaned_df = cleaned_df.loc[cleaned_df[best_val_loss_col] < cutoff]

    config_cols = [col for col in df.columns if "config" in col]
    print(f"Best validation loss ({df[best_val_loss_col].min():.2f}) found with config")
    print(df[config_cols].loc[df[best_val_loss_col].idxmin()])

    # plot training curves
    plt.figure(figsize=(15, 15))
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(10, 5))
    for tdf in training_dfs:
        ax1.plot(tdf["epoch"], tdf["val_loss"])
    ax1.set_ylabel("validation loss")
    ax1.set_xlabel("epoch")

    for tdf in training_dfs:
        ax2.plot(tdf["epoch"], tdf["loss"])
    ax2.set_ylabel("training loss")
    ax2.set_xlabel("epoch")
    plt.savefig(os.path.join(output_folder, f"loss_training_curves.png"), dpi=200)

    # plot best validation loss vs. config columns
    for col in config_cols:
        if df[col].dtype == "object":
            print("Skipping plotting", col)
            continue
        cleaned_col = col.replace("config/", "").replace("_", " ")
        plt.figure(figsize=(7, 7))
        if cleaned_df[col].nunique() > len(cleaned_df) / 2:
            x = cleaned_df[col].dropna()
            vals, bins = pd.qcut(
                x + np.random.randn(len(x)) * 1e-5,
                5,
                labels=False,
                retbins=True,
            )
            xtics = np.round((bins[:-1] + bins[1:]) / 2, 2)  # bin centers
            cleaned_df[col] = xtics[vals]
        best_across_runs = (
            cleaned_df.groupby(col)[best_val_loss_col].min().reset_index()
        )
        plt.scatter(
            best_across_runs[col],
            best_across_runs[best_val_loss_col],
            c="k",
            s=5,
        )
        plt.scatter(df[col], df[best_val_loss_col], c="gray")
        plt.plot(
            best_across_runs[col],
            best_across_runs[best_val_loss_col],
            c="k",
            linestyle="--",
        )
        plt.xlabel(cleaned_col)
        plt.savefig(
            os.path.join(output_folder, f"loss_vs_{cleaned_col.replace(' ', '_')}"),
            dpi=200,
        )
        plt.close("all")

    return df


@ray.remote(num_cpus=1)
def _subsample_cts(actual, pred):
    idx = np.random.randint(0, len(actual), len(actual))
    try:
        met = -mean_squared_error(actual[idx], pred[idx])
        if np.abs(met) > 1000:
            return np.nan
        return met
    except ValueError:
        return np.nan


@ray.remote(num_cpus=1)
def _subsample_cat(actual, pred):
    idx = np.random.randint(0, len(actual), len(actual))
    return roc_auc_score(actual[idx], pred[idx])


def _split_description(description: str) -> Dict[str, Any]:
    split = description.split(" ")
    out = {}

    label_idx = split.index("labels")
    out["n downstream labels"] = int(split[label_idx + 1].replace(".", ""))

    loss_idx = split.index("loss")
    out["pretraining loss"] = float(split[loss_idx + 1][:-1])

    model_idx = split.index("model.")
    out["task"] = split[model_idx - 1]

    out["pretrained"] = " ".join(split[: model_idx - 1])
    return out


def analyze_inference(inference_folder: str, output_folder: str) -> pd.DataFrame:
    """for now assumes cts predictions"""
    csvs = glob.glob(f"{inference_folder}/*.csv")
    out = []
    sns.set_theme(style="dark")
    os.makedirs(output_folder, exist_ok=True)
    for csv in csvs:
        description = os.path.splitext(os.path.basename(csv))[0]
        df = pd.read_csv(csv)
        for col in df.columns:
            if "actual" not in col:
                continue
            task = col.replace("_actual", "")
            pred = df[task + "_pred"]
            actual = df[col]
            remote_metric = _subsample_cts if df[col].nunique() > 5 else _subsample_cat
            print(f"Sampling metric for {description} {task}")
            metric = ray.get([remote_metric.remote(actual, pred) for _ in range(1000)])
            print("Done sampling")
            mean = np.mean(metric)
            lo, hi = np.quantile(metric, [0.025, 0.975])
            # fig = plt.figure(figsize=(7, 7))
            # plt.plot([pred.min(), pred.max()], [pred.min(), pred.max()], c='b', alpha=.5)
            # sns.kdeplot(x=actual, y=pred, levels=5, color='red', linewidths=1)
            # plt.scatter(x=actual, y=pred, s=1, c='k', marker='.')
            # plt.xlabel('actual')
            # plt.ylabel('pred')
            # plt.title(f'{description}\n{task} $R^2$ {mean:.3f} [{lo:.3f}, {hi:.3f}]')
            # plt.savefig(os.path.join(output_folder, f'{description}_{task}.png'), dpi=200)
            # fig.close()
            description = _split_description(description)
            metrics = {
                "metric": mean,
                "metric 2.5 percentile": lo,
                "metric 97.5 percentile": hi,
            }
            out.append({**description, **metrics})
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(output_folder, "metrics.csv"), index=False)

    # plot results by task
    for task, data in out.groupby("task"):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(task)
        best = data.sort_values(["n downstream labels", "metric"]).drop_duplicates(
            ["pretrained", "n downstream labels"],
            keep="last",
        )
        for pretrained, data in best.groupby("pretrained"):
            lo_error = data["metric"] - data["metric 2.5 percentile"]
            hi_error = data["metric 97.5 percentile"] - data["metric"]
            error = [lo_error, hi_error]
            ax.errorbar(
                data["n downstream labels"],
                data["metric"],
                label=f"{pretrained}",
                yerr=error,
                alpha=0.7,
            )
        ax.set_xlabel("n downstream labels")
        ax.set_xscale("log")
        ax.set_xticks(DOWNSTREAM_SIZES)
        ax.set_xticklabels(DOWNSTREAM_SIZES, rotation=90)
        ax.set_ylabel("metric")
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend()
        fig.savefig(
            os.path.join(output_folder, f"{task}_results.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # plot results by pretraining loss
    for (task, pretrained), data in out.groupby(["task", "pretrained"]):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(f"{task} Pretrained: {pretrained}")
        data = data.sort_values("n downstream labels")
        for pretrain_loss, data in data.groupby("pretraining loss"):
            lo_error = data["metric"] - data["metric 2.5 percentile"]
            hi_error = data["metric 97.5 percentile"] - data["metric"]
            error = [lo_error, hi_error]
            ax.errorbar(
                data["n downstream labels"],
                data["metric"],
                label=f"Pretraining loss {pretrain_loss}",
                yerr=error,
                alpha=0.7,
            )
        ax.set_xlabel("n downstream labels")
        ax.set_xscale("log")
        ax.set_xticks(DOWNSTREAM_SIZES)
        ax.set_xticklabels(DOWNSTREAM_SIZES, rotation=90)
        ax.set_ylabel("metric")
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend()
        fig.savefig(
            os.path.join(
                output_folder,
                f"{task}_pretrained_{pretrained}_model_compare.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    return out
