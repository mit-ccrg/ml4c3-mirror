# Imports: standard library
import os
from typing import List

# Imports: third party
import numpy as np
import pandas as pd
import seaborn as sns
from ray import tune
from matplotlib import pyplot as plt
from fully_train_pretrained import _get_training_dfs


def analyze_results(results_folder: str, output_folder: str) -> pd.DataFrame:
    os.makedirs(output_folder, exist_ok=True)
    df = tune.Analysis(results_folder).dataframe()
    training_dfs = _get_training_dfs(df)
    best_val_loss = [tdf["val_loss"].min() for tdf in training_dfs]
    best_val_loss_col = "best validation loss"
    df[best_val_loss_col] = best_val_loss
    cleaned_df = df.copy()
    cutoff = 10  # there are 7 standardized objectives with MSE loss -> guessing 0 should give loss ~= 7
    cleaned_df = cleaned_df.loc[cleaned_df[best_val_loss_col] < cutoff]

    config_cols = [col for col in df.columns if "config" in col]
    print(f"Best validation loss ({df[best_val_loss_col].min():.2f}) found with config")
    print(df[config_cols].loc[df[best_val_loss_col].idxmin()])

    # plot training curves
    plt.figure(figsize=(15, 15))
    for tdf in training_dfs:
        if (tdf["val_loss"] > cutoff).any():
            continue
        plt.plot(tdf["epoch"], tdf["val_loss"])
    tdf = training_dfs[np.nanargmin(best_val_loss)]
    plt.plot(
        tdf["epoch"],
        tdf["val_loss"],
        linestyle="--",
        c="k",
        label="best model",
        linewidth=6,
    )
    plt.legend()
    plt.ylabel("validation loss")
    plt.xlabel("epoch")
    plt.ylim(0, 5)
    plt.savefig(os.path.join(output_folder, f"val_loss_training_curves.png"), dpi=200)

    plt.figure(figsize=(15, 15))
    for tdf in training_dfs:
        if (tdf["val_loss"] > cutoff).any():
            continue
        plt.plot(tdf["epoch"], tdf["loss"])
    tdf = training_dfs[np.nanargmin(best_val_loss)]
    plt.plot(
        tdf["epoch"],
        tdf["loss"],
        linestyle="--",
        c="k",
        label="best model",
        linewidth=6,
    )
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.ylim(0, 5)
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
            vals, bins = pd.qcut(x, 10, labels=False, retbins=True)
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
