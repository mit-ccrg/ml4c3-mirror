# Imports: standard library
import os
from typing import List

# Imports: third party
import numpy as np
import pandas as pd
import seaborn as sns
from ray import tune
from matplotlib import pyplot as plt


def get_training_dfs(analysis_df: pd.DataFrame) -> List[pd.DataFrame]:
    return [
        pd.read_csv(os.path.join(folder, "progress.csv"))
        for folder in analysis_df["logdir"]
    ]


def analyze_results(results_folder: str, output_folder: str) -> pd.DataFrame:
    os.makedirs(output_folder, exist_ok=True)
    df = tune.Analysis(results_folder).dataframe()
    training_dfs = get_training_dfs(df)
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
    plt.ylabel("validation loss")
    plt.xlabel("epoch")
    plt.savefig(os.path.join(output_folder, f"training_curves.png"), dpi=200)

    # plot best validation loss vs. config columns
    for col in config_cols:
        cleaned_col = col.replace("config/", "").replace("_", " ")
        plt.figure(figsize=(7, 7))
        if cleaned_df[col].nunique() > len(cleaned_df) / 2:
            x = cleaned_df[col].dropna()
            vals, bins = pd.qcut(x, 10, labels=False, retbins=True)
            cleaned_col = f"{cleaned_col} bin center"
            xtics = np.round((bins[:-1] + bins[1:]) / 2, 2)  # bin centers
            cleaned_df[col] = xtics[vals]
        sns.lineplot(data=cleaned_df, x=col, y=best_val_loss_col)
        plt.xlabel(cleaned_col)
        plt.savefig(
            os.path.join(output_folder, f"loss_vs_{cleaned_col.replace(' ', '_')}"),
            dpi=200,
        )
        plt.close("all")

    return df
