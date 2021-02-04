# Imports: standard library
import os
import re
from typing import List, Tuple

# Imports: third party
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class RLPlotter:
    """
    Class to plot cool RL stuff.
    """

    def __init__(
        self,
        title: str = None,
        axis_labels: Tuple = None,
        axis_lims: List = None,
        path_store: str = "./saved_elements",
    ):
        """
        Init class to create a figure.

        :param title: <str> Title of the figure.
        :param axis_labels: <str> Axis labels.
        :param axis_lims: <str> Axis limits.
        """
        # Backend changed if class is used
        matplotlib.use("TkAgg")

        # Set fixed params for figure
        self.fig, self.axs = plt.subplots(1, 1)
        self.title = title
        self.axis_labels = axis_labels
        self.axis_lims = axis_lims

        # Path to save
        self.path_store = path_store

        # Register names when saving
        self.prev_name = ""

    def update_plot(
        self,
        data: np.ndarray,
        extra_info: str = None,
        hold: bool = False,
    ):
        """
        Update plot with data and/or text.

        :param data: <np.ndarray> Array with data to plot.
        :param extra_info: <str> Text to show in the figure.
        :param hold: <bool> If set true, plots are superimposed.
        """
        # Clean figure
        if not hold:
            self.axs.cla()

        # Set params to figure
        if self.title:
            self.axs.set_title(self.title)
        if self.axis_labels:
            self.axs.set_xlabel(self.axis_labels[0])
            self.axs.set_ylabel(self.axis_labels[1])
        if self.axis_lims:
            if self.axis_lims[0] >= min(data[:, 0]) or self.axis_lims[1] <= max(
                data[:, 0],
            ):
                self.axs.axis(
                    [
                        int(0.9 * min(data[:, 0])),
                        int(1.1 * max(data[:, 0])),
                        self.axis_lims[2],
                        self.axis_lims[3],
                    ],
                )

            elif self.axis_lims[2] >= min(data[:, 1]) or self.axis_lims[3] <= max(
                data[:, 1],
            ):
                self.axs.axis(
                    [
                        self.axis_lims[0],
                        self.axis_lims[1],
                        int(0.9 * min(data[:, 1])),
                        int(1.1 * max(data[:, 1])),
                    ],
                )
            else:
                self.axs.axis(self.axis_lims)

        # Plot data
        self.axs.plot(data[:, 0], data[:, 1])
        # Show text
        if extra_info:
            self.axs.text(
                0.65,
                0.93,
                extra_info,
                verticalalignment="center",
                transform=self.axs.transAxes,
            )
        self.fig.show()
        plt.pause(0.5)

    def save_figure(self, name: str):
        """
        Save figure.

        :param name: <str> Name used to save the figure (.png).
        """
        if not name.endswith(".png"):
            name += ".png"

        if not name[-6:] == "-0.png":
            name = name[:-4] + "-0.png"
        if os.path.isfile(os.path.join(self.path_store, "plots", self.prev_name)):
            name_parts = re.split("[-,.]", self.prev_name)
            name = name_parts[0] + "-" + str(int(name_parts[1]) + 1) + ".png"
        self.prev_name = name

        if not os.path.isdir(os.path.join(self.path_store, "plots")):
            os.makedirs(os.path.join(self.path_store, "plots"))
        self.fig.savefig(os.path.join(self.path_store, "plots", name))
