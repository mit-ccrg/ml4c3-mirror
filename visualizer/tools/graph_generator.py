# Imports: standard library
from typing import Dict

# pylint: disable=unused-import


class GraphGenerator:
    """
    Helper class to generate graphs.
    """

    color_palette = [
        "cyan",
        "green",
        "purple",
        "orange",
        "brown",
        "magenta",
        "blue",
        "white",
    ]

    @classmethod
    def update_graph(
        cls,
        cur_figure: Dict,
        data: Dict,
        optimized=False,
        markers="lines",
        no_time=False,
    ):
        # Reset previous
        cur_figure = cls.reset_graph(cur_figure)

        # Add signals
        signals = data["signals"]
        step = 1 / len(signals) if signals else 1

        for signal_idx, signal_data in enumerate(signals):
            # value charts
            axis_idx = str(signal_idx + 1) if signal_idx else ""
            cur_sig_data = {
                "x": signal_data.time,
                "y": signal_data.values,
                "mode": markers,
                "name": signal_data.name,
                "line": {"color": cls.color_palette[signal_idx]},
                "showlegend": True,
                "yaxis": "y" + axis_idx,
                "type": "scatergl" if optimized else None,
            }
            cur_figure["data"].append(cur_sig_data)
            cur_figure["layout"]["yaxis" + axis_idx] = GraphGenerator._yaxis_props(
                signal_idx,
                step,
                signal_data.units,
            )

        # Add events
        shapes, annotations = cls.add_events(data["events"])

        # Update layout
        cur_figure["layout"].update(
            {
                "height": 500 * (1 + 0.5 * (len(signals) - 1)) if signals else 500,
                "shapes": shapes,
                "annotations": annotations,
            },
        )

        # Configure x axis
        xaxis = cur_figure["layout"]["xaxis"]
        if no_time:
            xaxis["rangeslider"]["visible"] = False
            xaxis["type"] = "linear"
        else:
            xaxis["rangeslider"]["visible"] = True
            xaxis["rangeslider"]["range"] = [
                data["metadata"]["start_date"],
                data["metadata"]["end_date"],
            ]
            xaxis["type"] = "date"

        return cur_figure

    @classmethod
    def reset_graph(cls, cur_fig):
        # Reset data
        cur_fig["data"] = []

        # Reset layout
        cur_fig["layout"]["shapes"] = []
        cur_fig["layout"]["annotations"] = []
        cur_fig["layout"]["height"] = 500

        axis_to_rmv = [prop for prop in cur_fig["layout"] if prop.startswith("yaxis")]
        for axis in axis_to_rmv:
            cur_fig["layout"].pop(axis, None)

        return cur_fig

    @classmethod
    def time_graph(cls, cur_time_fig, data):
        # Reset previous
        cur_figure = cls.reset_graph(cur_time_fig)

        # Add signals
        signals = data["signals"]

        for signal_idx, signal_data in enumerate(signals):
            # value charts
            cur_sig_data = {
                "y": signal_data.time,
                "name": signal_data.name,
                "line": {"color": cls.color_palette[signal_idx]},
                "showlegend": True,
            }
            cur_figure["data"].append(cur_sig_data)

        # Configure x axis
        return cur_figure

    # PROPS
    @staticmethod
    def _yaxis_props(idx, step, units):
        props = {
            "anchor": "x",
            "autorange": True,
            "fixedrange": False,
            "domain": [idx * step, (idx + 1) * step],
            "linecolor": "#673ab7",
            "mirror": True,
            "showline": True,
            "side": "right",
            "tickfont": {"color": "#673ab7"},
            "tickmode": "auto",
            "titlefont": {"color": "#673ab7"},
            "type": "linear",
            "zeroline": False,
            "title": units,
        }
        return props

    # EVENTS
    @classmethod
    def add_events(cls, events):
        shapes = []
        annotations = []
        for event_idx, event in enumerate(events):
            start_dates = event.start_dates
            end_dates = event.end_dates
            color = cls.color_palette[event_idx]

            for rep_idx, st_date in enumerate(start_dates):
                end_date = end_dates[rep_idx] if end_dates is not None else None
                mean_date = st_date + (end_date - st_date) / 2 if end_date else st_date

                shapes.append(GraphGenerator._create_event(color, st_date, end_date))
                annotations.append(
                    GraphGenerator._create_annotation(event.name, mean_date),
                )

        return shapes, annotations

    @staticmethod
    def _create_annotation(name, position):
        props = {
            "x": position,
            "y": 1,
            "text": name,
            "xref": "x",
            "showarrow": False,
            "yanchor": "top",
            "yref": "paper",
        }
        return props

    @staticmethod
    def _create_event(color, x_start, x_end):
        props = {
            "fillcolor": color,
            "opacity": 0.2,
            "line": {"width": 5 if x_end else 10, "color": color},
            "type": "rect" if x_end else "line",
            "x0": x_start,
            "x1": x_end if x_end else x_start,
            "y0": 0,
            "y1": 1,
            "xref": "x",
            "yref": "paper",
        }
        return props
