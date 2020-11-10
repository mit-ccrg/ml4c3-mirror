# Imports: standard library
import os
import time
from typing import Dict

# Imports: third party
import dash
import h5py
import numpy as np
import pandas as pd
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, State, Output

# Imports: first party
from ml4c3.visualizer.properties import LAYOUT, MESSAGES, SIGNAL_INTERPRETATION
from ml4c3.visualizer.tools.tm_helper import TMapHelper
from ml4c3.visualizer.static_callbacks import get_static_data
from ml4c3.visualizer.tools.graph_generator import GraphGenerator

# pylint: disable=unused-variable, too-many-statements


def set_dynamic_callbacks(app, cache):
    """
    Sets the callbacks that depend on properties.py:

    * Visit ID selector (Signal dropdown population)
    * Graph generation
    """

    @app.callback(*signal_selector_ios(LAYOUT["graphs"]))
    def select_visit_id(visit_id: str, *args):
        """
        Call back to populate signal dropdowns.

        Triggered:
           * When the user selects a visit ID
        Causes:
           * Updates the top right text with the currently selected visit ID
           * Extracts the static data from the HD5, serializes it and stores in
           on the browser memory for later use.
           * Reads the HD5 file and populates the signal dropdowns.
        """

        # Update the Div text on the top right
        new_text = html.H6(
            f"Current Visit ID: {visit_id}",
            style={"margin-top": "3px", "float": "right"},
        )
        output = [new_text]

        # Don't update dropdowns if no visit ID or file path is set
        file_path, file_name = args[-2], args[-1]
        visit_not_selected = not (visit_id and file_path and file_name)
        if visit_not_selected:
            output.append({})
            for _ in range(LAYOUT["options"]["dropdown_num"]):
                output.append([])
                output.append(None)
            return output

        # Get complete file path
        file_path = os.path.join(file_path, file_name)

        graph_filters = args[:-2]
        with h5py.File(file_path, "r") as hd5_file:

            # Extract the serializable static data
            static_data = get_static_data(LAYOUT["statics"], hd5_file)
            output.append(static_data)

            # Extract signals to populate dropdowns
            for idx, (_graph_id, props) in enumerate(LAYOUT["graphs"].items()):

                # Filtered dropdowns is checked?
                filter_signals = graph_filters[idx]

                # Fill all dropdowns
                dropdowns = {**props["top_dropdowns"], **props["side_dropdowns"]}
                for signal_sources in dropdowns.values():
                    dd_options = fill_dropdown(
                        signal_sources,
                        hd5_file,
                        visit_id,
                        static_data,
                        filter_signals,
                    )
                    output.append(dd_options)
                    output.append(None)

        return output

    def make_graph_callback(app, cache, graph_id, props):
        # Get signal props
        dropdowns = list(props["top_dropdowns"]) + list(props["side_dropdowns"])
        outputs, inputs, states, opt_args = get_graph_ios(
            graph_id, dropdowns, **props["props"]
        )

        @app.callback(outputs, inputs, states)
        def update_medical_graph(*args):
            # Get properties
            visit_id, file_path, file_name, static_data, cur_fig = args[
                len(inputs) : len(inputs) + 5
            ]

            # Get file full path
            if file_path and file_name:
                file_path = os.path.join(file_path, file_name)
            else:
                file_path = None

            # Get signal and events
            dropdown_output = args[: len(dropdowns)]
            selected_signals = []
            selected_movements = []
            selected_events = []
            for dd_selected_signals in dropdown_output:
                if not dd_selected_signals:
                    continue
                for signal_id in dd_selected_signals:
                    signal_source, _ = signal_id.split("--")
                    if (
                        SIGNAL_INTERPRETATION[signal_source]["interpretation"]
                        == "event"
                    ):
                        selected_events.append(signal_id)
                    elif (
                        SIGNAL_INTERPRETATION[signal_source]["interpretation"]
                        == "static"
                    ):
                        selected_movements.append(signal_id)
                    else:
                        selected_signals.append(signal_id)

            def get_option(option_name, default):
                if option_name in opt_args:
                    idx = opt_args.index(option_name)
                    value = args[len(dropdowns) + 1 + idx]
                    return value if value is not None else default
                return default

            # default outputs
            outs = {
                "new_figure": dash.no_update,
                "pop_warning": dash.no_update,
                "warning_message": dash.no_update,
            }
            if len(outputs) >= 4:
                outs["new_time_fig"] = dash.no_update
            if len(outputs) >= 5:
                outs["time_fig_style"] = dash.no_update

            # Exit if no file or no signal is selected
            dd_selected = selected_signals or selected_movements or selected_events
            if not file_path or not dd_selected or not visit_id:
                if not cur_fig["data"]:
                    raise PreventUpdate
                outs["new_figure"] = GraphGenerator.reset_graph(cur_fig)
                return (*list(outs.values()),)

            # Extract the signal and events
            no_time = get_option("xaxis", "time") != "time"
            with h5py.File(file_path, "r") as hd5_file:
                data, errors = get_plot_data(
                    hd5_file=hd5_file,
                    visit_id=visit_id,
                    signals=selected_signals,
                    events=selected_events,
                    movements=selected_movements,
                    static_data=static_data,
                    no_time=no_time,
                )
            if errors:
                outs["pop_warning"] = True
                outs["warning_message"] = "\n\n".join(errors)
                return (*list(outs.values()),)

            # Crop the signal and events
            crop_data(
                data,
                percent_range=get_option("time_range", [0, 1]),
                downsample=get_option("downsampler", 1),
                optimize=props["optimized"],
                no_time=no_time,
            )

            # Throw warning if data is too large
            last_ok_ts = args[len(dropdowns)]
            warning_ok_clicked = (
                last_ok_ts / 1000 - time.time() < 2 if last_ok_ts else False
            )
            for signal in data["signals"]:
                large_signal = (
                    signal.values.size > LAYOUT["options"]["signal_too_large"]
                )
                if large_signal and not warning_ok_clicked:
                    outs["warning_message"] = MESSAGES["signal_too_large"]
                    outs["pop_warning"] = True
                    return (*list(outs.values()),)

            # Update the graph
            outs["new_figure"] = GraphGenerator.update_graph(
                cur_fig,
                data=data,
                optimized=props["optimized"],
                markers=get_option("marks", "lines"),
                no_time=no_time,
            )

            # Update time graph:
            plot_time_graph = get_option("time_graph_cb", None)
            if plot_time_graph:
                cur_time_graph = args[-1]
                outs["new_time_fig"] = GraphGenerator.time_graph(
                    cur_time_graph,
                    data=data,
                )
                outs["time_fig_style"] = {"display": "block"}
            elif plot_time_graph is not None:
                outs["time_fig_style"] = {"display": "None"}

            return (*list(outs.values()),)

        @cache.memoize()
        def get_plot_data(
            hd5_file,
            visit_id,
            signals,
            events,
            movements,
            no_time,
            static_data,
        ):
            """
            Extracts the data to be plotted.
            """
            data = {"signals": [], "events": []}
            errors = []
            # Extract events
            for event in events:
                event_source, event_name = event.split("--")
                event_info = TMapHelper.get_event(
                    hd5_file,
                    visit_id,
                    event_source,
                    event_name,
                )

                data["events"].append(event_info)

            for mvmnt in movements:
                _, mvmnt_name = mvmnt.split("--")
                event_mvmnt = TMapHelper.from_movement_to_event(mvmnt_name, static_data)
                data["events"].append(event_mvmnt)

            # Extract signals
            for signal in signals:
                sig_id, sig_name = signal.split("--")
                try:
                    sig_info = TMapHelper.get_signal(
                        hd5=hd5_file,
                        visit_id=visit_id,
                        signal_source=sig_id,
                        signal_name=sig_name,
                        no_time=no_time,
                    )

                    # check unplotable types
                    if isinstance(sig_info.values[0], (str, bytes)):
                        raise ValueError(f"{sig_info.values[0]} is a string")

                    data["signals"].append(sig_info)

                except ValueError as val_err:
                    error_msg = (
                        f"Could not extract the signal {signal} correctly. "
                        f"It may be stored as a string.\n MORE INFO: {str(val_err)}"
                    )
                    errors.append(error_msg)
                except KeyError as key_err:
                    error_msg = (
                        f"We couldn't find or create a tmap for signal {signal}."
                        f"Please, write a bug report at "
                        f"https://github.com/aguirre-lab/icu/issues/new/choose."
                        f"\n\nMORE INFO: {str(key_err)}"
                        f"\n\nPress cancel to exit"
                    )
                    errors.append(error_msg)

            admin_dict = static_data["Admission"]["fields"]
            data["metadata"] = (
                {
                    "start_date": pd.to_datetime(admin_dict["admin_date"]),
                    "end_date": pd.to_datetime(admin_dict["end_date"]),
                }
                if not no_time
                else {}
            )
            return data, errors

    for graph_id, props in LAYOUT["graphs"].items():
        make_graph_callback(app, cache, graph_id, props)


def signal_selector_ios(graphs: Dict):
    outputs = [
        Output("div-selected-csn", "children"),
        Output("static_data", "data"),
    ]
    inputs = [Input("visit_id_dd", "value")]

    for graph_id, props in graphs.items():
        dropdowns = list(props["top_dropdowns"]) + list(props["side_dropdowns"])
        for dropdown in dropdowns:
            outputs.append(Output(f"{graph_id}_{dropdown}_dd", "options"))
            outputs.append(Output(f"{graph_id}_{dropdown}_dd", "value"))
        inputs.append(Input(f"{graph_id}_filter_signals_list_cb", "value"))

    states = [State("input_path", "value"), State("files_dd", "value")]

    return outputs, inputs, states


def fill_dropdown(data_source, hd5_file, visit_id, static_data, filter_signals):
    """
    Populates a dropdown with a list of signals coming from the hd5's data
    source.
    """
    # Extract the signals
    try:
        # Add movements from static data
        if data_source == "static":
            mvmnt_fields = static_data["Movements"]["fields"]
            signal_list = mvmnt_fields["department_nm"]
        # Add all the other signals
        else:
            signal_list = TMapHelper.list_signal(
                hd5_file,
                data_source,
                visit_id,
                filter_signals,
            )
    except KeyError:  # Don't crash if the file is in an old format
        signal_list = []

    # Rename special signals
    special_signals = LAYOUT["options"]["special_signals"]
    for idx, signal in enumerate(signal_list):
        for special_signal in special_signals:
            if special_signal["pattern"] in signal:
                signal_list = np.delete(signal_list, idx)
                for suffix in special_signal["suffix"]:
                    signal_list = np.append(signal_list, f"{signal}_{suffix}")

    # Format the options for the dropdown
    dd_options = [
        {"label": sig, "value": f"{data_source}--{sig}"} for sig in signal_list
    ]
    return dd_options


def get_graph_ios(
    graph_id,
    dropdowns,
    time_range=False,
    data_events=False,
    time_graph=False,
    xaxis_choice=False,
    markers_choice=False,
    down_sampler=False,
):
    """
    Returns a list with the inputs, outputs and states for a graph with the
    input characteristics.
    """

    args = []
    outputs = [
        Output(f"{graph_id}-graph", "figure"),
        Output(f"{graph_id}-warning-dialog", "displayed"),
        Output(f"{graph_id}-warning-dialog", "message"),
    ]
    if time_graph:
        outputs.append(Output(f"{graph_id}_time-graph", "figure"))
        outputs.append(Output(f"{graph_id}_time_block", "style"))

    inputs = [Input(f"{graph_id}_{sig_type}_dd", "value") for sig_type in dropdowns]
    inputs.append(Input(f"{graph_id}-warning-dialog", "submit_n_clicks_timestamp"))

    if data_events:
        inputs.append(Input(f"{graph_id}_dataevents_cb", "value"))
        args.append("data_events")

    if time_graph:
        inputs.append(Input(f"{graph_id}_time_graph_cb", "value"))
        args.append("time_graph_cb")
        if data_events:
            inputs.append(Input(f"{graph_id}_time_dataevents_cb", "value"))
            args.append("time_graph_de_cb")

    if xaxis_choice:
        inputs.append(Input(f"{graph_id}_xaxis_cb", "value"))
        args.append("xaxis")

    if markers_choice:
        inputs.append(Input(f"{graph_id}_markers_cb", "value"))
        args.append("marks")

    if time_range:
        inputs.append(Input(f"{graph_id}_time_range", "value"))
        args.append("time_range")

    if down_sampler:
        inputs.append(Input(f"{graph_id}_downsampler", "value"))
        args.append("downsampler")

    states = [
        State("visit_id_dd", "value"),
        State("input_path", "value"),
        State("files_dd", "value"),
        State("static_data", "data"),
        State(f"{graph_id}-graph", "figure"),
    ]
    if time_graph:
        states.append(State(f"{graph_id}_time-graph", "figure"))

    return outputs, inputs, states, args


def crop_data(data, percent_range, downsample, optimize=False, no_time=False):
    for signal in data["signals"]:
        values = signal.values
        times = signal.time

        # Crop signals by event
        if optimize and data["events"] and not no_time:
            time_idx = np.logical_or.reduce(
                [
                    (times >= event.start_dates) & (times <= event.end_dates)
                    for event in data["events"]
                ],
            )
            times = times[time_idx]
            values = values[time_idx]

        # Downsample and crop
        idx_start = int(values.size * percent_range[0])
        idx_end = int(values.size * percent_range[1])
        signal.values = values[idx_start:idx_end:downsample]
        signal.time = times[idx_start:idx_end:downsample] if not no_time else None
