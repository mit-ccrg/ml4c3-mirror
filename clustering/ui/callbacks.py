# Imports: standard library
import os
import json
import base64
from typing import Dict, Optional

# Imports: third party
import dash
import pandas as pd
import dash_table
import plotly.express as px
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from plotly.subplots import make_subplots
from dash.dependencies import ALL, Input, State, Output

# Imports: first party
from clustering.main import get_data
from clustering.extras import modify_bundle
from clustering.ui.globals import STEPS, PROCESSES
from clustering.objects.finder import RequestReport
from clustering.objects.structures import Bundle

# BUNDLE: Bundle = Bundle.from_pickle("data/raw.pkl")
BUNDLE: Optional[Bundle] = None

BUNDLE_STEPS: Dict[str, Bundle] = {}


def set_callbacks(app, output_path, hd5_path):
    @app.callback(
        Output("reextract-data-dummy-div", "children"),
        [Input("reextract-data-btn", "n_clicks")],
        [State("stored-csv-dd", "value")],
    )
    def reextract_data(n_clicks, csv_file):
        if n_clicks is not None and csv_file is not None:
            global BUNDLE
            report_path = os.path.join(output_path, csv_file)
            BUNDLE = get_data(hd5_path, report_path)
            BUNDLE.store(f"{output_path}/raw.bundle")
            BUNDLE.name = "tmp"
            BUNDLE.store(f"{output_path}/tmp-raw.bpart")
            return []

    @app.callback(
        Output("raw-data-dummy-div", "children"),
        [Input("raw-data-btn", "n_clicks")],
    )
    def raw_data(n_clicks):
        global BUNDLE
        if BUNDLE is None:
            raise dash.exceptions.PreventUpdate

        BUNDLE = Bundle.from_pickle(f"{output_path}/raw.bundle")
        # modify_bundle(BUNDLE, only_signal="pa2d")
        # modify_bundle(BUNDLE, only_cores=True)
        # modify_bundle(BUNDLE, exclude_signals=["norepinephrine_infusion_syringe_in_swfi_80_mcg|ml_cmpd_central_mgh"])
        # modify_bundle(BUNDLE, only_signal="norepinephrine_infusion_syringe_in_swfi_80_mcg|ml_cmpd_central_mgh")
        # modify_bundle(BUNDLE, cluster_mult=1)
        BUNDLE.name = "tmp"
        BUNDLE.store(f"{output_path}/tmp-raw.bpart")
        return []

    @app.callback(
        Output("load-data-dummy-div", "children"),
        [Input("load-data-btn", "n_clicks")],
        [State("stored-bundle-dd", "value")],
    )
    def load_data(n_clicks, selected_file):
        if n_clicks is not None and selected_file:
            global BUNDLE
            BUNDLE = Bundle.from_pickle(f"{output_path}/{selected_file}")
            modify_bundle(BUNDLE, crop_time=2)
            return []

    @app.callback(
        Output("store-data-dummy-div", "children"),
        [Input("store-data-btn", "n_clicks")],
        [State("file-name-input", "value")],
    )
    def store_data(n_clicks, new_name):
        if n_clicks is not None and new_name:
            global BUNDLE
            BUNDLE.store(
                os.path.join("data", f"{new_name}.bundle"),
                rename_subfiles=True,
            )
            return []

    @app.callback(
        Output("current-bundle-processes-div", "children"),
        [Input("see-bundle-processes-btn", "n_clicks")],
    )
    def see_data_processes(n_clicks):
        if n_clicks is not None:
            global BUNDLE
            return [str(BUNDLE.processes)]

    @app.callback(
        Output("bundle-console-dummy-div", "children"),
        [Input("bundle-console-btn", "n_clicks")],
    )
    def launch_console(n_clicks):
        if n_clicks is not None:
            global BUNDLE
            bundle = BUNDLE
            try:
                # Imports: standard library
                import code

                code.interact(local=locals())
            except KeyboardInterrupt:
                pass
            return []

    @app.callback(
        [
            Output("step-preprocess-dd", "options"),
            Output("patient-preprocess-dd", "options"),
            Output("signal-preprocess-dd", "options"),
            Output("signal-summary-dd", "options"),
            Output("signal-clustering-dd", "options"),
        ],
        [Input("refresh-data-btn", "n_clicks")],
    )
    def populate_step_files_signal_dropdown(_):
        global BUNDLE
        if BUNDLE is None:
            raise dash.exceptions.PreventUpdate

        steps = [{"label": step, "value": step} for step in BUNDLE.processes]
        patients = [
            {"label": mrn, "value": path}
            for path, mrn in BUNDLE.patient_list(include_clean=True)
        ]
        signals = [
            {"label": signal, "value": signal} for signal in BUNDLE.signal_list()
        ]

        return steps, patients, signals, signals, signals

    def set_options(steps):
        def _set_options(selected_option):
            children = []

            if selected_option:
                dd_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
                selected_step = dd_id.replace("-dd", "")
                additional_opts = steps[selected_step].methods()[selected_option]
                for opt in additional_opts:
                    clean_name = opt.replace("_", " ")
                    clean_name = clean_name.capitalize()
                    children.append(
                        dcc.Input(
                            id={
                                "type": f"{selected_step}_kwargs",
                                "index": opt,
                            },
                            placeholder=clean_name,
                        ),
                    )

            return children if children else ["--"]

        return _set_options

    def parse_kwargs(selected_step):
        kwargs = {}
        states = dash.callback_context.states

        for state_id, value in states.items():
            state_id = state_id.replace(".value", "")
            if value is None:
                return

            if state_id[0] == "{":
                state_id = json.loads(state_id)
                arg = state_id["index"]
                kwarg = f"{selected_step}_kwargs"
                if kwarg not in kwargs:
                    kwargs[kwarg] = {arg: value}
                else:
                    kwargs[kwarg][arg] = value
            else:
                state_id = state_id.replace("-dd", "")
                kwargs[f"{state_id}_method"] = value

        return kwargs

    def process_data(selected_step):
        def _process_data(*_):
            kwargs = parse_kwargs(selected_step)
            if kwargs is None:
                return

            global BUNDLE
            PROCESSES[selected_step](BUNDLE, **kwargs)
            return []

        return _process_data

    for _process, _steps in STEPS.items():
        if _process == "cluster":
            continue
        for _step in _steps:
            app.callback(
                Output(f"{_step}-kwargs-div", "children"),
                [Input(f"{_step}-dd", "value")],
            )(set_options(_steps))

            app.callback(
                Output(f"dummy-div-{_step}", "children"),
                [Input(f"{_step}-btn", "n_clicks")],
                [State({"type": f"{_step}_kwargs", "index": ALL}, "value")]
                + [State(f"{_step}-dd", "value")],
            )(process_data(_step))

    @app.callback(
        [
            Output("cluster-kwargs-div", "children"),
            Output("cluster-distance-dd", "options"),
            Output("cluster-algo-dd", "options"),
        ],
        [Input("cluster-dd", "value")],
    )
    def cluster_options(cluster_method):
        if cluster_method is None:
            raise dash.exceptions.PreventUpdate
        kwargs = set_options(STEPS["cluster"])(cluster_method)

        cluster_inputs = STEPS["cluster"]["cluster"].ACCEPTED_DISTANCES[cluster_method]
        dist_options = [
            {"label": dist, "value": dist} for dist in cluster_inputs["distance"]
        ]
        algo_options = [
            {"label": dist, "value": dist} for dist in cluster_inputs["algo"]
        ]

        return kwargs, dist_options, algo_options

    @app.callback(
        Output("cluster-distance-kwargs-div", "children"),
        [Input("cluster-distance-dd", "value")],
        [State("cluster-dd", "value")],
    )
    def cluster_distance_options(distance_method, cluster_method):
        if cluster_method is None or distance_method is None:
            raise dash.exceptions.PreventUpdate

        inputs = []
        extra_kwargs = STEPS["cluster"]["cluster"].ACCEPTED_DISTANCES[cluster_method][
            "distance"
        ][distance_method]
        for kwarg in extra_kwargs:
            input = dcc.Input(
                id={
                    "type": "cluster-distance_kwargs",
                    "index": kwarg,
                },
                placeholder=f"Enter a {kwarg}",
            )
            inputs.append(input)
        return inputs

    @app.callback(
        Output("cluster-algo-kwargs-div", "children"),
        [Input("cluster-algo-dd", "value")],
        [State("cluster-dd", "value")],
    )
    def cluster_algo_options(distance_method, cluster_method):
        if cluster_method is None or distance_method is None:
            raise dash.exceptions.PreventUpdate

        inputs = []
        extra_kwargs = STEPS["cluster"]["cluster"].ACCEPTED_DISTANCES[cluster_method][
            "algo"
        ][distance_method]
        for kwarg in extra_kwargs:
            input = dcc.Input(
                id={
                    "type": "cluster-algo_kwargs",
                    "index": kwarg,
                },
                placeholder=f"Enter a {kwarg}",
            )
            inputs.append(input)
        return inputs

    @app.callback(
        Output("dummy-div-cluster", "children"),
        [
            Input("cluster-btn", "n_clicks"),
            Input("cluster-optim-btn", "n_clicks"),
        ],
        [State({"type": "cluster_kwargs", "index": ALL}, "value")]
        + [State({"type": "cluster-distance_kwargs", "index": ALL}, "value")]
        + [State({"type": "cluster-algo_kwargs", "index": ALL}, "value")]
        + [
            State("cluster-dd", "value"),
            State("cluster-distance-dd", "value"),
            State("cluster-algo-dd", "value"),
        ],
    )
    def cluster(
        _cluster_nclicks,
        _optimize_nclicks,
        cluster_kwargs,
        distance_kwargs,
        algo_kwargs,
        cluster_method,
        distance_metric,
        cluster_algo,
    ):
        if _cluster_nclicks is None and _optimize_nclicks is None:
            raise dash.exceptions.PreventUpdate

        kwargs = parse_kwargs("cluster")
        if kwargs is None or cluster_method is None:
            raise dash.exceptions.PreventUpdate

        if "cluster_kwargs" in kwargs:
            if any(list(kwargs["cluster_kwargs"].values())) == "":
                raise dash.exceptions.PreventUpdate

        optimize = (
            dash.callback_context.triggered[0]["prop_id"]
            == "cluster-optim-btn.n_clicks"
        )
        kwargs["optimize"] = optimize

        global BUNDLE
        PROCESSES["cluster"](BUNDLE, **kwargs)

        if optimize:
            img_path = "/home/ep479/repos/ul4icu/optimization.png"
            test_base64 = base64.b64encode(open(img_path, "rb").read()).decode("ascii")
            img = html.Img(
                src=f"data:image/png;base64,{test_base64}",
                style={
                    "margin-left": "auto",
                    "margin-right": "auto",
                    "display": "block",
                },
            )
            btn = html.Button(
                "Close",
                id="close-cluster-optim-btn",
                style={
                    "margin-left": "auto",
                    "margin-right": "auto",
                    "display": "block",
                },
            )
            layout = html.Div(
                id="cluster-optim-img-div",
                children=[
                    html.Div(img, className="row"),
                    html.Div(btn, className="row"),
                ],
            )
            return layout
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("cluster-optim-img-div", "children"),
        [Input("close-cluster-optim-btn", "n_clicks")],
    )
    def close_cluster_optim_image(n_clicks):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        return []

    @app.callback(
        Output("preprocessing-table-div", "children"),
        [Input("signal-summary-dd", "value")],
    )
    def summary_table(signal):
        if not BUNDLE or not signal:
            return []

        summary = BUNDLE.get_curation_table()
        signal_info = pd.DataFrame({file: summary[file][signal] for file in summary}).T
        signal_info.reset_index(level=0, inplace=True)
        signal_info = signal_info.rename(columns={"index": "file"})
        signal_info["values padded"] = signal_info["values padded"].astype(str)
        signal_info["downsample%"] = signal_info["downsample%"].astype(float).round(2)
        data_dict = signal_info.to_dict("records")

        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in signal_info.keys()],
            data=data_dict,
            filter_action="native",
            sort_action="native",
        )

        return table

    @app.callback(
        [
            Output("preprocessing-report-div", "children"),
            Output("preprocessing-report-btn", "children"),
        ],
        [Input("preprocessing-report-btn", "n_clicks")],
    )
    def summary_table(n_clicks):
        if not BUNDLE or not n_clicks:
            return dash.no_update
        if n_clicks % 2:
            report_dict = BUNDLE.get_bundle_report()
            report_str = "\n\n".join(
                [f"**{key}**: {value}" for key, value in report_dict.items()],
            )
            btn_text = "Hide report"
        else:
            report_str = ""
            btn_text = "Report"
        report_md = dcc.Markdown(report_str)
        return report_md, btn_text

    @app.callback(
        Output("visualize-container", "children"),
        [Input("plot-preprocess-graph-btn", "n_clicks")],
        [
            State("step-preprocess-dd", "value"),
            State("patient-preprocess-dd", "value"),
            State("signal-preprocess-dd", "value"),
            State("separate-cb", "value"),
        ],
    )
    def plot_graph(_, steps, selected_patient, selected_signals, separate):

        figure = get_figure(None, None)

        # n_patients = len(selected_patient)
        # n_row = int((n_patients - 1) / 3) + 1
        # n_col = math.ceil(n_patients / n_row)
        # figure = get_figure(n_row, n_col)

        if selected_signals and selected_patient:
            global BUNDLE
            global BUNDLE_STEPS

            bundles = []
            for step_file in steps:
                bundle_file = f"{output_path}/{BUNDLE.name}-{step_file}.bpart"
                if bundle_file in BUNDLE_STEPS:
                    bundles.append(BUNDLE_STEPS[bundle_file])
                else:
                    bundle = BUNDLE.from_pickle(bundle_file)
                    bundles.append(bundle)
                    BUNDLE_STEPS[bundle_file] = bundle
            for b_idx, bundle in enumerate(bundles):
                ax_idx = b_idx + 1 if b_idx else ""
                for signal_name in selected_signals:
                    signal = bundle[selected_patient].get_signal(signal_name)
                    figure.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(signal.time, unit="s"),
                            y=signal.values,
                            yaxis=f"y{ax_idx}" if separate else None,
                            name=signal.name,
                            showlegend=True,
                        ),
                        # row=p_row, col=p_col
                    )

                meta = bundle[selected_patient].meta
                for event in meta:
                    if event == "teoric_max_span":
                        continue
                    color = "red" if "blk08" in event else "white"
                    figure.add_shape(
                        opacity=0.2,
                        type="line",
                        line={"color": color},
                        x0=pd.to_datetime(meta[event]),
                        x1=pd.to_datetime(meta[event]),
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                    )

                margin = 1 / len(bundles) if bundles else 1
                if separate:
                    ax = f"yaxis{ax_idx}"
                    figure.update_layout(
                        **{ax: _yaxis_props(b_idx, margin, steps[b_idx])}
                    )

        return [dcc.Graph(figure=figure)]

    @app.callback(
        Output("signal-cluster-container", "children"),
        [Input("plot-signal-cluster-graph-btn", "n_clicks")],
        [
            State("signal-clustering-dd", "value"),
            State("signal-cluster-separate-cb", "value"),
        ],
    )
    def plot_signal_cluster_graph(_, signal_name, separate):
        figure = get_figure(None, None)

        if _ and signal_name:
            global BUNDLE
            clusters = BUNDLE.get_cluster_trajectories()

            for patient in BUNDLE.patients.values():
                ax_idx = patient.cluster + 2 if patient.cluster != -1 else ""
                signal = patient.get_signal(signal_name)
                figure.add_trace(
                    go.Scatter(
                        # x=pd.to_datetime(signal.time, unit="s"),
                        y=signal.values,
                        name=signal.name,
                        showlegend=False,
                        mode="lines",
                        line_width=0.5,
                        line_color=px.colors.qualitative.Plotly[patient.cluster + 1],
                        yaxis=f"y{ax_idx}" if separate else None,
                        opacity=0.1,
                        hoverinfo="skip",
                    ),
                )
            for cluster, signals in clusters.items():
                ax_idx = cluster + 2 if cluster != -1 else ""
                signal = signals[signal_name]
                figure.add_trace(
                    go.Scatter(
                        # x=pd.to_datetime(signal.time, unit="s"),
                        y=signal.values,
                        name=f"Cluster {cluster}",
                        showlegend=True,
                        mode="lines",
                        line_width=5,
                        line_color=px.colors.qualitative.Plotly[cluster + 1],
                        yaxis=f"y{ax_idx}" if separate else None,
                        opacity=1,
                    ),
                )
            if separate:
                margin = 1 / len(clusters) if clusters else 1
                for idx, cluster in enumerate(sorted(clusters)):
                    ax_idx = cluster + 2 if cluster != -1 else ""
                    ax = f"yaxis{ax_idx}"
                    figure.update_layout(
                        **{ax: _yaxis_props(idx, margin, f"Cluster {cluster}")}
                    )
                    figure.update_layout(**{ax: {"autorange": False, "range": [0, 1]}})
                    figure.update_xaxes(type="linear")

        return [dcc.Graph(figure=figure)]

    @app.callback(
        Output("clusters-info-table", "children"),
        [Input("cluster-info-refresh-btn", "n_clicks")],
    )
    def display_info(_):
        if _ is None:
            return dash.no_update
        try:
            global BUNDLE
            cluster_report = BUNDLE.analyze_clusters()
        except KeyError as ke:
            print(ke)
            return dash.no_update

        df = pd.DataFrame(cluster_report).T
        df.reset_index(inplace=True)
        df = df.rename(columns={"index": "cluster"})
        df = df.fillna(0)
        df = df.round(2)

        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict("records"),
            sort_action="native",
        )
        return table

    @app.callback(
        Output("pca-cluster-container", "children"),
        [Input("plot-cluster-graph-btn", "n_clicks")],
        [State("dim-reduction-dd", "value")],
    )
    def plot_pca(_, dim_red_meth):

        if _ is None:
            return dash.no_update
        try:
            global BUNDLES
            df, exp_var = BUNDLE.reduce_dimensions(dim_red_meth)
        except KeyError:
            return dash.no_update

        n_axes = len([col for col in df.columns if col.startswith("ax")])
        figure = make_subplots(rows=n_axes, cols=n_axes)

        if dim_red_meth == "meth_pca":
            ax_label = "PC"
            title = "Explained variance"
        elif dim_red_meth == "meth_tsne":
            ax_label = "tSNE"
            title = "kl_divergence"
        else:
            ax_label = "ax"
            title = "Variance"

        for i in range(1, n_axes + 1):
            for j in range(1, n_axes + 1):
                print(f"Position: {i}-{j}")
                for cluster in df["cluster"].unique():
                    cluster_df = df[df["cluster"] == cluster]
                    figure.add_trace(
                        go.Scatter(
                            x=cluster_df[f"ax{i}"],
                            y=cluster_df[f"ax{j}"],
                            mode="markers",
                            marker_color=cluster,
                            text=cluster_df["patient"],
                            showlegend=True,
                            name=f"Cluster {cluster}",
                        ),
                        row=i,
                        col=j,
                    )
                figure.update_xaxes(title_text=f"{ax_label}{i}", row=i, col=j)
                figure.update_yaxes(title_text=f"{ax_label}{j}", row=i, col=j)

        figure.update_layout(
            title=f"{title}: {exp_var:.2f}%",
            xaxis=None,
            template="icu",
            height=1000,
            margin=go.layout.Margin(l=50, r=50, b=50, t=50),
        )
        figure.update_xaxes(type="linear")

        return [dcc.Graph(figure=figure)]

    def get_figure(n_row, n_col):
        # figure = subplots.make_subplots(n_row, n_col)
        figure = go.Figure()
        figure.update_layout(
            xaxis={"type": "date"},
            margin=go.layout.Margin(l=50, r=50, b=50, t=50),  # noqa: E741
            height=500,
            template="icu",
        )
        return figure

    def _yaxis_props(idx, step, units):
        props = {
            "anchor": "x",
            "autorange": True,
            "fixedrange": False,
            "domain": [idx * step, (idx + 1) * step],
            "linecolor": "#673ab7",
            "mirror": True,
            "showline": True,
            "side": "left",
            "tickfont": {"color": "#673ab7"},
            "tickmode": "auto",
            "titlefont": {"color": "#673ab7"},
            "type": "linear",
            "zeroline": False,
            "title": units,
        }
        return props
