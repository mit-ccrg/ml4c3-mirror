# Imports: standard library
import os

# Imports: third party
import dash_core_components as dcc
import dash_html_components as html
from plotly import graph_objs as go

# Imports: first party
# pylint: disable=unused-import
from visualizer.assets import themes


class LayoutBuilder:
    """
    Tools to build the HTML layout.
    """

    @staticmethod
    def banner_layout():
        banner = html.Div(
            [
                html.H2(
                    "HD5 File visualizer",
                    id="title",
                    className="eight columns",
                    style={"margin-left": "3%"},
                ),
                html.Button(
                    id="learn-more-button",
                    className="two columns",
                    children=["Help"],
                ),
            ],
            className="banner row",
        )
        return banner

    @staticmethod
    def upload_div(files_dir):
        if files_dir:
            files = os.listdir(files_dir)
            options = [
                {"label": file, "value": file}
                for file in files
                if file.endswith(".hd5")
            ]
            value = files_dir
        else:
            options = []
            value = None
        upload_div = html.Div(
            children=[
                dcc.Input(
                    id="input_path",
                    type="text",
                    placeholder="input file path",
                    debounce=True,
                    className="six columns dropdown-box-first",
                    value=value,
                ),
                dcc.Dropdown(
                    id="files_dd",
                    options=options,
                    placeholder="Select a file",
                    searchable=True,
                    className="six columns dropdown-box-second",
                ),
            ],
        )
        return upload_div

    @staticmethod
    def visit_id_dd():
        visit_id_dd = html.Div(
            dcc.Dropdown(
                id="visit_id_dd",
                options=[],
                placeholder="Select a visit ID",
                searchable=False,
            ),
            className="twelve columns dropdown-box-first",
        )
        return visit_id_dd

    @staticmethod
    def signal_selector(root_id, dd_name, multi):
        signal_selector = html.Div(
            dcc.Dropdown(
                id=f"{root_id}_{dd_name}_dd",
                options=[],
                placeholder=f"Select {dd_name}",
                searchable=True,
                multi=multi,
                style={"margin-bottom": "5px"},
            ),
        )
        return signal_selector

    @staticmethod
    def time_cropper(name):
        time_cropper = html.Div(
            children=[
                html.Div("Crop time range: ", className="plot-display-text"),
                html.Div(
                    [
                        dcc.RangeSlider(
                            min=0,
                            max=1,
                            step=0.05,
                            value=[0, 1],
                            marks={i / 100: f"{i}%" for i in range(0, 101, 20)},
                            updatemode="drag",
                            id=f"{name}_time_range",
                        ),
                    ],
                    className="slider-smoothing",
                ),
            ],
            style={"margin-bottom": "10px"},
        )
        return time_cropper

    @staticmethod
    def down_sampler(name):
        time_cropper = html.Div(
            children=[
                html.Div(
                    "Down-sample: ",
                    style={"margin-top": "25px"},
                    className="plot-display-text",
                ),
                html.Div(
                    [
                        dcc.Slider(
                            min=1,
                            max=10,
                            step=1,
                            value=1,
                            marks={
                                int(i / 10): f"{int((i+1)/10)}"
                                for i in range(10, 101, 10)
                            },
                            updatemode="drag",
                            id=f"{name}_downsampler",
                        ),
                    ],
                    className="slider-smoothing",
                ),
            ],
        )
        return time_cropper

    @staticmethod
    def dataevents_cb(name):
        dataevents_cb = html.Div(
            [
                dcc.Checklist(
                    options=[{"label": " Dataevents", "value": "dataevents"}],
                    id=f"{name}_dataevents_cb",
                    className="checklist-smoothing",
                ),
            ],
            style={"margin": "10px"},
        )
        return dataevents_cb

    @staticmethod
    def mark_checklist(name):
        markers_cb = html.Div(
            [
                "Plot as:",
                dcc.RadioItems(
                    options=[
                        {"label": " Line", "value": "lines"},
                        {"label": " Points", "value": "markers"},
                        {"label": " Line+Points", "value": "lines+markers"},
                    ],
                    value="lines",
                    id=f"{name}_markers_cb",
                    className="checklist-smoothing",
                ),
            ],
            style={"margin": "10px"},
        )
        return markers_cb

    @staticmethod
    def yaxis_checklist(name):
        markers_cb = html.Div(
            [
                "Plot as:",
                dcc.Checklist(
                    options=[{"label": " Separate axis", "value": "separate"}],
                    id=f"{name}_separate_axis_cb",
                    className="checklist-smoothing",
                ),
            ],
            style={"margin": "10px"},
        )
        return markers_cb

    @staticmethod
    def x_axis_checklist(name):
        x_axis_cb = html.Div(
            [
                "X-axis: ",
                dcc.RadioItems(
                    options=[
                        {"label": " Time", "value": "time"},
                        {"label": " Index", "value": "index"},
                    ],
                    value="time",
                    id=f"{name}_xaxis_cb",
                    className="checklist-smoothing",
                ),
            ],
            style={"margin": "10px"},
        )
        return x_axis_cb

    @staticmethod
    def graph_time_cb(name):
        graph_time_cb = html.Div(
            [
                dcc.Checklist(
                    id=f"{name}_time_graph_cb",
                    options=[{"label": " Plot time graph", "value": "time_graph"}],
                    className="checklist-smoothing",
                ),
            ],
            style={"margin": "10px"},
        )
        return graph_time_cb

    @staticmethod
    def filter_signal_list_cb(name):
        filter_list_cb = html.Div(
            [
                dcc.Checklist(
                    options=[{"label": " Filter list", "value": "filter"}],
                    id=f"{name}_filter_signals_list_cb",
                    className="checklist-smoothing",
                ),
            ],
            style={"padding": "10px"},
        )
        return filter_list_cb

    @staticmethod
    def large_signal_alert(graph_id):
        alert = dcc.ConfirmDialog(id=f"{graph_id}-warning-dialog")
        return alert

    @staticmethod
    def graph_title(title):
        return html.H2(title)

    @classmethod
    def _empty_medical_graph(cls, name: str):
        figure = go.Figure()

        range_selector_props = {
            "buttons": [
                {"count": 1, "label": "1m", "step": "minute", "stepmode": "backward"},
                {
                    "count": 30,
                    "label": "30s",
                    "step": "second",
                    "stepmode": "backward",
                },
                {"count": 1, "label": "1h", "step": "hour", "stepmode": "backward"},
                {"count": 6, "label": "12h", "step": "hour", "stepmode": "backward"},
                {"step": "all"},
            ],
        }

        figure.update_layout(
            xaxis={
                "rangeselector": range_selector_props,
                "rangeslider": {"visible": True, "bgcolor": "#ebedeb"},
                "type": "date",
            },
            margin=go.layout.Margin(l=50, r=50, b=50, t=50),  # noqa: E741
            height=500,
            template="icu",
        )

        return dcc.Graph(figure=figure, id=f"{name}-graph")

    @staticmethod
    def graph_layout(
        name,
        side_dropdowns=None,
        time_range=False,
        data_events=False,
        time_graph=False,
        hidden=False,
        xaxis_choice=False,
        markers_choice=False,
        down_sampler=False,
        multi=True,
    ):
        slider_children = []
        if time_range:
            slider_children.append(LayoutBuilder.time_cropper(name))

        if down_sampler:
            slider_children.append(LayoutBuilder.down_sampler(name))

        slider_div = html.Div(
            children=slider_children,
            style={"margin-top": "20px", "margin-bottom": "100px"},
        )

        options_children = []
        if side_dropdowns:
            for signal in side_dropdowns:
                options_children.append(
                    LayoutBuilder.signal_selector(name, signal, multi=multi),
                )

        if data_events:
            options_children.append(LayoutBuilder.dataevents_cb(name))

        if time_graph:
            options_children.append(LayoutBuilder.graph_time_cb(name))

        if xaxis_choice:
            options_children.append(LayoutBuilder.x_axis_checklist(name))

        if markers_choice:
            options_children.append(LayoutBuilder.mark_checklist(name))

        options_div = html.Div(children=options_children)

        empty_graph = LayoutBuilder._empty_medical_graph(name)
        graph_div = html.Div(
            id=f"{name}_block",
            className="row",
            children=[
                html.Div(
                    className="two columns",
                    style={"padding-bottom": "5%"},
                    children=[slider_div, options_div],
                ),
                dcc.Loading(
                    id=f"{name}-graph-div",
                    className="ten columns",
                    children=[empty_graph],
                ),
            ],
            style={"display": "none" if hidden else "block"},
        )
        return graph_div

    @staticmethod
    def graph_block(graph_id, graph):
        # Top dropdowns
        signal_selectors = []
        for dropdown in graph["top_dropdowns"]:
            signal_selector = LayoutBuilder.signal_selector(graph_id, dropdown, True)
            signal_selectors.append(
                html.Div(
                    className="two columns",
                    children=[signal_selector],
                    style={"margin-right": "0.5%"},
                ),
            )
        signal_selectors.append(LayoutBuilder.filter_signal_list_cb(graph_id))

        # Main graph layout
        graphs = []
        main_graph = LayoutBuilder.graph_layout(
            f"{graph_id}", side_dropdowns=graph["side_dropdowns"], **graph["props"]
        )
        graphs.append(main_graph)

        # Time graph layout
        if graph["props"]["time_graph"]:
            time_graph = LayoutBuilder.graph_layout(
                f"{graph_id}_time",
                data_events=True,
                hidden=True,
            )
            graphs.append(time_graph)

        # Wrap all together
        layout = html.Div(
            className="container",
            children=[
                html.H4(graph["title"]),
                LayoutBuilder.large_signal_alert(graph_id),
                html.Div(className="row", children=signal_selectors),
                html.Div(children=graphs),
            ],
        )
        return layout
