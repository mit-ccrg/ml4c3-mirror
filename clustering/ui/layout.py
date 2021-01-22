# Imports: standard library
import os
from collections import OrderedDict

# Imports: third party
import dash_core_components as dcc
import dash_html_components as html

# Imports: first party
from clustering.ui.globals import STEPS, TITLE
from clustering.ui.layout_tools import LayoutBuilder


def banner():
    banner = html.Div(
        [
            html.H2(
                TITLE,
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


def data_selector_block(data_dir):
    stored_csv = [file for file in os.listdir(data_dir) if file.endswith(".csv")]
    csv_load_dd = dcc.Dropdown(
        id="stored-csv-dd",
        options=[{"label": file, "value": file} for file in stored_csv],
        className="two columns",
        placeholder="Select a file to load",
    )
    reextract_btn = html.Button("Reextract data", id="reextract-data-btn")

    raw_data_btn = html.Button("Use raw data", id="raw-data-btn")
    rext_dummy_div = html.Div(id="reextract-data-dummy-div")
    raw_dummy_div = html.Div(id="raw-data-dummy-div")
    ext_div = html.Div(
        [csv_load_dd, reextract_btn, raw_data_btn, rext_dummy_div, raw_dummy_div],
    )

    stored_bundles = [file for file in os.listdir(data_dir) if file.endswith(".bundle")]
    stored_data = dcc.Dropdown(
        id="stored-bundle-dd",
        options=[{"label": file, "value": file} for file in stored_bundles],
        className="two columns",
        placeholder="Select a file to load",
    )
    load_data_btn = html.Button(
        "Load data",
        id="load-data-btn",
        className="two columns",
    )
    load_data_dummy_div = html.Div(id="load-data-dummy-div", className="two columns")
    data_name_input = dcc.Input(
        id="file-name-input",
        placeholder="File name",
        className="two columns",
    )
    store_data_btn = html.Button(
        "Store data",
        id="store-data-btn",
        className="two columns",
    )
    store_data_dummy_div = html.Div(id="store-data-dummy-div", className="two columns")
    load_and_store_div = html.Div(
        [
            stored_data,
            load_data_btn,
            load_data_dummy_div,
            data_name_input,
            store_data_btn,
            store_data_dummy_div,
        ],
    )

    current_data_btn = html.Button(
        "Show bundle processes",
        id="see-bundle-processes-btn",
        className="three columns",
    )
    current_data_div = html.Div(
        id="current-bundle-processes-div",
        className="six columns",
    )
    current_bundle_div = html.Div([current_data_btn, current_data_div])

    console_btn = html.Button(
        "Launch console",
        id="bundle-console-btn",
        className="two columns",
    )
    console_btn_dummy_div = html.Div(
        id="bundle-console-dummy-div",
        className="two columns",
    )
    console_div = html.Div([console_btn, console_btn_dummy_div])

    layout = html.Div(
        className="container",
        children=[
            html.H4("Select data"),
            html.Div(className="row", children=[ext_div]),
            html.Div(className="row", children=[load_and_store_div]),
            html.Div(className="row", children=[current_bundle_div]),
            html.Div(className="row", children=[console_div]),
        ],
    )
    return layout


def preprocessing_block():
    dropdowns = []
    opts_div = []
    buttons = []
    dummies = []
    PREPROCESS_STEPS = STEPS["preprocess"]
    for step in PREPROCESS_STEPS:
        methods = []
        for method in PREPROCESS_STEPS[step].methods():
            method_name = method.replace("_", " ")
            method_name = method_name.replace("meth ", "")
            method_name = method_name.capitalize()
            methods.append({"label": method_name, "value": method})

        step_name = step.replace("_", " ")
        step_name = step_name.capitalize()

        dropdowns.append(
            dcc.Dropdown(
                id=f"{step}-dd",
                options=methods,
                placeholder=f"Select a {step_name} technique",
                searchable=True,
                multi=False,
                className="two columns",
                style={"margin-top": "5px"},
            ),
        )
        opts_div.append(html.Div(id=f"{step}-kwargs-div", className="two columns"))
        buttons.append(html.Button(step, id=f"{step}-btn", className="two columns"))
        dummies.append(html.Div(step, id=f"dummy-div-{step}"))

    layout = html.Div(
        className="container",
        children=[
            html.H4("Preprocess data"),
            html.Div(className="row", children=dropdowns),
            html.Div(className="row", children=opts_div),
            html.Div(className="row", children=buttons),
            html.Div(className="row", children=dummies),
        ],
    )
    return layout


def summary_block():
    div = html.Div(
        className="container",
        children=[
            html.H4("Preprocessing summary"),
            html.Div(
                className="row",
                children=[
                    dcc.Dropdown(id="signal-summary-dd"),
                ],
            ),
            html.Div(id="preprocessing-table-div"),
            html.Button(
                "Report",
                id="preprocessing-report-btn",
                style={"margin-top": "20px"},
            ),
            html.Div(id="preprocessing-report-div", style={"margin": "30px"}),
        ],
    )
    return div


def graph_block():
    signal_selectors = []
    for dropdown in ["step", "patient", "signal"]:
        dd = dcc.Dropdown(
            id=f"{dropdown}-preprocess-dd",
            options=[],
            placeholder=f"Select {dropdown}",
            searchable=True,
            multi=dropdown != "patient",
            style={"margin-bottom": "5px"},
        )
        signal_selectors.append(
            html.Div(
                className="two columns",
                children=[dd],
                style={"margin-right": "0.5%"},
            ),
        )
    signal_selectors.append(html.Button("Plot graph", id="plot-preprocess-graph-btn"))
    signal_selectors.append(html.Button("Refresh", id="refresh-data-btn"))
    signal_selectors.append(
        dcc.Checklist(
            id="separate-cb",
            options=[{"label": "Separate steps", "value": "Separate"}],
            labelStyle={"display": "inline-block"},
        ),
    )

    # Wrap all together
    layout = html.Div(
        className="container",
        children=[
            html.H4("Visualize"),
            html.Div(className="row", children=signal_selectors),
            html.Div(children=[LayoutBuilder.empty_graph("visualize")]),
        ],
    )

    return layout


def clustering_options():
    dropdowns = []
    opts_div = []

    steps = OrderedDict({**STEPS["distance"], **STEPS["cluster"]})
    for step in steps:
        methods = []
        for method in steps[step].methods():
            method_name = method.replace("_", " ")
            method_name = method_name.replace("meth ", "")
            method_name = method_name.capitalize()
            methods.append({"label": method_name, "value": method})

        step_name = step.replace("_", " ")
        step_name = step_name.capitalize()

        dropdowns.append(
            dcc.Dropdown(
                id=f"{step}-dd",
                options=methods if step != "cluster-distance" else [],
                placeholder=f"Select a {step_name} technique",
                searchable=True,
                multi=False,
                className="three columns",
                style={"margin-top": "5px"},
            ),
        )
        opts_div.append(
            html.Div(
                id=f"{step}-kwargs-div",
                className="three columns",
            ),
        )

    distances_btn = html.Button(
        "Distances",
        id="distance-btn",
        style={"margin": "15px"},
    )
    cluster_btn = html.Button("Cluster", id="cluster-btn", style={"margin": "15px"})
    cluster_optim_btn = html.Button(
        "Optimize",
        id="cluster-optim-btn",
        style={"margin": "15px"},
    )

    layout = html.Div(
        className="container",
        children=[
            html.H4("Precompute distances"),
            html.Div(className="row", children=dropdowns[0]),
            html.Div(className="row", children=opts_div[0]),
            html.Div(className="row", children=distances_btn),
            html.H4("Cluster data"),
            html.Div(className="row", children=dropdowns[1:]),
            html.Div(className="row", children=opts_div[1:]),
            html.Div(className="row", children=[cluster_btn, cluster_optim_btn]),
            html.Div(id="dummy-div-distance", children=[]),
            html.Div(id="dummy-div-cluster", children=[]),
        ],
    )
    return layout


def clustering_graph():
    # Signals tab
    dd = dcc.Dropdown(
        id="signal-clustering-dd",
        options=[],
        placeholder=f"Select signal",
        searchable=True,
        multi=False,
        style={"margin-bottom": "5px"},
    )
    dropdowns_div = html.Div(
        className="two columns",
        children=[dd],
        style={"margin-right": "0.5%"},
    )
    plot_btn = html.Button("Plot graph", id="plot-signal-cluster-graph-btn")
    separate_cb = dcc.Checklist(
        id="signal-cluster-separate-cb",
        options=[{"label": "Separate clusters", "value": "separate"}],
        labelStyle={"display": "inline-block"},
    )
    signal_tab_div = html.Div(
        [
            html.Div(className="row", children=[dropdowns_div, plot_btn, separate_cb]),
            html.Div(children=[LayoutBuilder.empty_graph("signal-cluster")]),
        ],
    )
    signal_tab = dcc.Tab(
        label="By signal",
        value="signal-tab",
        children=signal_tab_div,
    )

    # PCA Tab
    methods = []
    for method in STEPS["reduce"]["reduce"].methods():
        method_name = method.replace("_", " ")
        method_name = method_name.replace("meth ", "")
        method_name = method_name.upper()
        methods.append({"label": method_name, "value": method})
    dim_reduction_dd = dcc.Dropdown(
        id="dim-reduction-dd",
        placeholder="Dimension reduction method",
        options=methods,
        className="three columns",
    )
    plt_btn = html.Button("Plot graph", id="plot-cluster-graph-btn")
    pca_tab_div = html.Div(
        [
            html.Div(className="row", children=[dim_reduction_dd, plt_btn]),
            html.Div(children=[LayoutBuilder.empty_graph("pca-cluster")]),
        ],
    )
    pca_tab = dcc.Tab(
        label="PCA visualization",
        value="pca-tab",
        children=pca_tab_div,
    )

    # Clusters info tab
    refresh_btn = html.Button("Refresh", id="cluster-info-refresh-btn")
    info_tab_div = html.Div(
        [
            html.Div(className="row", children=refresh_btn),
            html.Div(id="clusters-info-table"),
        ],
    )
    info_tab = dcc.Tab(
        label="Cluster info",
        value="info-tab",
        children=info_tab_div,
    )

    # Wrap all together
    layout = html.Div(
        className="container",
        children=[
            html.H4("Clusters"),
            dcc.Tabs(children=[signal_tab, pca_tab, info_tab]),
        ],
    )
    return layout


def get_layout(data_dir):
    layout = html.Div(
        style={"height": "100%"},
        children=[
            banner(),
            data_selector_block(data_dir),
            preprocessing_block(),
            summary_block(),
            graph_block(),
            clustering_options(),
            clustering_graph(),
        ],
    )
    return layout
