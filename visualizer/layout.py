# Imports: third party
import dash_core_components as dcc
import dash_html_components as html

# Imports: first party
from visualizer.properties import LAYOUT
from visualizer.tools.layout_builder import LayoutBuilder


def create_options_menu(files_dir):
    options_div = html.Div(
        className="row",
        style={"margin": "8px 0px"},
        children=[
            html.Div(
                className="twelve columns",
                children=[
                    html.Div(
                        className="eight columns",
                        children=[
                            LayoutBuilder.upload_div(files_dir),
                            LayoutBuilder.visit_id_dd(),
                        ],
                    ),
                    html.Div(
                        className="four columns",
                        id="div-interval-control",
                        children=[
                            html.Div(
                                id="div-selected-mrn",
                                className="twelve columns",
                            ),
                            html.Div(
                                id="div-selected-csn",
                                className="twelve columns",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    return options_div


def create_layout(files_dir):

    layout = html.Div(
        style={"height": "100%"},
        children=[
            # Banner display
            LayoutBuilder.banner_layout(),
            html.Div(html.Div(id="help_div", children=[])),
            dcc.Store(id="static_data", storage_type="memory"),
            html.Div(
                className="container",
                style={"padding": "35px 25px"},
                children=[create_options_menu(files_dir)],
            ),
            html.Div(
                className="container",
                children=[
                    html.Button(id="display-patient-info-btn"),
                    html.Div(id="patient-info", children=[]),
                ],
            ),
            *[
                LayoutBuilder.graph_block(graph_id, graph_props)
                for graph_id, graph_props in LAYOUT["graphs"].items()
            ],
        ],
    )
    return layout
