# Imports: third party
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html

# Imports: first party
from clustering.ui.assets import themes


class LayoutBuilder:
    """
    Tools to build the HTML layout.
    """

    @classmethod
    def empty_graph(cls, name: str, data=None):
        figure = go.Figure()

        if data:
            for signal in data:
                figure.add_trace(
                    go.Scatter(
                        x=signal.time,
                        y=signal.values,
                        name=signal.name,
                        showlegend=True,
                    ),
                )

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

        graph = dcc.Graph(figure=figure, id=f"{name}-graph")
        graph_div = html.Div(
            id=f"{name}_block",
            className="row",
            children=[
                dcc.Loading(
                    id=f"{name}-container",
                    className="ten columns",
                    children=[graph],
                ),
            ],
        )
        return graph_div
