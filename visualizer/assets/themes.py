# Imports: third party
from plotly import io as pio
from plotly import graph_objects as go

pio.templates["icu"] = go.layout.Template(
    layout={
        "annotationdefaults": {"font": {"color": "white"}},
        "dragmode": "pan",
        "hovermode": "x",
        "paper_bgcolor": "white",
        "xaxis": {
            "type": "date",
            "automargin": True,
            "gridcolor": "#283442",
            "linecolor": "#506784",
            "ticks": "",
            "title": {"standoff": 15},
            "zerolinecolor": "#283442",
            "zerolinewidth": 2,
        },
        "yaxis": {
            "automargin": True,
            "gridcolor": "#283442",
            "linecolor": "#506784",
            "ticks": "",
            "title": {"standoff": 15},
            "zerolinecolor": "#283442",
            "zerolinewidth": 2,
        },
        "plot_bgcolor": "rgb(17,17,17)",
        "scene": {
            "xaxis": {
                "backgroundcolor": "rgb(17,17,17)",
                "gridcolor": "#506784",
                "gridwidth": 2,
                "linecolor": "#506784",
                "showbackground": True,
                "ticks": "",
                "zerolinecolor": "#C8D4E3",
            },
            "yaxis": {
                "backgroundcolor": "rgb(17,17,17)",
                "gridcolor": "#506784",
                "gridwidth": 2,
                "linecolor": "#506784",
                "showbackground": True,
                "ticks": "",
                "zerolinecolor": "#C8D4E3",
            },
        },
    },
)
