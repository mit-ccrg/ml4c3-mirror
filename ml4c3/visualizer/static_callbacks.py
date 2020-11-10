# Imports: standard library
import os
from typing import Dict

# Imports: third party
import h5py
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, State, Output

# Imports: first party
from ml4c3.definitions.icu import STATIC_UNITS, VISUALIZER_PATH
from ml4c3.visualizer.tools.tm_helper import TMapHelper

# pylint: disable=unused-variable, too-many-statements


def set_static_callbacks(app):
    """
    Sets the static callbacks: the ones that do not depend on properties.py.

    * Help Button
    * HD5 path input
    * HD5 file selector
    * Show patient info
    """

    @app.callback(
        [Output("help_div", "children"), Output("learn-more-button", "children")],
        [Input("learn-more-button", "n_clicks")],
    )
    def get_help(n_clicks):
        """
        Displays the help text when the 'Learn More Button' is clicked.
        """
        if n_clicks is None:
            n_clicks = 0

        if (n_clicks % 2) == 1:
            # Display the help message
            help_div = html.Div(
                className="container",
                style={"margin-bottom": "30px"},
                children=[read_help()],
            )
            help_btn_text = "Close"
        else:
            # Close the help message
            help_div = html.Div()
            help_btn_text = "Help"

        return help_div, help_btn_text

    @app.callback(
        Output("files_dd", "options"),
        [Input("input_path", "value")],
    )
    def list_files(dir_path: str):
        """
        Callback to populate files dropdown with the hd5 files on the input
        path.
        """
        files = []
        if dir_path:
            try:
                files = os.listdir(dir_path)
            except FileNotFoundError:
                pass
        options = [
            {"label": file, "value": file} for file in files if file.endswith(".hd5")
        ]
        return options

    @app.callback(
        [
            Output("visit_id_dd", "options"),
            Output("visit_id_dd", "value"),
            Output("div-selected-mrn", "children"),
        ],
        [Input("files_dd", "value")],
        [State("input_path", "value")],
    )
    def list_visit_ids(file: str, file_dir: str):
        """
        Callback to populate the visit IDs:
        Triggered:
          * When the user selects an HD5 from the hd5 dropdown
        Causes:
          * populates the visit_id dropdown with the visit ids on the selected hd5.
          * Updates the top right text with the selected file (MRN)
        """
        visit_ids = []
        file_name = "Not selected"

        # If the a file has been selected, extract visit IDs and the MRN
        if file and file_dir:
            file_path = os.path.join(file_dir, file)
            with h5py.File(file_path, "r") as hd5_file:
                visit_ids = list(hd5_file["bedmaster"])

            file_name = os.path.basename(file.split(".")[0])

        # Populate dropdown and change top-right text
        options = [{"label": vid, "value": vid} for vid in visit_ids]
        new_text = html.H6(
            f"Current MRN: {file_name}",
            style={"margin-top": "3px", "float": "right"},
        )
        return options, None, new_text

    @app.callback(
        [
            Output("patient-info", "children"),
            Output("display-patient-info-btn", "children"),
        ],
        [Input("display-patient-info-btn", "n_clicks")],
        [
            State("input_path", "value"),
            State("files_dd", "value"),
            State("visit_id_dd", "value"),
            State("static_data", "data"),
        ],
    )
    def patient_info(n_clicks, file_dir, file_name, visit_id, static_data):
        """
        Callback to display patient static data
        Triggered:
            * When the user clicks on 'Show Patient info'
        Causes:
            * Reads the static data stored on the browser and displays
              it in a markdown format.
        """
        demos_div = html.Div()
        btn_text = "Show Patient info"

        if file_dir and file_name and visit_id:
            if n_clicks is None:
                n_clicks = 0
            if (n_clicks % 2) == 1:
                n_clicks += 1
                demos_div = html.Div(children=display_static_data(static_data))
                btn_text = "Hide Patient info"
            n_clicks += 1

        return demos_div, btn_text


def read_help():
    """
    Creates a div in markdown format with the 'help.md' file content.
    """
    # Markdown files
    path = os.path.join(VISUALIZER_PATH, "help.md")
    with open(path, "r") as file:
        help_md = file.read()

    return dcc.Markdown(help_md, className="markdown", style={"margin": "10px"})


def get_static_data(static_template: Dict, hd5) -> Dict:
    """
    Extracts static data from the HD5 with tmaps.

    :param static_template: A dict with the structure of the static data. See
                            LAYOUT['statics'] on properties.py
    :param hd5: <h5py.File> the file to extract the data
    :param visit_id: <str> visit id of the data
    :return: <Dict> The static data coming from the hd5 structured as
             the static_template dict
    """
    statics_data = {}
    for category in static_template:
        values = {}
        for field in static_template[category]["fields"]:
            try:  # Extract data
                value = TMapHelper.get_static_data(field, hd5)
                values[field] = value
            except KeyError:  # Don't crash if the file does not contain a field
                pass

        category_type = static_template[category]["type"]

        if category == "Movements":
            department_list = values["department_nm"]
            department_list_no_rep = []
            for idx, dpt in enumerate(values["department_nm"]):
                if department_list.count(dpt) > 1:
                    cur_idx = department_list[:idx].count(dpt) + 1
                    dpt = f"{dpt} ({cur_idx})"
                department_list_no_rep.append(dpt)
            values["department_nm"] = department_list_no_rep

        statics_data[category] = {"fields": values, "type": category_type}

    return statics_data


def display_static_data(static_data):
    """
    Construct div with static data when 'Show Patient info' is clicked.

    :param static_data: <dict> the static data
    :return: div rows containing the formatted static data.
    """
    data_blocks = []
    for category, category_data in static_data.items():
        # Get the block title
        text_fields = [f"#### {category}"]

        # Generate tables
        if category_data["type"].startswith("table"):
            table_type = category_data["type"].split("-")[1]
            text_fields.append(_create_md_table(category_data["fields"], table_type))

        # Generate key-values
        elif category_data["type"] == "key-val":
            for field, value in category_data["fields"].items():
                field_name = (
                    field
                    if field not in STATIC_UNITS
                    else f"{field} ({STATIC_UNITS[field]})"
                )
                field_name = field_name.replace("_", " ").capitalize()
                if isinstance(value, list):
                    value = value[0]
                if isinstance(value, str):
                    value = value.replace("'", "")
                text_fields.append(f"**{field_name}**: {value}")

        # Unrecognized type
        else:
            raise ValueError(
                f"Invalid static category type "
                f"'{category_data['type']}'. Check properties.py ",
            )

        # Join info in a string in Markdown format
        div_text = "\n\n".join(text_fields)
        div = dcc.Markdown(
            div_text,
            className="markdown three columns",
            style={"margin": "20px"},
        )
        data_blocks.append(div)

    # Separate blocks in rows of three columns
    data_rows = [
        html.Div(data_blocks[i : i + 3], className="row")
        for i in range(0, len(data_blocks), 3)
    ]

    return data_rows


def _create_md_table(fields: Dict, category) -> str:
    """
    Converts static data into a Markdown table as a string.
    """

    def _row_to_dict(string_row):
        """
        Convert string table rows in dicts for easier conversion to df.
        """
        decoded_row = {}
        columns = string_row.split(";")

        for column in columns:
            split_column = column.split(":")
            if len(split_column) == 2:
                col_name, col_value = split_column
            else:
                col_name = split_column[0]
                col_value = "".join(split_column[1:])
            decoded_row[col_name] = col_value

        return decoded_row

    # Movements static data is already well formatted
    if category == "mov":
        dataframe = pd.DataFrame(fields)
        dataframe.reset_index(drop=True)

    # History static data is formatted as string. Needs _row_to_dict
    elif category == "hist":
        decoded_rows = []
        row_names = []
        for field_name, rows in fields.items():
            field_name = field_name.replace("_", " ").title()
            if isinstance(rows, str):
                rows = [rows]
            for row in rows:
                row_dict = _row_to_dict(row)
                decoded_rows.append(row_dict)
                row_names.append(field_name)

        dataframe = pd.DataFrame(decoded_rows, index=row_names)

    # Unknown category
    else:
        raise ValueError(
            f"Invalid table category 'table-{category}. " f"Check properties.py",
        )

    return dataframe.to_markdown()
