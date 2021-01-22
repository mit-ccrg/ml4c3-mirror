# Imports: standard library
import argparse

# Imports: first party
from clustering.ui import app
from clustering.ui.layout import get_layout
from clustering.ui.globals import TITLE
from clustering.ui.callbacks import set_callbacks


def run_cluster_ui(args):
    app.title = TITLE
    app.layout = get_layout(args.output_folder)
    set_callbacks(app, output_path=args.output_folder, hd5_path=args.tensors)
    app.run_server(host="0.0.0.0", port=args.port, debug=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", help="Port to use", default="8050")
    parsed_args = parser.parse_args()
    run_cluster_ui(parsed_args)
