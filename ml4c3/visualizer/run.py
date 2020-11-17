# Imports: third party
from flask_caching import Cache

# Imports: first party
from ml4c3.visualizer import app
from ml4c3.visualizer.properties import load_config
from ml4c3.visualizer.static_callbacks import set_static_callbacks


# pylint: disable=import-outside-toplevel
def run(debug, port, address, files_dir, options):
    app.title = "HD5 visualizer"

    cache = Cache(
        app.server,
        config={
            "CACHE_TYPE": "filesystem",
            "CACHE_DIR": "cache-directory",
            "CACHE_THRESHOLD": 200,
        },
    )

    load_config(user_files=options)
    # Imports: first party
    from ml4c3.visualizer.layout import create_layout
    from ml4c3.visualizer.graphs_callbacks import set_dynamic_callbacks

    app.layout = create_layout(files_dir)
    set_static_callbacks(app)
    set_dynamic_callbacks(app, cache)
    app.run_server(debug=debug, host=address, port=port)


def run_server(args):
    run(args.debug, args.port, args.address, args.tensors, args.options_file)


if __name__ == "__main__":
    run(debug=True, port=8050, address="0.0.0.0", files_dir=None, options=None)
