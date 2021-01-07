# Imports: standard library
from uuid import uuid4

# Imports: third party
import dash_auth
from flask_caching import Cache

# Imports: first party
from visualizer import app
from visualizer.properties import load_config
from visualizer.static_callbacks import set_static_callbacks


# pylint: disable=import-outside-toplevel
def run(debug, port, address, files_dir, options):
    app.title = "HD5 visualizer"

    random_token = str(uuid4())
    dash_auth.BasicAuth(
        app,
        {"aguirrelab": random_token},
    )

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
    from visualizer.layout import create_layout
    from visualizer.graphs_callbacks import set_dynamic_callbacks

    app.layout = create_layout(files_dir)
    set_static_callbacks(app)
    set_dynamic_callbacks(app, cache)

    print(
        f"""
    ****************
    To log in use:

    \t Username: aguirrelab
    \t Password: {random_token}

    ****************"
    """,
    )

    app.run_server(debug=debug, host=address, port=port)


def run_visualizer(args):
    run(args.debug, args.port, args.address, args.tensors, args.options_file)
