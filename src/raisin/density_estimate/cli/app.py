import sys

import typer
from density_estimate import streamlit_app
from streamlit.web import cli as stcli

app = typer.Typer()

@app.command()
def __version__():
    # Print the version of the app
    typer.echo("0.1.0")

@app.command()
def run():
    sys.argv = ["streamlit", "run", "density_estimate/streamlit_app.py"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    app()