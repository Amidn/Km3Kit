"""Console script for km3kit."""
import km3kit

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for km3kit."""
    console.print("Replace this message by putting your code into "
               "km3kit.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
