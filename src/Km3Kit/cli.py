"""Console script for Km3Kit."""
import Km3Kit

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for Km3Kit."""
    console.print("Replace this message by putting your code into "
               "Km3Kit.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
