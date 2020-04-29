"""Console script for persefone."""
import sys
import click


@click.command()
def main(args=None):
    """ djiasdjasoidj asoid

    Keyword Arguments:
        args {[type]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """
    click.echo("Replace this message by putting your code into "
               "persefone.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
