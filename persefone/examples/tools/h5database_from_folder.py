import click
from persefone.data.databases.h5 import H5DatabaseIO


@click.command()
@click.option('--folder', required=True, type=str, help='Input data folder.')
@click.option('--output_file', required=True, type=str, help='hdf5 output filename.')
def h5database_from_folder(folder, output_file):
    """ Convert data folder with Underscore notation in a H5Database. """

    H5DatabaseIO.generate_from_folder(output_file, folder)


if __name__ == '__main__':
    h5database_from_folder()
