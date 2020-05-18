from persefone.data.databases.pandas import PandasDatabase
from persefone.data.databases.h5 import H5SimpleDatabase


class H5SimpleDatabaseUtils(object):

    @classmethod
    def h5files_to_pandas_database(cls, h5files, include_filenames=True):
        merged_database = PandasDatabase()
        for h5file in h5files:
            database = H5SimpleDatabase(filename=h5file, readonly=True)
            with database:
                tabular = database.generate_tabular_representation(include_filename=include_filenames)
                merged_database += PandasDatabase(data=tabular)

        return merged_database
