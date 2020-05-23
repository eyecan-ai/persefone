from persefone.data.databases.pandas import PandasDatabase
from persefone.data.databases.h5 import H5SimpleDatabase


class H5SimpleDatabaseUtils(object):

    @classmethod
    def h5files_to_pandas_database(cls, h5files, include_filenames=True):
        """Generates a tabular representation of multiple hdf5 files

        :param h5files: list of hdf5 files
        :type h5files: list
        :param include_filenames: TRUE to add column with reference hdf5 filename, defaults to True
        :type include_filenames: bool, optional
        :return: tabular representation of multiple H5SimpleDatabase with single PandasDatabase 
        :rtype: PandasDatabase
        """
        merged_database = PandasDatabase()
        for h5file in h5files:
            database = H5SimpleDatabase(filename=h5file, readonly=True)
            with database:
                tabular = database.generate_tabular_representation(include_filename=include_filenames)
                merged_database += PandasDatabase(data=tabular)

        return merged_database
