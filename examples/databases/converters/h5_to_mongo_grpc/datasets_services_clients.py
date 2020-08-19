from persefone.interfaces.grpc.clients.datasets_services import DatasetsSimpleServiceClient
import logging
import click
from persefone.utils.cli_options import cli_base_options, cli_host_options
from persefone.data.databases.h5 import H5SimpleDatabase
from tqdm import tqdm
import numpy as np


@click.command("Launch Datasets Services Server")
@cli_base_options
@cli_host_options
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--driver_cfg', default='database_driver.yml', help="Database IO Driver configuration file", show_default=True)
@click.option('--h5_file', required=True, help="H5 Dataset file to convert")
@click.option('--new_dataset_name', required=True, help="New Dataset desired name")
@click.option('--new_dataset_category', required=True, help="New Dataset desired category name")
def h5_to_mongo_through_grpc(**options):

    logging.basicConfig(level=logging.DEBUG if options.get('debug') else logging.WARNING)

    new_dataset_name = options.get('new_dataset_name')
    new_dataset_category = options.get('new_dataset_category')

    client = DatasetsSimpleServiceClient(host=options.get('host'), port=options.get('port'))

    db = H5SimpleDatabase(filename=options.get('h5_file'))

    with db:

        dataset = client.new_dataset(new_dataset_name, new_dataset_category)
        assert dataset is not None, f"Problem creating new dataset with name [{new_dataset_name}]"

        for h5_sample in tqdm(db):

            metadata = {}  # TODO: Conversion with external mapper!!!
            for k, v in dict(h5_sample.attrs).items():
                if isinstance(v, np.int64):
                    v = int(v)
                metadata[k] = v

            sample = client.new_sample(dataset_name=new_dataset_name, metadata=metadata)
            assert sample is not None, f"Problem creating new sample"
            print(sample['sample_id'])

            # print(dict(item.attrs))
            for h5_item_name, h5_data in h5_sample.items():

                item = client.new_item(new_dataset_name, sample['sample_id'], h5_item_name, bytes(h5_data[...]), h5_data.attrs['_encoding'])
                assert item is not None, f"Proble creating item"


if __name__ == "__main__":
    h5_to_mongo_through_grpc()
