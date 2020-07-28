from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG
from persefone.data.databases.mongo.nodes.nodes import MNode
from persefone.data.databases.h5 import H5SimpleDatabase
from persefone.data.databases.mongo.nodes.buckets.datasets import DatasetsBucket
import numpy as np
import click
from tqdm import tqdm


@click.command("Converts H5 dataset into MongoDB Dataset")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--h5_file', required=True, help="H5 Dataset file to convert")
@click.option('--new_dataset_name', required=True, help="New Dataset desired name")
def h5_to_datasets_bucket(database_cfg, h5_file, new_dataset_name):

    bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG(filename=database_cfg))
    db = H5SimpleDatabase(filename=h5_file)

    dataset = bucket.new_dataset(new_dataset_name)
    assert dataset is not None

    with db:
        samples_counter = 0
        for h5_sample in tqdm(db):
            metadata = {}  # TODO: Conversion with external mapper!!!
            for k, v in dict(h5_sample.attrs).items():
                if isinstance(v, np.int64):
                    v = int(v)
                metadata[k] = v

            sample: MNode = bucket.new_sample(new_dataset_name, metadata=metadata, sample_id=samples_counter)
            samples_counter += 1
            sample_id = int(sample.last_name)

            # print(dict(item.attrs))
            for h5_item_name, h5_data in h5_sample.items():
                bucket.new_item(
                    new_dataset_name,
                    sample_id,
                    h5_item_name,
                    bytes(h5_data[...]),
                    h5_data.attrs['_encoding']
                )


if __name__ == "__main__":
    h5_to_datasets_bucket()
