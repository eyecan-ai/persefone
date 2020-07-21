from persefone.data.databases.h5 import H5SimpleDatabase
from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset
from persefone.data.io.drivers.safefs import SafeFilesystemDriver
import numpy as np
import click
from tqdm import tqdm


@click.command("Converts H5 dataset into MongoDB Dataset")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--driver_cfg', default='database_driver.yml', help="Database IO Driver configuration file", show_default=True)
@click.option('--h5_file', required=True, help="H5 Dataset file to convert")
@click.option('--new_dataset_name', required=True, help="New Dataset desired name")
@click.option('--new_dataset_category', required=True, help="New Dataset desired category name")
def h5_to_mongo(database_cfg, driver_cfg, h5_file, new_dataset_name, new_dataset_category):

    driver = SafeFilesystemDriver.create_from_configuration_file(driver_cfg)
    mongo_client = MongoDatabaseClient.create_from_configuration_file(filename=database_cfg)

    mongo_dataset = MongoDataset(
        mongo_client,
        new_dataset_name,
        new_dataset_category,
        {driver.driver_name(): driver}
    )

    db = H5SimpleDatabase(filename=h5_file)

    with db:
        for h5_sample in tqdm(db):

            metadata = {}  # TODO: Conversion with external mapper!!!
            for k, v in dict(h5_sample.attrs).items():
                if isinstance(v, np.int64):
                    v = int(v)
                metadata[k] = v

            mongo_sample = mongo_dataset.add_sample(metadata=metadata)

            # print(dict(item.attrs))
            for h5_item_name, h5_data in h5_sample.items():
                mongo_dataset.add_item(mongo_sample.sample_id, h5_item_name)
                mongo_dataset.push_resource_from_blob(
                    mongo_sample.sample_id,
                    h5_item_name,
                    h5_item_name,
                    h5_data[...],
                    h5_data.attrs['_encoding'],
                    driver.driver_name()
                )


if __name__ == "__main__":
    h5_to_mongo()
