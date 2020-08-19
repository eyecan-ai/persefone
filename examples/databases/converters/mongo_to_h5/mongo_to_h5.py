from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset
from persefone.data.databases.h5 import H5SimpleDatabase
from persefone.data.io.drivers.safefs import SafeFilesystemDriver
import click
from pathlib import Path
from tqdm import tqdm


@click.command("Converts H5 dataset into MongoDB Dataset")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--driver_cfg', default='database_driver.yml', help="Database IO Driver configuration file", show_default=True)
@click.option('--dataset_name', required=True, help="New Dataset desired name")
@click.option('--h5_filename', required=True, help="Output H5 filename")
@click.option('--zero_padding', default=6, help="Zero padding far sample renaming")
def mongo_datasets_viewer(database_cfg, driver_cfg, dataset_name, h5_filename, zero_padding):

    driver = SafeFilesystemDriver.create_from_configuration_file(driver_cfg)
    mongo_client = MongoDatabaseClient.create_from_configuration_file(filename=database_cfg)

    # Mongo Dataset handle
    mongo_dataset = MongoDataset(
        mongo_client,
        dataset_name,
        '',
        {driver.driver_name(): driver}
    )

    # Fetches ordered samples
    samples = mongo_dataset.get_samples(order_bys=['+sample_id'])
    n_samples = len(samples)

    database = H5SimpleDatabase(filename=h5_filename, readonly=False)

    with database:
        for sample_idx, sample in tqdm(enumerate(samples), total=n_samples):

            key = f'{database.DEFAULT_ROOT_ITEM}/{str(sample_idx).zfill(zero_padding)}'

            items = mongo_dataset.get_items(sample.sample_id)

            for item in items:

                for resource in item.resources:
                    data = mongo_dataset.fetch_resource_to_numpyarray(resource)
                    extension = Path(resource.uri).suffix
                    database.store_encoded_data(key, item.name, data, extension.replace('.', ''))
                    break

            group = database.get_group(key)
            group.attrs.update(sample.metadata)

    print(len(samples))


if __name__ == "__main__":
    mongo_datasets_viewer()
