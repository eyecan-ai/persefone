from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset
from persefone.data.databases.mongo.readers import MongoDatasetReader
from persefone.data.io.drivers.safefs import SafeFilesystemDriver
import click
import cv2


@click.command("Converts H5 dataset into MongoDB Dataset")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--driver_cfg', default='database_driver.yml', help="Database IO Driver configuration file", show_default=True)
@click.option('--dataset_name', required=True, help="New Dataset desired name")
def mongo_datasets_viewer(database_cfg, driver_cfg, dataset_name):

    driver = SafeFilesystemDriver.create_from_configuration_file(driver_cfg)
    mongo_client = MongoDatabaseClient.create_from_configuration_file(filename=database_cfg)

    # Mongo Dataset handle
    mongo_dataset = MongoDataset(
        mongo_client,
        dataset_name,
        '',
        {driver.driver_name(): driver}
    )

    # Dataset Reader
    dataset_reader = MongoDatasetReader.create_from_configuration_file(
        mongo_dataset=mongo_dataset,
        filename='reader_config.yml'  # TODO is wrong!
    )

    # Iterate retrieved samples
    for sample in dataset_reader:

        print(sample.keys())

        repr = ''
        for key, value in dataset_reader.data_mapping.items():
            if 'item' not in key:
                repr += f'{value}={sample[value]}; '
        print(repr)

        image = sample['x']

        cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        c = cv2.waitKey(0)
        if c == ord('q'):
            break


if __name__ == "__main__":
    mongo_datasets_viewer()
