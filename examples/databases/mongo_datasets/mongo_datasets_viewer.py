from persefone.data.databases.h5 import H5SimpleDatabase
from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset, MongoDatasetReader
from persefone.data.io.drivers.safefs import SafeFilesystemDriver
import numpy as np
import click
import cv2
import time


@click.command("Converts H5 dataset into MongoDB Dataset")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--driver_cfg', default='database_driver.yml', help="Database IO Driver configuration file", show_default=True)
@click.option('--dataset_name', required=True, help="New Dataset desired name")
def mongo_datasets_viewer(database_cfg, driver_cfg, dataset_name):

    driver = SafeFilesystemDriver.create_from_configuration_file(driver_cfg)
    mongo_client = MongoDatabaseClient.create_from_configuration_file(filename=database_cfg)

    mongo_dataset = MongoDataset(
        mongo_client,
        dataset_name,
        '',
        {driver.driver_name(): driver}
    )

    data_mapping = {
        'sample_id': 'sample_id',
        'metadata.classification_label': 'label',
        'metadata.hook_left': 'hook_left',
        'metadata.setup_name': 'setup_name',
        'items.image': 'x'
    }

    queries = [
        'sample_id > 0',
        'metadata.classification_label in [0]',
        'metadata.labeling_class in ["good","bad"]'
    ]

    orders = [
        '-metadata.hook_left'
    ]

    dataset_reader = MongoDatasetReader(
        mongo_dataset=mongo_dataset,
        data_mapping=data_mapping,
        queries=queries,
        orders=orders
    )

    for sample in dataset_reader:

        print(sample.keys())
        print("Sampleid:", sample['sample_id'], sample['label'], sample['hook_left'], sample['setup_name'])
        image = sample['x']

        cv2.imshow("image", image)
        c = cv2.waitKey(1)
        if c == ord('q'):
            break


if __name__ == "__main__":
    mongo_datasets_viewer()
