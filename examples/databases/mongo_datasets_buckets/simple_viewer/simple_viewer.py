from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG
from persefone.data.databases.mongo.nodes.nodes import MNode
from persefone.data.databases.h5 import H5SimpleDatabase
from persefone.data.databases.mongo.nodes.buckets.datasets import DatasetsBucket, DatasetsBucketReader, DatasetsBucketReaderCFG
import numpy as np
import click
from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset
from persefone.data.databases.mongo.readers import MongoDatasetReader
from persefone.data.io.drivers.safefs import SafeFilesystemDriver
import cv2
import time


@click.command("Converts H5 dataset into MongoDB Dataset")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--dataset_name', required=True, help="New Dataset desired name")
def mongo_datasets_viewer(database_cfg, dataset_name):

    bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG(filename=database_cfg))
    print("Dataset: ", bucket.get_dataset(dataset_name), bucket.count_samples(dataset_name))

    reader = DatasetsBucketReader.builds_from_configuration_file(bucket, dataset_name, 'reader_config.yml')

    for sample in reader:
        if 'label' in sample:
            print("Label:", sample['label'])

        if 'x' in sample:
            cv2.imshow("image", cv2.cvtColor(sample['x'], cv2.COLOR_RGB2BGR))
            c = cv2.waitKey(0)
            if c == ord('q'):
                break


if __name__ == "__main__":
    mongo_datasets_viewer()
