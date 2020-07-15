from persefone.data.databases.h5 import H5SimpleDatabase
from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset
from persefone.data.io.drivers.safefs import SafeFilesystemDriver, SafeFilesystemDriverCFG
import numpy as np
import click
import multiprocessing


def process_keys(job_id, keys, chunk_size, database_cfg, driver_cfg, h5_file, new_dataset_name, new_dataset_category):

    driver = SafeFilesystemDriver.create_from_configuration_file(driver_cfg)
    mongo_client = MongoDatabaseClient.create_from_configuration_file(filename=database_cfg)
    print(job_id, mongo_client)
    mongo_dataset = MongoDataset(
        mongo_client,
        new_dataset_name,
        new_dataset_category,
        {driver.driver_name(): driver}
    )

    db = H5SimpleDatabase(filename=h5_file, swmr=True)

    sample_id_counter = chunk_size * job_id
    with db:
        for key in keys:

            h5_sample = db[key]

            metadata = {}  # TODO: Conversion with external mapper!!!
            for k, v in dict(h5_sample.attrs).items():
                if isinstance(v, np.int64):
                    v = int(v)
                metadata[k] = v

            # print(job_id, "Counter: ", sample_id_counter)
            mongo_sample = mongo_dataset.add_sample(metadata=metadata, sample_id=sample_id_counter)
            # print(job_id, "Sample: ", sample_id_counter, mongo_sample.sample_id)
            sample_id_counter += 1

            if sample_id_counter % 100 == 0:
                print(job_id, '->', sample_id_counter)
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


@click.command("Converts H5 dataset into MongoDB Dataset")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--driver_cfg', default='database_driver.yml', help="Database IO Driver configuration file", show_default=True)
@click.option('--h5_file', required=True, help="H5 Dataset file to convert")
@click.option('--new_dataset_name', required=True, help="New Dataset desired name")
@click.option('--new_dataset_category', required=True, help="New Dataset desired category name")
@click.option('--workers', default=10, help="How many workers to use")
def h5_to_mongo(database_cfg, driver_cfg, h5_file, new_dataset_name, new_dataset_category, workers):

    keys = []
    db = H5SimpleDatabase(filename=h5_file)
    with db:
        keys = list(db.keys)

    driver = SafeFilesystemDriver.create_from_configuration_file(driver_cfg)
    mongo_client = MongoDatabaseClient.create_from_configuration_file(filename=database_cfg)
    MongoDataset(
        mongo_client,
        new_dataset_name,
        new_dataset_category,
        {driver.driver_name(): driver}
    )
    mongo_client.disconnect()

    print("Dataset ready...")
    import time
    time.sleep(1)

    job_numbers = workers
    total = len(keys)
    chunk_size = total // job_numbers
    chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]

    jobs = []
    for job_id, keys_slice in enumerate(chunks):
        print("JOB ID CREATED", job_id, len(keys_slice))
        j = multiprocessing.Process(target=process_keys, args=(
            job_id,
            keys_slice,
            chunk_size,
            database_cfg,
            driver_cfg,
            h5_file,
            new_dataset_name,
            new_dataset_category
        ))
        jobs.append(j)

    for j in jobs:
        j.start()


if __name__ == "__main__":
    h5_to_mongo()
