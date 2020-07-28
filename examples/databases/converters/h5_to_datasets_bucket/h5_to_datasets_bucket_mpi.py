import multiprocessing
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG
from persefone.data.databases.mongo.nodes.nodes import MNode
from persefone.data.databases.h5 import H5SimpleDatabase
from persefone.data.databases.mongo.nodes.buckets.datasets import DatasetsBucket
import numpy as np
import click
from tqdm import tqdm


def process_keys(database_cfg, h5_file, new_dataset_name, job_id, keys, chunk_size):

    bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG(filename=database_cfg))
    dataset = bucket.get_dataset(new_dataset_name)
    db = H5SimpleDatabase(filename=h5_file)

    sample_id_counter = chunk_size * job_id
    internal_counter = 0
    with db:
        for key in keys:

            h5_sample = db[key]

            metadata = {}  # TODO: Conversion with external mapper!!!
            for k, v in dict(h5_sample.attrs).items():
                if isinstance(v, np.int64):
                    v = int(v)
                metadata[k] = v

            sample: MNode = bucket.new_sample(new_dataset_name, metadata=metadata, sample_id=sample_id_counter)
            sample_id = int(sample.last_name)
            assert sample_id == sample_id_counter
            sample_id_counter += 1
            internal_counter += 1

            if sample_id_counter % 100 == 0:
                print("Job", job_id, f'{(internal_counter/chunk_size)*100:.2f}%')

            # print(dict(item.attrs))
            for h5_item_name, h5_data in h5_sample.items():
                bucket.new_item(
                    new_dataset_name,
                    sample_id,
                    h5_item_name,
                    bytes(h5_data[...]),
                    h5_data.attrs['_encoding']
                )


@click.command("Converts H5 dataset into MongoDB Dataset")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--h5_file', required=True, help="H5 Dataset file to convert")
@click.option('--new_dataset_name', required=True, help="New Dataset desired name")
@click.option('--workers', default=1, help="How many workers to use")
def h5_to_datasets_bucket(database_cfg, h5_file, new_dataset_name, workers):

    bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG(filename=database_cfg))
    dataset = bucket.new_dataset(new_dataset_name)
    del bucket

    db = H5SimpleDatabase(filename=h5_file)

    keys = []
    db = H5SimpleDatabase(filename=h5_file)
    with db:
        keys = list(db.keys)

    # Compute workers chunks
    job_numbers = workers
    total = len(keys)
    chunk_size = total // job_numbers
    chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]

    # Builds Jobs
    jobs = []
    for job_id, keys_slice in enumerate(chunks):
        print("JOB ID CREATED", job_id, len(keys_slice))
        j = multiprocessing.Process(target=process_keys, args=(
            database_cfg,
            h5_file,
            new_dataset_name,
            job_id,
            keys_slice,
            chunk_size
        ))
        jobs.append(j)

    for j in jobs:
        j.start()


if __name__ == "__main__":
    h5_to_datasets_bucket()
