from persefone.data.databases.filesystem.underfolder import UnderfolderDatabase
import time
import shutil
import json
import copy
import math
from pathlib import Path
from queue import Queue
from threading import Thread
import multiprocessing as mp

import click
from tqdm import tqdm

from persefone.data.databases.mongo.nodes.buckets.datasets import DatasetsBucket
from persefone.data.databases.mongo.nodes.nodes import MNode
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG
import yaml


@click.command("Exports Mongo Buckets dataset into folder")
@click.option('--database_host', default='localhost', help="Database host")
@click.option('--database_port', default=27017, help="Database post")
@click.option('--database_user', default='root', help="Database username")
@click.option('--database_pass', default='', help="Database password")
@click.option('--database_name', required=True, help="Database name")
@click.option('--folder', required=True, help="Output dataset folder")
@click.option('--dataset_name', required=True, help="Input dataset name")
@click.option('--overwrite', default=False, type=click.BOOL, help="Overwrite folder with same name")
@click.option('--workers', '-w', default=10, type=click.INT, help='Number of processes')
@click.option('--ids_type', default='int', help='"int" or "str"')
@click.option('--plain_output', default=False, help='TRUE to use plain output folder format')
@click.option('--metadata_format', default='yml', help='"json" or "yml"')
def mongo_to_folder(
    database_host,
    database_port,
    database_user,
    database_pass,
    database_name,
    folder,
    dataset_name,
    ids_type,
    overwrite,
    workers,
    plain_output,
    metadata_format
):

    database_cfg = {
        'host': database_host,
        'port': database_port,
        'user': database_user,
        'pass': database_pass,
        'db': database_name
    }

    t1 = time.time()

    # get dataset
    bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG.from_dict(database_cfg))
    dataset = bucket.get_dataset(dataset_name)
    assert dataset is not None, f"Dataset '{dataset_name}' not found!"

    folder = Path(folder).expanduser()

    # Make folder
    if folder.is_dir():
        if overwrite:
            # folder.unlink()
            shutil.rmtree(folder)
        else:
            raise ValueError('Output folder already exists')
    folder.mkdir(parents=True, exist_ok=True)

    # data folder for generated files
    data_folder = folder

    # data folder in subfolder for UnderFolder format if plain_output is set to FALSE
    if not plain_output:
        data_folder = folder / UnderfolderDatabase.DATA_SUBFOLDER
        data_folder.mkdir(parents=True, exist_ok=True)

        metadata = dataset.to_mongo()['metadata']
        import pprint
        print("XXX"*100)
        for k, v in metadata.items():
            print(k, type(v))
        pprint.pprint(metadata)
        if metadata:
            if metadata_format == 'yml':
                yaml.safe_dump(metadata, open(folder / 'metadata.yml', 'w'))
            elif metadata_format == 'json':
                json.dump(metadata, open(folder / 'metadata.json', 'w'))

    # Get samples queue
    samples = bucket.get_samples(dataset_name)
    samples_queue = mp.JoinableQueue()  # Queue(maxsize=0)
    [samples_queue.put(sample) for sample in tqdm(samples)]

    total_size = samples_queue.qsize()
    worker_approx_size = total_size // workers

    # Compute padding if ids_type is int
    if ids_type == 'int':
        padding = math.ceil(math.log(len(samples) + 1, 10))
    else:
        padding = 0

    def exporter(w: int, worker_size: int, q: mp.JoinableQueue):
        # Reconnect after fork
        bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG.from_dict(database_cfg))

        with tqdm(total=worker_size, position=w) as pbar:
            while not q.empty():
                # Retrieve info
                sample = q.get()
                sample_id = sample.metadata['#sample_id']
                items = bucket.get_items(dataset_name, sample_id)

                # Compute sample id name
                if ids_type == 'int':
                    sample_id_name = str(int(sample_id)).zfill(padding)
                else:
                    sample_id_name = str(sample_id)

                # Save all items
                for item in items:
                    item: MNode
                    data, enc = item.get_data()
                    with open(data_folder / ('%s_%s.%s' % (sample_id_name, item.last_name, enc)), 'wb') as f:
                        f.write(data)

                # Sanitize metadata
                metadata = sample.plain_metadata  # copy.deepcopy(sample.metadata)
                for k in sample.metadata:
                    if k.startswith('#'):
                        del metadata[k]

                # Save metadata
                with open(data_folder / ('%s_%s.%s' % (sample_id_name, 'metadata', metadata_format)), 'w') as f:
                    if metadata_format == 'yml':
                        yaml.safe_dump(metadata, f)
                    elif metadata_format == 'json':
                        json.dump(metadata, f)

                # Next sample
                q.task_done()
                pbar.update(1)

    # Disconnect before fork
    bucket.disconnect()

    # Spawn workers
    for wid in range(workers):
        worker = mp.Process(target=exporter, args=(wid, worker_approx_size, samples_queue,))
        worker.start()

    samples_queue.join()
    t2 = time.time()
    print("Export time: ", t2 - t1)


if __name__ == "__main__":
    mongo_to_folder()
