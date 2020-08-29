from pathlib import Path
from threading import Thread
from typing import Tuple
from persefone.utils.filesystem import tree_from_underscore_notation_files
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG
from persefone.data.databases.mongo.nodes.nodes import MNode
from persefone.data.databases.mongo.nodes.buckets.datasets import DatasetsBucket
import click
import yaml
from tqdm import tqdm
from mongoengine.errors import DoesNotExist
from queue import Queue
import time
from progress.bar import ShadyBar


@click.command("Converts Folder dataset into MongoDB Dataset")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--folder', required=True, help="Input Dataset folder")
@click.option('--new_dataset_name', required=True, help="New Dataset desired name")
@click.option('--overwrite', default=False, help="Overwrite dataset with same name")
def folder_to_datasets_bucket(database_cfg, folder, new_dataset_name, overwrite):

    num_workers = 8
    bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG(filename=database_cfg))

    tree = tree_from_underscore_notation_files(folder)

    try:
        old_dataset = bucket.get_dataset(new_dataset_name)
        if overwrite:
            print("Deleting previous dataset ...")
            t1 = time.time()
            bucket.delete_dataset(old_dataset.last_name, num_workers=num_workers)
            t2 = time.time()
            print("Deletion time: ", t2 - t1)
        else:
            print("Dataset with same name exists!")
            return
    except Exception as e:
        print(e)

    # return
    t1 = time.time()
    dataset = bucket.new_dataset(new_dataset_name)
    assert dataset is not None

    samples_queue = Queue(maxsize=0)
    [samples_queue.put((sample_id, item_data)) for sample_id, item_data in tqdm(tree.items())]

    def inserter(q: Queue, bar: ShadyBar):
        while not q.empty():
            sample_id, item_data = q.get()
            metadata = {}
            if 'metadata' in item_data:
                metadata = yaml.safe_load(open(item_data['metadata'], 'r'))

            sample: MNode = bucket.new_sample(new_dataset_name, metadata=metadata, sample_id=int(sample_id))

            for item_name, filename in item_data.items():
                if item_name != 'metadata':
                    bucket.new_item(
                        new_dataset_name,
                        sample_id,
                        item_name,
                        open(filename, 'rb').read(),
                        Path(filename).suffix.replace('.', '')
                    )
            bar.next()
            q.task_done()

    # Loading BAR
    bar = ShadyBar('Inserting data', max=len(tree), suffix='%(percent).1f%% - %(remaining)ds')

    workers = []
    for w in range(num_workers):
        worker = Thread(target=inserter, args=(samples_queue, bar,), daemon=True)
        worker.start()

    samples_queue.join()
    t2 = time.time()
    print("Inserting time: ", t2 - t1)


if __name__ == "__main__":
    folder_to_datasets_bucket()
