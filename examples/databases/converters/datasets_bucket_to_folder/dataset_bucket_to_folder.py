import time
import shutil
import json
import copy
import math
from pathlib import Path
from queue import Queue
from threading import Thread

import click
from tqdm import tqdm
from progress.bar import ShadyBar

from persefone.data.databases.mongo.nodes.buckets.datasets import DatasetsBucket
from persefone.data.databases.mongo.nodes.nodes import MNode
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG


@click.command("Exports Mongo Buckets dataset into folder")
@click.option('--database_cfg', default='database.yml', help="Database configuration file", show_default=True)
@click.option('--folder', required=True, help="Output dataset folder")
@click.option('--dataset_name', required=True, help="Input dataset name")
@click.option('--overwrite', default=False, type=click.BOOL, help="Overwrite folder with same name")
@click.option('--workers', '-w', default=8, type=click.INT, help='Number of processes')
@click.option('--ids_type', default='int', help='"int" or "str"')
def datasets_bucket_to_folder(database_cfg, folder, dataset_name, ids_type, overwrite, workers):
    t1 = time.time()
    bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG(filename=database_cfg))
    folder = Path(folder).expanduser()

    # Make folder
    if folder.is_dir():
        if overwrite:
            shutil.rmtree(folder)
        else:
            raise ValueError('Output folder already exists')
    folder.mkdir(parents=True, exist_ok=True)

    # Get samples queue
    samples = bucket.get_samples(dataset_name)
    samples_queue = Queue(maxsize=0)
    [samples_queue.put(sample) for sample in tqdm(samples)]

    # Compute padding if ids_type is int
    if ids_type == 'int':
        padding = math.ceil(math.log(len(samples) + 1, 10))
    else:
        padding = 0

    def exporter(q: Queue, bar: ShadyBar):
        # Reconnect after fork
        bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG(filename=database_cfg))

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
                with open(folder / ('%s_%s.%s' % (sample_id_name, item.last_name, enc)), 'wb') as f:
                    f.write(data)

            # Sanitize metadata
            metadata = copy.deepcopy(sample.metadata)
            for k in sample.metadata:
                if k.startswith('#'):
                    del metadata[k]

            # Save metadata
            with open(folder / ('%s_%s.%s' % (sample_id_name, 'metadata', 'json')), 'w') as f:
                json.dump(metadata, f, indent=4)

            # Next sample
            bar.next()
            q.task_done()

    # Loading BAR
    bar = ShadyBar('Exporting data', max=len(samples), suffix='%(percent).1f%% - %(remaining)ds')

    # Disconnect before fork
    bucket.disconnect()

    # Spawn workers
    for _ in range(workers):
        worker = Thread(target=exporter, args=(samples_queue, bar,), daemon=True)
        worker.start()

    samples_queue.join()
    t2 = time.time()
    print("Export time: ", t2 - t1)


if __name__ == "__main__":
    datasets_bucket_to_folder()
