from pathlib import Path
from persefone.utils.filesystem import tree_from_underscore_notation_files
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG
from persefone.data.databases.mongo.nodes.nodes import MNode
from persefone.data.databases.mongo.nodes.buckets.datasets import DatasetsBucket
import click
import yaml
from tqdm import tqdm
import time
import multiprocessing as mp
from persefone.data.databases.filesystem.underfolder import UnderfolderDatabase


@click.command("Converts Folder dataset into MongoDB Dataset")
@click.option('--database_host', default='localhost', help="Database host")
@click.option('--database_port', default=27017, help="Database post")
@click.option('--database_user', default='root', help="Database username")
@click.option('--database_pass', default='', help="Database password")
@click.option('--database_name', required=True, help="Database name")
@click.option('--folder', required=True, help="Input Dataset folder")
@click.option('--new_dataset_name', required=True, help="New Dataset desired name")
@click.option('--overwrite', default=False, help="Overwrite dataset with same name")
def folder_to_mongo(
        database_host,
        database_port,
        database_user,
        database_pass,
        database_name,
        folder,
        new_dataset_name,
        overwrite):

    num_workers = 8
    database_cfg = {
        'host': database_host,
        'port': database_port,
        'user': database_user,
        'pass': database_pass,
        'db': database_name
    }
    bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG.from_dict(database_cfg))

    ufolder = UnderfolderDatabase(folder)

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

    dataset = bucket.new_dataset(new_dataset_name, metadata=ufolder.metadata)
    assert dataset is not None
    bucket.disconnect()

    samples_queue = mp.JoinableQueue()  # maxsize=0)

    [samples_queue.put(item_data) for item_data in ufolder.skeleton]

    total_size = samples_queue.qsize()
    worker_approx_size = total_size // num_workers

    def inserter(worker_id: int, worker_size: int, q: mp.JoinableQueue):

        bucket = DatasetsBucket(client_cfg=MongoDatabaseClientCFG.from_dict(database_cfg))

        with tqdm(total=worker_size, position=w) as pbar:
            while not q.empty():
                item_data = q.get()
                sample_id = item_data['_id']
                metadata = {}

                if 'metadata' in item_data:
                    metadata = ufolder.load_data(item_data['metadata'])

                sample: MNode = bucket.new_sample(new_dataset_name, metadata=metadata, sample_id=sample_id)
                assert sample is not None

                for item_name, filename in item_data.items():
                    if item_name != 'metadata' and item_name != '_id':
                        bucket.new_item(
                            new_dataset_name,
                            sample_id,
                            item_name,
                            open(filename, 'rb').read(),
                            Path(filename).suffix.replace('.', '')
                        )
                pbar.update(1)
                q.task_done()

    for w in range(num_workers):
        worker = mp.Process(target=inserter, args=(w, worker_approx_size, samples_queue,))  # , daemon=True)
        worker.start()

    samples_queue.join()


if __name__ == "__main__":
    folder_to_mongo()
