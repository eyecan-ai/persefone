

from pathlib import Path
import yaml
from persefone.data.databases.mongo.nodes.nodes import MLink, MNode
from mongoengine.errors import DoesNotExist
from persefone.data.databases.mongo.clients import MongoDatabaseClient
from persefone.data.databases.mongo.nodes.buckets.datasets import (
    DatasetsBucket,
    DatasetsBucketReader,
    DatasetsBucketSamplesListReader,
    DatasetsBucketSnapshot,
    DatasetsBucketSnapshotCFG
)
from persefone.utils.filesystem import tree_from_underscore_notation_files
from persefone.utils.bytes import DataCoding
import pytest
import numpy as np


class TestDatasetsBucket(object):

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_datasets(self, temp_mongo_database: MongoDatabaseClient, minimnist_folder):

        tree = tree_from_underscore_notation_files(minimnist_folder)
        n_samples = len(tree.items())
        n_items = -1
        impossible_dataset_name = "_ASDasdasdjasdas_IMPOSSIBLE!"

        R = DatasetsBucket(client_cfg=temp_mongo_database.cfg)
        print(R)

        datasets_names = ['Data_A', 'Data_B']
        dataset_metadata_map = {
            'Data_A': {
                'dataset_version': 1.0,
                'test_mode': True
            },
            'Data_B': None
        }

        for dataset_name in datasets_names:
            dataset = R.new_dataset(dataset_name, metadata=dataset_metadata_map[dataset_name])
            assert dataset.node_type == R.NODE_TYPE_DATASET, "dataset type is wrong!"

            with pytest.raises(NameError):
                dataset = R.new_dataset(dataset_name)

            assert dataset is not None, "Dataset creation should be valid"
            dataset_r = R.get_dataset(dataset_name)

            if dataset_metadata_map[dataset_name]:
                assert dataset_r.metadata

            assert dataset_r.node_type == R.NODE_TYPE_DATASET, "dataset type is wrong!"

            with pytest.raises(DoesNotExist):
                R.get_dataset(impossible_dataset_name)

            assert dataset == dataset_r, "Retrieved dataset is wrong"

            created_items = 0

            for sample_str, items in tree.items():
                n_items = len(items.items())

                sample: MNode = R.new_sample(dataset_name, {
                    'sample': sample_str,
                    'items': items.keys(),
                    'even': int(sample_str) % 2 == 0,
                    'sample_number': int(sample_str)
                })

                sample_id = sample.last_name
                sample_r = R.get_sample(dataset_name, sample_id)
                assert sample_r.node_type == R.NODE_TYPE_SAMPLE, "dataset type is wrong!"

                with pytest.raises(DoesNotExist):
                    R.get_sample(impossible_dataset_name, sample_id)

                with pytest.raises(DoesNotExist):
                    R.get_sample(dataset_name, n_samples * 10)

                with pytest.raises(DoesNotExist):
                    R.get_sample(impossible_dataset_name, n_samples * 10)

                assert sample == sample_r, "Retrieved sample is wrong"

                for item_name, filename in items.items():
                    blob, encoding = DataCoding.file_to_bytes(filename)

                    item: MNode = R.new_item(dataset_name, sample_id, item_name, blob_data=blob, blob_encoding=encoding)
                    created_items += 1

                    with pytest.raises(NameError):
                        R.new_item(dataset_name, sample_id, item_name, blob_data=blob, blob_encoding=encoding)

                    item_r: MNode = R.get_item(dataset_name, sample_id, item_name)
                    assert item_r.node_type == R.NODE_TYPE_ITEM, "dataset type is wrong!"
                    assert item == item_r, "Retrieved item is wrong!"

                    with pytest.raises(DoesNotExist):
                        R.get_item(impossible_dataset_name, sample_id, item_name)
                    with pytest.raises(DoesNotExist):
                        R.get_item(dataset_name, n_samples * 10, item_name)
                    with pytest.raises(DoesNotExist):
                        R.get_item(dataset_name, sample_id, impossible_dataset_name)
                    with pytest.raises(DoesNotExist):
                        R.get_item(impossible_dataset_name, n_samples * 10, impossible_dataset_name)

                    blob_r, encoding_r = item_r.get_data()
                    assert blob_r is not None, "Retrieved Blob is empty!"
                    assert encoding_r is not None, "Retrieved Blob encoding is empty!"

                    assert blob_r == blob, "Retrievd Blob is different from original one"
                    assert encoding_r == encoding, "Retrievd Blob encoding is different from original onw"

                    a = DataCoding.bytes_to_data(blob, encoding)
                    b = DataCoding.bytes_to_data(blob_r, encoding_r)

                    assert type(a) == type(b), "Decoding must produces same data!"
                    if isinstance(a, np.ndarray):
                        assert np.array_equal(a, b), "If data is an array, it should be consistent after decoding!"

                    # item = dataset.add_item(sample_idx, item_name)
                    # assert item is not None, "Item should be not None!"
                    # item_r = dataset.get_item(sample_idx, item_name)
                    # assert item_r is not None, "Retrieved Item should be not None!"

            assert len(R.get_items_by_query(dataset_name)) == created_items, "Number of items is wrong"

            assert len(R.get_samples(dataset_name)) == n_samples, "Number of samples is wrong"
            assert len(R.get_samples_by_query(dataset_name)) == n_samples, "Number of  queryed samples is wrong"

            samples_sub = R.get_samples_by_query(dataset_name, queries=['metadata.even == True'])
            assert len(samples_sub) > 0, "Event samples must be not empty"
            assert len(samples_sub) < n_samples, "Event samples must be not empty"

            samples_sub = R.get_samples_by_query(dataset_name, queries=['metadata.even == False'])
            assert len(samples_sub) > 0, "Event samples must be not empty"
            assert len(samples_sub) < n_samples, "Event samples must be not empty"

            samples_sub = R.get_samples_by_query(dataset_name, queries=['metadata.even == True', 'metadata.sample contains "0"'])
            assert len(samples_sub) > 0, "Event samples must be not empty"
            assert len(samples_sub) < n_samples, "Event samples must be not empty"

        n_datasets = len(datasets_names)
        datasets = R.get_datasets()
        assert len(datasets) == n_datasets, "Number of datasets is wrong"
        assert len(MLink.outbound_nodes_of(R.get_namespace_node())) == n_datasets, "Datasets must be children of namespace"

        whole_samples = MNode.get_by_node_type(R.NODE_TYPE_SAMPLE)
        assert len(whole_samples) == n_samples * len(datasets_names), "Number of whole samples is wrong"
        assert len(MLink.links_by_type(DatasetsBucket.LINK_TYPE_DATASET2SAMPLE)) == n_samples * n_datasets, (
            "Number of linked samples is wrong"
        )

        whole_items = MNode.get_by_node_type(R.NODE_TYPE_ITEM)
        assert len(whole_items) == n_items * n_samples * n_datasets, "Number of whole items is wrong"
        assert len(MLink.links_by_type(DatasetsBucket.LINK_TYPE_SAMPLE2ITEM)) == n_samples * n_datasets * n_items, (
            "Number of linked samples is wrong"
        )

        data_mapping = {
            'metadata.sample_number': 's',
            'metadata.even': 'e',
            'items.image': 'x',
            'items.image_mask': 'y'
        }

        types_map = {
            's': int,
            'e': bool,
            'x': np.ndarray,
            'y': np.ndarray
        }

        for sample in whole_samples:
            remapped = R.remap_sample_with_items(sample, data_mapping)
            for k, v in data_mapping.items():
                assert v in remapped, f"Remapped key {v} is missing"
            for k, v in types_map.items():
                assert isinstance(remapped[k], v), f"Type of {k} is wrong. Must be {v}!"

        # DELETION
        for dataset in datasets:
            dataset: MNode

            samples = R.get_samples(dataset.last_name)
            assert len(samples) == n_samples, "Number of retrieved samples is wrong"

            for sample in samples:

                items = R.get_items(dataset.last_name, sample.last_name)

                assert len(items) == n_items, "Number of items is wrong"

            with pytest.raises(DoesNotExist):
                R.delete_dataset(impossible_dataset_name)

            R.delete_dataset(dataset.last_name)

        datasets = R.get_datasets()
        assert len(datasets) == 0, "No datasets should be there"
        assert len(MNode.get_by_node_type(R.NODE_TYPE_SAMPLE)) == 0, "No samples should be there"
        assert len(MNode.get_by_node_type(R.NODE_TYPE_ITEM)) == 0, "No items should be there"

        assert R.get_namespace_node() is not None, "Namespace None should be valid!"
        R.get_namespace_node().delete()

    def dataset_from_tree(self, tree, dataset_name: str, bucket: DatasetsBucket):

        bucket.new_dataset(dataset_name)
        for sample_str, items in tree.items():
            sample = bucket.new_sample(dataset_name,
                                       {
                                           'sample': int(sample_str),
                                           'sample_f': float(sample_str),
                                           'sample_points': [float(sample_str)] * 10,
                                           'odd': int(sample_str) % 2 != 0,
                                           'even': int(sample_str) % 2 == 0,
                                       }
                                       )
            sample_id = sample.last_name

            for item_name, filename in items.items():
                data, encoding = DataCoding.file_to_bytes(filename)
                bucket.new_item(
                    dataset_name,
                    sample_id,
                    item_name,
                    data,
                    encoding
                )

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_dataset_readers(self, temp_mongo_database: MongoDatabaseClient, minimnist_folder, tmpdir_factory):

        tree = tree_from_underscore_notation_files(minimnist_folder)
        assert len(tree) == 20, "Someone altered miniminst folder!!!!"

        dataset_name = "Temp_dataset_for_reader"
        bucket = DatasetsBucket(client_cfg=temp_mongo_database.cfg)
        self.dataset_from_tree(tree, dataset_name, bucket)
        data_mapping = {
            'metadata.#sample_id': 'id',
            'metadata.sample': 'y0',
            'metadata.sample_f': 'y1',
            'metadata.sample_points': 'y2',
            'metadata.odd': 'odd',
            'metadata.even': 'even',
            'items.image': 'x',
            'items.image_mask': 'x_mask',
            'items.points': 'pts'
        }

        # Plain Reader
        reader = DatasetsBucketReader(datasets_bucket=bucket, dataset_name=dataset_name, data_mapping=data_mapping)
        assert reader.data_mapping == data_mapping, "Data mapping manipualted!"
        assert len(reader.queries) == 0, "No query dict should be there!"
        assert len(reader.orders) == 0, "No oders by should be there!"
        for sample in reader:
            for source_key, sample_key in data_mapping.items():
                assert sample_key in sample, f"Missing key [{source_key}->{sample_key}]"
            assert sample['x'].shape == sample['x_mask'].shape, "Shapes of images/masks is wrong!"
        assert len(reader) == len(tree), "Samples count is wrong!"

        # Filtered Reader
        queries = [
            'metadata.odd = True'
        ]
        reader = DatasetsBucketReader(
            datasets_bucket=bucket,
            dataset_name=dataset_name,
            data_mapping=data_mapping,
            queries=queries
        )
        assert len(reader) != len(tree), "Samples count is wrong!"

        # Filtered Reader
        mid_point = len(tree) // 2
        reminder = len(tree) - mid_point
        queries = [
            f'metadata.sample >= {mid_point}'
        ]
        reader = DatasetsBucketReader(
            datasets_bucket=bucket,
            dataset_name=dataset_name,
            data_mapping=data_mapping,
            queries=queries
        )
        assert len(reader) == reminder, "Samples count is wrong!"

        # Filtered Reader (EMPTY)
        queries = [
            'metadata.odd == True',
            'metadata.even = True',
        ]
        reader = DatasetsBucketReader(
            datasets_bucket=bucket,
            dataset_name=dataset_name,
            data_mapping=data_mapping,
            queries=queries
        )
        assert len(reader) == 0, "Samples count is wrong!"

        # Filtered Reader (1)
        queries = [
            'metadata.sample > 0',
            'metadata.sample >= 1',
            'metadata.sample < 5',
            'metadata.sample <= 4',
            'metadata.sample != 3',
            'metadata.sample in [2]',
            'metadata.sample not_in [22]',
        ]
        reader = DatasetsBucketReader(
            datasets_bucket=bucket,
            dataset_name=dataset_name,
            data_mapping=data_mapping,
            queries=queries
        )
        assert len(reader) == 1, "Samples count is wrong!"

        # Filtered Reader (ORDERED 0 ... 10)
        queries = [
            'metadata.odd == True',
        ]
        orders = ['+metadata.sample']
        reader = DatasetsBucketReader(
            datasets_bucket=bucket,
            dataset_name=dataset_name,
            data_mapping=data_mapping,
            queries=queries,
            orders=orders
        )
        assert len(reader) == 10, "Samples count is wrong!"
        samples_ids = [x['y0'] for x in reader]
        assert samples_ids[-1] > samples_ids[0], "Samples ids order is wrong!"

        # Filtered Reader (ORDERED 10 ... 0)

        orders = ['-metadata.sample']
        reader = DatasetsBucketReader(
            datasets_bucket=bucket,
            dataset_name=dataset_name,
            data_mapping=data_mapping,
            orders=orders
        )
        assert len(reader) == len(tree), "Samples count is wrong!"
        samples_ids = [x['y0'] for x in reader]
        assert samples_ids[-1] < samples_ids[0], "Samples ids order is wrong!"

        # Filtered Reader (1)
        queries = [
            'metadata.sample > 0',
        ]
        orders = ['+metadata.sample']

        out_cfg = {
            'data_mapping': data_mapping,
            'queries': queries,
            'orders': orders
        }

        cfg_filename = Path(tmpdir_factory.mktemp("reader_folder")) / 'reader_cfg.yml'
        yaml.safe_dump(out_cfg, open(str(cfg_filename), 'w'))

        reloaded_reader = DatasetsBucketReader.builds_from_configuration_file(
            bucket,
            dataset_name,
            cfg_filename
        )
        plain_reader = DatasetsBucketReader(
            datasets_bucket=bucket,
            dataset_name=dataset_name,
            data_mapping=data_mapping,
            queries=queries,
            orders=orders
        )
        assert len(reloaded_reader) == len(plain_reader), "Loading and creating not consistent"
        for idx, plain_sample in enumerate(plain_reader):
            for key, _ in plain_sample.items():
                assert key in reloaded_reader[idx], f"Samples keys do not match on {key}!"
            assert plain_sample['id'] == reloaded_reader[idx]['id'], "sample ids do not match!"

        samples_reader = DatasetsBucketSamplesListReader(plain_reader.samples, data_mapping)
        for idx, sample in enumerate(samples_reader):
            assert sample['id'] == plain_reader[idx]['id'], "Samples does not matches"

        # TEARDOWN DATASET
        bucket.delete_dataset(dataset_name)

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_dataset_snapshots(self, temp_mongo_database, minimnist_folder, tmpdir_factory):

        tree = tree_from_underscore_notation_files(minimnist_folder)
        assert len(tree) == 20, "Someone altered miniminst folder!!!!"

        data_mapping_A = {
            'metadata.#sample_id': 'id',
            'metadata.odd': 's',
            'metadata.#dataset_name': 'dataset_name',
            'items.image': 'x',
        }

        data_mapping_B = {
            'metadata.#sample_id': 'id',
            'metadata.even': 's',
            'metadata.#dataset_name': 'dataset_name',
            'items.image': 'x',
        }

        reader_A = {
            'data_mapping': data_mapping_A,
            'queries': [
                'metadata.odd == True'
            ]
        }
        reader_B = {
            'data_mapping': data_mapping_B,
            'queries': [
                'metadata.even == True'
            ]
        }

        pipeline = """def generate(self, ops, d):
        gA, gB = ops.split(d['good'], 0.8)
        bA, bB = ops.split(d['bad'], 0.8)
        x = gA + bA
        y = gB + bB
        x = ops.shuffle(x,10)
        y = ops.shuffle(y,10)
        return {'x':x, 'y': y}
        """

        snapshot_cfg = {
            'datasets': [
                {'good': {
                    'dataset': {'name': 'Data_A'},
                    'reader': reader_A
                }},
                {'bad': {
                    'dataset': {'name': 'Data_B'},
                    'reader': reader_B
                }}
            ],
            'pipeline': pipeline
        }

        bucket = DatasetsBucket(temp_mongo_database.cfg)
        total_size = 0
        for out_dataset in snapshot_cfg['datasets']:

            dataset_name = out_dataset[list(out_dataset.keys())[0]]['dataset']['name']

            self.dataset_from_tree(tree, dataset_name, bucket)
            print(dataset_name)
            total_size += bucket.count_samples(dataset_name)

        # SNAPSHOT TEST
        cfg = DatasetsBucketSnapshotCFG.from_dict(snapshot_cfg)
        snapshot = DatasetsBucketSnapshot(bucket, cfg=cfg)

        outputs = snapshot.output_data
        assert 'good' not in outputs, "good key can't be there"
        assert 'bad' not in outputs, "bad key can't be there"
        assert 'x' in outputs, "x key is missing"
        assert 'y' in outputs, "y key is missing"

        # Total size is half the total size because A takes only odds samples and B only evens
        assert len(outputs['x']) + len(outputs['y']) == total_size // 2, "Total size is wrong!"

        unique_keys = set()  # checks uniquness of samples after sum/split/shuffle
        for sample in outputs['x']:

            sample_key = (sample['id'], sample['dataset_name'])
            assert sample_key not in unique_keys, "Duplicate sample is very strange!!"
            unique_keys.add(sample_key)

            assert 'id' in sample, "id key is missing"
            assert 's' in sample, "s key is missing"
            assert sample['s'] is True, "Odd field needs to be True"
            assert 'x' in sample, "x key is missing"

        for sample in outputs['y']:

            sample_key = (sample['id'], sample['dataset_name'])
            assert sample_key not in unique_keys, "Duplicate sample is very strange!!"
            unique_keys.add(sample_key)

            assert 'id' in sample, "id key is missing"
            assert 's' in sample, "s key is missing"
            assert sample['s'] is True, "Odd field needs to be True"
            assert 'x' in sample, "x key is missing"

        print(unique_keys)

        # TEARDOWN DATASET
        for out_dataset in snapshot_cfg['datasets']:
            dataset_name = out_dataset[list(out_dataset.keys())[0]]['dataset']['name']
            bucket.delete_dataset(dataset_name)
