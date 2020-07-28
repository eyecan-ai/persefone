

from persefone.data.databases.mongo.nodes.nodes import MLink, MNode
from mongoengine.errors import DoesNotExist
from persefone.data.databases.mongo.clients import MongoDatabaseClient
from persefone.data.databases.mongo.nodes.buckets.datasets import DatasetsBucket
from persefone.utils.filesystem import tree_from_underscore_notation_files
from persefone.utils.bytes import DataCoding
import pytest
import numpy as np


class TestDatasetsBucket(object):

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_datasets(self, temp_mongo_persistent_database: MongoDatabaseClient, minimnist_folder):

        tree = tree_from_underscore_notation_files(minimnist_folder)
        n_samples = len(tree.items())
        n_items = -1
        impossible_dataset_name = "_ASDasdasdjasdas_IMPOSSIBLE!"

        R = DatasetsBucket(client_cfg=temp_mongo_persistent_database.cfg)
        print(R)

        datasets_names = ['Data_A', 'Data_B']

        for dataset_name in datasets_names:
            dataset = R.new_dataset(dataset_name)

            with pytest.raises(NameError):
                dataset = R.new_dataset(dataset_name)

            assert dataset is not None, "Dataset creation should be valid"
            dataset_r = R.get_dataset(dataset_name)

            with pytest.raises(DoesNotExist):
                R.get_dataset(impossible_dataset_name)

            assert dataset == dataset_r, "Retrieved dataset is wrong"

            for sample_str, items in tree.items():
                n_items = len(items.items())

                sample: MNode = R.new_sample(dataset_name, {'sample': sample_str, 'items': items.keys(), 'even': int(sample_str) % 2 == 0})

                sample_id = int(sample.last_name)
                sample_r = R.get_sample(dataset_name, sample_id)

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

                    with pytest.raises(NameError):
                        R.new_item(dataset_name, sample_id, item_name, blob_data=blob, blob_encoding=encoding)

                    item_r: MNode = R.get_item(dataset_name, sample_id, item_name)
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

        for dataset in datasets:
            dataset: MNode

            samples = R.get_samples(dataset.last_name)
            assert len(samples) == n_samples, "Number of retrieved samples is wrong"

            for sample in samples:

                items = R.get_items(dataset.last_name, int(sample.last_name))

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
