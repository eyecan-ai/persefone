

from pathlib import Path

import deepdiff
from persefone.data.databases.mongo.nodes.buckets.networks import NetworksBucket
from cv2 import data
from numpy.lib.arraysetops import isin
import yaml
from persefone.data.databases.mongo.nodes.nodes import MLink, MNode
from mongoengine.errors import DoesNotExist
from persefone.data.databases.mongo.clients import MongoDatabaseClient
from persefone.data.databases.mongo.nodes.buckets.datasets import DatasetsBucket, DatasetsBucketReader, DatasetsBucketSamplesListReader, DatasetsBucketSnapshot, DatasetsBucketSnapshotCFG
from persefone.utils.filesystem import tree_from_underscore_notation_files
from persefone.utils.bytes import DataCoding
import pytest
import numpy as np


class TestNetworksBucket(object):

    @pytest.fixture
    def random_metadata(self):
        return {
            'one': 1,
            'two': [[[0], [2], [3]]],
            'three': 3.3,
            '4': 'four',
            'dd': {
                'a': 1.1,
                'b': True,
                'c': {
                    'c.1': 0000,
                    'c.2': 'c1',
                    'c.3': [1, 2, 3, 4, 5, 5, .33]
                }
            }
        }

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_networks(self, temp_mongo_database: MongoDatabaseClient, random_metadata):

        db_cfg = temp_mongo_database.cfg

        datasets_bucket = DatasetsBucket(client_cfg=db_cfg)

        datasets_names = ['Data_A', 'Data_B', 'Data_C', 'Data_D']
        datasets = []
        for dataset_name in datasets_names:
            dataset = datasets_bucket.new_dataset(dataset_name)
            assert dataset is not None, "Dataset must be ok!"
            datasets.append(dataset)

        trainable_items = [
            {'name': 'AE_A', 'metadata': random_metadata},
            {'name': 'AE_B', 'metadata': random_metadata},
            {'name': 'OBJECT_DETECTOR_YOLO'},
        ]

        model_weights = np.random.uniform(-1, 1, (500, 500)).astype(np.float32)
        n_models = 4
        n_tasks = 4
        impossible_name = "IMPOSSIBLE_NAME!!!@@@"
        networks_bucket = NetworksBucket(client_cfg=db_cfg)

        for trainable_item in trainable_items:

            trainable_name = trainable_item.get('name', '')
            trainbale_meta = trainable_item.get('metadata', {})

            trainable = networks_bucket.new_trainable(trainable_name, trainbale_meta)

            with pytest.raises(NameError):
                networks_bucket.new_trainable(trainable_name, trainbale_meta)

            assert trainable is not None, "Trainable must be ok"
            assert trainable.node_type == networks_bucket.NODE_TYPE_TRAINABLE

        for trainable_item in trainable_items:

            trainable_name = trainable_item.get('name', '')
            trainbale_meta = trainable_item.get('metadata', {})

            trainable: MNode = networks_bucket.get_trainable(trainable_name)
            assert trainable is not None, "Retrieved Trainable must be ok"

            diff = deepdiff.DeepDiff(trainable.plain_metadata, trainbale_meta, ignore_order=True, ignore_type_subclasses=True)

            assert not diff, "Loaded metadata are wrong!"

            with pytest.raises(DoesNotExist):
                networks_bucket.get_trainable(impossible_name)

            for model_idx in range(n_models):

                model_name = f'model_{model_idx}'

                model: MNode = networks_bucket.new_model(trainable_name, model_name, random_metadata)
                assert model is not None, "Model should be ok!"
                assert model.node_type == networks_bucket.NODE_TYPE_MODEL, "Model type is wrong"

                no_data, no_encoding = model.get_data()
                assert no_data is None, "No bytes should be there!"
                assert no_encoding is None, "No bytes means no encoding !"

                with pytest.raises(NameError):
                    networks_bucket.new_model(trainable_name, model_name, random_metadata)

                weights_encoding = 'npy'
                weiths_data = DataCoding.numpy_array_to_bytes(model_weights, data_encoding=weights_encoding)
                model.put_data(weiths_data, weights_encoding)

                model_r: MNode = networks_bucket.get_model(trainable_name, model_name)
                assert model_r is not None, "Retrieved model must be ok"
                assert model_r.node_type == networks_bucket.NODE_TYPE_MODEL, "Retrieved Model type is wrong"

                with pytest.raises(DoesNotExist):
                    networks_bucket.get_model(trainable_name, impossible_name)
                with pytest.raises(DoesNotExist):
                    networks_bucket.get_model(impossible_name, impossible_name)

                weights_data_r, weights_encoding_r = model_r.get_data()
                weights_r = DataCoding.bytes_to_data(weights_data_r, weights_encoding_r)
                assert np.array_equal(model_weights, weights_r), "Retrieved weights are inconsistent!"

                for dataset in datasets:
                    dataset: MNode
                    model.link_to(dataset, link_type=networks_bucket.LINK_TYPE_MODEL2DATASET)

            for task_idx in range(n_tasks):

                task_name = f'task_{task_idx}'

                task: MNode = networks_bucket.new_task(trainable_name, task_name, random_metadata)
                assert task is not None, "Task should be ok!"
                assert task.node_type == networks_bucket.NODE_TYPE_TASK, "Task type is wrong"

                with pytest.raises(NameError):
                    networks_bucket.new_task(trainable_name, task_name, random_metadata)

                task_r: MNode = networks_bucket.get_task(trainable_name, task_name)
                assert task_r is not None, "Retrieved task must be ok"
                assert task_r.node_type == networks_bucket.NODE_TYPE_TASK, "Retrieved Task type is wrong"

                with pytest.raises(DoesNotExist):
                    networks_bucket.get_task(trainable_name, impossible_name)
                with pytest.raises(DoesNotExist):
                    networks_bucket.get_task(impossible_name, impossible_name)

                for dataset in datasets:
                    dataset: MNode
                    task.link_to(dataset, link_type=networks_bucket.LINK_TYPE_TASK2DATASET)

        n_trainables = len(trainable_items)
        trainables = networks_bucket.get_trainables()
        assert len(trainables) == n_trainables, "Number of retrieved trainables is wrong"

        n_whole_models = n_trainables * n_models
        n_whole_tasks = n_trainables * n_tasks

        whole_models = []
        whole_tasks = []
        for trainable in trainables:
            trainable: MNode
            whole_models.extend(networks_bucket.get_models(trainable.last_name))
            whole_tasks.extend(networks_bucket.get_tasks(trainable.last_name))

        assert len(whole_models) == n_whole_models, "Number of models is wrong"
        assert len(whole_tasks) == n_whole_tasks, "Number of models is wrong"

        for task in whole_tasks:
            task: MNode

            r_datasets = task.outbound_nodes_by_node_type(datasets_bucket.NODE_TYPE_DATASET)
            assert len(r_datasets) == len(datasets), "NUmber of dataset associated with task is wrong!"
            r_datasets2 = task.outbound_nodes(networks_bucket.LINK_TYPE_TASK2DATASET)
            assert len(r_datasets2) == len(datasets), "NUmber of dataset associated with task is wrong!"

        for model in whole_models:
            model: MNode
            r_datasets = model.outbound_nodes_by_node_type(datasets_bucket.NODE_TYPE_DATASET)
            assert len(r_datasets) == len(datasets), "NUmber of dataset associated with model is wrong!"
            r_datasets2 = model.outbound_nodes(networks_bucket.LINK_TYPE_MODEL2DATASET)
            assert len(r_datasets2) == len(datasets), "NUmber of dataset associated with model is wrong!"

        for model in whole_models:
            model.delete()

        assert len(MNode.objects(node_type=networks_bucket.NODE_TYPE_MODEL)) == 0, "No models should be there"
        assert len(MLink.objects(link_type=networks_bucket.LINK_TYPE_TRAINABLE2MODEL)) == 0, "No links to models should be there"
