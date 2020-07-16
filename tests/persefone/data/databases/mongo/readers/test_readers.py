from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset
from persefone.data.databases.mongo.readers import MongoDatasetReader
from persefone.data.databases.mongo.repositories import DatasetCategoryRepository
from persefone.data.io.drivers.safefs import SafeFilesystemDriver, SafeFilesystemDriverCFG
import pytest
from persefone.utils.filesystem import tree_from_underscore_notation_files
import yaml
from pathlib import Path


class TestMongoDatasetReader(object):

    @pytest.mark.mongo_mock_server  # NOT EXECUTE IF --mongo_real_server option is passed
    def test_reader_mock(self,
                         temp_mongo_mock_database: MongoDatabaseClient,
                         safefs_sample_configuration,
                         minimnist_folder,
                         driver_temp_base_folder):
        pass

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_reader(self,
                    temp_mongo_database: MongoDatabaseClient,
                    safefs_sample_configuration,
                    minimnist_folder,
                    driver_temp_base_folder,
                    tmpdir_factory):
        self._test_reader(
            temp_mongo_database,
            safefs_sample_configuration,
            minimnist_folder,
            driver_temp_base_folder,
            tmpdir_factory
        )

    def dataset_from_tree(self, tree, dataset):
        for sample_str, items in tree.items():
            sample = dataset.add_sample(
                {
                    'sample': int(sample_str),
                    'sample_f': float(sample_str),
                    'sample_points': [float(sample_str)] * 10,
                    'odd': int(sample_str) % 2 != 0,
                    'even': int(sample_str) % 2 == 0,
                }
            )
            sample_idx = sample.sample_id

            for item_name, filename in items.items():
                dataset.add_item(sample_idx, item_name)

                dataset.push_resource(
                    sample_idx,
                    item_name,
                    item_name,
                    filename,
                    SafeFilesystemDriver.driver_name()
                )

    def _test_reader(self,
                     mongo_client,
                     safefs_sample_configuration,
                     minimnist_folder,
                     driver_temp_base_folder,
                     tmpdir_factory):

        tree = tree_from_underscore_notation_files(minimnist_folder)
        assert len(tree) == 20, "Someone altered miniminst folder!!!!"

        dataset_name = 'FAKEDATASET_TEMPFORREADER'
        category_name = 'FAKEDATASET_TEMPFORREADER_CATEGORY_TEMP'
        cfg = SafeFilesystemDriverCFG.from_dict({'base_folder': driver_temp_base_folder})
        drivers = {
            SafeFilesystemDriver.driver_name(): SafeFilesystemDriver(cfg)
        }
        dataset = MongoDataset(mongo_client, dataset_name, category_name, drivers=drivers)
        self.dataset_from_tree(tree, dataset)
        data_mapping = {
            'sample_id': 'id',
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
        reader = MongoDatasetReader(mongo_dataset=dataset, data_mapping=data_mapping)
        assert reader.data_mapping == data_mapping, "Data mapping manipualted!"
        assert len(reader.query_dict) == 0, "No query dict should be there!"
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
        reader = MongoDatasetReader(mongo_dataset=dataset, data_mapping=data_mapping, queries=queries)
        assert len(reader) != len(tree), "Samples count is wrong!"

        # Filtered Reader
        mid_point = len(tree) // 2
        reminder = len(tree) - mid_point
        queries = [
            f'metadata.sample >= {mid_point}'
        ]
        reader = MongoDatasetReader(mongo_dataset=dataset, data_mapping=data_mapping, queries=queries)
        assert len(reader) == reminder, "Samples count is wrong!"

        # Filtered Reader (EMPTY)
        queries = [
            f'metadata.odd == True',
            f'metadata.even = True',
        ]
        reader = MongoDatasetReader(mongo_dataset=dataset, data_mapping=data_mapping, queries=queries)
        assert len(reader) == 0, "Samples count is wrong!"

        # Filtered Reader (1)
        queries = [
            f'metadata.sample > 0',
            f'metadata.sample >= 1',
            f'metadata.sample < 5',
            f'metadata.sample <= 4',
            f'metadata.sample != 3',
            f'metadata.sample in [2]',
            f'metadata.sample not_in [22]',
        ]
        reader = MongoDatasetReader(mongo_dataset=dataset, data_mapping=data_mapping, queries=queries)
        assert len(reader) == 1, "Samples count is wrong!"

        # Filtered Reader (ORDERED 0 ... 10)
        queries = [
            f'metadata.odd == True',
        ]
        orders = ['+metadata.sample']
        reader = MongoDatasetReader(mongo_dataset=dataset, data_mapping=data_mapping, queries=queries, orders=orders)
        assert len(reader) == 10, "Samples count is wrong!"
        samples_ids = [x['y0'] for x in reader]
        assert samples_ids[-1] > samples_ids[0], "Samples ids order is wrong!"

        # Filtered Reader (ORDERED 10 ... 0)
        orders = ['-metadata.sample']
        reader = MongoDatasetReader(mongo_dataset=dataset, data_mapping=data_mapping, orders=orders)
        assert len(reader) == len(tree), "Samples count is wrong!"
        samples_ids = [x['y0'] for x in reader]
        assert samples_ids[-1] < samples_ids[0], "Samples ids order is wrong!"

        # Filtered Reader (1)
        queries = [
            f'metadata.sample > 0',
        ]
        orders = ['+metadata.sample']

        out_cfg = {
            'data_mapping': data_mapping,
            'queries': queries,
            'orders': orders
        }

        cfg_filename = Path(tmpdir_factory.mktemp("reader_folder")) / 'reader_cfg.yml'
        yaml.safe_dump(out_cfg, open(str(cfg_filename), 'w'))

        reloaded_reader = MongoDatasetReader.create_from_configuration_file(mongo_dataset=dataset, filename=cfg_filename)
        plain_reader = MongoDatasetReader(mongo_dataset=dataset, data_mapping=data_mapping, queries=queries, orders=orders)
        assert len(reloaded_reader) == len(plain_reader), "Loading and creating not consistent"
        print(len(plain_reader))
        for idx, plain_sample in enumerate(plain_reader):
            for key, _ in plain_sample.items():
                assert key in reloaded_reader[idx], f"Samples keys do not match on {key}!"
            assert plain_sample['id'] == reloaded_reader[idx]['id'], "sample ids do not match!"

        # TEARDOWN DATASET
        dataset.delete(security_name=dataset_name)
        DatasetCategoryRepository.delete_category(name=category_name)
