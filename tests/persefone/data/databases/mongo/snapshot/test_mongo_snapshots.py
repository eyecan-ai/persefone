from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDataset
from persefone.data.databases.mongo.snapshots import MongoSnapshot, MongoSnapshotCFG
from persefone.data.databases.mongo.repositories import DatasetCategoryRepository
from persefone.data.io.drivers.safefs import SafeFilesystemDriver, SafeFilesystemDriverCFG
import pytest
from persefone.utils.filesystem import tree_from_underscore_notation_files


class TestMongoDatasetSnapshots(object):

    @pytest.mark.mongo_mock_server  # NOT EXECUTE IF --mongo_real_server option is passed
    def test_reader_mock(self,
                         temp_mongo_mock_database: MongoDatabaseClient,
                         safefs_sample_configuration,
                         minimnist_folder,
                         driver_temp_base_folder,
                         tmpdir_factory):
        self._test_reader(
            temp_mongo_mock_database,
            safefs_sample_configuration,
            minimnist_folder,
            driver_temp_base_folder,
            tmpdir_factory
        )

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

        data_mapping_A = {
            'sample_id': 'id',
            'metadata.odd': 's',
            'dataset.name': 'dataset_name',
            'items.image': 'x',
        }

        data_mapping_B = {
            'sample_id': 'id',
            'metadata.even': 's',
            'dataset.name': 'dataset_name',
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

        drivers_cfg = SafeFilesystemDriverCFG.from_dict({'base_folder': driver_temp_base_folder})
        drivers = {
            SafeFilesystemDriver.driver_name(): SafeFilesystemDriver(drivers_cfg)
        }

        total_size = 0
        for out_dataset in snapshot_cfg['datasets']:

            dataset_name = out_dataset[list(out_dataset.keys())[0]]['dataset']['name']
            category_name = dataset_name + "_cat"
            dataset = MongoDataset(mongo_client, dataset_name, dataset_name, drivers=drivers)
            self.dataset_from_tree(tree, dataset)
            print(dataset_name)
            total_size += dataset.count_samples()

        # SNAPSHOT TEST
        cfg = MongoSnapshotCFG.from_dict(snapshot_cfg)
        snapshot = MongoSnapshot(mongo_client, drivers, cfg)

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
            dataset.delete(security_name=dataset_name)
            DatasetCategoryRepository.delete_category(name=category_name)
