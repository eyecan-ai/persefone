
import numpy as np
import pytest
from mongoengine import connect
from persefone.data.databases.mongo.model import MItem, MResource, MSample, MDataset, MDatasetCategory
from persefone.data.databases.mongo.repositories import (RepositoryTestingData,
                                                         DatasetsRepository,
                                                         DatasetCategoryRepository,
                                                         SamplesRepository,
                                                         ItemsRepository)


class TestRepositories(object):

    @pytest.fixture(scope='function')
    def temp_database(self):
        db_name = '##_temp_database_@@'
        db = connect(db_name)
        yield db
        db.drop_database(db_name)

    def _compare_dicts(self, d1: dict, d2: dict):
        for k, v in d1.items():
            assert k in d2, f"Metadata {k} is missing!"
            # print("Compare", d1[k], d2[k])
            if isinstance(d1[k], dict):
                self._compare_dicts(d1[k], d2[k])
            else:
                # print("Comparing", d1[k], d2[k])
                assert self._compare_param(d1[k], d2[k]), f"{ d1[k]} != {d2[k]}"

    def _compare_param(self, p1, p2):
        if isinstance(p1, list) or isinstance(p1, tuple):
            return np.all(np.isclose(np.array(p1), np.array(p2)))
        else:
            return p1 == p2

    def test_repo(self, temp_database):

        n_categories = 3
        n_datasets = 2
        n_samples = 20
        n_items = 5
        n_resources = 2

        RepositoryTestingData.generate_test_data(
            n_categories=n_categories,
            n_datasets=n_datasets,
            n_samples=n_samples,
            n_items=n_items,
            n_resources=n_resources
        )

        for dataset_idx in range(n_datasets):

            dataset: MDataset

            dataset_name = RepositoryTestingData.dataset_name(dataset_idx)
            dataset = DatasetsRepository.get_dataset(dataset_name)
            assert dataset is not None, f"Dataset {dataset_name} not found!"

            none_name = dataset_name + "_INP0SS1BLE!"
            dataset_none = DatasetsRepository.get_dataset(none_name)
            assert dataset_none is None, f"Dataset {none_name} should be None!"

            dataset_category: MDatasetCategory = dataset.category
            not_unique_category = DatasetCategoryRepository.new_category(dataset_category.name)
            assert not_unique_category == dataset_category, "New Not-unique category attempt should return conflicting category instead"

            dataset_samples = SamplesRepository.get_samples(dataset)
            assert len(dataset_samples) == n_samples, f"Number of samples is wrong! {len(dataset_samples) }/{n_samples}"

            for sample in dataset_samples:

                sample: MSample
                not_unique_sample = SamplesRepository.new_sample(dataset, sample.sample_id)
                assert not_unique_sample is None, "A not unique Sample should not be possible!"

                assert sample.dataset == dataset, "Dataset of sample is wrong!"

                metadata = RepositoryTestingData.metadata(sample.sample_id)
                sample_metadata = sample.metadata
                self._compare_dicts(metadata, sample_metadata)

                sample_items = ItemsRepository.get_items(sample)
                assert len(sample_items) == n_items, f"Number of items is wrong! {len(sample_items)}/{n_items}"

                for item_idx, item in enumerate(sample_items):

                    item: MItem
                    assert item.sample == sample, "Item sample reference is wrong!"
                    assert item.name == RepositoryTestingData.item_name(item_idx)

                    not_unique_item = ItemsRepository.new_item(sample, item.name)
                    assert not_unique_item == None, "Not unique Item should be None after creation"

                    resources = item.resources
                    assert len(resources) == n_resources, f"Number of resources is wrong! {len(resources)}/{n_resources}"

                    for resource in resources:

                        resource: MResource
                        for resource_idx in range(n_resources):
                            driver_name = RepositoryTestingData.driver_name(resource_idx)
                            if resource.driver == driver_name:
                                assert resource.uri == RepositoryTestingData.uri(item_idx, resource_idx)

            print(len(dataset_samples))

        print("WOW"*20, RepositoryTestingData)
