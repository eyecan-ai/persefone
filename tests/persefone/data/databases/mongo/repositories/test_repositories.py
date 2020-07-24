
import numpy as np
import pytest
from persefone.data.databases.mongo.model import MItem, MResource, MSample, MDataset, MDatasetCategory, MModel, MModelCategory
from persefone.data.databases.mongo.repositories import (RepositoryTestingData,
                                                         DatasetsRepository,
                                                         DatasetCategoryRepository,
                                                         SamplesRepository,
                                                         ModelsRepository,
                                                         ModelCategoryRepository,
                                                         ItemsRepository)


class TestDatasetManagement(object):

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

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_real_database_data_creation(self, temp_mongo_database):
        self._data_creation_on_active_dataset(temp_mongo_database)

    @pytest.mark.mongo_mock_server  # NOT EXECUTE IF --mongo_real_server option is passed
    def test_mock_database_data_creation(self, temp_mongo_mock_database):
        self._data_creation_on_active_dataset(temp_mongo_mock_database)

    # def test_real_database_data_and_keep_alive(self, temp_mongo_database_keep_alive):
    #     self._data_creation_on_active_dataset(temp_mongo_database_keep_alive)

    def _data_creation_on_active_dataset(self, database):

        n_categories = 2
        n_datasets = 4
        n_samples = 10
        n_items = 3
        n_resources = 2
        n_models = 5
        n_models_categories = 2
        n_model_resources = 2

        RepositoryTestingData.generate_test_data(
            n_categories=n_categories,
            n_datasets=n_datasets,
            n_samples=n_samples,
            n_items=n_items,
            n_resources=n_resources,
            n_models_categories=n_models_categories,
            n_models=n_models,
            n_model_resources=n_model_resources
        )

        created_datasets = []
        for dataset_idx in range(n_datasets):

            dataset: MDataset

            dataset_name = RepositoryTestingData.dataset_name(dataset_idx)
            dataset = DatasetsRepository.get_dataset(dataset_name)
            created_datasets.append(dataset)
            assert dataset is not None, f"Dataset {dataset_name} not found!"
            assert DatasetsRepository.new_dataset(dataset_name, dataset.category) is None, "No duplicate datasets allowed!"

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
                    assert not_unique_item is None, "Not unique Item should be None after creation"

                    resources = item.resources
                    assert len(resources) == n_resources, f"Number of resources is wrong! {len(resources)}/{n_resources}"

                    for resource in resources:

                        assert ItemsRepository.create_item_resource(item, resource.name, '', '') is None, "Duplicate resource not allowed!"

                        resource: MResource
                        for resource_idx in range(n_resources):
                            driver_name = RepositoryTestingData.driver_name(resource_idx)
                            if resource.driver == driver_name:
                                assert resource.uri == RepositoryTestingData.uri(item_idx, resource_idx)

        assert len(created_datasets) == n_datasets, "Some dataset was lost!"

        # Check model query set
        created_models = list(ModelsRepository.get_models())
        assert len(created_models) == n_models, "Number of retrieved models is wrong!"
        models_categories = set()
        for model in created_models:
            models_categories.add(model.category)
        cat_sum = 0
        for model_cat in models_categories:
            sub_models = list(ModelsRepository.get_models(model_category=model_cat))
            assert len(sub_models) != n_models, "Number of specific cat models is wrong!"
            cat_sum += len(sub_models)
        assert cat_sum == n_models, "Sum of sub category models is wrong!"

        for model_idx in range(n_models):

            model_name = RepositoryTestingData.model_name(model_idx)
            model: MModel = ModelsRepository.get_model(model_name)
            assert ModelsRepository.get_model(model_name + "!IMPOSSIBLE!!!") is None, "No model should be found!"
            assert ModelsRepository.new_model(model.name, model.category) is None, "Duplicates models are not allowed!"
            assert model is not None, "Model creation fails!"
            assert model.task is not None, "Testing models should have a task!"
            assert model == ModelsRepository.get_model_by_task(model.task), "Model should be equal to its task model"

            assert len(model.task.datasets) == len(created_datasets), "Each task should have each dataset as child!"
            for task_dataset in model.task.datasets:
                assert task_dataset in created_datasets, "Task dataset not found in previously created datasets"

            model_category: MModelCategory = model.category
            not_unique_category = ModelCategoryRepository.new_category(model_category.name)
            assert not_unique_category == model_category, "New Not-unique category attempt should return conflicting category instead"

            for r_dataset in model.task.datasets:
                assert r_dataset in created_datasets, "Retrived Model>Dataset not found!"
                assert model in ModelsRepository.get_models_by_dataset(r_dataset), "Model should be in query which involves itself"

            assert model in ModelsRepository.get_models_by_dataset(model.task.datasets), "Model should be in query which involves itself"

            for resource in model.resources:

                assert ModelsRepository.create_model_resource(model, resource.name, '', '') is None, "Duplicate resource not allowed!"

                resource: MResource
                for resource_idx in range(n_model_resources):
                    driver_name = RepositoryTestingData.driver_name(resource_idx)
                    if resource.driver == driver_name:
                        assert resource.uri == RepositoryTestingData.uri(model_idx, resource_idx)

        for model_idx in range(n_models):
            model_name = RepositoryTestingData.model_name(model_idx)
            model: MModel = ModelsRepository.get_model(model_name)
            assert ModelsRepository.delete_model(model.name) is True, "Deletion of MModel should be ok!"
            assert ModelsRepository.delete_model(model.name) is False, "Double Deletion of MModel should be invalid!"
