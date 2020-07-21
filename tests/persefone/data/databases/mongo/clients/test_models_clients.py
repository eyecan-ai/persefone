from persefone.data.databases.mongo.model import MModel
from persefone.data.databases.mongo.clients import MongoModelsManager, MongoDatabaseTaskManager, MongoDatabaseTaskManagerType
from persefone.data.databases.mongo.repositories import RepositoryTestingData, DatasetsRepository
from pathlib import Path
import pytest
import numpy as np


class TestModelsManager(object):

    def _test_manager(self, mongo_client):

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

        # Creation of new Tasks
        n_tasks = 10
        task_creator = MongoDatabaseTaskManager(
            mongo_client=mongo_client,
            manager_type=MongoDatabaseTaskManagerType.TASK_CREATOR
        )

        datasets = DatasetsRepository.get_datasets()
        tasks = []
        for task_idx in range(n_tasks):
            task_name = f'modelclient_task_{task_idx}'
            task = task_creator.new_task(name=task_name)
            assert task is not None, "Task should be ok!"
            task.datasets = datasets
            task.save()
            tasks.append(task)

        # Models manager
        models_manager = MongoModelsManager(mongo_client=mongo_client)
        assert models_manager is not None, "It should be obvious!"

        custom_cat = "CAT_EGO"
        custom_models = []
        for task_id, task in enumerate(tasks):
            model = models_manager.new_model(f"NewModel_{task_id}", custom_cat, task)
            assert model is not None, "Model should be ok!"
            custom_models.append(model)
            assert model == models_manager.get_model_by_task(model.task.name), "Model should be equal to ist task's model"

            dataset_names = [x.name for x in task.datasets]
            assert model in models_manager.get_models_by_datasets(dataset_names), "Model should be in query whit its task dataset"
            for d_name in dataset_names:
                assert model in models_manager.get_models_by_datasets(d_name), "Model should be in query whit its task dataset"

        assert len(models_manager.get_models()) > len(tasks), "Overall models set should be greater than jit created!"
        assert len(models_manager.get_models(model_category=custom_cat)) == len(tasks), "Each task should have a model!"

        for model in custom_models:
            assert models_manager.delete_model(model.name) is True, "Model deletion should be ok!"
            assert models_manager.delete_model(model.name) is False, "Model double deletion should be wrong!"

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_manager(self, temp_mongo_database):
        self._test_manager(temp_mongo_database)

    @pytest.mark.mongo_mock_server  # NOT EXECUTE IF --mongo_real_server option is passed
    def test_manager_mock(self, temp_mongo_mock_database):
        self._test_manager(temp_mongo_mock_database)
