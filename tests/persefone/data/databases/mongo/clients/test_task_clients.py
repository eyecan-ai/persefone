
from persefone.data.databases.mongo.model import MTask, MTaskStatus
from persefone.data.databases.mongo.clients import MongoDatabaseClient, DatabaseTaskManager, DatabaseTaskManagerType
from pathlib import Path
import pytest
import numpy as np


class TestDatabaseTaskManager(object):

    @pytest.fixture(scope='function')
    def mongo_client(self, mongo_configurations_folder):
        cfg_file = Path(mongo_configurations_folder) / 'mongo_test_client_cfg.yml'
        client = MongoDatabaseClient.create_from_configuration_file(filename=cfg_file)
        yield client
        client.drop_database(key0=client.DROP_KEY_0, key1=client.DROP_KEY_1)
        client.disconnect()

    @pytest.fixture(scope='function')
    def mongo_client_mock(self, mongo_configurations_folder):
        cfg_file = Path(mongo_configurations_folder) / 'mongo_test_client_cfg_mock.yml'
        client = MongoDatabaseClient.create_from_configuration_file(filename=cfg_file)
        yield client
        client.drop_database(key0=client.DROP_KEY_0, key1=client.DROP_KEY_1)
        client.disconnect()

    def _test_manager(self, mongo_client):

        creator = DatabaseTaskManager(mongo_client=mongo_client, manager_type=DatabaseTaskManagerType.TASK_CREATOR)
        worker = DatabaseTaskManager(mongo_client=mongo_client, manager_type=DatabaseTaskManagerType.TASK_WORKER)
        god = DatabaseTaskManager(mongo_client=mongo_client, manager_type=DatabaseTaskManagerType.TASK_GOD)

        with pytest.raises(PermissionError):
            worker.new_task('impossible_task', {'a': 2.2})

        wrong_task = god.new_task("wrong")
        canceled_task = worker.cancel_task(wrong_task.name)
        with pytest.raises(PermissionError):
            creator.cancel_task(wrong_task.name)
        assert canceled_task.status == MTaskStatus.CANCELED.name, "Canceled task is not so canceled!"

        task = creator.new_task('my_task', input_payload={'a': 'b'})
        assert task is not None, "Task should be valid"
        assert god.new_task('my_task_2') is not None, "God can do it!"
        assert god.new_task('my_task') is None, "Neither God can do duplicates!"

        tasks = worker.get_tasks(status=MTaskStatus.READY)
        assert len(tasks) == 2, "Wrong number of retrieved tasks!"  # READY TASK SHOULD BE ONLY 2

        target_task: MTask = tasks[0]
        started_task = worker.start_task(target_task.name, input_payload={'start': True})
        assert started_task is not None, "Started task should be valid"
        with pytest.raises(PermissionError):
            creator.start_task(target_task.name)
        assert god.start_task(target_task.name) is None, "Neither God should start a started tasks!"

        for j in range(10):
            current_percentage = 100 / (j + 1)
            worked_task = worker.work_on_task(started_task.name, work_payload={'percentage': current_percentage})
            assert worked_task is not None, "Worked task should be valid"
            with pytest.raises(PermissionError):
                creator.work_on_task(started_task.name)

            r_worked_task = creator.get_task(worked_task.name)
            assert 'percentage' in r_worked_task.working_payload, "Work payload does not contain percentage"
            assert np.isclose(r_worked_task.working_payload['percentage'], current_percentage), "Percentage is wrong"

        with pytest.raises(PermissionError):
            creator.complete_task(r_worked_task.name, output_payload={})

        completed_task = worker.complete_task(r_worked_task.name, output_payload={'done': True})
        assert completed_task is not None, "Completed task should be valid"

        tasks = god.get_tasks()
        print(tasks)
        assert len(tasks) == 3, "Wrong number of retrieved tasks!"  # TOTAL TASK SHOULD BE 3

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_manager(self, mongo_client):
        self._test_manager(mongo_client)

    def test_manager_mock(self, mongo_client_mock):
        self._test_manager(mongo_client_mock)
