
from persefone.data.databases.mongo.model import MTask, MTaskStatus
from persefone.data.databases.mongo.clients import MongoDatabaseTaskManager, MongoDatabaseTaskManagerType
import pytest
import numpy as np


class TestDatabaseTaskManager(object):

    def _test_manager(self, mongo_client):

        creator = MongoDatabaseTaskManager(mongo_client=mongo_client, manager_type=MongoDatabaseTaskManagerType.TASK_CREATOR)
        worker = MongoDatabaseTaskManager(mongo_client=mongo_client, manager_type=MongoDatabaseTaskManagerType.TASK_WORKER)
        god = MongoDatabaseTaskManager(mongo_client=mongo_client, manager_type=MongoDatabaseTaskManagerType.TASK_GOD)

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
        assert len(worker.get_tasks(status=MTaskStatus.READY, negate=True)) == 1, "Only one canceled task was there!"
        assert len(worker.get_tasks(status=[MTaskStatus.READY, MTaskStatus.CANCELED], negate=True)) == 0, "No tasks other than ready"
        assert len(tasks) == 2, "Wrong number of retrieved tasks!"  # READY TASK SHOULD BE ONLY 2

        target_task: MTask = tasks[0]
        started_task = worker.start_task(target_task.name)
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

        tasks = god.get_tasks()
        for task in tasks:
            with pytest.raises(PermissionError):
                creator.remove_task(task.name)
            with pytest.raises(PermissionError):
                worker.remove_task(task.name)
            assert god.remove_task(task.name), "Remove task failed"

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_manager(self, temp_mongo_database):
        self._test_manager(temp_mongo_database)

    @pytest.mark.mongo_mock_server  # NOT EXECUTE IF --mongo_real_server option is passed
    def test_manager_mock(self, temp_mongo_mock_database):
        self._test_manager(temp_mongo_mock_database)
