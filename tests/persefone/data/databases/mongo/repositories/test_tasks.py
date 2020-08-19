
from persefone.data.databases.mongo.repositories import TasksRepository
from persefone.data.databases.mongo.model import MTask, MTaskStatus
import time
import pytest


class TestTaskManagement(object):

    def _generate_tasks(self, N: int, name_prefix: str, source: str, payload={}, dt: float = 0.):
        for i in range(N):
            TasksRepository.new_task(
                name=f'{name_prefix}{i}',
                source=source,
                input_payload=payload
            )
            time.sleep(dt)

    @pytest.mark.mongo_real_server  # EXECUTE ONLY IF --mongo_real_server option is passed
    def test_lifecycle(self, temp_mongo_database):
        self._test_lifecycle(temp_mongo_database)

    @pytest.mark.mongo_mock_server  # NOT EXECUTE IF --mongo_real_server option is passed
    def test_lifecycle_mock(self, temp_mongo_mock_database):
        self._test_lifecycle(temp_mongo_mock_database)

    def _test_lifecycle(self, database):

        N_ready = 20
        N_cancel = 20
        self._generate_tasks(N_ready, 'ready_to_start_', 'tester', payload={'this': 'is', 'a': 'meta', 'data': 2.2})
        self._generate_tasks(N_cancel, 'to_cancel_', 'tester', payload={'this': 'is', 'a': 'meta', 'data': 2.2})

        whole_tasks = TasksRepository.get_tasks(status=MTaskStatus.READY)
        assert len(TasksRepository.get_tasks(status=MTaskStatus.READY, negate=True)) == 0, "No tasks other than ready"
        assert len(TasksRepository.get_tasks(status=[MTaskStatus.DONE, MTaskStatus.WORKING])) == 0, "No tasks other than ready"
        assert len(TasksRepository.get_tasks(status=[MTaskStatus.DONE, MTaskStatus.WORKING], negate=True)) != 0, "Should be not empty!"

        assert len(whole_tasks) == N_cancel + N_ready, "Task size is wrong!"

        for task in whole_tasks:
            task: MTask

            if 'ready_to_start_' in task.name:
                pre_working_task = TasksRepository.work_on_task(task, {})
                assert pre_working_task is None, "Working on ready task is not allowed"

                started_task = TasksRepository.start_task(task)
                assert started_task == task, "Started task is different from original one!"
                restarted_task = TasksRepository.start_task(task)
                assert restarted_task is None, "Restarting task is not allowed"

        whole_tasks = TasksRepository.get_tasks()
        for task in whole_tasks:
            task: MTask

            if 'ready_to_start_' in task.name:

                for wj in range(10):
                    working_task = TasksRepository.work_on_task(task, working_payload={'percentage': wj})
                    assert working_task == task, "Working on task failed!"

                done_task = TasksRepository.complete_task(task, output_payload={'done': True})
                assert done_task == task, "Completing task fails"

        whole_tasks = TasksRepository.get_tasks()
        for task in whole_tasks:
            task: MTask

            if 'ready_to_start_' in task.name:
                assert TasksRepository.start_task(task) is None, "Starting should be impossible!"
                assert TasksRepository.work_on_task(task) is None, "Working should be impossible!"
                assert TasksRepository.complete_task(task) is None, "Completing should be impossible!"

            if 'to_cancel_' in task.name:
                canceled_task = TasksRepository.cancel_task(task)
                assert canceled_task == task, "Canceled task is different from original one!"
                assert TasksRepository.cancel_task(canceled_task) is None, "Double cancel is not Allowed!"
                assert TasksRepository.start_task(canceled_task) is None, "Start not allowd on canceled task!"
                assert TasksRepository.work_on_task(canceled_task) is None, "Work not allowd on canceled task!"
                assert TasksRepository.complete_task(canceled_task) is None, "Complete not allowd on canceled task!"

        whole_tasks = TasksRepository.get_tasks()
        for task in whole_tasks:
            assert TasksRepository.delete_task(task), "Remove task failed!"

        assert len(TasksRepository.get_tasks()) == 0, "All tasks should be removed"
