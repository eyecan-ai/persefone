from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDatabaseTaskManager, MongoDatabaseTaskManagerType
from persefone.data.databases.mongo.model import MTask, MTaskStatus
import click
import time


@click.command("Task Daemon")
@click.option('--database_cfg', default='database.yml', help="Database configuration file")
@click.option('--name', required=True, type=str, help=f"Daemon Name")
@click.option('--work_time', default=10, type=int, help=f"Task work time [s]")
def manage_tasks(database_cfg, name, work_time):

    client = MongoDatabaseClient.create_from_configuration_file(filename=database_cfg)
    manager = MongoDatabaseTaskManager(mongo_client=client, manager_type=MongoDatabaseTaskManagerType.TASK_WORKER)

    print(f'Daemon [{name}] started...')

    while True:

        tasks = manager.get_tasks(status=MTaskStatus.READY)
        print(f"Polling [{len(tasks)}]...")

        for task in tasks:

            if name in task.name:
                started_task: MTask = manager.start_task(task.name)
                print(f"Task [{started_task.name}] started...")
                time.sleep(2)

                dt = work_time / 100
                for j in range(100 + 1):
                    percentage = j / 100
                    print(f"\tWorking on [{started_task.name}]: ", {'daemon_name': name, 'percentage': percentage})
                    manager.work_on_task(started_task.name, {'daemon_name': name, 'percentage': percentage})
                    time.sleep(dt)

                print(f"Task [{name}] completed!")
                manager.complete_task(started_task.name, {'daemon_name': name, 'output': '/tmp/networks/hell.txt'})
                break

        time.sleep(1)


if __name__ == "__main__":
    manage_tasks()
