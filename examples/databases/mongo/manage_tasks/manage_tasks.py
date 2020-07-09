from persefone.data.databases.mongo.clients import MongoDatabaseClient, DatabaseTaskManager
from persefone.data.databases.mongo.model import MTask
from persefone.utils.images.drawing import ConsoleImage
import click
from tabulate import tabulate

_AVAILABLE_COMMANDS = ['LIST', 'NEW', 'START', 'WORK', 'CANCEL', 'COMPLETE', 'DELETE', 'ARMAGEDDON']


def print_header(length=2):
    print("\n" * length)
    ConsoleImage.print_image('logo.png', scale=0.3)
    print("\n" * length)
    print("=" * 20, "List", "=" * 20)
    print("\n")


def print_footer():
    print("\n")
    print("=" * 44)
    print("\n")


def print_task(task: MTask):
    print(f"Task [{task.name}]", "=" * 20)
    table = []
    table.append(['name', task.name])
    table.append(['status', task.status])
    table.append(['input metadata', task.input_payload])
    table.append(['working metadata', task.working_payload])
    table.append(['output metadata', task.output_payload])
    print(tabulate(table))
    print('\n')


@click.command("Manage tasks")
@click.option('--database_cfg', default='database.yml', help="Database configuration file")
@click.option('--command', required=True, type=click.Choice(_AVAILABLE_COMMANDS), help=f"Command to execute")
def manage_tasks(database_cfg, command):

    client = MongoDatabaseClient.create_from_configuration_file(filename=database_cfg)
    manager = DatabaseTaskManager(mongo_client=client)

    # ==================== TASKS LIST ==================================
    if command == 'LIST':
        print_header()
        tasks = manager.get_tasks()  # status=MTaskStatus.CANCELED, negate=True)
        if len(tasks) == 0:
            print("No tasks found!")
        else:
            for task in tasks:
                print_task(task)
        print_footer()

    # ==================== NEW TASK ==================================
    if command == 'NEW':
        name = input("Inser task name:")
        task = manager.new_task(name)
        if task is None:
            print("Task creation fail! Check name!")
            return
        print(f"New task created:")
        print_task(task)

    # ==================== START TASK ==================================
    if command == 'START':
        name = input("Inser task name:")
        task = manager.start_task(name)
        if task is None:
            print(f"No task with name '{name}' can be started'")
        else:
            print("Task started:")
            print_task(task)

    # ==================== WORK ON TASK ==================================
    if command == 'WORK':
        name = input("Inser task name:")
        metadata = input("Insert metadata:")
        metadata = {metadata.split(':')[0]: metadata.split(':')[1]}
        task = manager.work_on_task(name, work_payload=metadata)
        if task is None:
            print(f"No task with name '{name}' can be worked")
        else:
            print("Task worked:")
            print_task(task)

    # ==================== COMPLETE TASK ==================================
    if command == 'COMPLETE':
        name = input("Inser task name:")
        task = manager.complete_task(name)
        if task is None:
            print(f"No task with name '{name}' can be completed")
        else:
            print("Task completed:")
            print_task(task)

    # ==================== CANCEL TASK ==================================
    if command == 'CANCEL':
        name = input("Inser task name:")
        task = manager.cancel_task(name)
        if task is None:
            print(f"No task with name '{name}' can be canceled")
        else:
            print("Task canceled:")
            print_task(task)

    # ==================== DELETE TASK ==================================
    if command == 'DELETE':
        name = input("Inser task name:")
        result = manager.remove_task(name)
        if result is None:
            print(f"No task with name '{name}' can be deleted")
        else:
            print("Task deleted")

    # ==================== DESTROY ALL TASKs ==================================
    if command == 'ARMAGEDDON':
        whole_tasks = manager.get_tasks()
        for task in whole_tasks:
            manager.remove_task(task.name)
        print(f"Armageddon complete. Remaining tasks: {len(manager.get_tasks())}")


if __name__ == "__main__":
    manage_tasks()
