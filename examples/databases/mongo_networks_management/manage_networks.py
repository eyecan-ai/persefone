
from mongoengine.errors import DoesNotExist
from persefone.data.databases.mongo.nodes.nodes import MNode
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG
from persefone.data.databases.mongo.nodes.buckets.networks import NetworksBucket
import click


def clear_context(ctx):
    del ctx.obj['bucket']


def print_header(name):
    print(f"========= {name} =========")


def print_footer():
    print(f"==========================")


@click.group()
@click.option("--database_cfg", required=True, help="Database configuration file")
@click.pass_context
def cli(ctx, database_cfg):
    ctx.obj['bucket'] = NetworksBucket(client_cfg=MongoDatabaseClientCFG(database_cfg))
    print(ctx.obj['bucket'])


############## TRAINABLES #################
###########################################
###########################################
###########################################

@cli.group()
@click.pass_context
def trainables(ctx):
    pass
    """First Command"""


@trainables.command('list')
@click.pass_context
def list_trainables(ctx):
    bucket: NetworksBucket = ctx.obj['bucket']
    trainables = bucket.get_trainables()
    print_header("Trainables list")
    for t in trainables:
        t: MNode
        print(t.last_name)
    print_footer()
    clear_context(ctx)


@trainables.command('new')
@click.option("--trainable_name", required=True, help="Trainable Network name")
@click.pass_context
def new_trainable(ctx, trainable_name):
    bucket: NetworksBucket = ctx.obj['bucket']
    if bucket.new_trainable(trainable_name) is not None:
        print(f"Trainable {trainable_name} created")
    clear_context(ctx)


@trainables.command('get')
@click.option("--trainable_name", required=True, help="Trainable Network name")
@click.pass_context
def get_trainable(ctx, trainable_name):
    bucket: NetworksBucket = ctx.obj['bucket']
    try:
        trainable = bucket.get_trainable(trainable_name)
        print(trainable, trainable.last_name)
    except DoesNotExist:
        print(f"Trainable {trainable_name} not found!")
    clear_context(ctx)


############## MODELS #################
###########################################
###########################################
###########################################


@cli.group()
@click.pass_context
def models(ctx):
    pass
    """First Command"""


@models.command('list')
@click.option("--trainable_name", required=True, help="Trainable Network name")
@click.pass_context
def list_models(ctx, trainable_name):
    bucket: NetworksBucket = ctx.obj['bucket']
    try:
        models = bucket.get_models(trainable_name)
        print_header(f"{trainable_name} / Models list")
        for t in models:
            t: MNode
            print(t.last_name)
        print_footer()
    except DoesNotExist:
        print(f"Trainable {trainable_name} not found!")
    clear_context(ctx)


@models.command('new')
@click.option("--trainable_name", required=True, help="Trainable Network name")
@click.option("--model_name", required=True, help="Trained Model name")
@click.pass_context
def new_model(ctx, trainable_name, model_name):
    bucket: NetworksBucket = ctx.obj['bucket']

    try:
        model = bucket.new_model(trainable_name, model_name)
        print(f"Model {trainable_name}/{model_name} created!")
    except DoesNotExist:
        print(f"Trainable {trainable_name} not found!")
    except NameError:
        print(f"Model {trainable_name}/{model_name} exists!")
    clear_context(ctx)

############## TASKS #################
###########################################
###########################################
###########################################


@cli.group()
@click.pass_context
def tasks(ctx):
    pass
    """First Command"""


@tasks.command('list')
@click.option("--trainable_name", required=True, help="Trainable Network name")
@click.pass_context
def list_tasks(ctx, trainable_name):
    bucket: NetworksBucket = ctx.obj['bucket']
    try:
        tasks = bucket.get_tasks(trainable_name)
        print_header(f"{trainable_name} / Tasks list")
        for t in tasks:
            t: MNode
            print(t.last_name)
        print_footer()
    except DoesNotExist:
        print(f"Trainable {trainable_name} not found!")
    clear_context(ctx)


@tasks.command('new')
@click.option("--trainable_name", required=True, help="Trainable Network name")
@click.option("--task_name", required=True, help="Task name")
@click.pass_context
def new_task(ctx, trainable_name, task_name):
    bucket: NetworksBucket = ctx.obj['bucket']

    try:
        task = bucket.new_task(trainable_name, task_name)
        print(f"Task {trainable_name}/{task_name} created!")
    except DoesNotExist:
        print(f"Trainable {trainable_name} not found!")
    except NameError:
        print(f"Task {trainable_name}/{task_name} exists!")
    clear_context(ctx)


############## UTILS #################
###########################################
###########################################
###########################################

@cli.group()
@click.pass_context
def utils(ctx):
    pass
    """First Command"""


@utils.command('graph')
@click.pass_context
def draw_graph(ctx):
    bucket: NetworksBucket = ctx.obj['bucket']

    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    trainables = bucket.get_trainables()

    # Links from base to trainables
    base_edges = []
    trainable_nodes = []
    base = bucket.get_namespace_node()
    for trainable in trainables:
        base_edges.append(('root', trainable.last_name))
        trainable_nodes.append(trainable.last_name)
    # G.add_edges_from(base_edges)
    G.add_edges_from(base_edges)

    # Links from trainable to models
    models_edges = []
    models_nodes = []
    for trainable in trainables:
        for model in bucket.get_models(trainable.last_name):
            models_edges.append((trainable.last_name, model.last_name))
            models_nodes.append(model.last_name)
    G.add_edges_from(models_edges)

    # Links from trainable to models
    tasks_edges = []
    task_nodes = []
    for trainable in trainables:
        for task in bucket.get_tasks(trainable.last_name):
            tasks_edges.append((trainable.last_name, task.last_name))
            task_nodes.append(task.last_name)
    G.add_edges_from(tasks_edges)

    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=['root'], node_color='black', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=trainable_nodes, node_color='r', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=models_nodes, node_color='g', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=task_nodes, node_color='#ffff00', node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=base_edges, edge_color='r', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=models_edges, edge_color='g', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=tasks_edges, edge_color='#ffff00', arrows=True)
    plt.show()


# wp = click.CommandCollection(sources=[trainables], help="This is the help")
if __name__ == '__main__':
    cli(obj={}, standalone_mode=True)
