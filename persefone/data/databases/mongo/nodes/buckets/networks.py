

from persefone.data.databases.mongo.nodes.nodes import MNode, NodesBucket
from mongoengine.errors import DoesNotExist
from persefone.data.databases.mongo.clients import MongoDatabaseClientCFG


class NetworksBucket(NodesBucket):  # TODO: doc missing

    NETWORKS_NAMSPACE_NAME = '$NETWORKS'
    NODE_TYPE_TRAINABLE = 'trainable_model'
    NODE_TYPE_MODEL = 'trained_model'
    NODE_TYPE_TASK = 'training_task'
    LINK_TYPE_TRAINABLE2MODEL = 'trainable_model_2_trained_model'
    LINK_TYPE_TRAINABLE2TASK = 'trainable_model_2_training_task'
    LINK_TYPE_TASK2MODEL = 'training_task_2_trained_model'
    LINK_TYPE_TASK2DATASET = 'training_task_2_dataset'
    LINK_TYPE_MODEL2DATASET = 'model_2_dataset'

    DEFAULT_ZERO_PADDING = 7

    def __init__(self, client_cfg: MongoDatabaseClientCFG, namespace: str = None):
        super(NetworksBucket, self).__init__(client_cfg, self.NETWORKS_NAMSPACE_NAME)

    def get_trainable(self, trainable_name: str):
        return self.get_node_by_name(self.namespace / trainable_name)

    def get_trainables(self):
        namespace_node: MNode = self.get_namespace_node()
        return namespace_node.outbound_nodes(link_type=self.LINK_TYPE_NAMESPACE2GENERIC)

    def new_trainable(self, trainable_name: str, metadata: dict = None):

        try:
            self.get_trainable(trainable_name)
            raise NameError(f"Trainable with same name '{trainable_name}' already exists")
        except DoesNotExist:

            namespace_node: MNode = self.get_namespace_node()
            trainable_node: MNode = self[self.namespace / trainable_name]
            trainable_node.node_type = self.NODE_TYPE_TRAINABLE
            trainable_node.set_metadata(metadata)
            trainable_node.save()

            namespace_node.link_to(trainable_node, link_type=self.LINK_TYPE_NAMESPACE2GENERIC)
            return trainable_node

    def get_model(self, trainable_name: str, model_name: str):
        return self.get_node_by_name(self.namespace / trainable_name / model_name)

    def new_model(self, trainable_name: str, model_name: str, metadata: dict = None):

        trainable_node: MNode = self.get_trainable(trainable_name)

        try:
            self.get_model(trainable_name, model_name)
            raise NameError(f"Model with same name '{trainable_name}/{model_name}' already exists")
        except DoesNotExist:

            model_node: MNode = self[self.namespace / trainable_name / model_name]
            model_node.node_type = self.NODE_TYPE_MODEL
            model_node.set_metadata(metadata)
            model_node.save()

            trainable_node.link_to(model_node, link_type=self.LINK_TYPE_TRAINABLE2MODEL)
            return model_node

    def get_models(self, trainable_name: str):
        trainable_node: MNode = self.get_trainable(trainable_name)
        return trainable_node.outbound_nodes(link_type=self.LINK_TYPE_TRAINABLE2MODEL)

    def get_task(self, trainable_name: str, task_name: str):
        return self.get_node_by_name(self.namespace / trainable_name / task_name)

    def get_tasks(self, trainable_name: str):
        trainable_node: MNode = self.get_trainable(trainable_name)
        return trainable_node.outbound_nodes(link_type=self.LINK_TYPE_TRAINABLE2TASK)

    def get_datasets_of_task(self, trainable_name: str, task_name: str):
        task_node: MNode = self.get_task(trainable_name, task_name)
        return task_node.outbound_nodes(link_type=self.LINK_TYPE_TASK2DATASET)

    def new_task(self, trainable_name: str, task_name: str, metadata: dict = None):

        trainable_node: MNode = self.get_trainable(trainable_name)

        try:
            self.get_task(trainable_name, task_name)
            raise NameError(f"Task with same name '{trainable_name}/{task_name}' already exists")
        except DoesNotExist:

            task_node: MNode = self[self.namespace / trainable_name / task_name]
            task_node.node_type = self.NODE_TYPE_TASK
            task_node.set_metadata(metadata)
            task_node.save()

            trainable_node.link_to(task_node, link_type=self.LINK_TYPE_TRAINABLE2TASK)
            return task_node

    def delete_trainable(self, trainable_name: str):

        tranable_node: MNode = self.get_trainable(trainable_name)

        tasks = self.get_tasks(trainable_name)
        for task in tasks:
            task: MNode
            task.delete()

        models = self.get_models(trainable_name)
        for model in models:
            model: MNode
            model.delete()

        tranable_node.delete()

    def delete_task(self, trainable_name: str, task_name: str):
        task = self.get_task(trainable_name, task_name)
        task.delete()

    def delete_model(self, trainable_name: str, model_mame: str):
        model = self.get_model(trainable_name, model_mame)
        model.delete()
