

from persefone.data.databases.mongo.clients import MongoDatabaseClient
import pickle
from typing import Any, Union
from mongoengine.document import Document
from mongoengine.fields import FileField, GenericLazyReferenceField, MapField, StringField
import logging


class MNode(Document):
    """ Model model """

    namespace_f = StringField(required=True)
    category_f = StringField()
    name_f = StringField(required=True, unique_with='namespace_f')
    data_f = FileField()
    links_f = MapField(field=GenericLazyReferenceField())

    @property
    def namespace(self):
        return self.namespace_f

    @namespace.setter
    def namespace(self, namespace):
        self.namespace_f = namespace
        self.save()

    @property
    def name(self):
        return self.name_f

    @name.setter
    def name(self, name):
        self.name_f = name
        self.save()

    @property
    def category(self):
        return self.category_f

    @category.setter
    def category(self, category):
        self.category_f = category
        self.save()

    @property
    def data(self):
        self.data_f.seek(0)
        d = self.data_f.read()
        if d is not None:
            return pickle.loads(d)
        return None

    @data.setter
    def data(self, data):
        if data is not None:
            self.data_f = pickle.dumps(data)
            self.save()

    @property
    def links(self):
        return self.links_f

    def link_to(self, node: 'MNode', link_name: str):
        self.links_f[link_name] = node
        self.save()


class NodesRealm(object):

    def __init__(self, mongo_client: MongoDatabaseClient):
        self._mongo_client = mongo_client
        self._mongo_client.connect()

    def __getitem__(self, fullname):

        fullname = str(fullname)

        if '.' in fullname or '/' in fullname:
            fullname = fullname.replace('/', '.')
            namespace, name = fullname.split('.')
            node = self.get_node(namespace, name)
            if node is None:
                node = self.new_node(namespace, name)
            return node
        else:
            return self[f'_.{fullname}']

    def new_node(self, namespace: str, name: str, category: str = '', obj: Any = None) -> Union[MNode, None]:

        node = MNode(namespace_f=namespace, category_f=category, name_f=name)
        node.data = obj
        node.save()
        return node

    def get_node(self, namespace: str, name: str) -> Union[MNode, None]:

        try:
            node = MNode.objects.get(namespace_f=namespace, name_f=name)
            return node
        except Exception as e:
            logging.error(e)
            return None

    def get_nodes_by_category(self, category: str) -> Union[MNode, None]:

        try:
            return MNode.objects(category_f__contains=category)
        except Exception as e:
            logging.error(e)
            return []

    def link_node(self, node: MNode, document: Document, link_name: str) -> MNode:

        try:
            node.links[link_name] = document
            node.save()
        except Exception as e:
            logging.error(e)
        return node

    def delete_node(self, node: MNode) -> bool:

        try:
            node.delete()
            return True
        except Exception as e:
            logging.error(e)
        return False
