

from enum import unique
from persefone.data.databases import mongo

import mongoengine
from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDatabaseClientCFG
import pickle
from typing import Any, List, Union
from mongoengine.document import Document
from mongoengine.fields import DictField, FileField, GenericLazyReferenceField, LazyReferenceField, ListField, MapField, ReferenceField, StringField
import logging


class NodesPath(object):
    PATH_SEPARATOR = '/'

    def __init__(self, value):
        self._value = value
        if self._value.startswith(self.PATH_SEPARATOR):
            self._value = self._value[:1]
        self._chunks = value.split(self.PATH_SEPARATOR)
        self._chunks = [x for x in self._chunks if len(x) > 0]
        self._valid = False
        if len(self._chunks) > 2:
            self._valid = True

    @property
    def value(self):
        return self._value

    @property
    def valid(self):
        return self._valid

    @property
    def namespace(self):
        if self.valid:
            return self._chunks[0]
        return None

    @property
    def category(self):
        if self.valid:
            return self._chunks[1]
        return None

    @property
    def items(self):
        if self.valid:
            return self._chunks[2:]
        return []

    @classmethod
    def builds_path(cls, namespace: str, category: str, items: List[str]):
        p = f'{namespace}{cls.PATH_SEPARATOR}{category}'
        for item in items:
            p += f'{cls.PATH_SEPARATOR}{item}'
        return NodesPath(p)


class MLink(Document):
    """ Link model """

    start_node_ = LazyReferenceField("MNode")
    end_node_ = LazyReferenceField("MNode")
    link_type_ = StringField()
    metadata_ = DictField()

    meta = {
        'indexes': [
            ('start_node_', 'end_node_')  # text index
        ]
    }

    @property
    def start_node(self):
        return self.start_node_

    @property
    def end_node(self):
        return self.end_node_

    @property
    def link_type(self):
        return self.link_type_

    @link_type.setter
    def link_type(self, tp: str):
        self.link_type_ = tp
        self.save()

    @property
    def metadata(self):
        return self.metadata_

    @metadata.setter
    def metadata(self, d: dict):
        self.metadata_ = d
        self.save()

    @classmethod
    def outbound_of(cls, node: 'MNode'):
        return MLink.objects(start_node_=node).order_by('start_node___name', 'end_node___name', 'link_type')

    @classmethod
    def inbound_of(cls, node: 'MNode'):
        return MLink.objects(end_node_=node).order_by('start_node___name', 'end_node___name', 'link_type')


class MNode(Document):
    """ Node model """

    name_ = StringField(required=True)
    node_type_ = StringField()
    data_ = FileField()
    metadata_ = DictField()

    meta = {
        'indexes': [
            '$name_'  # text index
        ]
    }

    @property
    def name(self):
        return self.name_

    @name.setter
    def name(self, name):
        self.name_ = name
        self.save()

    @property
    def node_type(self):
        return self.node_type_

    @node_type.setter
    def node_type(self, tp: str):
        self.node_type_ = tp
        self.save()

    @property
    def path(self):
        return NodesPath(self.name)

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

    def link_to(self, node: 'MNode', metadata={}, link_type: str = ''):
        link = MLink()
        link.start_node_ = self
        link.end_node_ = node
        link.metadata_ = metadata
        link.link_type_ = link_type
        link.save()

    @property
    def outbound(self):
        return MLink.outbound_of(self)

    @property
    def inbound(self):
        return MLink.inbound_of(self)

    @classmethod
    def get_by_name(cls, name: str):
        return MNode.objects.get(name_=name)

    @classmethod
    def create(cls, name: str):
        path = NodesPath(name)
        if not path.valid:
            raise NameError(f'Node name "{name}" is not valid')

        node = MNode(name)
        node.save()
        return node


# Register MNOde - MLink reverse delete rules
MNode.register_delete_rule(MLink, 'start_node_', mongoengine.CASCADE)
MNode.register_delete_rule(MLink, 'end_node_', mongoengine.CASCADE)


class NodesRealm(object):

    def __init__(self, client_cfg: MongoDatabaseClientCFG):
        self._client_cfg = client_cfg
        self._mongo_client = MongoDatabaseClient(cfg=self._client_cfg)
        self._mongo_client.connect()

    def __getitem__(self, path):

        np = NodesPath(path)
        if not np.valid:
            return None

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
