

from enum import unique
from os import link
import pathlib

from mongoengine.errors import DoesNotExist
from numpy.lib.function_base import piecewise
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
    PRIVATE_NAME_FOR_NAMESPACE = '$$NAMESPACE'
    PRIVATE_NAME_FOR_CATEGORY = '$$CATEGORY'

    def __init__(self, value):
        if isinstance(value, pathlib.Path):
            value = str(value)

        self._value = value

        if self._value.startswith(self.PATH_SEPARATOR):
            self._value = self._value[1:]

        if self._value.endswith(self.PATH_SEPARATOR):
            self._value = self._value[:-1]

        self._chunks = self._value.split(self.PATH_SEPARATOR)

        self._valid = False
        if len(self._chunks) >= 1:
            self._valid = True
            for chunk in self._chunks:
                if len(chunk) == 0:
                    self._valid = False

    @property
    def value(self):
        return self._value

    @property
    def valid(self):
        return self._valid

    @property
    def items(self):
        return self._chunks

    @property
    def parent_path(self):
        return self.builds_path(*self.items[:-1])

    def subpaths(self, reverse=True):
        pieces = []
        for i in range(1, len(self.items) + 1):
            pieces.append(self.builds_path(*self.items[0:i]))
        if reverse:
            pieces.reverse()
        return pieces

    @classmethod
    def builds_path(cls, *items):
        p = ''
        for item in items:
            p += f'{item}{cls.PATH_SEPARATOR}'
        return NodesPath(p)


class MLink(Document):
    """ Link model """

    start_node_ = LazyReferenceField("MNode", passthrough=True)
    end_node_ = LazyReferenceField("MNode", passthrough=True)
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
    def outbound_of(cls, node: 'MNode', link_type: str = None):
        try:
            if link_type is None:
                return MLink.objects(start_node_=node)  # .order_by('start_node___name', 'end_node___name', 'link_type_')
            else:
                return MLink.objects(start_node_=node, link_type_=link_type)  # .order_by('start_node___name', 'end_node___name')
        except:
            return []

    @classmethod
    def inbound_of(cls, node: 'MNode', link_type: str = None):
        try:
            if link_type is None:
                return MLink.objects(end_node_=node).order_by('start_node___name', 'end_node___name', 'link_type_')
            else:
                return MLink.objects(end_node_=node, link_type_=link_type).order_by('start_node___name', 'end_node___name')
        except:
            return []

    @classmethod
    def outbound_nodes_of(cls, node: 'MNode', link_type: str = None):
        return [x.end_node.fetch() for x in cls.outbound_of(node, link_type=link_type)]

    @classmethod
    def links_of(cls, node_0: 'MNode', node_1: 'MNode', link_type: str = None):
        if link_type is None:
            return MLink.objects(start_node_=node_0, end_node_=node_1).order_by(
                'start_node___name', 'end_node___name', 'link_type_')
        else:
            return MLink.objects(start_node_=node_0, end_node_=node_1, link_type_=link_type).order_by(
                'start_node___name', 'end_node___name')

    @classmethod
    def links_by_type(cls, link_type: str):
        return MLink.objects(link_type_=link_type).order_by(
            'start_node___name', 'end_node___name')


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

    @property
    def last_name(self):
        np = NodesPath(self.name)
        if np.valid:
            return np.items[-1]
        return None

    # @ name.setter
    # def name(self, name):
    #     self.name_ = name
    #     self.save()

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

    # @data.setter
    # def data(self, data):
    #     if data is not None:
    #         self.data_.put(data)
    #         self.save()

    def put_data(self, data, data_encoding):
        if data is not None:
            self.data_.put(data, content_type=data_encoding)
            self.save()

    def get_data(self):
        self.data_.seek(0)
        d = self.data_.read()
        if d is not None:
            return d, self.data_.content_type
        return None, None

    def link_to(self, node: 'MNode', metadata={}, link_type: str = ''):
        link = MLink()
        link.start_node_ = self
        link.end_node_ = node
        link.metadata_ = metadata
        if len(link_type) == 0:
            link_type = f'{self.node_type}_2_{node.node_type}'
        link.link_type_ = link_type
        link.save()
        return link

    def outbound(self, link_type: str = None):
        return MLink.outbound_of(self, link_type=link_type)

    def outbound_nodes(self, link_type: str = None):
        return MLink.outbound_nodes_of(self, link_type=link_type)

    def inbound(self, link_type: str = None):
        return MLink.inbound_of(self, link_type=link_type)

    @classmethod
    def get_by_name(cls, name: str, create_if_none: bool = False, metadata: dict = None, node_type: str = None) -> 'MNode':
        """ Get MNode by name

        :param name: node name
        :type name: str
        :param create_if_none: TRUE to create node if not found, defaults to False
        :type create_if_none: bool, optional
        :param metadata: attached metadata, defaults to None
        :type metadata: dict, optional
        :param node_type: node type string representation, defaults to None
        :type node_type: str, optional
        :raises DoesNotExist: raises Exception if not exists and create_if_none flag is False
        :return: retrieved MNode
        :rtype: MNode
        """

        name = NodesPath(name).value
        try:
            return MNode.objects.get(name_=str(name))
        except DoesNotExist:
            if create_if_none:
                return MNode.create(str(name), metadata=metadata, node_type=node_type)
            else:
                raise DoesNotExist()

    @classmethod
    def get_by_node_type(cls, node_type: str):
        return MNode.objects(node_type_=node_type)

    @classmethod
    def create(cls, name: str, metadata: dict = None, node_type: str = None):
        # path = NodesPath(name)
        # if not path.valid:
        #     raise NameError(f'Node name "{name}" is not valid')
        name = NodesPath(name).value
        node = MNode(name_=name, metadata_=metadata, node_type_=node_type)
        node.save()
        return node


# Register MNOde - MLink reverse delete rules
MNode.register_delete_rule(MLink, 'start_node_', mongoengine.CASCADE)
MNode.register_delete_rule(MLink, 'end_node_', mongoengine.CASCADE)


class NodesBucket(object):
    DEFAULT_NAMESPACE = 'VAR'
    LINK_TYPE_NAMESPACE2GENERIC = 'namespace_2_any'

    def __init__(self, client_cfg: MongoDatabaseClientCFG, namespace: str = None):
        if not isinstance(client_cfg, MongoDatabaseClientCFG):
            raise TypeError(f"Configuration '{client_cfg}' is invalid!")
        self._namespace = namespace if namespace is not None else self.DEFAULT_NAMESPACE
        self._client_cfg = client_cfg
        self._mongo_client = MongoDatabaseClient(cfg=self._client_cfg)
        self._mongo_client.connect()

    @property
    def namespace(self):
        return pathlib.Path(self._namespace)

    @property
    def category(self):
        return self._category

    def get_node_by_name(self, name):
        return MNode.get_by_name(name)

    def get_namespace_node(self):
        return self[self.namespace]

    def __getitem__(self, path):

        node_path = NodesPath(path)
        if not node_path.valid:
            return None

        if not node_path.items[0] == str(self.namespace):
            raise PermissionError(f"Namespace '{self.namespace}' cannot manages nodes starting with '{node_path.items[0]}'")

        namespace_node = MNode.get_by_name(node_path.items[0], True, None, NodesPath.PRIVATE_NAME_FOR_NAMESPACE)

        current_node = namespace_node

        for sub_idx, sub_path in enumerate(node_path.subpaths(reverse=False)[1:]):

            sub: NodesPath
            node = MNode.get_by_name(sub_path.value, create_if_none=True)

            if sub_idx == 0:
                namespace_link = MLink.links_of(namespace_node, node)
                if namespace_link is None:
                    namespace_link = namespace_node.link_to(node, link_type=self.LINK_TYPE_NAMESPACE2GENERIC)

            current_node = node

        return current_node

    def __del__(self):
        if self._mongo_client is not None:
            self._mongo_client.disconnect()
