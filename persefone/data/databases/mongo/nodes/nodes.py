

from persefone.utils.configurations import XConfiguration
from mongoengine.errors import DoesNotExist
from mongoengine.queryset.queryset import QuerySet
import mongoengine
from persefone.data.databases.mongo.clients import MongoDatabaseClient, MongoDatabaseClientCFG
from typing import Sequence, Tuple, Union
from mongoengine.document import Document
from mongoengine.fields import DictField, FileField, LazyReferenceField, StringField


class CustomPath(str):

    def __init__(self, s: str):
        self._s = s

    def __repr__(self) -> str:
        return self._s

    def __str__(self) -> str:
        return self._s

    def __truediv__(self, other: 'CustomPath') -> str:
        return CustomPath(str(self).rstrip('/') + '/' + str(other).lstrip('/'))

    def __rtruediv__(self, other: 'CustomPath') -> str:
        return CustomPath(str(self).rstrip('/') + '/' + str(other).lstrip('/'))


class NodesPath(object):
    PATH_SEPARATOR = '/'
    PRIVATE_NAME_FOR_NAMESPACE = '$$NAMESPACE'
    PRIVATE_NAME_FOR_CATEGORY = '$$CATEGORY'

    def __init__(self, value: Union[CustomPath, str]):
        """ Creats a Node CustomPath represenation checking consistency

        :param value: desired path name (e.g. '/one/two/three')
        :type value: Union[CustomPath, str]
        """

        if isinstance(value, CustomPath):
            value = str(value)

        self._value = value

        # Removes front separator
        if self._value.startswith(self.PATH_SEPARATOR):
            self._value = self._value[1:]

        # Removes trailing separator
        if self._value.endswith(self.PATH_SEPARATOR):
            self._value = self._value[:-1]

        # Split path in chunks
        self._chunks = self._value.split(self.PATH_SEPARATOR)

        # Checks for path consistency
        self._valid = False
        if len(self._chunks) >= 1:
            self._valid = True
            for chunk in self._chunks:
                if len(chunk) == 0:
                    self._valid = False

    @property
    def value(self) -> str:
        return self._value

    @property
    def valid(self) -> bool:
        return self._valid

    @property
    def items(self) -> Sequence[str]:
        return self._chunks

    @property
    def parent_path(self) -> 'NodesPath':
        """ Retrieves a NodesPath representing current parent path

        :return: NodesPath parent representation
        :rtype: NodesPath
        """

        return self.builds_path(*self.items[:-1])

    def subpaths(self, reverse: bool = True) -> Sequence['NodesPath']:
        """ Builds a list of NodesPath representing recursive split from current to root

        :return: List of NodesPath
        :rtype:  Sequence['NodesPath']
        """

        pieces = []
        for i in range(1, len(self.items) + 1):
            pieces.append(self.builds_path(*self.items[0:i]))
        if reverse:
            pieces.reverse()
        return pieces

    @classmethod
    def builds_path(cls, *items) -> 'NodesPath':
        """ Builds a NodesPath from strings

        :return: NOdesPath chaining input strings
        :rtype: NodesPath
        """

        p = ''
        for item in items:
            p += f'{item}{cls.PATH_SEPARATOR}'
        return NodesPath(p)


class MLink(Document):
    """ Link Model representing relation between two Nodes """

    start_node = LazyReferenceField("MNode")
    end_node = LazyReferenceField("MNode")
    link_type = StringField()
    metadata = DictField()

    meta = {
        'indexes': [
            ('start_node', 'end_node')  # text index
        ]
    }

    def set_link_type(self, tp: str):
        """ Sets and Saves link type

        :param tp: Link type as string
        :type tp: str
        """

        self.link_type = tp
        self.save()

    def set_metadata(self, d: dict):
        """ Sets and Saves metadata field

        :param d: metadata to store
        :type d: dict
        """

        self.metadata = d
        self.save()

    @classmethod
    def outbound_of(cls, node: 'MNode', link_type: str = None) -> QuerySet:
        """ Retrieves a list of outbound MLink of target node.

        :param node: source MNode
        :type node: MNode
        :param link_type: If not None, filters links by link_type, defaults to None
        :type link_type: str, optional
        :return: list of queried MLink
        :rtype: QuerySet
        """

        try:
            if link_type is None:
                return MLink.objects(start_node=node)
            else:
                return MLink.objects(start_node=node, link_type=link_type)
        except DoesNotExist:
            return []

    @classmethod
    def outbound_of_by_node_type(cls, node: 'MNode', node_type: str) -> Sequence['MLink']:
        """ Retrieves a list of outbound MLink of target node based on node_type

        :param node: source MNode
        :type node: MNode
        :param node_type: used to filter link towards targets with specified node_type
        :type node_type: str
        :return: list of queried MLink
        :rtype: Sequence['MLink']
        """

        return [x for x in cls.outbound_of(node) if x.end_node.fetch().node_type == node_type]

    @classmethod
    def inbound_of(cls, node: 'MNode', link_type: str = None) -> QuerySet:
        """  Retrieves a list of inbound MLink of target node.

        :param node: source MNode
        :type node: MNode
        :param link_type: If not None, filters links by link_type, defaults to None
        :type link_type: str, optional
        :return: list of queried MLink
        :rtype: QuerySet
        """

        try:
            if link_type is None:
                return MLink.objects(end_node=node).order_by('start_node__name', 'end_node__name', 'link_type')
            else:
                return MLink.objects(end_node=node, link_type=link_type).order_by('start_node__name', 'end_node__name')
        except DoesNotExist:
            return []

    @classmethod
    def outbound_nodes_of(cls, node: 'MNode', link_type: str = None) -> Sequence['MNode']:
        """ Retrieves a list of outbound Mnode of target node based on node_type

        :param node: source MNode
        :type node: MNode
        :param link_type: used to filter link towards targets with specified node_type
        :type link_type: str, optional
        :return: list of queried MNode
        :rtype: Sequence['MNode']
        """

        return [x.end_node.fetch() for x in cls.outbound_of(node, link_type=link_type)]

    @classmethod
    def outbound_nodes_of_by_node_type(cls, node: 'MNode', node_type: str) -> Sequence['MNode']:
        """ Retrieves a list of outbound MNode of target node based on node_type

        :param node: source MNode
        :type node: MNode
        :param node_type: used to filter link towards targets with specified node_type
        :type node_type: str
        :return: list fo queried MNode
        :rtype: Sequence['MNode']
        """

        return [x.end_node.fetch() for x in cls.outbound_of_by_node_type(node, node_type)]

    @classmethod
    def links_of(cls, node_0: 'MNode', node_1: 'MNode', link_type: str = None) -> QuerySet:
        """ Retrieves a list of all MLink connecting two MNode s

        :param node_0: source MNode
        :type node_0: MNode
        :param node_1: target MNode
        :type node_1: MNode
        :param link_type: If not None, filters MLink by link_type, defaults to None
        :type link_type: str, optional
        :return: list of queried MLink
        :rtype: QuerySet
        """

        if link_type is None:
            return MLink.objects(start_node=node_0, end_node=node_1).order_by(
                'start_node__name', 'end_node__name', 'link_type')
        else:
            return MLink.objects(start_node=node_0, end_node=node_1, link_type=link_type).order_by(
                'start_node__name', 'end_node__name')

    @classmethod
    def links_by_type(cls, link_type: str) -> QuerySet:
        """ Retrieves all MLInk by link_type

        :param link_type: filtered link_type
        :type link_type: str
        :return: list fo queried MLink
        :rtype: QuerySet
        """

        return MLink.objects(link_type=link_type).order_by(
            'start_node__name', 'end_node__name')


class MNode(Document):
    """ Node model """

    name = StringField(required=True)
    node_type = StringField()
    data = FileField()
    metadata = DictField()
    meta = {
        'indexes': [
            '$name'  # text index
        ]
    }

    @property
    def last_name(self) -> str:
        """ Fetches last chunk of full name path

        :return: last chunk of full name path
        :rtype: str
        """
        np = NodesPath(self.name)
        if np.valid:
            return np.items[-1]
        return None

    def set_node_type(self, tp: str):
        """ Sets and Saves node_type filed

        :param tp: node_type string representation
        :type tp: str
        """

        self.node_type = tp
        self.save()

    def set_metadata(self, d: dict):
        """ Sets and Saves metadata field

        :param d: metadata to store
        :type d: dict
        """

        if d is not None:
            self.metadata = d
            self.save()

    @property
    def plain_metadata(self) -> dict:
        return self.to_mongo()['metadata']

    @property
    def path(self) -> NodesPath:
        return NodesPath(self.name)

    def put_data(self, data: bytes, data_encoding: str):
        """ Puts and Saves blob data into FileField

        :param data: source bytes
        :type data: bytes
        :param data_encoding: source encoding
        :type data_encoding: str
        """

        if data is not None:
            if self.data:
                self.data.replace(data, content_type=data_encoding)
            else:
                self.data.put(data, content_type=data_encoding)
            self.save()

    def get_data(self) -> Tuple[bytes, str]:
        """  Retrieves FielField stored data

        :return: tuple representing (data, encoding)
        :rtype: Tuple[bytes, str]
        """

        if self.data:
            self.data.seek(0)
            d = self.data.read()
            if d is not None:
                return d, self.data.content_type
        return None, None

    def link_to(self, node: 'MNode', metadata={}, link_type: str = '') -> MLink:
        """ Connects current MNode with target MNode creating an MLink

        :param node: target MNode
        :type node: MNode
        :param metadata: optional metadata to store in MLink object, defaults to {}
        :type metadata: dict, optional
        :param link_type: new link type, defaults to ''
        :type link_type: str, optional
        :return: created MLink
        :rtype: MLink
        """

        link = MLink()
        link.start_node = self
        link.end_node = node
        link.metadata_ = metadata
        if len(link_type) == 0:
            link_type = f'{self.node_type}_2_{node.node_type}'
        link.link_type = link_type
        link.save()
        return link

    def outbound(self, link_type: str = None) -> QuerySet:
        """ Retrieves output MLink of current MNode

        :param link_type: filters by link type, defaults to None
        :type link_type: str, optional
        :return: list of retreived MLink
        :rtype: QuerySet
        """

        return MLink.outbound_of(self, link_type=link_type)

    def outbound_by_node_type(self, node_type) -> Sequence[MLink]:
        """ Retrieves outbound MLink of current MNode

            :param node_type: filtering target node types
            :type node_type: node type string representation
            :return: list of retrieved MNode s
            :rtype: Sequence['MNode']
        """

        return MLink.outbound_of_by_node_type(self, node_type)

    def outbound_nodes(self, link_type: str = None) -> Sequence['MNode']:
        """ Retrieves output MLink of current MNode

        :param link_type: filters by link type, defaults to None
        :type link_type: str, optional
        :return: list of retreived MLink
        :rtype: QuerySet
        """

        return MLink.outbound_nodes_of(self, link_type=link_type)

    def outbound_nodes_by_node_type(self, node_type: str) -> Sequence['MNode']:
        """ Retrives outbound MNode by node type

        :param node_type: filtering target node types
        :type node_type: str
        :return: list of retrived MNode
        :rtype: Sequence['MNode']
        """
        return MLink.outbound_nodes_of_by_node_type(self, node_type)

    def inbound(self, link_type: str = None):
        """ Retrieves inbound MLink of current MNode

        :param link_type: filters by link type, defaults to None
        :type link_type: str, optional
        :return: list of retreived MLink
        :rtype: QuerySet
        """

        return MLink.inbound_of(self, link_type=link_type)

    @classmethod
    def get_by_name(cls, name: str, create_if_none: bool = False, metadata: dict = None, node_type: str = None) -> 'MNode':
        """ Get MNode by name

        : param name: node name
        : type name: str
        : param create_if_none: TRUE to create node if not found, defaults to False
        : type create_if_none: bool, optional
        : param metadata: attached metadata, defaults to None
        : type metadata: dict, optional
        : param node_type: node type string representation, defaults to None
        : type node_type: str, optional
        : raises DoesNotExist: raises Exception if not exists and create_if_none flag is False
        : return: retrieved MNode
        : rtype: MNode
        """

        name = NodesPath(name).value
        try:
            return MNode.objects.get(name=str(name))
        except DoesNotExist:
            if create_if_none:
                return MNode.create(str(name), metadata=metadata, node_type=node_type)
            else:
                raise DoesNotExist(f"Node with name '{name}' does not exist!")

    @classmethod
    def get_by_node_type(cls, node_type: str) -> QuerySet:
        """ Retrives list of MNode by node type

        :param node_type: filtering node tyhpe
        :type node_type: str
        :return: list of queried MNode
        :rtype: QuerySet
        """

        return MNode.objects(node_type=node_type)

    @classmethod
    def get_by_queries(cls, query_dict: dict = {}, orders_bys: list = None) -> QuerySet:
        """ Queries on MNode documents

        :param query_dict: dictionary of mongonegine-like kwargs arguments, defaults to {}
        :type query_dict: dict, optional
        :param orders_bys: list of mongonengine-like orders string, defaults to None
        :type orders_bys: list, optional
        :return: list of retrieved MNode s
        :rtype: QuerySet
        """

        if orders_bys is None:
            orders_bys = []

        return MNode.objects(**query_dict).order_by(*orders_bys)

    @classmethod
    def create(cls, name: str, metadata: dict = None, node_type: str = None) -> 'MNode':
        """ Creates new MNode

        :param name: target MNode name
        :type name: str
        :param metadata: target MNode metadata, defaults to None
        :type metadata: dict, optional
        :param node_type: target MNode type, defaults to None
        :type node_type: str, optional
        :raises NameError: raise exception if name collision occurs
        :return: built MNode
        :rtype: 'MNode'
        """

        name = NodesPath(name).value
        node = MNode(name=name, metadata=metadata, node_type=node_type)
        node.save()
        return node


# Register MNOde - MLink reverse delete rules
MNode.register_delete_rule(MLink, 'start_node', mongoengine.CASCADE)
MNode.register_delete_rule(MLink, 'end_node', mongoengine.CASCADE)


class NodesBucket(object):
    DEFAULT_NAMESPACE = 'VAR'
    LINK_TYPE_NAMESPACE2GENERIC = 'namespace_2_any'

    def __init__(self, client_cfg: MongoDatabaseClientCFG, namespace: str = None):
        """ Creates a Nodes Bucket

        :param client_cfg: database configuration object
        :type client_cfg: MongoDatabaseClientCFG
        :param namespace: bucket namespace, defaults to None
        :type namespace: str, optional
        :raises TypeError: raise Exception if configuration is wrong!
        """

        if not isinstance(client_cfg, XConfiguration):
            raise TypeError(f"Configuration '{client_cfg}' is invalid!")
        self._namespace = namespace if namespace is not None else self.DEFAULT_NAMESPACE
        self._client_cfg = client_cfg
        self._mongo_client = MongoDatabaseClient(cfg=self._client_cfg)
        self._mongo_client.connect()

    @property
    def namespace(self) -> CustomPath:
        """ Path representation of namespace, used to start a chain of '/' operators """
        return CustomPath(self._namespace)

    @property
    def category(self) -> str:
        return self._category

    def get_node_by_name(self, name: str) -> MNode:
        """ Retrives a node by name

        :param name: search for name
        :type name: str
        :return: retrived MNode if any
        :rtype: MNode
        """

        return MNode.get_by_name(name)

    def get_namespace_node(self) -> MNode:
        """ Retrieves the root namespace Node

        :return: [description]
        :rtype: MNode
        """

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

    def disconnect(self):
        if self._mongo_client is not None:
            self._mongo_client.disconnect()
