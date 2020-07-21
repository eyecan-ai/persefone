
from persefone.interfaces.grpc.tensor_services_pb2_grpc import SimpleTensorServiceStub
from persefone.interfaces.proto.data_pb2 import DTensorBundle
from persefone.interfaces.proto.utils.dtensor import DTensorUtils
import grpc
import json
from typing import List, Union, Tuple
import numpy as np


class TensorServiceClientCFG(object):
    DEFAULT_MAX_MESSAGE_LENGTH = -1

    def __init__(self):
        self.options = [
            ('grpc.max_send_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
        ]


class BaseTensorServiceClient(object):

    def __init__(self, host='localhost', port=50051, cfg=TensorServiceClientCFG()):
        if isinstance(port, str):
            port = int(port)
        self._channel = grpc.insecure_channel(f'{host}:{port}', options=cfg.options)
        self._stub = SimpleTensorServiceStub(self._channel)


class SimpleTensorServiceClient(BaseTensorServiceClient):

    def __init__(self, host='localhost', port=50051, cfg=TensorServiceClientCFG()):
        super(SimpleTensorServiceClient, self).__init__(host=host, port=port, cfg=cfg)

    def consume(self, bundle: DTensorBundle) -> DTensorBundle:
        """ GRPC -> Consume endpoint

        :param bundle: request protobuf DTensorBundle object
        :type bundle: DTensorBundle
        :return: protobuf DTensorBundle reply
        :rtype: DTensorBundle
        """

        return self._stub.Consume(bundle)


class MetaImagesServiceClient(BaseTensorServiceClient):

    def __init__(self, host='localhost', port=50051, cfg=TensorServiceClientCFG()):
        super(MetaImagesServiceClient, self).__init__(host=host, port=port, cfg=cfg)

    def create_bundle(self,
                      images: Union[List[np.ndarray], np.ndarray],
                      metadata: dict = {},
                      codecs: Union[str, List[str]] = None) -> DTensorBundle:
        """ Creates bundles from a list of images, their codecs and an action string

        :param images: list of np.ndarray images
        :type images: Union[List[np.ndarray], np.ndarray]
        :param metadata: attached metadata as dict, defaults to {}
        :type metadata: dict, optional
        :param codecs: images encoding/s, defaults to None
        :type codecs: Union[str, List[str]], optional
        :return: corresponding DTensorBundle object
        :rtype: DTensorBundle
        """

        if not isinstance(images, list):
            images = [images]

        action = json.dumps(metadata)
        if codecs is None or len(codecs) == 0:
            bundle = DTensorUtils.numpy_to_dtensor_bundle(images, action)
        else:
            bundle = DTensorUtils.images_to_dtensor_bundle(images, codecs, action)

        return bundle

    def send_bundle(self, bundle: DTensorBundle) -> Tuple[List[np.ndarray], str]:
        """ Sends DTensorBundle through client

        :param bundle: input bundle
        :type bundle: DTensorBundle
        :return: a list of reply images as np.ndarray with corresponding metadata
        :rtype: [type]
        """

        # GRPC call
        reply_bundle = self._stub.Consume(bundle)

        # Fetches data from reply bundle
        reply_arrays, reply_action = DTensorUtils.dtensor_bundle_to_numpy(reply_bundle)
        reply_metadata = json.loads(reply_action)

        return reply_arrays, reply_metadata

    def send_meta_images(self,
                         images: Union[List[np.ndarray], np.ndarray], metadata: dict = {},
                         codecs: Union[str, List[str]] = None) -> Tuple[List[np.ndarray], str]:
        """ Sends images with metadata to server. Metadata will be converted in a JSON string in order
        to be encapsulated in the action string field of the generic protobuf DTensorBundle message

        :param images: list of images or single images as np.ndarray
        :type images: Union[List[np.ndarray], np.ndarray]
        :param metadata: attached metadata, defaults to {}
        :type metadata: dict, optional
        :return: a list of reply images as np.ndarray with corresponding metadata
        :rtype: [type]
        """

        # create Bundle
        bundle = self.create_bundle(images, metadata, codecs)
        return self.send_bundle(bundle)
