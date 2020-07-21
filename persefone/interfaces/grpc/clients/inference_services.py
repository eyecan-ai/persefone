from persefone.interfaces.grpc.inference_services_pb2_grpc import InferenceServiceStub
from persefone.interfaces.grpc.inference_services_pb2 import (
    DModelRequest, DModelResponse, DInferenceRequest, DInferenceResponse
)
from persefone.interfaces.proto.utils.comm import ResponseStatusUtils
from persefone.interfaces.proto.utils.dtensor import DTensorUtils
import grpc
from typing import List, Tuple
import logging
import numpy as np
import json


class InferenceServiceClientCFG(object):
    DEFAULT_MAX_MESSAGE_LENGTH = -1

    def __init__(self):
        self.options = [
            ('grpc.max_send_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
        ]


class InferenceServiceClient(object):

    def __init__(self, host='localhost', port=50051, cfg=InferenceServiceClientCFG()):
        if isinstance(port, str):
            port = int(port)
        self._channel = grpc.insecure_channel(f'{host}:{port}', options=cfg.options)
        self._stub = InferenceServiceStub(self._channel)

    def ModelsList(self, request: DModelRequest) -> DModelResponse:
        return self._stub.ModelsList(request)

    def ActivateModel(self, request: DModelRequest) -> DModelResponse:
        return self._stub.ActivateModel(request)

    def DeactivateModel(self, request: DModelRequest) -> DModelResponse:
        return self._stub.DeactivateModel(request)

    def Inference(self, request: DInferenceRequest) -> DInferenceResponse:
        return self._stub.Inference(request)


class InferenceSimpleServiceClient(InferenceServiceClient):

    def __init__(self, host='localhost', port=50051, cfg=InferenceServiceClientCFG()):
        super(InferenceSimpleServiceClient, self).__init__(host=host, port=port, cfg=cfg)

    def models_list(self, category_name: str = '') -> List[str]:
        """ Retrives list of available models names

        :param category_name: category name for query (can be empty), defaults to ''
        :type category_name: str, optional
        :return: list of available models names
        :rtype: List[str]
        """

        request = DModelRequest()
        request.model_category = category_name

        response = self.ModelsList(request=request)
        results = []
        for model in response.models:
            results.append(model.name)

        return results

    def activate_model(self, model_name: str) -> bool:

        request = DModelRequest()
        request.model_name = model_name

        response = self.ActivateModel(request=request)
        if response.status.code == ResponseStatusUtils.STATUS_CODE_OK:
            return True
        else:
            logging.error(response.status.message)
            raise SystemError(response.status.message)

    def deactivate_model(self, model_name: str) -> bool:

        request = DModelRequest()
        request.model_name = model_name

        response = self.DeactivateModel(request=request)
        if response.status.code == ResponseStatusUtils.STATUS_CODE_OK:
            return True
        else:
            logging.error(response.status.message)
            raise SystemError(response.status.message)

    def inference(self, arrays: List[np.ndarray], action: str) -> Tuple[List[np.ndarray], str]:

        request = DInferenceRequest()
        request.bundle.CopyFrom(DTensorUtils.numpy_to_dtensor_bundle(arrays, action))

        response = self.Inference(request)
        if response.status.code == ResponseStatusUtils.STATUS_CODE_OK:
            return DTensorUtils.dtensor_bundle_to_numpy(response.bundle)
        else:
            logging.error(response.status.message)
            raise SystemError(response.status.message)

    def inference_with_metadata(self, arrays: List[np.ndarray], metadata: dict) -> Tuple[List[np.ndarray], dict]:

        request = DInferenceRequest()
        action = json.dumps(metadata)
        request.bundle.CopyFrom(DTensorUtils.numpy_to_dtensor_bundle(arrays, action))

        response = self.Inference(request)
        if response.status.code == ResponseStatusUtils.STATUS_CODE_OK:
            reply_arrays, reply_action = DTensorUtils.dtensor_bundle_to_numpy(response.bundle)
            reply_metadata = json.loads(reply_action)
            return reply_arrays, reply_metadata
        else:
            logging.error(response.status.message)
            raise SystemError(response.status.message)
