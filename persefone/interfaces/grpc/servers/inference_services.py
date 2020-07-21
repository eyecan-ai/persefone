
from persefone.data.io.drivers.common import AbstractFileDriver
from persefone.data.databases.mongo.model import MModel
from persefone.data.databases.mongo.clients import MongoModelsManager, MongoDatabaseClient
from persefone.interfaces.proto.utils.comm import ResponseStatusUtils
from persefone.interfaces.grpc.inference_services_pb2_grpc import InferenceServiceServicer, add_InferenceServiceServicer_to_server
from persefone.interfaces.proto.models_pb2 import (
    DModel
)
from persefone.interfaces.grpc.inference_services_pb2 import (
    DModelRequest, DModelResponse, DInferenceRequest, DInferenceResponse
)
from persefone.interfaces.proto.utils.dtensor import DTensorUtils
from typing import Callable
import grpc
from abc import ABC, abstractmethod


class InferenceServiceCFG(object):
    DEFAULT_MAX_MESSAGE_LENGTH = -1

    def __init__(self):
        self.options = [
            ('grpc.max_send_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
        ]


class InferenceService(ABC, InferenceServiceServicer):

    def __init__(self):
        pass

    def register(self, grpc_server):
        add_InferenceServiceServicer_to_server(self, grpc_server)

    @abstractmethod
    def ModelsList(self, request: DModelRequest, context: grpc.ServicerContext) -> DModelResponse:
        pass

    @abstractmethod
    def ActivateModel(self, request: DModelRequest, context: grpc.ServicerContext) -> DModelResponse:
        pass

    @abstractmethod
    def DeactivateModel(self, request: DModelRequest, context: grpc.ServicerContext) -> DModelResponse:
        pass

    @abstractmethod
    def Inference(self, request: DInferenceRequest, context: grpc.ServicerContext) -> DInferenceResponse:
        pass


class MongoInferenceService(InferenceService):

    def __init__(self,
                 mongo_client: MongoDatabaseClient,
                 driver: AbstractFileDriver,
                 lifecycle_callbacks: Callable = None,
                 inference_callback: Callable = None
                 ):
        super(MongoInferenceService, self).__init__()

        self._mongo_client = mongo_client
        assert isinstance(driver, AbstractFileDriver), f"Invalid driver: {driver}"
        self._drivers = [driver]
        self._lifecycle_callback = lifecycle_callbacks
        self._inference_callback = inference_callback

    def set_lifecycle_callback(self, cb: Callable):
        self._lifecycle_callback = cb

    def set_inference_callback(self, cb: Callable):
        self._inference_callback = cb

    def _get_models(self, request: DModelRequest):
        model_category = request.model_category
        return MongoModelsManager(self._mongo_client).get_models(model_category=model_category)

    def _get_model(self, request: DModelRequest):
        model_name = request.model_name
        return MongoModelsManager(self._mongo_client).get_model(name=model_name)

    def _create_dmodel(self, model: MModel) -> DModel:
        dmodel = DModel()
        dmodel.name = model.name
        dmodel.category = model.category.name
        return dmodel

    def ModelsList(self, request: DModelRequest, context: grpc.ServicerContext) -> DModelResponse:

        # Fetches list of models s if any
        models = self._get_models(request)

        # Inits Response
        response = DModelResponse()

        # Build response status
        response.status.CopyFrom(ResponseStatusUtils.create_ok_status("all right!"))

        # Fills response with DDataset s if any
        for model in models:
            response.models.append(self._create_dmodel(model))

        return response

    def ActivateModel(self, request, context):

        # Fetches model s if any
        model = self._get_model(request)

        # Inits Response
        response = DModelResponse()

        if model is not None:

            callback_response = False
            if self._lifecycle_callback is not None:
                callback_response = self._lifecycle_callback(model.name)

            if callback_response:
                response.status.CopyFrom(ResponseStatusUtils.create_ok_status(f"Model [{request.model_name}] active!"))
            else:
                response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Model [{request.model_name}] activation failed!"))

        else:
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Model [{request.model_name}] not found!"))
        return response

    def DeactivateModel(self, request, context):

        # Fetches model s if any
        model = self._get_model(request)

        # Inits Response
        response = DModelResponse()

        if model is not None:

            callback_response = False
            if self._lifecycle_callback is not None:
                callback_response = self._lifecycle_callback(model.name)

            if callback_response:
                response.status.CopyFrom(ResponseStatusUtils.create_ok_status(f"Model [{request.model_name}] deactivated!"))
            else:
                response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Model [{request.model_name}] deactivation failed!"))

        else:
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Model [{request.model_name}] not found!"))
        return response

    def Inference(self, request: DInferenceRequest, context: grpc.ServicerContext) -> DInferenceResponse:

        arrays, action = DTensorUtils.dtensor_bundle_to_numpy(request.bundle)

        # Inits Response
        response = DInferenceResponse()

        if self._inference_callback is None:
            response.status.CopyFrom(ResponseStatusUtils.create_error_status(f"Inference gone wrong, no callback implemented!"))
        else:
            reply_arrays, reply_action = self._inference_callback(arrays, action)
            reply_bundle = DTensorUtils.numpy_to_dtensor_bundle(reply_arrays, reply_action)
            response.status.CopyFrom(ResponseStatusUtils.create_ok_status(f""))
            response.bundle.CopyFrom(reply_bundle)

        return response
