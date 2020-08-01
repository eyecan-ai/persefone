from persefone.interfaces.grpc.clients.deep_services import DeepServicePack
from typing import Callable
from schema import Schema
from persefone.interfaces.proto.utils.comm import ResponseStatusUtils
from persefone.interfaces.grpc.deep_services_pb2_grpc import DeepServiceServicer, add_DeepServiceServicer_to_server
from persefone.interfaces.grpc.deep_services_pb2 import (
    DDeepServiceRequest, DDeepServiceResponse
)
import grpc
from abc import ABC, abstractmethod


class DeepService(ABC, DeepServiceServicer):

    def __init__(self):
        pass

    def register(self, grpc_server):
        add_DeepServiceServicer_to_server(self, grpc_server)

    @abstractmethod
    def DeepServe(self, request: DDeepServiceRequest, context: grpc.ServicerContext) -> DDeepServiceResponse:
        pass


class EchoDeepService(ABC, DeepServiceServicer):

    def __init__(self):
        pass

    def register(self, grpc_server):
        add_DeepServiceServicer_to_server(self, grpc_server)

    def DeepServe(self, request: DDeepServiceRequest, context: grpc.ServicerContext) -> DDeepServiceResponse:
        response = DDeepServiceResponse()
        response.status.CopyFrom(ResponseStatusUtils.create_ok_status("ok!"))
        response.metadata.CopyFrom(request.metadata)
        response.bundle.CopyFrom(request.bundle)
        return response


class EndpointDeepService(ABC, DeepServiceServicer):

    def __init__(self):
        self._callbacks_map = {}
        self._schema_map = {}
        self._output_schema_map = {}

    def register_endpoint(self,
                          name: str,
                          schema: Schema,
                          callback: Callable[[DeepServicePack], DeepServicePack],
                          output_schema: Schema = None
                          ):

        self._callbacks_map[name] = callback
        self._schema_map[name] = schema
        if output_schema is not None:
            self._output_schema_map[name] = output_schema

    def register(self, grpc_server):
        add_DeepServiceServicer_to_server(self, grpc_server)

    def _error_response(self, message: str):
        response = DDeepServiceResponse()
        response.status.CopyFrom(ResponseStatusUtils.create_error_status(message))
        return response

    def DeepServe(self, request: DDeepServiceRequest, context: grpc.ServicerContext) -> DDeepServiceResponse:

        # Pack from request
        try:
            pack = DeepServicePack.from_deep_service_request(request)
        except Exception as e:
            return self._error_response(str(e))

        # Iterate available schema
        for name, schema in self._schema_map.items():
            # If schema is valid, invoke corresponding callback if any
            if schema.is_valid(pack.metadata):

                # Callback found
                if name in self._callbacks_map:

                    reply_pack = self._callbacks_map[name](pack)

                    # Validate output metadata before send
                    if name in self._output_schema_map:
                        # try:
                        #     print(self._output_schema_map[name].validate(reply_pack.metadata))
                        # except Exception as e:
                        #     return self._error_response(str(e))
                        if not self._output_schema_map[name].is_valid(reply_pack.metadata):
                            return self._error_response(f'Server produces an invalid response for [{name}]')

                    return reply_pack.to_response()

                # Callback not found
                else:
                    return self._error_response(f"Service logic ['{name}'] is not implemented yet!")

        # No valid schema found!
        return self._error_response(f"Invalid input metadata for [{pack.metadata}]")
