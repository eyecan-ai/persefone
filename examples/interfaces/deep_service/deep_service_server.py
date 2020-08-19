from abc import ABC
from concurrent import futures

import numpy
from persefone.interfaces.grpc.clients.deep_services import DeepServiceCFG, DeepServicePack
from typing import Callable, Dict
from persefone.interfaces.proto.utils.comm import ResponseStatusUtils
import grpc
from persefone.interfaces.grpc.deep_services_pb2 import DDeepServiceRequest, DDeepServiceResponse
from persefone.interfaces.grpc.deep_services_pb2_grpc import DeepServiceServicer, add_DeepServiceServicer_to_server
from schema import Optional, Schema


class CallbackableDeepService(ABC, DeepServiceServicer):

    def __init__(self, proxy_schema_map: Dict[str, Schema]):
        self._callbacks_map = {}
        self._proxy_schema_map = proxy_schema_map

    def register_callback(self, name: str, callback: Callable[[DeepServicePack], DeepServicePack]):
        if name not in self._proxy_schema_map:
            raise NotImplementedError(f'No schema validator for "{name}"')
        self._callbacks_map[name] = callback

    def register(self, grpc_server):
        add_DeepServiceServicer_to_server(self, grpc_server)

    def DeepServe(self, request: DDeepServiceRequest, context: grpc.ServicerContext) -> DDeepServiceResponse:

        # Pack from request
        try:
            pack = DeepServicePack.from_deep_service_request(request)
            # if len(pack.arrays) > 0:
            #     import cv2
            #     cv2.imshow("image", pack.arrays[0])
            #     cv2.waitKey(0)
        except Exception as e:
            # No valid schema found!
            response = DDeepServiceResponse()
            response.status.CopyFrom(
                ResponseStatusUtils.create_error_status(str(e))
            )
            return response

        # Iterate available schema
        for name, schema in self._proxy_schema_map.items():

            # If schema is valid, invoke corresponding callback if any
            if schema.is_valid(pack.metadata):

                # Callback found
                if name in self._callbacks_map:
                    return self._callbacks_map[name](pack).to_response()

                # Callback not found
                else:
                    response = DDeepServiceResponse()
                    response.status.CopyFrom(
                        ResponseStatusUtils.create_error_status(f"Service logic ['{name}'] is not implemented yet!")
                    )
                    return response

        # No valid schema found!
        response = DDeepServiceResponse()
        response.status.CopyFrom(
            ResponseStatusUtils.create_error_status(f"Invalid metadata!")
        )
        return response


def action_callback(pack: DeepServicePack) -> DeepServicePack:
    reply_pack = DeepServicePack()
    reply_pack.metadata = {'response_for': pack.metadata['command']}
    return reply_pack


def ping_callback(pack: DeepServicePack) -> DeepServicePack:
    reply_pack = DeepServicePack()
    reply_pack.metadata = pack.metadata
    reply_pack.metadata['_pong'] = True
    for i in range(len(pack.arrays)):
        reply_pack.arrays.append(numpy.rot90(pack.arrays[i]))
    return reply_pack


# Create Callbackable service with Schemas map
service = CallbackableDeepService(proxy_schema_map={
    # Action schema
    'action': Schema({
        '_schema': 'action',
        'command': str,
        Optional('params'): dict
    }),

    # Ping schema
    'ping': Schema({
        '_schema': 'ping',
        Optional('time'): float
    })
})

# Register callbacks for each schema
service.register_callback('action', action_callback)
service.register_callback('ping', ping_callback)

# Create server
host = 'localhost'
port = 50051
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=DeepServiceCFG().options)
server.add_insecure_port(f'{host}:{port}')
service.register(server)
server.start()
server.wait_for_termination()
