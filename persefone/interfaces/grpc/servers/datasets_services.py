from persefone.interfaces.grpc.datasets_services_pb2_grpc import DatasetsServiceServicer, add_DatasetsServiceServicer_to_server
from persefone.interfaces.grpc.datasets_services_pb2 import (
    DDatasetRequest, DDatasetResponse,
    DSampleRequest, DSampleResponse,
    DItemRequest, DItemResponse
)

import threading
import grpc
from concurrent import futures
from typing import Callable
import json
from abc import ABC, abstractmethod


class DatasetsServiceServerCFG(object):
    DEFAULT_MAX_MESSAGE_LENGTH = -1

    def __init__(self):
        self.options = [
            ('grpc.max_send_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', self.DEFAULT_MAX_MESSAGE_LENGTH),
        ]


class DatasetsServiceServer(ABC, DatasetsServiceServicer):

    def __init__(self, host='0.0.0.0', port=50051, max_workers=10, use_threads=False, options: DatasetsServiceServerCFG = DatasetsServiceServerCFG().options):

        self._host = host
        self._port = port
        self._options = options
        self._max_workers = max_workers
        self._server = None
        self._server_thread = None
        self._use_threads = use_threads
        self._started = False
        self._failed_connection = False

    @property
    def active(self) -> bool:
        """
        :return: Is Server active?
        :rtype: bool
        """

        return self._started

    def start(self):
        """ Starts server thread
        """

        if self._use_threads:
            self._server_thread = threading.Thread(target=self._serve, daemon=True)
            self._server_thread.start()
        else:
            self._serve()

    def wait_for_termination(self):
        """ Waits for main thread termination
        """

        if self._server_thread is not None:
            self._server_thread.join()

    def stop(self):
        """ Force stop
        """

        if self._server is not None:
            self._server.stop(0)
            self._started = False

    def _serve(self):
        """ Serve gRPC internal server """

        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=self._max_workers), options=self._options)
        add_DatasetsServiceServicer_to_server(self, self._server)
        port = self._server.add_insecure_port(f'{self._host}:{self._port}')
        if port == 0:
            raise Exception("GRPC Port is not valid!")
            self._failed_connection = True
            return False
        self._server.start()
        self._started = True
        self._server.wait_for_termination()
        return True

    @abstractmethod
    def DatasetsList(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:
        pass

    @abstractmethod
    def GetDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:
        pass

    @abstractmethod
    def DeleteDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:
        pass

    @abstractmethod
    def NewDataset(self, request: DDatasetRequest, context: grpc.ServicerContext) -> DDatasetResponse:
        pass

    @abstractmethod
    def GetSample(self, request: DSampleRequest, context: grpc.ServicerContext) -> DSampleResponse:
        pass

    @abstractmethod
    def NewSample(self, request: DSampleRequest, context: grpc.ServicerContext) -> DSampleResponse:
        pass

    @abstractmethod
    def GetItem(self, request: DItemRequest, context: grpc.ServicerContext) -> DItemResponse:
        pass

    @abstractmethod
    def NewItem(self, request: DItemRequest, context: grpc.ServicerContext) -> DItemResponse:
        pass
