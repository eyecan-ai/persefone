# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from persefone.interfaces.grpc import datasets_services_pb2 as persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2


class DatasetsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.DatasetsList = channel.unary_unary(
                '/persefone.DatasetsService/DatasetsList',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.FromString,
                )
        self.NewDataset = channel.unary_unary(
                '/persefone.DatasetsService/NewDataset',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.FromString,
                )
        self.DeleteDataset = channel.unary_unary(
                '/persefone.DatasetsService/DeleteDataset',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.FromString,
                )
        self.GetDataset = channel.unary_unary(
                '/persefone.DatasetsService/GetDataset',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.FromString,
                )
        self.GetSample = channel.unary_unary(
                '/persefone.DatasetsService/GetSample',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleResponse.FromString,
                )
        self.UpdateSample = channel.unary_unary(
                '/persefone.DatasetsService/UpdateSample',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleResponse.FromString,
                )
        self.NewSample = channel.unary_unary(
                '/persefone.DatasetsService/NewSample',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleResponse.FromString,
                )
        self.GetItem = channel.unary_unary(
                '/persefone.DatasetsService/GetItem',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemResponse.FromString,
                )
        self.NewItem = channel.unary_unary(
                '/persefone.DatasetsService/NewItem',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemResponse.FromString,
                )
        self.UpdateItem = channel.unary_unary(
                '/persefone.DatasetsService/UpdateItem',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemResponse.FromString,
                )


class DatasetsServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def DatasetsList(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def NewDataset(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteDataset(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDataset(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetSample(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateSample(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def NewSample(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetItem(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def NewItem(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateItem(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DatasetsServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'DatasetsList': grpc.unary_unary_rpc_method_handler(
                    servicer.DatasetsList,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.SerializeToString,
            ),
            'NewDataset': grpc.unary_unary_rpc_method_handler(
                    servicer.NewDataset,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.SerializeToString,
            ),
            'DeleteDataset': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteDataset,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.SerializeToString,
            ),
            'GetDataset': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDataset,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.SerializeToString,
            ),
            'GetSample': grpc.unary_unary_rpc_method_handler(
                    servicer.GetSample,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleResponse.SerializeToString,
            ),
            'UpdateSample': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateSample,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleResponse.SerializeToString,
            ),
            'NewSample': grpc.unary_unary_rpc_method_handler(
                    servicer.NewSample,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleResponse.SerializeToString,
            ),
            'GetItem': grpc.unary_unary_rpc_method_handler(
                    servicer.GetItem,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemResponse.SerializeToString,
            ),
            'NewItem': grpc.unary_unary_rpc_method_handler(
                    servicer.NewItem,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemResponse.SerializeToString,
            ),
            'UpdateItem': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateItem,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'persefone.DatasetsService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class DatasetsService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def DatasetsList(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.DatasetsService/DatasetsList',
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def NewDataset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.DatasetsService/NewDataset',
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteDataset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.DatasetsService/DeleteDataset',
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDataset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.DatasetsService/GetDataset',
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DDatasetResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetSample(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.DatasetsService/GetSample',
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateSample(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.DatasetsService/UpdateSample',
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def NewSample(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.DatasetsService/NewSample',
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DSampleResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetItem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.DatasetsService/GetItem',
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def NewItem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.DatasetsService/NewItem',
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateItem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.DatasetsService/UpdateItem',
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_datasets__services__pb2.DItemResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
