# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from persefone.interfaces.grpc import inference_services_pb2 as persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2


class InferenceServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.TrainableModels = channel.unary_unary(
                '/persefone.InferenceService/TrainableModels',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelCategoryRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelCategoryResponse.FromString,
                )
        self.ModelsList = channel.unary_unary(
                '/persefone.InferenceService/ModelsList',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelResponse.FromString,
                )
        self.ActivateModel = channel.unary_unary(
                '/persefone.InferenceService/ActivateModel',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelResponse.FromString,
                )
        self.DeactivateModel = channel.unary_unary(
                '/persefone.InferenceService/DeactivateModel',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelResponse.FromString,
                )
        self.Inference = channel.unary_unary(
                '/persefone.InferenceService/Inference',
                request_serializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DInferenceRequest.SerializeToString,
                response_deserializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DInferenceResponse.FromString,
                )


class InferenceServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def TrainableModels(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ModelsList(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ActivateModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeactivateModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Inference(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'TrainableModels': grpc.unary_unary_rpc_method_handler(
                    servicer.TrainableModels,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelCategoryRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelCategoryResponse.SerializeToString,
            ),
            'ModelsList': grpc.unary_unary_rpc_method_handler(
                    servicer.ModelsList,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelResponse.SerializeToString,
            ),
            'ActivateModel': grpc.unary_unary_rpc_method_handler(
                    servicer.ActivateModel,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelResponse.SerializeToString,
            ),
            'DeactivateModel': grpc.unary_unary_rpc_method_handler(
                    servicer.DeactivateModel,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelResponse.SerializeToString,
            ),
            'Inference': grpc.unary_unary_rpc_method_handler(
                    servicer.Inference,
                    request_deserializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DInferenceRequest.FromString,
                    response_serializer=persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DInferenceResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'persefone.InferenceService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class InferenceService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def TrainableModels(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.InferenceService/TrainableModels',
            persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelCategoryRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelCategoryResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ModelsList(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.InferenceService/ModelsList',
            persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ActivateModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.InferenceService/ActivateModel',
            persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeactivateModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.InferenceService/DeactivateModel',
            persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DModelResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Inference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/persefone.InferenceService/Inference',
            persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DInferenceRequest.SerializeToString,
            persefone_dot_interfaces_dot_grpc_dot_inference__services__pb2.DInferenceResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
