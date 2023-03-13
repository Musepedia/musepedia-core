# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from src.rpc.proto import ES_pb2 as src_dot_rpc_dot_proto_dot_ES__pb2


class ESServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetOpenDocument = channel.unary_unary(
                '/ESService/GetOpenDocument',
                request_serializer=src_dot_rpc_dot_proto_dot_ES__pb2.OpenDocumentRequest.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )


class ESServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetOpenDocument(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ESServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetOpenDocument': grpc.unary_unary_rpc_method_handler(
                    servicer.GetOpenDocument,
                    request_deserializer=src_dot_rpc_dot_proto_dot_ES__pb2.OpenDocumentRequest.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ESService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ESService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetOpenDocument(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ESService/GetOpenDocument',
            src_dot_rpc_dot_proto_dot_ES__pb2.OpenDocumentRequest.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
