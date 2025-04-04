# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/rpc/proto/ES.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='src/rpc/proto/ES.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x16src/rpc/proto/ES.proto\x1a\x1bgoogle/protobuf/empty.proto\"3\n\x13OpenDocumentRequest\x12\r\n\x05label\x18\x01 \x01(\t\x12\r\n\x05texts\x18\x02 \x03(\t2N\n\tESService\x12\x41\n\x0fGetOpenDocument\x12\x14.OpenDocumentRequest\x1a\x16.google.protobuf.Empty\"\x00\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,])




_OPENDOCUMENTREQUEST = _descriptor.Descriptor(
  name='OpenDocumentRequest',
  full_name='OpenDocumentRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='label', full_name='OpenDocumentRequest.label', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='texts', full_name='OpenDocumentRequest.texts', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=55,
  serialized_end=106,
)

DESCRIPTOR.message_types_by_name['OpenDocumentRequest'] = _OPENDOCUMENTREQUEST
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

OpenDocumentRequest = _reflection.GeneratedProtocolMessageType('OpenDocumentRequest', (_message.Message,), {
  'DESCRIPTOR' : _OPENDOCUMENTREQUEST,
  '__module__' : 'src.rpc.proto.ES_pb2'
  # @@protoc_insertion_point(class_scope:OpenDocumentRequest)
  })
_sym_db.RegisterMessage(OpenDocumentRequest)



_ESSERVICE = _descriptor.ServiceDescriptor(
  name='ESService',
  full_name='ESService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=108,
  serialized_end=186,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetOpenDocument',
    full_name='ESService.GetOpenDocument',
    index=0,
    containing_service=None,
    input_type=_OPENDOCUMENTREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_ESSERVICE)

DESCRIPTOR.services_by_name['ESService'] = _ESSERVICE

# @@protoc_insertion_point(module_scope)
