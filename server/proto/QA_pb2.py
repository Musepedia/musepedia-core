# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: QA.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x08QA.proto\"*\n\x0eRpcExhibitText\x12\n\n\x02id\x18\x01 \x01(\x03\x12\x0c\n\x04text\x18\x02 \x01(\t\"P\n\x0cHelloRequest\x12\x10\n\x08question\x18\x01 \x01(\t\x12\x1e\n\x05texts\x18\x02 \x03(\x0b\x32\x0f.RpcExhibitText\x12\x0e\n\x06status\x18\x03 \x01(\x05\"2\n\x10\x41nswerWithTextId\x12\x0e\n\x06\x61nswer\x18\x01 \x01(\t\x12\x0e\n\x06textId\x18\x02 \x01(\x03\"9\n\nHelloReply\x12+\n\x10\x61nswerWithTextId\x18\x01 \x01(\x0b\x32\x11.AnswerWithTextId\"M\n\x15QuestionWithExhibitId\x12\x14\n\x0c\x65xhibitLabel\x18\x01 \x01(\t\x12\x10\n\x08question\x18\x02 \x01(\t\x12\x0c\n\x04\x66req\x18\x03 \x01(\x03\"P\n\x17QuestionAnalysisRequest\x12\x35\n\x15questionWithExhibitId\x18\x01 \x03(\x0b\x32\x16.QuestionWithExhibitId\"j\n\x15QuestionAnalysisReply\x12\x11\n\tquestions\x18\x01 \x03(\t\x12\x14\n\x0cquestionFreq\x18\x02 \x03(\x03\x12\x15\n\rexhibitLabels\x18\x03 \x03(\t\x12\x11\n\tlabelFreq\x18\x04 \x03(\x03\"6\n\x0cUserQuestion\x12\x14\n\x0c\x65xhibitLabel\x18\x01 \x01(\t\x12\x10\n\x08question\x18\x02 \x01(\t\":\n\x13UserAnalysisRequest\x12#\n\x0cuserQuestion\x18\x01 \x03(\x0b\x32\r.UserQuestion\"*\n\x11UserAnalysisReply\x12\x15\n\rexhibitLabels\x18\x01 \x03(\t\"8\n\x13InfoAnalysisMessage\x12\x0b\n\x03\x61ge\x18\x01 \x01(\x03\x12\x14\n\x0c\x65xhibitLabel\x18\x02 \x01(\t\"H\n\x13InfoAnalysisRequest\x12\x31\n\x13infoAnalysisMessage\x18\x01 \x03(\x0b\x32\x14.InfoAnalysisMessage\"\"\n\tTopKLabel\x12\x15\n\rexhibitLabels\x18\x02 \x03(\t\"\x93\x01\n\x11InfoAnalysisReply\x12<\n\rageWithLabels\x18\x01 \x03(\x0b\x32%.InfoAnalysisReply.AgeWithLabelsEntry\x1a@\n\x12\x41geWithLabelsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x19\n\x05value\x18\x02 \x01(\x0b\x32\n.TopKLabel:\x02\x38\x01\x32\x35\n\tMyService\x12(\n\x08SayHello\x12\r.HelloRequest\x1a\x0b.HelloReply\"\x00\x62\x06proto3')



_RPCEXHIBITTEXT = DESCRIPTOR.message_types_by_name['RpcExhibitText']
_HELLOREQUEST = DESCRIPTOR.message_types_by_name['HelloRequest']
_ANSWERWITHTEXTID = DESCRIPTOR.message_types_by_name['AnswerWithTextId']
_HELLOREPLY = DESCRIPTOR.message_types_by_name['HelloReply']
_QUESTIONWITHEXHIBITID = DESCRIPTOR.message_types_by_name['QuestionWithExhibitId']
_QUESTIONANALYSISREQUEST = DESCRIPTOR.message_types_by_name['QuestionAnalysisRequest']
_QUESTIONANALYSISREPLY = DESCRIPTOR.message_types_by_name['QuestionAnalysisReply']
_USERQUESTION = DESCRIPTOR.message_types_by_name['UserQuestion']
_USERANALYSISREQUEST = DESCRIPTOR.message_types_by_name['UserAnalysisRequest']
_USERANALYSISREPLY = DESCRIPTOR.message_types_by_name['UserAnalysisReply']
_INFOANALYSISMESSAGE = DESCRIPTOR.message_types_by_name['InfoAnalysisMessage']
_INFOANALYSISREQUEST = DESCRIPTOR.message_types_by_name['InfoAnalysisRequest']
_TOPKLABEL = DESCRIPTOR.message_types_by_name['TopKLabel']
_INFOANALYSISREPLY = DESCRIPTOR.message_types_by_name['InfoAnalysisReply']
_INFOANALYSISREPLY_AGEWITHLABELSENTRY = _INFOANALYSISREPLY.nested_types_by_name['AgeWithLabelsEntry']
RpcExhibitText = _reflection.GeneratedProtocolMessageType('RpcExhibitText', (_message.Message,), {
  'DESCRIPTOR' : _RPCEXHIBITTEXT,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:RpcExhibitText)
  })
_sym_db.RegisterMessage(RpcExhibitText)

HelloRequest = _reflection.GeneratedProtocolMessageType('HelloRequest', (_message.Message,), {
  'DESCRIPTOR' : _HELLOREQUEST,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:HelloRequest)
  })
_sym_db.RegisterMessage(HelloRequest)

AnswerWithTextId = _reflection.GeneratedProtocolMessageType('AnswerWithTextId', (_message.Message,), {
  'DESCRIPTOR' : _ANSWERWITHTEXTID,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:AnswerWithTextId)
  })
_sym_db.RegisterMessage(AnswerWithTextId)

HelloReply = _reflection.GeneratedProtocolMessageType('HelloReply', (_message.Message,), {
  'DESCRIPTOR' : _HELLOREPLY,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:HelloReply)
  })
_sym_db.RegisterMessage(HelloReply)

QuestionWithExhibitId = _reflection.GeneratedProtocolMessageType('QuestionWithExhibitId', (_message.Message,), {
  'DESCRIPTOR' : _QUESTIONWITHEXHIBITID,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:QuestionWithExhibitId)
  })
_sym_db.RegisterMessage(QuestionWithExhibitId)

QuestionAnalysisRequest = _reflection.GeneratedProtocolMessageType('QuestionAnalysisRequest', (_message.Message,), {
  'DESCRIPTOR' : _QUESTIONANALYSISREQUEST,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:QuestionAnalysisRequest)
  })
_sym_db.RegisterMessage(QuestionAnalysisRequest)

QuestionAnalysisReply = _reflection.GeneratedProtocolMessageType('QuestionAnalysisReply', (_message.Message,), {
  'DESCRIPTOR' : _QUESTIONANALYSISREPLY,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:QuestionAnalysisReply)
  })
_sym_db.RegisterMessage(QuestionAnalysisReply)

UserQuestion = _reflection.GeneratedProtocolMessageType('UserQuestion', (_message.Message,), {
  'DESCRIPTOR' : _USERQUESTION,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:UserQuestion)
  })
_sym_db.RegisterMessage(UserQuestion)

UserAnalysisRequest = _reflection.GeneratedProtocolMessageType('UserAnalysisRequest', (_message.Message,), {
  'DESCRIPTOR' : _USERANALYSISREQUEST,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:UserAnalysisRequest)
  })
_sym_db.RegisterMessage(UserAnalysisRequest)

UserAnalysisReply = _reflection.GeneratedProtocolMessageType('UserAnalysisReply', (_message.Message,), {
  'DESCRIPTOR' : _USERANALYSISREPLY,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:UserAnalysisReply)
  })
_sym_db.RegisterMessage(UserAnalysisReply)

InfoAnalysisMessage = _reflection.GeneratedProtocolMessageType('InfoAnalysisMessage', (_message.Message,), {
  'DESCRIPTOR' : _INFOANALYSISMESSAGE,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:InfoAnalysisMessage)
  })
_sym_db.RegisterMessage(InfoAnalysisMessage)

InfoAnalysisRequest = _reflection.GeneratedProtocolMessageType('InfoAnalysisRequest', (_message.Message,), {
  'DESCRIPTOR' : _INFOANALYSISREQUEST,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:InfoAnalysisRequest)
  })
_sym_db.RegisterMessage(InfoAnalysisRequest)

TopKLabel = _reflection.GeneratedProtocolMessageType('TopKLabel', (_message.Message,), {
  'DESCRIPTOR' : _TOPKLABEL,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:TopKLabel)
  })
_sym_db.RegisterMessage(TopKLabel)

InfoAnalysisReply = _reflection.GeneratedProtocolMessageType('InfoAnalysisReply', (_message.Message,), {

  'AgeWithLabelsEntry' : _reflection.GeneratedProtocolMessageType('AgeWithLabelsEntry', (_message.Message,), {
    'DESCRIPTOR' : _INFOANALYSISREPLY_AGEWITHLABELSENTRY,
    '__module__' : 'QA_pb2'
    # @@protoc_insertion_point(class_scope:InfoAnalysisReply.AgeWithLabelsEntry)
    })
  ,
  'DESCRIPTOR' : _INFOANALYSISREPLY,
  '__module__' : 'QA_pb2'
  # @@protoc_insertion_point(class_scope:InfoAnalysisReply)
  })
_sym_db.RegisterMessage(InfoAnalysisReply)
_sym_db.RegisterMessage(InfoAnalysisReply.AgeWithLabelsEntry)

_MYSERVICE = DESCRIPTOR.services_by_name['MyService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _INFOANALYSISREPLY_AGEWITHLABELSENTRY._options = None
  _INFOANALYSISREPLY_AGEWITHLABELSENTRY._serialized_options = b'8\001'
  _RPCEXHIBITTEXT._serialized_start=12
  _RPCEXHIBITTEXT._serialized_end=54
  _HELLOREQUEST._serialized_start=56
  _HELLOREQUEST._serialized_end=136
  _ANSWERWITHTEXTID._serialized_start=138
  _ANSWERWITHTEXTID._serialized_end=188
  _HELLOREPLY._serialized_start=190
  _HELLOREPLY._serialized_end=247
  _QUESTIONWITHEXHIBITID._serialized_start=249
  _QUESTIONWITHEXHIBITID._serialized_end=326
  _QUESTIONANALYSISREQUEST._serialized_start=328
  _QUESTIONANALYSISREQUEST._serialized_end=408
  _QUESTIONANALYSISREPLY._serialized_start=410
  _QUESTIONANALYSISREPLY._serialized_end=516
  _USERQUESTION._serialized_start=518
  _USERQUESTION._serialized_end=572
  _USERANALYSISREQUEST._serialized_start=574
  _USERANALYSISREQUEST._serialized_end=632
  _USERANALYSISREPLY._serialized_start=634
  _USERANALYSISREPLY._serialized_end=676
  _INFOANALYSISMESSAGE._serialized_start=678
  _INFOANALYSISMESSAGE._serialized_end=734
  _INFOANALYSISREQUEST._serialized_start=736
  _INFOANALYSISREQUEST._serialized_end=808
  _TOPKLABEL._serialized_start=810
  _TOPKLABEL._serialized_end=844
  _INFOANALYSISREPLY._serialized_start=847
  _INFOANALYSISREPLY._serialized_end=994
  _INFOANALYSISREPLY_AGEWITHLABELSENTRY._serialized_start=930
  _INFOANALYSISREPLY_AGEWITHLABELSENTRY._serialized_end=994
  _MYSERVICE._serialized_start=996
  _MYSERVICE._serialized_end=1049
# @@protoc_insertion_point(module_scope)
