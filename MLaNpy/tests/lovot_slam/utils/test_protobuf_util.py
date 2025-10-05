from google.protobuf import timestamp_pb2

from lovot_slam.utils.protobuf_util import unix_time_to_pb_timestamp


def test_unix_time_to_pb_timestamp():
    stamp_pb = unix_time_to_pb_timestamp(0.0)
    assert stamp_pb == timestamp_pb2.Timestamp(seconds=0, nanos=0)

    stamp_pb = unix_time_to_pb_timestamp(1.0)
    assert stamp_pb == timestamp_pb2.Timestamp(seconds=1, nanos=0)

    stamp_pb = unix_time_to_pb_timestamp(1.1)
    assert stamp_pb == timestamp_pb2.Timestamp(seconds=1, nanos=100000000)
