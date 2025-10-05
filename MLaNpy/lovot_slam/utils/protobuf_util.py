from google.protobuf import timestamp_pb2


def unix_time_to_pb_timestamp(unix_time: float) -> timestamp_pb2.Timestamp:
    seconds = int(unix_time)
    nanos = int((unix_time - seconds) * 1e9)
    return timestamp_pb2.Timestamp(seconds=seconds, nanos=nanos)
