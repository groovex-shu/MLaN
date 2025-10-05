import json
import uuid

from websocket import create_connection, WebSocketTimeoutException


class SimpleLovotalkClient:
    def __init__(self, hostname):
        self.url = "ws://" + hostname + ":38001"
        self.ws = create_connection(self.url)
        self.ws.settimeout(2)

    def keys(self, database, pattern):
        qid = str(uuid.uuid4())
        query = f'?{database},{qid},KEYS,{pattern};'
        self.ws.send(query)
        while True:
            try:
                # ?STM,qid,KEYS,key1,key2,...;
                response = self.ws.recv().strip(';')
                response = response.split(',')
                if response[1] == qid:
                    return response[3:]
            except WebSocketTimeoutException:
                return

    def delete(self, database, key):
        query = f'!{database},DEL,{key};'
        self.ws.send(query)

    def mset(self, database, key, value):
        return self._query(f'!{database},MSET', {"items": {key: value}})

    def mget(self, database, keys):
        result = self._query_and_receive(f'?{database},MGET', {"keys": keys})
        return result

    def publish(self, database, channel, value):
        return self._query(f'!{database},PUBLISH', {"channel": channel, "value": value})

    def hmset(self, database, key, items):
        return self._query(f'!{database},HMSET', {"key": key, "items": items})

    def hgetall(self, database, keys):
        result = self._query_and_receive(f'?{database},HGETALL', {"keys": keys})
        return result

    def _query(self, cmd, attrs):
        qid = str(uuid.uuid4())
        query = {
            "cmd": cmd,
            "qid": qid,
            "attrs": attrs
        }
        self.ws.send(json.dumps(query))

    def _query_and_receive(self, cmd, attrs):
        qid = str(uuid.uuid4())
        query = {
            "cmd": cmd,
            "qid": qid,
            "attrs": attrs
        }
        self.ws.send(json.dumps(query))
        while True:
            try:
                response = json.loads(self.ws.recv())
                if "qid" in response and response["qid"] == qid:
                    if "result" in response:
                        return response["result"]
                    return
            except WebSocketTimeoutException:
                return
