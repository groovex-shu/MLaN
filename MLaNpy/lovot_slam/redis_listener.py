import queue
import threading
from logging import getLogger

logger = getLogger(__name__)


class RedisListener:
    def __init__(self, redis_cli, key):
        self.redis_cli = redis_cli
        self.pubsub = self.redis_cli.pubsub()
        self.pubsub.subscribe([key])

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run)

        self._queue = queue.Queue()

    def start(self):
        self.thread.start()

    def stop(self):
        logger.debug("stopping listener thread")
        self.stop_event.set()
        self.pubsub.unsubscribe()
        self.thread.join()

    def _run(self):
        while not self.stop_event.is_set():
            for item in self.pubsub.listen():
                # logger.debug(f'received: {item}')
                self._queue.put(item)
        logger.debug("listener thread terminated")

    def empty(self):
        return self._queue.empty()

    def get(self):
        return self._queue.get()
