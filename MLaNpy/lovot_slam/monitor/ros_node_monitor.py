from logging import getLogger

import json
import threading
from functools import partial

import numpy as np
import trio
from prometheus_client import (Counter,
                               Histogram,
                               start_http_server)

from lovot_slam.redis import create_stm_client

logger = getLogger(__name__)

class RosNodeMarkerLocalizationMonitor:
    """Monitor ROS node status and push the result to prometheus.
    """
    MARKER_ACTION_CHANNEL = 'slam:marker_node_action'

    def __init__(self) -> None:
        self._stm_client = create_stm_client()

        self.prometheus_detection_count = Counter('localization_marker_detection_count',
                                                  'Count of detected marker',
                                                  labelnames=['marker_id'])
        
        # 0.25: minimum distance
        # 1.0: threshold distance
        # 1.75: working distance
        # 2.0: maximum distance
        self.prometheus_detection_distance_histogram = Histogram('localization_marker_detection_distance_histogram',
                                                                'Distance of detected marker histogram',
                                                                labelnames=['marker_id'],
                                                                buckets=[0.25, 1.0, 1.75, 2.0])
        
        self.prometheus_registration_count = Counter('localization_marker_registration_count',
                                                     'Count of registered marker',
                                                     labelnames=['marker_id'])
        
        self.prometheus_relocalization_count = Counter('localization_marker_relocalization_count',
                                                       'Count of relocalizate marker',
                                                       labelnames=['marker_id'])

    async def listen_to_redis(self):
        """Listen to Redis messages asynchronously."""
        pubsub = self._stm_client.pubsub()
        pubsub.subscribe(self.MARKER_ACTION_CHANNEL)

        while True:
            # Make get_message block until a message is received by passing a large timeout
            # NOTE: This is a workaround for the fact that get_message is non-blocking
            # and will return None if no message is available, which causes a busy loop.
            # The argument cancellable=True is used to allow the task to be cancelled
            # even if it's waiting for a message.
            message = await trio.to_thread.run_sync(partial(pubsub.get_message, timeout=1000.0),
                                                    cancellable=True)
            if message and message["type"] == "message":
                self.handle_marker_action(message["data"])

    def handle_marker_action(self, msg:str):
        """Process received marker action messages."""
        msg = json.loads(msg)
        marker_id = msg.get("marker_id", "unknown")
        action = msg.get("action", "unknown")

        if action == "DETECT":
            self.prometheus_detection_count.labels(marker_id).inc()
            marker_pose = msg.get("marker_pose_in_camera", None)
            if marker_pose:
                marker_pose = np.array(marker_pose)
            
                distance = np.linalg.norm(marker_pose[:3, -1])
                logger.debug(f"Distance of detected marker: {distance}")

                self.prometheus_detection_distance_histogram.labels(marker_id).observe(distance)

        elif action == "REGISTER":
            self.prometheus_registration_count.labels(marker_id).inc()
        elif action == "RELOCALIZATION":
            self.prometheus_relocalization_count.labels(marker_id).inc()
        else:
            assert False, f"Unknown action: {action}"

    async def run(self):
        await self.listen_to_redis()


def start_prometheus_server(port=8000):
    """Start Prometheus HTTP server in a separate thread."""
    thread = threading.Thread(target=start_http_server, args=(port,), daemon=True)
    thread.start()
    print(f"Prometheus metrics available at: http://localhost:{port}/")


if __name__ == "__main__":
    # init service
    start_prometheus_server(port=8999)

    # create monitor
    monitor = RosNodeMarkerLocalizationMonitor()
    trio.run(monitor.run)