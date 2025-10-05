"""
Define redis key names used in localizer and builder.

This module provides redis key names used in both localizer and builder.
Some keys that are used only in either localizer or builder are defined in each module.
"""
import os

# Keys provided by other services
# STM
PHYSICAL_STATE_KEY = "neodm:physical_state"
INTENTION_KEY = "neodm:intention"

# LTM
COLONY_ID_KEY = 'config:colony_id'
GHOST_ID_KEY = 'config:ghost_id'
COLONY_LOVOTS_KEY = 'colony:lovots'

# DEVICE
ROBOT_MODEL_KEY = 'robot:model'
DYNAMIC_VARIANT_TRACKING_MODULE_MODEL_KEY = 'robot:variant:tracking_module_model'

# Keys provided by this service
class RedisKeyRepository:
    """Redis key name repository
    these keys are commonly used in localizer and builder,
    and are synchronized via gRPC communication.

    On coro2 both localizer and builder are running on the same machine,
    so we need to distinguish keys used in each process.
    - on coro1 (tom/spike): prefix is "slam"
    - on coro2 localizer  : prefix is "slam"
    - on coro2 builder    : prefix is "slam:builder"
    The prefix is defined in environment variable `LOCALIZATION_REDIS_KEY_PREFIX`.
    """
    KEY_PREFIX = os.getenv('LOCALIZATION_REDIS_KEY_PREFIX', 'slam')

    def __init__(self) -> None:
        pass

    @property
    def prefix(self) -> str:
        return self.KEY_PREFIX

    @property
    def command(self) -> str:
        return f'{self.prefix}:command'

    @property
    def response(self) -> str:
        return f'{self.prefix}:response'

    @property
    def state(self) -> str:
        return f'{self.prefix}:state'

    @property
    def map(self) -> str:
        return f'{self.prefix}:map'

    def spot(self, spot_id: str) -> str:
        return f'{self.prefix}:spot:{spot_id}'

    @property
    def unwelcomed_area(self) -> str:
        return f'{self.prefix}:unwelcomed_area'

    @property
    def unwelcomed_area_hash(self) -> str:
        return f'{self.prefix}:unwelcomed_area_hash'

    @property
    def is_busy(self) -> str:
        return f'{self.prefix}:is_busy'


