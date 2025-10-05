from contextvars import ContextVar
from typing import TYPE_CHECKING

from attr import attrs

# import on type check only to work around circular dependencies
if TYPE_CHECKING:
    from lovot_slam.client.localization_client import LocalizationClient
    from lovot_slam.client.lovot_tf_client import LovotTfClient
    from lovot_slam.client.wifi_service_client import WifiServiceClient
    from lovot_slam.spike_client import SlamServicerClient
    from lovot_slam.wifi.mapping.mapping import RadioMap
    from lovot_slam.wifi.updater import FingerprintSync, WiFiScan


@attrs(auto_attribs=True)
class Context:
    slam_servicer_client: 'SlamServicerClient'
    wifi_client: 'WifiServiceClient'
    lovot_tf_client: 'LovotTfClient'
    localization_client: 'LocalizationClient'
    fingerprint_sync: 'FingerprintSync'
    wifi_scan: 'WiFiScan'
    radio_map: 'RadioMap'


context: ContextVar[Context] = ContextVar('lovot-localization-context')


class ContextMixin:
    """Mixin which provides context property to a class"""

    @property
    def context(self) -> Context:
        return context.get()
