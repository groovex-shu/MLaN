"""Convert wifi_fingerprint(.pb) to pandas DataFrame pickle file.

DataFrame columns:
    'stamp': fingerprint.transform.stamp,
    'px': fingerprint.transform.transform.translation.x,
    'py': fingerprint.transform.transform.translation.y,
    'yaw': 0,
    'localizer': fingerprint.transform.localizer.value,
    'reliability': fingerprint.transform.reliability.reliability,
    'detection': fingerprint.transform.reliability.detection,
    'likelihood': fingerprint.transform.reliability.likelihood,
    'cov_0': fingerprint.transform.covariance.matrix.flatten()[0]
    'cov_1': fingerprint.transform.covariance.matrix.flatten()[1]
    ...
    'cov_8': fingerprint.transform.covariance.matrix.flatten()[8]
    # pairs of SSID and sginal strength
    '00:00:00:00:00:00': 82
    '11:11:11:11:11:11': 81
    ...
"""
import argparse
import pathlib
from datetime import datetime
from functools import partial
from logging import getLogger

import pandas as pd
import trio

from lovot_apis.lovot_localization.wifi_fingerprint_pb2 import RadioMap as RadioMap_pb

from lovot_slam.wifi.mapping.mapping import RadioMap

_logger = getLogger(__name__)


async def _load_radio_map(path) -> RadioMap:
    async with await trio.open_file(path, 'rb') as f:
        message = RadioMap_pb()
        message.ParseFromString(await f.read())
    try:
        radio_map = await RadioMap.from_proto(message)
    except ValueError as e:
        _logger.error(f'Failed to load radio map: {e}')
        import sys
        sys.exit(1)
    _logger.info(f'Radio map is loaded: '
                 f'{len(radio_map.fingerprints)} fingerprints '
                 f'with {len(radio_map.ssids)} SSIDs')
    return radio_map


async def async_main(args):
    path = pathlib.Path(args.protobuf)
    radio_map = await _load_radio_map(path)

    def unixtime_to_human_readable(unixtime: float) -> str:
        dt = datetime.fromtimestamp(unixtime)
        return dt.strftime("%Y/%m/%d %H:%M:%S")

    df = pd.DataFrame()

    for fingerprint in radio_map.fingerprints:
        print(f'{unixtime_to_human_readable(fingerprint.transform.stamp)}: '
              f'[{fingerprint.transform.transform.translation.x:7.2f}, '
              f'{fingerprint.transform.transform.translation.y:7.2f}], '
              f'{len(fingerprint.access_points):3d} APs')

        # Save to pickle
        series_dict = {
            'stamp': fingerprint.transform.stamp,
            'px': fingerprint.transform.transform.translation.x,
            'py': fingerprint.transform.transform.translation.y,
            'yaw': 0,
            'localizer': fingerprint.transform.localizer.value,
            'reliability': fingerprint.transform.reliability.reliability,
            'detection': fingerprint.transform.reliability.detection,
            'likelihood': fingerprint.transform.reliability.likelihood,
        }

        cov_dict = {f'cov_{key}': fingerprint.transform.covariance.matrix.flatten()[key]
                    for key in range(9)}
        series_dict.update(cov_dict)

        # e.g. '00:00:00:00:00:00': 82
        fingerprint_dict = {key.bssid_str(): value.strength
                            for key, value in fingerprint.access_points.items()}
        series_dict.update(fingerprint_dict)

        s = pd.Series(series_dict)
        df = pd.concat([df, pd.DataFrame([s])], ignore_index=True)

    # Save to pickle
    last_stamp = df['stamp'].iloc[-1]
    dt = datetime.fromtimestamp(last_stamp)
    filename = f'wifi_fingerprints_{dt.strftime("%Y%m%d_%H%M%S")}.pkl'
    _logger.info(f'Saving to {filename} ...')
    df.to_pickle(path.parent / filename)


def parse_args():
    parser = argparse.ArgumentParser(description="convert wifi_fingerprints.pb to csv")
    parser.add_argument("protobuf", help="protobuf file path")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    trio.run(partial(async_main, args))
