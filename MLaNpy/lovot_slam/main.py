import argparse
import logging
import platform
import signal
import shutil

# import prometheus_client
# import sentry_sdk
import anyio
from importlib.metadata import version

from lovot_slam.env import WEBVIEW_PORT, data_directories
# from lovot_slam.env import PROMETHEUS_PORT, get_sentry_info
from lovot_slam.utils.logging_util import setup_logging

# SENTRY_DSN = "https://f758954bd08447ffa8abd89135b75a00@sentry.dev2.groove-x.io/18"

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='lovot localization slam manager')
    parser.add_argument('device', choices=['lovot', 'nest'], type=str, help='device')
    parser.add_argument('--default_map', default='', help='default map')
    parser.add_argument('--debug', action='store_true', default=False, help='output debug log')
    parser.add_argument('--journal', action='store_true', default=False,
                        help='remove timestamps from log, for journal logging')
    parser.add_argument('--enable-sentry', action='store_true')
    return parser.parse_args()


async def run_main(slam_manager, webview, logger):
    async with anyio.create_task_group() as tg:
        async def _signal_receiver():
            with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
                async for signum in signals:
                    logger.info(f"signal {signal.Signals(signum).name} received")
                    tg.cancel_scope.cancel()

        tg.start_soon(slam_manager.run)
        tg.start_soon(webview.serve, '0.0.0.0', WEBVIEW_PORT)
        tg.start_soon(_signal_receiver)


def run():
    args = parse_args()
    setup_logging('lovot-localization', args.debug, args.journal)
    logger.info('slam manager starting...')

    # sentry_info = get_sentry_info(logger, args.device)

    slam_manager = None

    try:
        # create temporary directory
        data_directories.tmp.mkdir(exist_ok=True, parents=True)

        if args.device == "lovot":
            from lovot_slam.lovot_slam_manager import LovotSlamManager
            slam_manager = LovotSlamManager(default_map=args.default_map, debug=args.debug, journal=args.journal)

            from lovot_slam.viewer.lovot_webview import WebViewServer
            webview = WebViewServer(slam_manager)

        elif args.device == "nest":
            from lovot_slam.nest_slam_manager import NestSlamManager
            slam_manager = NestSlamManager(debug=args.debug, journal=args.journal)

            from lovot_slam.viewer.nest_webview import WebViewServer
            webview = WebViewServer(slam_manager)

        else:
            raise RuntimeError("Invalid device name.")

        # prometheus_client.start_http_server(PROMETHEUS_PORT)
        anyio.run(run_main, slam_manager, webview, logger)

    except Exception as e:
        # if args.enable_sentry:
            # environment_str = f"{platform.system()}-{platform.machine()}"
            # sentry_sdk.init(
            #     SENTRY_DSN,
            #     release=version("lovot-slam"),
            #     environment=environment_str
            # )
            # with sentry_sdk.configure_scope() as scope:
            #     scope.user = {"id": sentry_info.pop("device_id")}
            #     for key, value in sentry_info.items():
            #         scope.set_tag(key, value)
            #     if slam_manager:
            #         if args.device == "tom":
            #             scope.set_tag("map_name", slam_manager.map_name)
            #         elif args.device == "spike":
            #             scope.set_tag("processing_map_name",
            #                           slam_manager._map_builder.processing_map_name)
            #     sentry_sdk.capture_exception(e)
        raise e
    finally:
        # remove temporary directory
        if data_directories.tmp.exists():
            shutil.rmtree(data_directories.tmp)

    logger.info("main thread terminated")
