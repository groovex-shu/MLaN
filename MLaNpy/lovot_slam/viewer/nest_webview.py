from collections import OrderedDict
from logging import getLogger

from lovot_slam.exploration.exploration_status import ExplorationStatusMonitor
from lovot_slam.viewer import webview_base
from lovot_slam.viewer.webview_base import ServerName, get_drawpose_js, get_drawul_js

logger = getLogger(__name__)


class WebViewServer(webview_base.WebViewServerBase):
    def __init__(self, nest_slam_manager):
        super().__init__(ServerName.nest.name, nest_slam_manager)

    def get_map_metadata(self) -> OrderedDict:
        metadata = super().get_map_metadata()

        exploration_status_monitor: ExplorationStatusMonitor = self.slam_manager._exploration_status_monitor
        metadata['exploration status'] = str(exploration_status_monitor.status)
        metadata['ready'] = str(exploration_status_monitor.status.is_ready())
        metadata['well explored'] = str(exploration_status_monitor.status.is_well_explored())
        metadata['gave up'] = str(exploration_status_monitor.status.is_gave_up())
        metadata['can explore'] = str(exploration_status_monitor.status.can_explore())
        return metadata

    def get_custom_html_content(self):
        return ""  # TODO add spike specific html content

    def _get_js(self, entrance, unwelcomed_area):
        draw_poses_str = ''
        draw_unwelomed_area_str = ''

        if entrance is not None:
            draw_poses_str += get_drawpose_js(entrance, "entrance")

        if unwelcomed_area is not None:
            for u in unwelcomed_area:
                draw_unwelomed_area_str += get_drawul_js(u)

        template = self._env.get_template('html/index_nest.js')
        return template.render({
            'drawPoses': draw_poses_str,
            'drawUnwelcomedArea': draw_unwelomed_area_str
        })

    def get_custom_js(self):
        entrance = self.get_slam_spot_on_map_img_origin("entrance")
        unwelcomed_area = self.get_slam_unwelcomed_area()

        return self._get_js(entrance, unwelcomed_area)
