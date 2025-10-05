from logging import getLogger

from lovot_slam.viewer import webview_base
from lovot_slam.viewer.webview_base import ServerName, get_drawpose_js, get_drawul_js

logger = getLogger(__name__)


class WebViewServer(webview_base.WebViewServerBase):
    def __init__(self, lovot_slam_manager):
        super().__init__(ServerName.lovot.name, lovot_slam_manager)

    def get_custom_html_content(self):
        content = '<input type="button" value="change map" onclick="slamview.client.changeMap()">\n'
        content += '<input type="button" value="undeploy map" onclick="slamview.client.undeployMap()">'
        return content

    def _get_js(self, entrance, nest, unwelcomed_area):
        draw_poses_str = ''
        draw_unwelomed_area_str = ''

        if entrance is not None:
            draw_poses_str += get_drawpose_js(entrance, "entrance")

        if nest is not None:
            draw_poses_str += get_drawpose_js(nest, "nest")

        if unwelcomed_area is not None:
            for u in unwelcomed_area:
                draw_unwelomed_area_str += get_drawul_js(u)

        map_origin = self.get_map_origin()
        map_origin_str = f'slamview.mapOrigin = [{map_origin[0]}, {map_origin[1]}];'

        template = self._env.get_template('html/index_lovot.js')
        return template.render({
            'drawPoses': draw_poses_str,
            'drawUnwelcomedArea': draw_unwelomed_area_str,
            'mapOrigin': map_origin_str
        })

    def get_custom_js(self):
        entrance = self.get_slam_spot_on_map_img_origin("entrance")
        nest = self.get_slam_spot_on_map_img_origin("nest")
        unwelcomed_area = self.get_slam_unwelcomed_area()

        return self._get_js(entrance, nest, unwelcomed_area)
    