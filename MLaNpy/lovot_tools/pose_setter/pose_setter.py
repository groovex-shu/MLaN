import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from matplotlib.patches import Arrow
from matplotlib.animation import FuncAnimation

from grid_map_util.occupancy_grid import OccupancyGrid

from localization_tools.utils.math_util import quaternion_to_euler


def update(event_handler):
    """post processing updating plot"""

    def event_handler_decorated(self, *args, **kwargs):
        event_handler(self, *args, **kwargs)
        if self.arrow:
            self.arrow.remove()
        d = 20 * np.array([math.cos(self.yaw), math.sin(self.yaw)])
        arrow = Arrow(self.point[1], self.point[0], d[0], d[1], width=10)
        self.arrow = self.ax.add_patch(arrow)
        self.fig.canvas.draw()
        # print('({}, {}), yaw={}'.format(self.point[0], self.point[1], self.yaw))
    return event_handler_decorated


class PoseSetter:
    def __init__(self, oc_map: OccupancyGrid):
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

        self.oc_map = oc_map
        self.point = np.zeros(2)
        self.position = np.zeros(3)
        self.orientation = quaternion.one
        self.yaw = 0

        self.is_dragged = False
        self.arrow = None
        self.preset_arrows = {}

        self.callback = None
        self.use_image_pose = False

    @update
    def _on_motion(self, event):
        if not self.is_dragged:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.yaw = math.atan2(
            event.ydata - self.point[1], event.xdata - self.point[0])
        # y axis is reversed (matplotlib coords vs image_pose coords)
        self.orientation = quaternion.from_euler_angles(
            np.array([0, 0, -self.yaw]))

    @update
    def _on_click(self, event):
        self.is_dragged = True
        if event.xdata is None or event.ydata is None:
            return
        self.point = np.array([event.ydata, event.xdata])
        self.position[:2] = self.oc_map.npcoords_to_realcoords(self.point)

    @update
    def _on_release(self, event):
        self.is_dragged = False
        if self.use_image_pose:
            pose_str = self._transform_to_image_pose_str(
                self.position, self.orientation)
        else:
            pose_str = self._transform_to_pose_str(
                self.position, self.orientation)
        if self.callback is None:
            return
        self.callback(pose_str)

    def _transform_to_pose_str(self, position, orientation):
        pose_str = ','.join(map(str, np.concatenate([
            position,
            # order of element is converted: ow,ox,oy,oz => ox,oy,oz,ow
            np.roll(quaternion.as_float_array(orientation), -1)])))
        return pose_str

    def _transform_to_image_pose_str(self, position, orientation):
        pose_str = ','.join(map(str, np.concatenate([
            position - np.append(self.oc_map.origin, [0.]),
            # order of element is converted: ow,ox,oy,oz => ox,oy,oz,ow
            np.roll(quaternion.as_float_array(orientation), -1)])))
        return pose_str

    def _pose_str_to_transform(self, pose_str: str):
        pose = list(map(float, pose_str.split(",")))
        if len(pose) != 7:
            print(f"invalid pose length {len(pose)} != 7")
            return None, None
        position = np.array(pose[:3])
        orientation = quaternion.from_float_array(np.roll(pose[3:], 1))
        return position, orientation

    def pose_str(self):
        return self._transform_to_pose_str(self.position, self.orientation)

    def plot(self):
        img = cv2.cvtColor(self.oc_map.img, cv2.COLOR_GRAY2RGB)
        plt.imshow(img)
        plt.show()

    def _update(self, frame):
        if self.animation_callback:
            pose_str = self.animation_callback()
            self.update_arrow('current', pose_str)
        img = cv2.cvtColor(self.oc_map.img, cv2.COLOR_GRAY2RGB)
        im = plt.imshow(img, animated=True)
        return [im]

    def plot_animation(self, callback=None):
        self.animation_callback = callback
        self.current_pose_arrow = None
        _ = FuncAnimation(self.fig, self._update, blit=False)
        plt.show()

    def update_arrow(self, name, pose_str, color="#aa0088"):
        position, orientation = self._pose_str_to_transform(pose_str)
        if position is None:
            return
        if name in self.preset_arrows and self.preset_arrows[name]:
            self.preset_arrows[name].remove()
        point = self.oc_map.realcoords_to_npcoords(position[:2])
        q = orientation
        _, _, yaw_angle = quaternion_to_euler(q.x, q.y, q.z, q.w)
        # y axis is reversed (matplotlib coords vs image_pose coords)
        d = 20 * np.array([math.cos(-yaw_angle), math.sin(-yaw_angle)])
        arrow = Arrow(point[1], point[0], d[0], d[1], width=10, color=color)
        self.preset_arrows[name] = self.ax.add_patch(arrow)

    def set_callback(self, func, use_image_pose=False):
        self.callback = func
        self.use_image_pose = use_image_pose
