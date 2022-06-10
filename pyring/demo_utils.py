from matplotlib import patches
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
import pyrealtime as prt
import numpy as np

FINGERTIP_LEN_MM = -40
HAND_LEN_MM = -120
ARM_LEN_MM = -300


@prt.transformer
def get_tip_pos(pose):
    pos = pose['pos']
    rot = R.from_quat(pose['rot'][[1,2,3,0]])
    tip_pos = pos + rot.apply(np.array([0,0,FINGERTIP_LEN_MM]), inverse=True)
    return tip_pos


def get_filtered_tip_pos(pose,alpha=.1):
    tip_pos = get_tip_pos(pose)
    filtered_tip_pos = prt.ExponentialFilter(tip_pos, alpha=alpha)
    return filtered_tip_pos


@prt.transformer
def get_tip_pos_with_arm(data):
    pos = data['pose']['pos']
    rot = R.from_quat(data['pose']['rot'][[1,2,3,0]])
    tip_pos = pos + rot.apply(np.array([0,0,FINGERTIP_LEN_MM]), inverse=True)

    arm_pos = data['vive']['pos']
    arm_rot = R.from_quat(data['vive']['rot'])

    final_tip_pos = arm_pos*1000 + arm_rot.apply(tip_pos + np.array([0,0,ARM_LEN_MM]), inverse=True)#
    return final_tip_pos

@prt.transformer
def get_tip_pos_with_arm_no_mag(data):
    tip_pos = np.array([0, 0, HAND_LEN_MM + FINGERTIP_LEN_MM])

    arm_pos = data['vive']['pos']
    arm_rot = R.from_quat(data['vive']['rot'])

    final_tip_pos = arm_pos*1000 + arm_rot.apply(tip_pos + np.array([0,0,ARM_LEN_MM]), inverse=True)#
    return final_tip_pos


def get_filtered_tip_pos_with_arm(data, use_mag_pose=True, alpha=.1):
    if use_mag_pose:
        tip_pos = get_tip_pos_with_arm(data)
    else:
        tip_pos = get_tip_pos_with_arm_no_mag(data)
    filtered_tip_pos = prt.ExponentialFilter(tip_pos, alpha=alpha)
    return filtered_tip_pos

@prt.transformer
def get_xy_projection(pos):
    return pos[[0,1]]


class ButtonPlotLayer(prt.PlotLayer):
    def __init__(self, port_in, *args, **kwargs):
        super().__init__(port_in, *args, **kwargs)
        self.h_text = None

    def draw_empty_plot(self, ax):
        ax.set_axis_off()
        ax.set_xlim([0,1])
        self.rect1 = patches.Rectangle((0, 0), .5, 30, facecolor='none')
        self.rect2 = patches.Rectangle((.5, 0), .5, 30, facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(self.rect1)
        ax.add_patch(self.rect2)

        self.t1 = ax.text(0.25, 0.5, "Left", horizontalalignment='center', verticalalignment='center', fontsize=28, color="white")
        self.t2 = ax.text(0.75, 0.5, "Right", horizontalalignment='center', verticalalignment='center', fontsize=28, color="white")

        return self.rect1, self.rect2, self.t1, self.t2

    def init_fig(self):
        return self.update_fig(1)

    def update_fig(self, data):
        if data & 0x01:
            self.rect1.set_color("#1f77b4")
        else:
            self.rect1.set_color("black")

        if data & 0x02:
            self.rect2.set_color("#1f77b4")
        else:
            self.rect2.set_color("black")

        return self.rect1, self.rect2, self.t1, self.t2

@prt.transformer
def format_button_plot(data):
    out = 0
    if data['touch']:
        if "default" not in data:
            return None
        x = data['default'][0]
        print(data['default'])
        if x < 0:
            out |= 0x01
        else:
            out |= 0x02

    return out


class DrawingPane(prt.AggregateScatterPlotLayer):
    def __init__(self, port_in, *args, x_proj=None, y_proj=None, scroll_speed=0, **kwargs):
        super().__init__(port_in, *args, scatter_kwargs={'color': "#1f77b4", "s": 60}, **kwargs)
        self.x_proj = x_proj if x_proj is not None else np.array([1,0,0])
        self.y_proj = y_proj if y_proj is not None else np.array([0,1,0])
        self.do_clear = False
        self.clear_button = None
        self.scroll_speed = scroll_speed
        self.colors = cm.rainbow(np.linspace(0, 1, self.buffer_size))

    def clear(self, _):
        self.do_clear = True

    def draw_empty_plot(self, ax):
        ax_clear = ax.figure.add_axes([0.81, 0.005, 0.1, 0.075], label="clear button"+ax.get_label())
        self.clear_button = Button(ax_clear, 'Clear', color="#000000")
        self.clear_button.on_clicked(self.clear)
        ax.set_axis_off()
        ax.figure.patch.set_facecolor('black')
        ax.figure.canvas.toolbar.pack_forget()
        return []

    def post_init(self, data):
        super().post_init(data)
        self.series[0].set_color(self.colors)

    def transform(self, data):
        data = np.atleast_2d(data)
        # x = np.matmul(data, self.x_proj)
        # y = np.matmul(data, self.y_proj)
        xyz = np.linalg.norm(data, axis=1)
        xz = np.linalg.norm(data[:, [0, 2]], axis=1)
        yaw = np.arcsin(data[:, 0] / xz)
        pitch = np.arcsin(data[:,1] / xyz)
        data_2d = np.hstack((yaw, pitch))

        self.buffer[:, :, 0] += self.scroll_speed
        super().transform(data_2d)
        if self.do_clear:
            self.do_clear = False
            self.buffer[:, :, :] = None

    def handle_signal(self, signal):
        if signal == 2:
            self.do_clear = True

    def update_fig(self, data):
        tmp = super().update_fig(data)
        return tmp

@prt.transformer
def detect_contact(final_tip_pos):
    if final_tip_pos[2] > -350:
        print(final_tip_pos[2])
        return None
    else:
        return final_tip_pos
