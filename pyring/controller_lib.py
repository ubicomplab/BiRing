import os
from settings import PORT, BAUD, DATA_ROOT
import serial
import pyrealtime as prt
import struct
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from matplotlib.widgets import Button
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R

rigid_body_calib = 'exp5'
calib_rigid = 1
OSR = 256
BYTES_PER_SAMPLE = 2
RX_CHANNELS_ADC = 3
BUFFER_SIZE = 8
AGGREGATE = 1
RX_CHANNELS = 3  # was 6

NUM_SIGN_BYTES = 1
TRIAL_ON = "exp2"
CALIB_ON = "exp2"

NUM_BYTES = BYTES_PER_SAMPLE * RX_CHANNELS_ADC * AGGREGATE + NUM_SIGN_BYTES
PACKET_FORMAT = '<' + 'B' * RX_CHANNELS_ADC * AGGREGATE * BYTES_PER_SAMPLE + ("B" * NUM_SIGN_BYTES)

USE_NATIVE = True
RECORD = False

VICON_PORT = 9987

FRAME_FORMAT = "IHhhhB"
FRAME_SIZE = 13
FRAMES_PER_PACKET = 40


def convert_to_volts(data):
    center = 0x800000 >> int(((256 / OSR) - 1) * 3)
    if BYTES_PER_SAMPLE == 2:
        center = center >> 8
    return np.sign(data) * (np.abs(data) - center) / (center - 1) * -1.2


def fix_signs(data, counts, buffer_size):
    signs = [(1 if x > buffer_size / 2 else -1) for x in counts]
    return data * signs


def fix_signs_single_byte(data, sign_byte):
    values = [(sign_byte >> 0) & 0x03, (sign_byte >> 2) & 0x03, (sign_byte >> 4) & 0x03]
    # print(values)
    signs = [(1 if x >= 2 else -1) for x in values]
    return data * signs


BYTES_PER_GROUP = 3
AGGREGATE = 1
DEMO = False


def process(data):
    try:
        data = struct.unpack('>' + 'B' * 11, data)
        data = np.array([int(x) for x in data])

    except (ValueError, struct.error):
        print("Parse error")
        return None

    def convert_adc_bottom(data):
        all_data = np.zeros((3, 1))
        for i in range(3):
            adc_b = data[1 + i * 2] + ((data[0 + i * 2] & 0x0f) << 8)
            adc_b = adc_b - 2 ** 12 if adc_b > 2 ** 11 - 1 else adc_b
            all_data[i] = -adc_b
        result = all_data.T / 2048 * 1.2
        return result

    def convert_adc_top(data):

        all_data = np.zeros((6, 1))
        for i in range(3):
            adc_a = (data[0 + i * BYTES_PER_GROUP] << 4) + ((data[1 + i * BYTES_PER_GROUP] & 0xf0) >> 4)
            adc_b = data[2 + i * BYTES_PER_GROUP] + ((data[1 + i * BYTES_PER_GROUP] & 0x0f) << 8)
            adc_a = adc_a - 2 ** 12 if adc_a > 2 ** 11 - 1 else adc_a
            adc_b = adc_b - 2 ** 12 if adc_b > 2 ** 11 - 1 else adc_b
            all_data[i * 2 + 0] = -adc_a
            all_data[i * 2 + 1] = -adc_b

        result = all_data.T / 2048 * 1.2
        return result

    # top = convert_adc(data[1:10], 'top')
    # bottom = convert_adc(data[11:17], 'bottom')
    top = convert_adc_top(data[1:11])
    # bottom = convert_adc_bottom(data[11:17])
    # bottom = bottom[0,[1,3,5]]
    all_data = np.hstack((np.atleast_2d(data[0]), top[:, [0,2,4]]))
    if np.any(all_data < -.02):
        # print("bad packet")
        return None
    return all_data


@prt.transformer
def decimate(layer):
    return layer[0, :]


@prt.transformer(multi_output=True)
def split(data):
    return {"frame_num": data[:, 0:1], "adc": data[:, 1:]}

last_data = [0,0]
@prt.transformer
def diff(data):
    global last_data
    d1 = data[0,:] - last_data
    d = np.diff(data, axis=0)
    last_data = data[-1,:]
    # print(data)
    return np.vstack((d1, d))


@prt.transformer
def get_channel(data):
    return data[:,0]


@prt.transformer
def get_energy(x):
    return np.atleast_2d(np.sum(x ** 2, axis=1)).T


@prt.transformer
def get_taps(energy_thresholds):
    energy = energy_thresholds['energy']
    if "thresholds" in energy_thresholds:
        thresholds = energy_thresholds['thresholds']
    else:
        thresholds = (0.000008, 0.000015)
    if np.max(energy) > thresholds[1]:  # 0.000015:
        # print(2)
        return np.array([2])
    # else:
    #     return False
    #     return np.array([1])
    elif np.max(energy) > thresholds[0]:  # 0.000008:
        # print(1)
        return np.array([1])
    else:
        return np.array([0])


def decode_mag_file(line):
    data = np.array(eval(line.decode('utf-8')))
    return data


def playback_device_data(key, show_plot=True, strip_frame_num=True):
    filename = os.path.join(DATA_ROOT, "recordings", f"mag_{key}.txt")
    if not os.path.exists(filename):
        filename = f"mag_{key}.txt"
    data = prt.PlaybackLayer(filename, rate=59, decoder=decode_mag_file, print_fps=True, loop=True)
    if strip_frame_num:
        split_data = split(data)
        adc = split_data.get_port("adc")
    else:
        adc = data

    if show_plot:
        fm = prt.FigureManager(fps=10000)
        prt.TimePlotLayer(split_data.get_port("adc"), window_size=5000, n_channels=RX_CHANNELS, ylim=(-0.1, 1), lw=1,
                          fig_manager=fm)
    return data, adc


def get_device_data(show_plot=True, buffer_size=BUFFER_SIZE):
    serial_port = serial.Serial(prt.find_serial_port(PORT), BAUD, timeout=5)
    serial_buffer = prt.FixedBuffer(buffer_size, use_np=True, shape=(RX_CHANNELS + 1,), axis=0)
    raw_data = prt.ByteSerialReadLayer.from_port(serial=serial_port, decoder=process, print_fps=True, preamble=b'UW',
                                                 num_bytes=11, buffer=serial_buffer, multi_output=False)
    split_data = split(raw_data)
    # prt.PrintLayer(raw_data)
    if show_plot:
        fm = prt.FigureManager(fps=10000)
        if DEMO:
            filtered = prt.ExponentialFilter(split_data.get_port("adc"), alpha=.1, batch=True)
            prt.TimePlotLayer(filtered, window_size=5000, n_channels=RX_CHANNELS, ylim=(0, 0.2), lw=3, fig_manager=fm)
        else:
            # import scipy.signal
            # # filtered = prt.ExponentialFilter(split_data.get_port("adc"), alpha=1, batch=True)
            # prt.TimePlotLayer(split_data.get_port("adc"), window_size=5000, n_channels=RX_CHANNELS, ylim=(0, 1), lw=1, fig_manager=fm)
            # sos = scipy.signal.butter(5, [25,55], fs=472, btype="bandpass", output='sos')
            # filtered = prt.SOSFilter(split_data.get_port("adc"), sos, axis=0, shape=(9,))
            # energy = get_energy(filtered)
            #
            # prt.TimePlotLayer(energy, window_size=1000, n_channels=1, ylim=(0, .00005), lw=1)#, fig_manager=fm)
            # filtered2 = prt.ExponentialFilter(energy, alpha=.3, batch=True)
            # prt.TimePlotLayer(filtered2, window_size=1000, n_channels=1, ylim=(0, .00005), lw=1)#, fig_manager=fm)
            # # prt.Spectrogram(get_channel(filtered))
            # taps = get_taps(filtered2)
            # prt.TextPlotLayer(taps)

            prt.TimePlotLayer(split_data.get_port("adc"), window_size=5000, n_channels=RX_CHANNELS, ylim=(-0.1, 1), lw=1, fig_manager=fm)
            # prt.TimePlotLayer(diff(split_data.get_port("frame_num")), window_size=5000, n_channels=2, ylim=(0, 5), lw=1)

    return raw_data, split_data.get_port("adc")


def encode_touch(x):
    return "Touch" if x else ""


class ThresholdPlot(prt.TimePlotLayer):
    def __init__(self, port_in, thresholds, *args, **kwargs):
        super().__init__(port_in, *args, **kwargs)
        self.thresholds = thresholds
        self.threshold_series = []

    def transform(self, data):
        super().transform(data)
        return self.thresholds

    def post_init(self, data):

        for i, threshold in enumerate(self.thresholds):
            handle, = self.ax.plot([], [], '-', lw=self.lw, label=f"threshold_{i}")
            self.threshold_series.append(handle)

        self.fig_manager.fig.canvas.mpl_connect('button_press_event', self.on_click)
        super().post_init(data)

    def on_click(self, event):
        if event.xdata > self.window_size / 2:
            self.thresholds[1] = event.ydata
        else:
            self.thresholds[0] = event.ydata

    def update_fig(self, data):
        for (i, series) in enumerate(self.threshold_series):
            series.set_data([self.x_data[0], self.x_data[-1]], [self.thresholds[i], self.thresholds[i]])

        return super().update_fig(data) + self.threshold_series


def detect_touch(data, show_plot=True):

    import scipy.signal
    # filtered = prt.ExponentialFilter(split_data.get_port("adc"), alpha=1, batch=True)
    sos = scipy.signal.butter(5, [25,80], fs=365, btype="bandpass", output='sos')
    filtered = prt.SOSFilter(data, sos, axis=0, shape=(3,))
    energy = get_energy(filtered)

    smoothed_energy = prt.ExponentialFilter(energy, alpha=.2, batch=True)
    # prt.TimePlotLayer(filtered2, window_size=1000, n_channels=1, ylim=(0, .000005), lw=1)#, fig_manager=fm)
    # prt.Spectrogram(get_channel(filtered), fs=472)
    energy_thresholds = prt.MergeLayer(None)

    energy_thresholds.set_input(smoothed_energy, "energy")
    if show_plot:
        thresholds = ThresholdPlot(smoothed_energy, thresholds=[.000002, .00001], window_size=1000, n_channels=1, ylim=(0, .00008), lw=1)
        thresholds.fig_manager.fps = 10
        energy_thresholds.set_input(thresholds, key="thresholds")
    taps = get_taps(energy_thresholds)
    # prt.TextPlotLayer(taps, encoder=encode_touch)
    return taps, filtered, smoothed_energy


def fit_data(data, model_mag2sens, model_sens2pos, model_mag2sens_z, model_sens2pos_z):
    mag_data = data[0:3]
    mag_data = mag_data[:, [0, 2, 1]]
    sensor_predict = model_mag2sens.predict(mag_data)
    pos_predict = model_sens2pos.predict(sensor_predict)

    sensor_predict_z = model_mag2sens_z.predict(mag_data)
    pos_predict_z = model_sens2pos_z.predict(sensor_predict_z)
    pos_predict = np.append(pos_predict[0], pos_predict_z[0][2])
    # print(pos_predict)
    return pos_predict


class ModelLayer(prt.TransformMixin, prt.ThreadLayer):
    def __init__(self, *arg, **kwargs):
        self.data = None
        # self.model_mag2sens = pickle.load(open(os.path.join(DATA_ROOT, 'model',
        #                                                     'model_mag2sensor_'+TRIAL_ON+'_calib'+CALIB_ON+'_'), 'rb'))
        # self.model_sens2pos = pickle.load(open(os.path.join(DATA_ROOT, 'model',
        #                                                     'model_sensor2pos_'+TRIAL_ON+'_calib'+CALIB_ON+'_'), 'rb'))

        self.model_mag2sens_z = pickle.load(open(os.path.join(DATA_ROOT, 'model', 'model_mag2sensor_[exp10]_calibexp11__dof3'), 'rb'))
        self.model_sens2pos_z = pickle.load(open(os.path.join(DATA_ROOT, 'model', 'model_sensor2pos_[exp10]_calibexp11__dof3'), 'rb'))
        self.model_mag2sens = pickle.load(open(os.path.join(DATA_ROOT, 'model', 'model_mag2sensor_[exp10]_calibexp11__dof3'), 'rb'))
        self.model_sens2pos = pickle.load(open(os.path.join(DATA_ROOT, 'model', 'model_sensor2pos_[exp10]_calibexp11__dof3'), 'rb'))

        self.offset = np.array([0, 0])
        self.gain = np.array([150, 150])
        self.max_angles = np.array([0, 0])
        super().__init__(*arg, **kwargs)

    def transform(self, data):
        pos = fit_data(data, self.model_mag2sens, self.model_sens2pos, self.model_mag2sens_z, self.model_sens2pos_z)
        return pos


class DrawingPane2(prt.AggregateScatterPlotLayer):
    def __init__(self, port_in, *args, x_proj=None, y_proj=None, scroll_speed=0, **kwargs):
        super().__init__(port_in, *args, scatter_kwargs={'color': "#1f77b4", "l": 1000}, **kwargs)
        self.x_proj = x_proj if x_proj is not None else np.array([1,0,0])
        self.y_proj = y_proj if y_proj is not None else np.array([0,1,0])
        self.do_clear = False
        self.clear_button = None
        self.scroll_speed = scroll_speed

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

    def transform(self, data):
        # data = data[0]
        data_2d = np.hstack((2 * data[0], -data[1]))
        # y = np.matmul(data, self.x_proj)
        # x = np.matmul(data, self.y_proj)
        if data[2] < -60:
            data_2d = np.hstack((-1000, -10000))

        self.buffer[:, :, 0] += self.scroll_speed
        super().transform(data_2d)
        if self.do_clear:
            self.do_clear = False
            self.buffer[:, :, :] = None

    def handle_signal(self, signal):
        if signal == 2:
            self.do_clear = True


def predict_pos(data, train_on, calib_on):
    global TRIAL_ON
    global CALIB_ON

    TRIAL_ON = train_on
    CALIB_ON = calib_on
    pos = ModelLayer(data)
    return pos


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
        data = data[0]
        data = np.atleast_2d(data)
        # x = np.matmul(data, self.x_proj)
        # y = np.matmul(data, self.y_proj)
        xyz = np.linalg.norm(data, axis=1)
        xz = np.linalg.norm(data[:, [2, 0]], axis=1)
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


def adjust_bodies(body_p, body_q, r, t):
    # print("Adjusting rigid bodies")
    np_body_q = R.from_quat(body_q[[1, 2, 3, 0]])
    np_r = R.from_dcm(r)
    adjust_q = np_body_q * np_r.inv()
    vicon_cad_origin = -np.matmul(r.T, t).T
    adjust_p = body_p + np_body_q.apply(vicon_cad_origin)

    return adjust_p, R.as_quat(adjust_q)[[3, 0, 1, 2]]


def adjust_rigid_bodies(vicon_data, rigid_body_calib):
    global calib_rigid
    global pickle_data
    if calib_rigid:
        pickle_data = pickle.load(open(os.path.join(DATA_ROOT, 'rigid_body_calibration', rigid_body_calib), 'rb'))
        print(pickle_data)
        calib_rigid = 0
    # pickle_data['wrist_q'] = np.eye(3)
    wrist_p, wrist_q = adjust_bodies(vicon_data['wrist']['pos'], vicon_data['wrist']['rot'],
                                     pickle_data['wrist_r'], pickle_data['wrist_t'])
    finger_p, finger_q = adjust_bodies(vicon_data['finger']['pos'], vicon_data['finger']['rot'],
                                       pickle_data['finger_r'], pickle_data['finger_t'])

    vicon_data['finger']['pos'] = finger_p
    vicon_data['finger']['rot'] = finger_q
    vicon_data['wrist']['pos'] = wrist_p
    vicon_data['wrist']['rot'] = wrist_q
    return vicon_data


@prt.transformer
def sp_extract_hand_coord(data):

    data = adjust_rigid_bodies(data, f"calib_{rigid_body_calib}.pkl")

    finger_pos = data['finger']['pos']
    finger_q = R.from_quat(data['finger']['rot']).inv()  # using opti_rot to put in x, y, z, w order
    wrist_pos = data['wrist']['pos']
    wrist_q = R.from_quat(data['wrist']['rot']).inv()  # using opti_rot to put in x, y, z, w order

    rel_pos_wrist_world = finger_pos - wrist_pos
    rel_pos_wrist = wrist_q.apply(rel_pos_wrist_world)

    return rel_pos_wrist*1000