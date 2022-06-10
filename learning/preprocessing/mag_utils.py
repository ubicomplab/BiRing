import os
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from settings import DATA_ROOT


@dataclass(init=False)
class Calibration:
    base_sensor_offset: np.array
    base_sensor_rot_offset: R
    sensor_offset: np.array
    sensor_rot_offset: R
    global_gain: float
    gain: np.array
    per_channel_gain: np.array
    noise: np.array
    bias: np.array
    crosstalk: np.array
    kxy: float = 0
    kxz: float = 0
    kyz: float = 0

    @classmethod
    def parse(cls, data):
        coil = cls()
        i = 0
        coil.global_gain = data[i]
        i += 1
        coil.gain = data[i:i+3]
        i += 3
        coil.per_channel_gain = data[i:i+3]
        i += 3
        coil.bias = data[i:i+3]
        i += 3
        coil.noise = data[i:i+3]
        i += 3
        coil.sensor_offset = data[i:i+3]
        i += 3
        coil.sensor_rot_offset = R.from_quat([data[i+1],data[i+2],data[i+3],data[i]])  # x, y, z, w
        i += 4
        coil.ring_offset = data[i:i+3]
        i += 3
        coil.ring_rot_offset = R.from_quat([data[i+1],data[i+2],data[i+3],data[i]])  # x, y, z, w
        i += 4
        coil.crosstalk = data[i:i+3]
        i += 3
        coil.base_sensor_offset = data[i:i+3]
        i += 3
        coil.base_sensor_rot_offset = R.from_quat([data[i+1],data[i+2],data[i+3],data[i]])  # x, y, z, w
        i += 4
        return coil


@dataclass(init=False)
class DeviceCalibration:
    coil1: Calibration = Calibration()
    # coil2: Calibration = Calibration()

    @classmethod
    def from_trial(cls, trial):
        calib = cls()
        filename = os.path.join(DATA_ROOT, "ceres", f"calibrate__{trial}.csv")
        data = np.loadtxt(filename)
        calib.coil1 = Calibration.parse(data)
        # calib.coil1 = Calibration.parse(data[0, :])
        # calib.coil2 = Calibration.parse(data[1, :])
        # calib.coil3 = Calibration.parse(data[2, :])
        return calib


def dipole_model(pos):
    # https://ccmc.gsfc.nasa.gov/RoR_WWW/presentations/Dipole.pdf
    # slide 2
    pos = np.atleast_2d(pos)
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    Bx = 3*x*z/(r**5)
    By = 3*y*z/(r**5)
    Bz = (3*z**2-r**2)/(r**5)
    field = np.vstack((Bx, By, Bz)).T

    return field * 1e5


def get_sensor_pos(ring_pos, ring_q, calib):
    ring_q = np.atleast_2d(ring_q)[:, [1,2,3,0]]
    sensor = calib.base_sensor_offset + calib.sensor_offset
    sensor_rot = calib.sensor_rot_offset * calib.base_sensor_rot_offset

    ring_qs = R.from_quat(ring_q)
    ring_pos_adj = ring_pos + ring_qs.apply(calib.ring_offset, inverse=True)
    ring_rot_adj = ring_qs * calib.ring_rot_offset

    sensor_ring = (ring_qs * calib.ring_rot_offset).apply(sensor-ring_pos_adj)
    return sensor_ring


def get_field(ring_pos, ring_q, calib):
    ring_q = np.atleast_2d(ring_q)[:, [1,2,3,0]]
    sensor = calib.base_sensor_offset + calib.sensor_offset
    sensor_rot = calib.sensor_rot_offset * calib.base_sensor_rot_offset

    ring_qs = R.from_quat(ring_q)
    ring_pos_adj = ring_pos + ring_qs.apply(calib.ring_offset, inverse=True)
    ring_rot_adj = ring_qs * calib.ring_rot_offset

    sensor_ring = (ring_qs * calib.ring_rot_offset).apply(sensor-ring_pos_adj)
    field = ring_rot_adj.apply(dipole_model(sensor_ring), inverse=True)

    return sensor_rot.apply(field)


def compute_sensor(field, calib):
    field_adj = field * calib.global_gain / calib.per_channel_gain
    coeffs = np.array([  [1, calib.kxy ** 2, calib.kxz ** 2],
                [calib.kxy ** 2, 1, calib.kyz ** 2],
                [calib.kxz ** 2, calib.kyz ** 2, 1],
                [2 * calib.kxy * calib.kxz, 0, 0],
                [0, 2 * calib.kxy * calib.kyz, 0],
                [0, 0, 2 * calib.kxz * calib.kyz],
                calib.noise ** 2])

    features_1 = np.hstack((field_adj ** 2,
    np.array([field_adj[:, 1]*field_adj[:, 2],
    field_adj[:, 0]*field_adj[:, 2],
    field_adj[:, 0]*field_adj[:, 1]]).T, np.ones((field_adj.shape[0], 1))))
    sensors = calib.gain * np.sqrt(np.matmul(features_1, coeffs)) - calib.bias

    sensors[sensors < 0] = 0
    return sensors


def plot_correlation(fields, field_obs):
    fig = plt.figure()
    axis = 'xyz'
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.scatter(fields[:, i], field_obs[:, i], marker='.')
        ax.set_title(f"C{i}: {axis[i%3]}")
    # plt.show()

if __name__ == "__main__":
    p = np.array([-14.4915, -35.7031, -113.7257])
    q = np.array([0.0737, 0.2439, -0.9371, 0.2384])
    print(get_field(p, q, state.coil1))
    print(compute_sensor(get_field(p, q, state.coil1), state.coil1))