import pickle
from local_settings import DATA_ROOT
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal
import seaborn as sns
from sklearn.neural_network import MLPRegressor
import os
from preprocessing.mag_utils import DeviceCalibration
from utils import load_resampled_data, save_predictions, save_pos_data

TRAIN_ON = "exp6"
CALIB_ON = 'exp6'
TRIAL_TEST = "hello"
TEST_FRAC = 0.2
NDOF = 2
CUTOFF = 700
USE_BIAS = True


def split_data(data):
    split_point = round(len(data) * (1 - TEST_FRAC))
    data_train = data[0:split_point, :]
    data_test = data[split_point:, :]
    return data_train, data_test


def main():
    mag_data, pos, rot = load_resampled_data(TRIAL_TEST)
    mag_data = mag_data[:-CUTOFF]
    pos = pos[:-CUTOFF]
    rot = rot[:-CUTOFF]
    model_mag2sens = pickle.load(
        open(os.path.join(DATA_ROOT, 'model', 'model_mag2sensor_' + TRAIN_ON + '_calib' + CALIB_ON + '__dof' + str(NDOF)), 'rb'))
    model_sens2pos = pickle.load(
        open(os.path.join(DATA_ROOT, 'model', 'model_sensor2pos_' + TRAIN_ON + '_calib' + CALIB_ON + '__dof' + str(NDOF)), 'rb'))

    sensor_predict = model_mag2sens.predict(mag_data)
    pos_predict = model_sens2pos.predict(sensor_predict)
    if NDOF == 2:
        pos = pos[:, :-1]
    if USE_BIAS:
        pos_initial = np.mean(pos[:50], axis=0)
        pos_predict_initial = np.mean(pos_predict[:50], axis=0)
        bias = pos_initial - pos_predict_initial
        pos_predict += bias

    error = pos_predict - pos
    dummy_pos = np.mean(pos, axis=0)
    dummy_pos = dummy_pos.reshape(1, np.size(dummy_pos, 0))
    dummy_pos = np.repeat(dummy_pos, np.size(pos, 0), axis=0)
    dummy_pos_error = dummy_pos - pos

    mean_error = np.mean(np.abs(error), axis=0)
    print("Mean error (mm): ", mean_error)
    print("Mean absolute error (mm): ", np.linalg.norm(mean_error))
    mean_error = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
    print("Mean square absolute error (mm): ", mean_error)
    dummy_mean_error = np.mean(np.abs(dummy_pos_error), axis=0)
    print("Baseline test error: ", dummy_mean_error)
    print("Mean absolute error (mm): ", np.linalg.norm(dummy_mean_error))

    plt.figure()
    plt.plot(pos)
    plt.plot(pos_predict)
    plt.title(TRIAL_TEST)
    plt.show()

    save_pos_data(TRAIN_ON, TRIAL_TEST, pos)
    save_predictions(TRAIN_ON, TRIAL_TEST, pos_predict)


if __name__ == '__main__':
        main()