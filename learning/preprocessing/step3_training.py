import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal
import seaborn as sns
from sklearn.neural_network import MLPRegressor

from preprocessing.mag_utils import DeviceCalibration, compute_sensor, get_field, plot_correlation, get_sensor_pos
from utils import load_resampled_data, save_mode, save_pos_data, save_predictions

TRIAL = "par5_1"
# TRIAL_SAVE = "1245"
TRIAL = ["par1_2", "par2_2", "par5_2", "par4_1", "par6_2", "par3_2", "par7_1", "par8_2"]
TRIAL = ["exp10"]
TRIAL_CALIBRATE = 'exp11_'
TEST_FRAC = 0.1
NDOF = 3


CUTOFF = 500
state = DeviceCalibration.from_trial(TRIAL_CALIBRATE)
USE_BIAS = True
SHOW_FIGURE = True


def split_data(data):

    split_point = round(len(data) * (1 - TEST_FRAC))
    data_train = data[0:split_point, :]
    data_test = data[split_point:, :]
    return data_train, data_test


def train_model(x_train, y_train, x_test, y_test):
    regr = MLPRegressor(hidden_layer_sizes=(50, ), random_state=1, max_iter=140, verbose=False).fit(x_train, y_train)  #20
    y_train_predict = regr.predict(x_train)
    y_test_predict = regr.predict(x_test)
    error_train = y_train_predict - y_train
    error_test = y_test_predict - y_test

    mean_error_train = np.mean(np.sqrt(np.sum(np.square(error_train), axis=1)))
    mean_error_test = np.mean(np.sqrt(np.sum(np.square(error_test), axis=1)))
    print(mean_error_train)
    print(mean_error_test)

    if SHOW_FIGURE:
        plt.figure()
        plt.plot(y_train_predict)
        plt.plot(y_train)

        plt.figure()
        plt.plot(y_test_predict)
        plt.plot(y_test)
        # plt.show()
    return regr, y_train_predict, y_test_predict


def main():
    mag_data_all = []
    pos_all = []
    rot_all = []
    mean_error_all = []

    for trial_test in TRIAL:
        trial_train = TRIAL.copy()
        # trial_train = TRIAL
        # trial_train.remove(trial_test)
        print("Training on: ", trial_train)
        print("Testing on: ", trial_test)

        for user in trial_train:
            mag_data, pos, rot = load_resampled_data(user)
            mag_data = mag_data[:-CUTOFF]
            pos = pos[:-CUTOFF]
            rot = rot[:-CUTOFF]
            try:
                mag_data_all = np.append(mag_data_all, mag_data, axis=0)
                pos_all = np.append(pos_all, pos, axis=0)
                rot_all = np.append(rot_all, rot, axis=0)
            except:
                mag_data_all = mag_data
                pos_all = pos
                rot_all = rot
            # sensor_pos = get_sensor_pos(pos, rot, state.coil1)
            # sensor_pos_train, sensor_pos_test = split_data(sensor_pos)

        pos_train, pos_test = split_data(pos_all)
        rot_train, rot_test = split_data(rot_all)
        mag_train, mag_test = split_data(mag_data_all)

        sensor_train = compute_sensor(get_field(pos_train, rot_train, state.coil1), state.coil1)
        sensor_test = compute_sensor(get_field(pos_test, rot_test, state.coil1), state.coil1)

        if SHOW_FIGURE:
            plt.figure()
            plt.plot(mag_train)
            plt.figure()
            plt.plot(sensor_train)
            plot_correlation(sensor_train, mag_train)

        model_mag2sens, sensors_pred_train, sensors_pred_test = train_model(mag_train, sensor_train, mag_test, sensor_test)

        if NDOF == 2:
            pos_train = pos_train[:, :-1]
            pos_test = pos_test[:, :-1]

        model_sens2pos, pos_pred_train, pos_pred_test = train_model(sensor_train, pos_train, sensor_test, pos_test)

        pos_pred_train = model_sens2pos.predict(sensors_pred_train)
        pos_pred_test = model_sens2pos.predict(sensors_pred_test)

        error_train = pos_pred_train - pos_train
        error_test = pos_pred_test - pos_test

        mean_error_train = np.mean(np.sqrt(np.sum(np.square(error_train), axis=1)))
        mean_error_test = np.mean(np.sqrt(np.sum(np.square(error_test), axis=1)))
        print("Mean train error (mm): ", mean_error_train)
        print("Mean test error (mm): ", mean_error_test)

        if SHOW_FIGURE:
            plt.figure()
            plt.plot(pos_pred_train)
            plt.plot(pos_train)
            plt.title("TRAIN")

            plt.figure()
            plt.plot(pos_pred_test)
            plt.plot(pos_test)
            plt.title("TEST")
            plt.show()

        save_mode(model_mag2sens, "mag2sensor", TRIAL, TRIAL_CALIBRATE, NDOF)
        save_mode(model_sens2pos, "sensor2pos", TRIAL, TRIAL_CALIBRATE, NDOF)

        mag_data, pos, rot = load_resampled_data(trial_test)
        mag_data = mag_data[:-CUTOFF]
        pos = pos[:-CUTOFF]
        rot = rot[:-CUTOFF]
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
        mean_sqaure_error = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
        print("Mean square absolute error (mm): ", mean_sqaure_error)
        dummy_mean_error = np.mean(np.abs(dummy_pos_error), axis=0)
        print("Baseline test error: ", dummy_mean_error)
        print("Mean absolute error (mm): ", np.linalg.norm(dummy_mean_error))
        mean_error_all.append(mean_error)

        if SHOW_FIGURE:
            plt.figure()
            plt.plot(pos)
            plt.plot(pos_predict)
            plt.title(trial_test)
            plt.show()

        save_pos_data(trial_train, trial_test, pos, str(NDOF))
        save_predictions(trial_train, trial_test, pos_predict, str(NDOF))
    # pickle.dump(model_mag2sens, open("mag2sensor_"+TRIAL+".sav", 'wb'))
    # pickle.dump(model_sens2pos, open("sensor2pos_"+TRIAL+".sav", 'wb'))

    print("Cross User Mean Absolute Error (mm):", np.mean(mean_error_all, axis=0))


if __name__ == "__main__":
    main()
