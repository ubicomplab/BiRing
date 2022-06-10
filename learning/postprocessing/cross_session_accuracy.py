import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal
import seaborn as sns
from sklearn.neural_network import MLPRegressor

from preprocessing.mag_utils import compute_sensor, get_field, DeviceCalibration
from preprocessing.step3_training import train_model, split_data
from utils import load_resampled_data


TRIAL_CALIBRATE = 'exp11_'
state = DeviceCalibration.from_trial(TRIAL_CALIBRATE)
TRIAL = [ "par2", "par3", "par5", "par6", "par7", "par8"]
TEST_FRAC = 0.1
NDOF = 2
CUTOFF = 500
USE_BIAS = True
mean_error_all = []


def main():
    for trial in TRIAL:
        mag_data, pos, rot = load_resampled_data(trial+"_1")
        mag_data = mag_data[:-CUTOFF]
        pos = pos[:-CUTOFF]
        rot = rot[:-CUTOFF]

        pos_train, pos_test = split_data(pos)
        rot_train, rot_test = split_data(rot)
        mag_train, mag_test = split_data(mag_data)

        sensor_train = compute_sensor(get_field(pos_train, rot_train, state.coil1), state.coil1)
        sensor_test = compute_sensor(get_field(pos_test, rot_test, state.coil1), state.coil1)

        model_mag2sens, sensors_pred_train, sensors_pred_test = train_model(mag_train, sensor_train,
                                                                            mag_test, sensor_test)

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

        mag_data_test, pos_test, rot_test = load_resampled_data(trial + "_2")
        mag_data_test = mag_data_test[:-CUTOFF]
        pos_test = pos_test[:-CUTOFF]
        rot_test = rot_test[:-CUTOFF]

        sensor_predict = model_mag2sens.predict(mag_data_test)
        pos_predict = model_sens2pos.predict(sensor_predict)
        if NDOF == 2:
            pos_test = pos_test[:, :-1]
        if USE_BIAS:
            pos_initial = np.mean(pos_test[:50], axis=0)
            pos_predict_initial = np.mean(pos_predict[:50], axis=0)
            bias = pos_initial - pos_predict_initial
            pos_predict += bias

        error = pos_predict - pos_test
        dummy_pos = np.mean(pos_test, axis=0)
        dummy_pos = dummy_pos.reshape(1, np.size(dummy_pos, 0))
        dummy_pos = np.repeat(dummy_pos, np.size(pos_test, 0), axis=0)
        dummy_pos_error = dummy_pos - pos_test

        mean_error = np.mean(np.abs(error), axis=0)
        print("Mean error (mm): ", mean_error)
        print("Mean absolute error (mm): ", np.linalg.norm(mean_error))
        mean_sqaure_error = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
        print("Mean square absolute error (mm): ", mean_sqaure_error)
        dummy_mean_error = np.mean(np.abs(dummy_pos_error), axis=0)
        print("Baseline test error: ", dummy_mean_error)
        print("Mean absolute error (mm): ", np.linalg.norm(dummy_mean_error))
        mean_error_all.append(mean_error)

    print("Cross User Mean Absolute Error (mm):", np.mean(mean_error_all, axis=0))


if __name__ == '__main__':
    main()