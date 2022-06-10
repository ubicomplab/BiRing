import os
import pickle
import matplotlib as mpl
import pandas as pd

mpl.rcParams['pdf.fonttype'] = 42
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import load_predictions, load_pos
from local_settings import DATA_ROOT

# 1684536 data points
MODEL_NAME = "linearReg"
TRIAL = ["par1_2", "par2_2", "par5_2", "par4_1", "par6_2", "par3_2"]
TRIAL = ["par1_2", "par2_2", "par5_2", "par4_1", "par6_2", "par3_2", "par7_1", "par8_2"]


def make_cdf_xyz(data, label1="X", label2="Y"):
    A = np.random.randint(data.shape[1], size=(1000,))
    data_samp = data[:, A]

    sns.kdeplot(np.abs(data_samp[0, :]), cumulative=True, lw=4, label=label1)
    sns.kdeplot(np.abs(data_samp[1, :]), cumulative=True, lw=4, label=label2)
    plt.xlabel("Error (mm)")
    plt.ylabel("CDF")
    plt.xlim((0, 25))
    plt.ylim((0, 1))
    plt.legend()


def make_cdf_xyz_rmse(data, label1="RMSE", label2="Y"):
    A = np.random.randint(data.shape[0], size=(1000,))
    data_samp = data[A]

    sns.kdeplot(np.abs(data_samp), cumulative=True, lw=4, label=label1)
    # sns.kdeplot(np.abs(data_samp[:, 1]), cumulative=True, lw=4, label=label2)
    plt.xlabel("Error (mm)")
    plt.ylabel("CDF")
    plt.xlim((0, 25))
    plt.ylim((0, 1))
    plt.legend()


def make_cdf_pitch(data, label, color):
    A = np.random.randint(data.shape[0], size=(10000,))
    data_samp = data[A]
    sns.kdeplot(np.abs(data_samp), cumulative=True, lw=4, label=label, color=color)
    plt.xlabel("Error (mm)")
    plt.ylabel("CDF")
    plt.xlim((0, 25))
    plt.ylim((0, 1))
    plt.legend()


def make_cdf_yaw(data, label, color):
    A = np.random.randint(data.shape[0], size=(10000,))
    data_samp = data[A]
    sns.kdeplot(np.abs(data_samp), cumulative=True, lw=4, label=label, color=color)
    plt.xlabel("Error (mm)")
    plt.ylabel("CDF")
    plt.xlim((0, 25))
    plt.ylim((0, 1))
    plt.legend()


def main():
    data = pd.DataFrame(columns={'x' 'y', 'error_'})
    opti_y_all = []
    opti_x_all = []
    opti_error_all = []
    opti_error_all_x = []
    opti_error_all_y = []
    rmse_all = []
    for trial_test in TRIAL:
        trial_train = TRIAL.copy()
        trial_train.remove(trial_test)
        for user in trial_train:
            preds = load_predictions(trial_train, trial_test)
            pos = load_pos(trial_train, trial_test)
            rmse = np.sqrt(np.mean(np.square(preds - pos), axis=1))
            # rmse = np.abs(preds - pos)
            x_preds = preds[:, 0]
            y_preds = preds[:, 1]
            x_opti = pos[:, 0]
            y_opti = pos[:, 1]
            error_x = np.abs(x_preds - x_opti)
            error_y = np.abs(y_preds - y_opti)

            rmse_all = np.append(rmse_all, rmse)
            opti_error_all_x = np.append(opti_error_all_x, error_x)
            opti_error_all_y = np.append(opti_error_all_y, error_y)

            opti_x_all = np.append(opti_x_all, x_opti)
            opti_y_all = np.append(opti_y_all, y_opti)

    data = np.vstack((opti_error_all_x, opti_error_all_y))
    fig = plt.figure(figsize=(6.25, 4.2))
    make_cdf_pitch(opti_error_all_x, "X", "r")
    make_cdf_yaw(opti_error_all_y, "Y", "b")
    fig.savefig(os.path.join(DATA_ROOT, "figures", "CDF_xy.pdf"))
    fig = plt.figure(figsize=(6.25, 4.2))
    make_cdf_xyz_rmse(rmse_all, "x", "y")
    fig.savefig(os.path.join(DATA_ROOT, "figures", "CDF_rmse_xy.pdf"))
    plt.show()

    # with open(os.path.join(DATA_ROOT, "results", "LEAVE-1-OUT_all_0_1"), 'rb') as f:
    #     error_leave_noCalib = pickle.load(f)[:, 0:2]
    #     norm_error = np.linalg.norm(error_leave_noCalib, axis=1)
    # with open(os.path.join(DATA_ROOT, "results", "LEAVE-1-OUT_all_supervised"), 'rb') as f:
    #     error_supervise = pickle.load(f)[:, 0:2]
    # with open(os.path.join(DATA_ROOT, "results", "UNSUPERVISED_all_1_0"), 'rb') as f:
    #     error_supervise = pickle.load(f)[:, 0:2]
    #     norm_error = np.linalg.norm(error_supervise, axis=1)
    # with open(os.path.join(DATA_ROOT, "results", "BETWEEN_SESSION_all_1_0"), 'rb') as f:
    #     error_between_sesssion = pickle.load(f)[:, 0:2]
    #     norm_error = np.linalg.norm(error_between_sesssion, axis=1)

    # error_un_normalize = np.vstack([error_un_small_normalize, error_un_large_normalize])
    # error_re_calib = np.vstack([error_re_calib_small, error_re_calib_large])
    # error_leave_noCalib = np.vstack([error_un_large_noCalib, error_un_small_noCalib])

    # sns.set(context="paper", style="ticks", font="Lato", font_scale=.9)
    # fig = plt.figure(figsize=(6.25, 3.95))
    # ax = fig.add_subplot(1, 1, 1)
    # make_cdf_xyz(error_re_calib, "Pitch-Method1", "Yaw-Method1")
    # make_cdf_xyz(error_leave_noCalib, "Pitch-Method2", "Yaw-Method2")
    # make_cdf_xyz(error_un_normalize, "Pitch-Method3", "Yaw-Method3")
    # make_cdf_xyz(error_between_sesssion, "Pitch-Method4", "Yaw-Method4")
    #
    # # ax.set_title('Angle Cumulative Distribution Function')
    # # ax = fig.add_subplot(2, 2, 4)
    # # plt.xlabel("Error (mm)")
    # # plt.ylabel("CDF")
    # # plt.xlim((0, 10))
    # # plt.ylim((0, 1))
    # # plt.legend()
    # # fig.subplots_adjust(bottom=0.16, left=0.1, right=0.97, top=0.93, wspace=.34, hspace=.84)
    #
    # plt.show()
    # print("Saving figure")
    # fig.savefig(os.path.join(DATA_ROOT, "figures", "CDF"))
    sns.set(color_codes=False, style="whitegrid")
    p = sns.color_palette()

    fig = plt.figure(figsize=(6.25, 4.2))
    ax = fig.add_subplot(1, 1, 1)


    # # make_cdf_pitch(error_re_calib, "Between User (Re-Calib")
    # make_cdf_pitch(error_between_sesssion, "Across Session", p[0])
    # make_cdf_pitch(error_leave_noCalib, "Cross User-No Calb", p[1])
    # make_cdf_pitch(error_supervise, "Cross User-Supervise", p[2])
    #
    # plt.show()
    # print("Saving figure")
    # fig.savefig(os.path.join(DATA_ROOT, "figures", "CDF_pitch.pdf"))
    #
    # sns.set(color_codes=False, style="whitegrid")
    # p = sns.color_palette()
    # fig = plt.figure(figsize=(6.25, 4.2))
    # ax = fig.add_subplot(1, 1, 1)
    # # make_cdf_yaw(error_re_calib, "Between User (Re-Calib")
    # make_cdf_yaw(error_between_sesssion, "Between Session", p[0])
    # make_cdf_yaw(error_leave_noCalib, "Cross User-No Calib", p[1])
    # make_cdf_yaw(error_supervise, "Cross User-Supervise", p[2])
    # plt.show()
    # print("Saving figure")
    # fig.savefig(os.path.join(DATA_ROOT, "figures", "CDF_yaw.pdf"))


    def make_cdf_yaw_grid(data, label, color, ax):
        A = np.random.randint(data.shape[0], size=(10000,))
        data_samp = data[A, :]
        g = sns.kdeplot(np.abs(data_samp[:, 0]), cumulative=True, lw=4, label=label, color=color, ax=ax)
        # plt.xlabel("Error (degrees)")
        # plt.ylabel("CDF of Yaw")
        # plt.xlim((0, 12))
        # plt.ylim((0, 1))
        return g

    def make_cdf_pitch_grid(data, label, color, ax):
        A = np.random.randint(data.shape[0], size=(10000,))
        data_samp = data[A, :]
        g = sns.kdeplot(np.abs(data_samp[:, 1]), cumulative=True, lw=4, color=color, ax=ax)
        # plt.xlabel("Error (degrees)")
        # plt.ylabel("CDF of Pitch")
        # plt.xlim((0, 12))
        # plt.ylim((0, 1))
        return g

    f, (ax1, ax2) = plt.subplots(1,2, figsize=(6, 4.2))

    ax1.get_shared_y_axes().join(ax2)
    g1 = make_cdf_pitch_grid(error_leave_noCalib, "Cross-User", p[0], ax1)
    g2 = make_cdf_pitch_grid(error_between_sesssion, "Cross-Session", p[1], ax1)
    g3 = make_cdf_pitch_grid(error_supervise, "Per-Session", p[2], ax1)
    g4 = make_cdf_yaw_grid(error_leave_noCalib, "Cross-User", p[0], ax2)
    g5 = make_cdf_yaw_grid(error_between_sesssion, "Cross-Session", p[1], ax2)
    g6 = make_cdf_yaw_grid(error_supervise, "Per-Session", p[2], ax2)

    g1.set_xlim([0, 12])
    g1.set_ylim((0, 1))

    g4.set_xlim([0, 12])
    g4.set_ylim((0, 1))
    # g3.set_yticks([])
    # g4.set_yticks([])
    # g5.set_yticks([])
    # g6.set_yticks([])
    g1.set_title('Pitch')
    g4.set_title('Yaw')
    g2.set_ylabel('')
    g1.set_ylabel("CDF")
    g1.set_xlabel("Error (degrees)")
    g4.set_xlabel("Error (degrees)")

    f.tight_layout()
    plt.show()
    print("Saving figure")
    f.savefig(os.path.join(DATA_ROOT, "figures", "CDF_all.pdf"))


if __name__ == "__main__":
    main()
