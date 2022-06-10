import os

import numpy as np
import matplotlib.pyplot as plt
from utils import load_predictions, load_resampled_data, load_pos, DATA_ROOT
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns


TRAIN_ON = "exp6"
TEST_ON = "exp6"


TRIAL = ["par1_2", "par2_2", "par5_2", "par4_1", "par6_2", "par3_2", "par7_1", "par8_2"]

TRIAL = ["par1_2", "par2_2", "par5_2", "par4_1", "par6_2", "par3_2", "par7_1"]

TEST = "par8_2"


def main():
    preds = load_predictions(TRIAL, TEST)
    pos = load_pos(TRIAL, TEST)

    b, a = scipy.signal.butter(5, 0.1, btype='lowpass')
    preds_filt = scipy.signal.filtfilt(b, a, preds, axis=0)
    pos = scipy.signal.filtfilt(b, a, pos, axis=0)

    rmse = np.sqrt(np.mean(np.square(preds_filt - pos), axis=0))
    print("RMSE is: ", rmse[0], rmse[1])
    mean_error = np.mean(np.sqrt(np.sum(np.square(preds_filt - pos), axis=1)))
    print("Mean square absolute error (mm): ", mean_error)

    x_lim = (4000, 5994)
    x_lim = (9000, 13000)
    t = np.linspace(0, (x_lim[1] - x_lim[0]) / 50, x_lim[1] - x_lim[0])

    fig = plt.figure(figsize=(4.8, 3.7))
    # fig = plt.figure()
    p = sns.color_palette()  # 'colorblind')
    ax = fig.add_subplot(211)
    h_preds_x, = ax.plot(t, preds_filt[x_lim[0]:x_lim[1], 0], color=p[1])
    h_gnd, = ax.plot(t, pos[x_lim[0]:x_lim[1], 0], color=p[2])
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Samples")
    sns.despine()
    ax.set_ylabel("X (mm)")

    ax = fig.add_subplot(212)
    h_preds_y, = ax.plot(t, preds_filt[x_lim[0]:x_lim[1], 1], color=p[1])
    h_gnd_y, = ax.plot(t, pos[x_lim[0]:x_lim[1], 1], color=p[2])
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Samples")
    sns.despine()
    ax.set_ylabel("Y (mm)")
    plt.figlegend((h_preds_x, h_gnd), ('Ground Truth', 'DualRing'), loc="upper center", ncol=3)

    plt.figure()
    plt.plot(pos[:, 0], label='gnd')
    plt.plot(preds_filt[:, 0], label='pred')
    plt.ylabel("X")
    plt.legend()
    plt.figure()
    plt.plot(pos[:, 1], label='gnd')
    plt.plot(preds_filt[:, 1], label='pred')
    plt.ylabel("Y")
    plt.legend()

    plt.show()
    fig.savefig(os.path.join(DATA_ROOT, "figures", "trace.pdf"))


if __name__ == '__main__':
    main()
