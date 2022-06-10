
import numpy as np
import matplotlib.pyplot as plt
from utils import load_predictions, load_pos
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import load_predictions
plt.rcParams['pdf.fonttype'] = 42

nCut = 20

TRIAL = ["par1_2", "par2_2", "par5_2", "par4_1", "par6_2", "par3_2"]
TRIAL = ["par1_2", "par2_2", "par5_2", "par4_1", "par6_2", "par3_2", "par7_1", "par8_2"]
TRAIN_ON = "exp2"
TEST_ON = "exp2"



def plot_2d_subplot_scatter_color_coded(ax, index, x, y, data, label, title, legend_title):
    scatter = ax[int(index / 4 % 4)][index % 4].scatter(x, y, label=label, marker='.', c=data)
    ax[int(index / 4 % 4)][index % 4].set_xlabel("Picth")
    ax[int(index / 4 % 4)][index % 4].set_ylabel("Yaw")
    ax[int(index / 4 % 4)][index % 4].set_title(title)
    ax[int(index / 4 % 4)][index % 4].legend(*scatter.legend_elements(), loc="lower left",
                                             title=legend_title+str(index))


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

    # for trial in range(1):
    #     preds = load_predictions(TRAIN_ON, TEST_ON)
    #     pos = load_pos(TRAIN_ON, TEST_ON)
    #     rmse = np.sqrt(np.mean(np.square(preds - pos), axis=1))
    #     # rmse = np.abs(preds - pos)
    #     x_preds = preds[:, 0]
    #     y_preds = preds[:, 1]
    #     x_opti = pos[:, 0]
    #     y_opti = pos[:, 1]
    #     error_x = np.abs(x_preds - x_opti)
    #     error_y = np.abs(y_preds - y_opti)
    #
    #     opti_error_all_x = np.append(opti_error_all_x, error_x)
    #     opti_error_all_y = np.append(opti_error_all_y, error_y)
    #
    #     opti_x_all = np.append(opti_x_all, x_opti)
    #     opti_y_all = np.append(opti_y_all, y_opti)

    data_all = pd.DataFrame({'x': opti_x_all, 'y': opti_y_all, 'error_': rmse_all})
    data_all = data_all[data_all["y"].between(30, 140)]

    # df = pd.pivot_table(data_all, index = 'x', columns='y', values='error_')
    df2_all = data_all.groupby(['x', 'y'], as_index=False).mean()

    cuts_all = pd.DataFrame({str(feature) + 'Bin': pd.cut(df2_all[feature], nCut) for feature in ['x', 'y']})

    means_all = df2_all.join(cuts_all).groupby(list(cuts_all)).mean()
    means_all = means_all.unstack(level=0)  # Use level 0 to put 0Bin as columns.
    # Reverse the order of the rows as the heatmap will print from top to bottom.
    means_all = means_all.iloc[::-1]
    # sns.set(color_codes=False, style="whitegrid", font="Lato_bold")#, context="paper")
    # mask = np.where(rmse > 20)
    # mask[np.triu_indices_from(mask)] = True
    fig = plt.figure(figsize=(6.2, 5.7))
    plt.clf()
    sns.heatmap(means_all['error_'], xticklabels=means_all['error_'].columns.map(lambda x: x.left),
                yticklabels=means_all['error_'].index.map(lambda x: x.left), robust=True, square=True)
    plt.title('Error distribution (cross section)')
    plt.ylabel("Y (mm)")
    plt.xlabel("X (mm)")
    plt.tight_layout()

    plt.savefig('heatMap_cross-user_xy.pdf')
    plt.show()
    # ############################################
    data_pitch = pd.DataFrame({'x': opti_x_all, 'y': opti_y_all, 'error_': opti_error_all_x})
    df2_pitch = data_pitch.groupby(['x', 'y'], as_index=False).mean()
         
    cuts_pitch = pd.DataFrame({str(feature) + 'Bin': pd.cut(df2_pitch[feature], nCut) for feature in ['x', 'y']})

    means_pitch = df2_pitch.join(cuts_pitch).groupby(list(cuts_pitch)).mean()
    means_pitch = means_pitch.unstack(level=0)  # Use level 0 to put 0Bin as columns.
    # Reverse the order of the rows as the heatmap will print from top to bottom.
    means_pitch = means_pitch.iloc[::-1]
    # sns.set(color_codes=False, style="whitegrid", font="Lato_bold")#, context="paper")

    fig = plt.figure(figsize=(6.2, 5.7))
    plt.clf()
    sns.heatmap(means_pitch['error_'], xticklabels=means_pitch['error_'].columns.map(lambda x : x.left),
                yticklabels=means_pitch['error_'].index.map(lambda x : x.left), robust=True, vmax=20)
    plt.title('Error distribution for Y (cross section)')
    plt.ylabel("Y (mm)")
    plt.xlabel("X (mm)")
    plt.tight_layout()

    plt.savefig('heatMap_cross-user_pitch.pdf')
    # ############################################
    data_yaw = pd.DataFrame({'x': opti_x_all, 'y': opti_y_all, 'error_': opti_error_all_y})
    df2_yaw = data_yaw.groupby(['x', 'y'], as_index=False).mean()

    cuts_yaw = pd.DataFrame({str(feature) + 'Bin': pd.cut(df2_yaw[feature], nCut) for feature in ['x', 'y']})

    means_yaw = df2_yaw.join(cuts_yaw).groupby(list(cuts_yaw)).mean()
    means_yaw = means_yaw.unstack(level=0)  # Use level 0 to put 0Bin as columns.
    # Reverse the order of the rows as the heatmap will print from top to bottom.
    means_yaw = means_yaw.iloc[::-1]
    # sns.set(color_codes=False, style="whitegrid", font="Lato_bold")#, context="paper")

    fig = plt.figure(figsize=(6.2, 5.7))
    plt.clf()
    sns.heatmap(means_yaw['error_'], xticklabels=means_yaw['error_'].columns.map(lambda x: x.left),
                yticklabels=means_yaw['error_'].index.map(lambda x: x.left), robust=True, vmax=20)
    plt.title('Error distribution for X (cross section)')
    plt.ylabel("Y (mm)")
    plt.xlabel("X (mm)")
    plt.tight_layout()

    plt.savefig('heatMap_cross-user_yaw.pdf')
    # ############################################
    f, (ax1, ax2,axcb) = plt.subplots(1,3, figsize=(7.8, 7.4), gridspec_kw={'width_ratios':[1,1,0.08]})

    ax1.get_shared_y_axes().join(ax2)
    g1 = sns.heatmap(means_pitch['error_'], xticklabels=means_pitch['error_'].columns.map(lambda x: x.left),
                yticklabels=means_pitch['error_'].index.map(lambda x: x.left), robust=True, vmax=20, ax=ax1, cbar=False)
    g2 = sns.heatmap(means_yaw['error_'], xticklabels=means_yaw['error_'].columns.map(lambda x: x.left),
                yticklabels=means_yaw['error_'].index.map(lambda x: x.left), robust=True, vmax=20, ax=ax2,cbar_ax=axcb)
    fmt = '{:1.0f}'
    xticklabels1 = []
    xticklabels2 = []
    yticklabels1 = []
    yticklabels2 = []
    for item in g1.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels1 += [item]
    for item in g2.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels2 += [item]
    for item in g1.get_yticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        yticklabels1 += [item]
    for item in g2.get_yticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        yticklabels2 += [item]
    g1.set_xticklabels(xticklabels1)
    g2.set_xticklabels(xticklabels2)
    g1.set_yticklabels(yticklabels1)

    g2.set_yticks([])
    g1.set_title('Error distribution for X')
    g2.set_title('Error distribution for Y')
    g2.set_ylabel('')
    g1.set_ylabel("Y (mm)")
    g1.set_xlabel("X (mm)")
    g2.set_xlabel("X (mm)")
    # fig.suptitle("Yaw (degrees)")
    f.tight_layout()
    plt.savefig('heatMap_cross-user_all.pdf')
    plt.show()


if __name__ == '__main__':
    main()