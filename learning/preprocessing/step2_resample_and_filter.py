import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal
import seaborn as sns

from preprocessing.mag_utils import get_field, compute_sensor, DeviceCalibration, \
    plot_correlation
from utils import MAG_RAW_NAMES, save_resampled_data, \
    load_extracted_opti_data, POS, \
    load_raw_mag, ROT_STANDARD, PALM_OPTI_POS, WRIST_ROT, WRIST_OPTI_POS, load_opti_metadata

TRIAL = "par8_1"

state = DeviceCalibration.from_trial('exp6_')

F_MOCAP = 240
F_MAG = 366

RING_VECTOR = [0, 0, -1]

USE_SECOND_STAGE_ALIGNMENT = True

# Positive alignment at start means increase first number by 2x
# Negative alignment at start means reduce first number by 2x
# (how much to take off of mag data start, how much to take off end of pos, how much to take off of pos start)
ALIGNMENTS = {"triband-train": (1330, 0),
              "ring1wb1-freeform": (3350, 0, 1000),
              "ring1wb1-test": (4400, 0, 1800),
              "ring1wb1-test_2": (6700, 0, 2000),
              "ring1wb1-freeform_2": (1850, 0, 0),
              "ring2wb1-freeform_2": (1400, 0, 0),
              "ring2wb1-test_2": (1860, 0, 0),
              "ring2wb2-freeform_2": (1850, 0, 0),
              "ring2wb2-test_2": (1400, 0, 0),
              "ring1wb1-freeform_4": (3350, 0, 1000),
              "ring3wb3-freeform_4": (2350, 0, 500),
              "ring3wb1-freeform_4": (2450, 0, 800),
              "ring3wb2-freeform_4": (4110, 0, 1627),
              "t1": (400,0,0),
              "t2": (390,0,0),
              "exp1": (350,0,0),
              "exp2": (379,0,0),
              "exp3": (423,19,0),
              "exp5": (428,0,0),
              "exp6": (436,0,0),
              "exp7": (420,0,0),
              "exp8": (430,0,0),
              "exp9": (430,0,0),
              "exp10": (404,0,0),
              "exp11": (409,0,0),
              "par1_1": (406,0,0),
              "par1_2": (428,0,0),
              "par2_1": (434,8,0),
              "par2_2": (428,10,0),
              "par3_1": (446,0,0),
              "par3_2": (444,10,0),
              "par4_1": (444,0,0),
              "par5_1": (432,0,0),
              "par5_2": (430,8,0),
              "par6_1": (415,0,0),
              "par6_2": (420,0,0),
              "par7_1": (428,0,0),
              "par7_2": (412,0,0),
              "par8_1": (419,0,0),
              "par8_2": (425,0,0)}


def filter_in_box(filtered_data):
    x = (filtered_data.x < -.02) & (filtered_data.x > -.03)
    y = (filtered_data.y < -.50) & (filtered_data.y > -.51)
    z = (filtered_data.z < .285) & (filtered_data.z > .275)
    filtered = x & y & z
    return filtered_data[filtered]


def filter_good_data(filtered_data, drop_high=False, drop_low=False, only_best=False):
    total_points = len(filtered_data)
    if only_best:
        drop_high = True
    valid_data = filtered_data[filtered_data.is_valid]

    if drop_high:
        is_valid = np.all(valid_data[MAG_RAW_NAMES] < np.cbrt(17000), axis=1)
        valid_data = valid_data[is_valid]
    if drop_low:
        is_valid = np.partition(valid_data[MAG_RAW_NAMES], 4, axis=1)[:, 4] > np.cbrt(
            200)  # note using valid_data here instead of raw_data, just because it's easier
        valid_data = valid_data[is_valid]
    if only_best:
        norm = np.linalg.norm(valid_data[MAG_RAW_NAMES].values, axis=1)
        # is_valid = np.all(valid_data[MAG_RAW_NAMES] > 100, axis=1)
        is_valid = norm > np.cbrt(2000)
        valid_data = valid_data[is_valid]

    remaining_points = len(valid_data)
    print("Kept %d out of %d points. (%d%%)" % (remaining_points, total_points, remaining_points / total_points * 100))
    return valid_data


def next_power_of_two(n):
    y = np.floor(np.log2(n))
    return (int)(np.power(2, y + 1))


def trim_to_times(data, start, stop):
    start_index = np.argmin(abs(data.time - start))
    stop_index = np.argmin(abs(data.time - stop))
    print("Reducing 0:%d to %d:%d" % (len(data) - 1, start_index, stop_index))
    return data.iloc[start_index:stop_index + 1].reset_index()


def plot_alignment(data, pos, rot):
    pass
    # opti_distance = np.linalg.norm(pos, axis=1)
    # # mag_distance = 5/np.cbrt(np.sqrt(np.sum(raw_data_slow[MAG_RAW_NAMES]**2,axis=1)))
    # mag_distance2 = 5 / np.linalg.norm(data, axis=1)
    #
    # b, a = scipy.signal.butter(2, .1 / (217 / 2), btype='highpass')
    # b, a = scipy.signal.butter(2, .1 / (217 / 2), btype='lowpass')
    #
    # filtered_mag_diff = np.diff(scipy.signal.filtfilt(b, a, np.abs(data), axis=0), axis=0)
    # mag_sig = np.linalg.norm(filtered_mag_diff, axis=1)
    # # opti_distance = scipy.signal.filtfilt(b, a, opti_distance)
    # # mag_distance2 = scipy.signal.filtfilt(b, a, mag_distance2)
    # x = (opti_distance - np.mean(opti_distance)) / np.std(opti_distance)
    # y = (mag_distance2 - np.mean(mag_distance2)) / np.std(mag_distance2)
    #
    # # from scipy.spatial.distance import euclidean
    # #
    # # from fastdtw import fastdtw
    # #
    # # distance, path = fastdtw(x, y, dist=lambda a, b: np.sum(1 - np.sign(a) * np.sign(b)))
    # # print(distance)
    # # px, py = zip(*path)
    #
    # xs = np.array_split(x, 100)
    # ys = np.array_split(y, 100)
    # cs = []
    # for i in range(len(xs)):
    #     c = scipy.signal.correlate(xs[i], ys[i])
    #     cs.append(np.argmax(c))
    #
    # plt.figure()
    # plt.plot(cs)
    #
    # plt.figure()
    # plt.plot(data)
    # plt.figure()
    # plt.plot(x)
    # plt.plot(y)
    #
    # #
    # # plt.figure()
    # # plt.plot(px, py)
    # # plt.figure()
    # # plt.figure()
    # # plt.plot(x[np.array(px)])
    # # plt.plot(y[np.array(py)])


def transform_pos(mag_data, pos, rot):
    field_obs1 = mag_data[:, [0, 2, 1]]
    field1 = compute_sensor(get_field(pos * 1000, rot, state.coil1), state.coil1)
    plot_correlation(field1, field_obs1)
    return (field_obs1 - np.mean(field_obs1, axis=0)) / np.std(field_obs1, axis=0), (
                field1 - np.mean(field_obs1, axis=0)) / np.std(field_obs1, axis=0)
    # field_obs1 = mag_data[:, [0, 4, 2]]
    # field_obs2 = mag_data[:, [1, 5, 3]]
    # field_obs3 = mag_data[:, [6, 8, 7]]
    # field1 = compute_sensor(get_field(pos * 1000, rot, state.coil1), state.coil1)
    # field2 = compute_sensor(get_field(pos * 1000, rot, state.coil2), state.coil2)
    # field3 = compute_sensor(get_field(pos * 1000, rot, state.coil3), state.coil3)
    # field_obs = np.hstack((field_obs1, field_obs2, field_obs3))
    # field = np.hstack((field1, field2, field3))
    # plot_correlation(field, field_obs)
    # return (field_obs - np.mean(field_obs, axis=0)) / np.std(field_obs, axis=0), (
    #             field - np.mean(field_obs, axis=0)) / np.std(field_obs, axis=0)


#
# def find_shift(mag, opti):
#     plt.figure()
#     plt.plot(mag)
#     plt.plot(opti)
#
#     corr = scipy.signal.correlate(mag, opti)
#     print(np.argmax(corr))
#
#
#     plt.figure()
#     plt.plot(mag)
#     plt.plot(opti)
#     plt.show()


def remove_bad_data(mag_interp, pos, rot):
    diff = np.diff(mag_interp[:, 2], axis=-1)
    plt.figure()
    plt.plot(diff)
    index_to_delet = np.where(np.abs(diff) > 0.05)
    start_index = 0
    end_index = 0
    while end_index < len(index_to_delet[0]):
        end_index = start_index + 1
        while index_to_delet[0][end_index] - index_to_delet[0][end_index - 1] < 100:
            end_index += 1
            if end_index == len(index_to_delet[0]):
                break
        mag_interp[index_to_delet[0][start_index]:index_to_delet[0][end_index - 1]] = np.NaN
        pos[index_to_delet[0][start_index]:index_to_delet[0][end_index - 1]] = np.NaN
        rot[index_to_delet[0][start_index]:index_to_delet[0][end_index - 1]] = np.NaN
        start_index = end_index
    mag_interp = mag_interp[~np.isnan(mag_interp)]
    pos = pos[~np.isnan(pos)]
    rot = rot[~np.isnan(rot)]
    return mag_interp, pos, rot


def transform_pos_old(mag_data, pos, rot):
    opti_distance = np.linalg.norm(pos, axis=1)
    mag_distance = 5 / np.linalg.norm(mag_data, axis=1)
    opti_sig = (opti_distance - np.mean(opti_distance)) / np.std(opti_distance)
    mag_sig = (mag_distance - np.mean(mag_distance)) / np.std(mag_distance)

    return mag_sig, opti_sig


def resample_and_align(mag_data, pos, rot, markers, preprocessor, mag_rate=90.90, opti_rate=120, wrist_pos=None,
                       wrist_rot=None):
    # filter the raw file
    # b, a = scipy.signal.butter(8, 10 / (217/2))
    # b, a = scipy.signal.butter(8, 0.1)
    # data_filt = scipy.signal.filtfilt(b, a, np.cbrt(scipy.signal.filtfilt(b, a, mag_data, axis=0)), axis=0)

    # start_time = max(data_filt.time.values[0], opti_data.time.values[0])
    # stop_time = min(data_filt.time.values[-1], opti_data.time.values[-1])

    mag_data, pos, rot, markers, wrist_pos, wrist_rot = preprocessor(mag_data, pos, rot, markers, wrist_pos, wrist_rot)
    mag_data = np.clip(mag_data, 0, None)

    # filter mag_data
    b, a = scipy.signal.butter(5, 100 / (F_MAG / 2), btype='lowpass')
    mag_data_filt = scipy.signal.filtfilt(b, a, mag_data, axis=0)

    # t_mag_est = np.linspace(0, len(mag_data), len(mag_data))
    # t_rot_est = np.linspace(0, len(mag_data), len(rot))
    # plt.figure()
    # plt.plot(t_mag_est, mag_sig)
    # # plt.figure()
    # plt.plot(t_rot_est, opti_sig)
    # # plt.show()

    pos_filt = scipy.signal.filtfilt(b, a, pos, axis=0)
    rot_filt = scipy.signal.filtfilt(b, a, rot, axis=0)
    rot_filt = (rot_filt.T / np.linalg.norm(rot_filt, axis=1)).T

    pos = pos_filt
    rot = rot_filt

    print(f"Downsample factor: {len(mag_data)} / {len(pos)} = {len(mag_data) / len(pos)}")

    # downsample magnetic data to roughly the optitrack rate (240 Hz)
    f_mag_interp = scipy.interpolate.interp1d(list(range(len(mag_data))), mag_data_filt, axis=0, assume_sorted=True,
                                              fill_value='extrapolate')
    mag_interp = f_mag_interp(np.linspace(0, len(mag_data) - 1, len(pos)))

    # mag_interp, pos, rot = remove_bad_data(mag_interp, pos, rot)
    # mag_sig, opti_sig = transform_pos_old(mag_interp, pos, rot)
    mag_sig, opti_sig = transform_pos_old(mag_interp, pos, rot)



    # plt.figure()
    # plt.plot(mag_data)
    plt.figure()
    plt.plot(mag_sig, label="mag")
    plt.plot(opti_sig, label="opti")
    # plt.plot(mag_sig[:, 0:3], label="mag")
    # plt.plot(opti_sig[:, 0:3], label="opti")
    plt.legend()
    # plt.show()

    WINDOW = 1000
    PADDING = 30
    SKIP = 25

    def test_alignment(start_pos):
        corr = scipy.signal.correlate(mag_sig[start_pos:start_pos + WINDOW],
                                      opti_sig[start_pos - PADDING:start_pos + WINDOW + PADDING], 'valid')
        return np.argmax(corr) - PADDING, corr

    print("Start alignment: ", test_alignment(PADDING)[0])
    alignments = []
    all_corr = []
    sample_x = range(PADDING, len(mag_sig) - WINDOW - PADDING * 2, SKIP)
    for i in sample_x:
        peak, corr = test_alignment(i)
        alignments.append(peak)
        all_corr.append(corr)
    plt.figure()
    plt.plot(sample_x, alignments)
    all_corr = np.array(all_corr)
    print(all_corr.shape)
    # plt.figure()
    # plt.imshow(all_corr)
    plt.show()
    print("End alignment: ", test_alignment(-WINDOW - PADDING - 1)[0])

    if USE_SECOND_STAGE_ALIGNMENT:
        b, a = scipy.signal.butter(5, .025 / (F_MOCAP / SKIP / 2), btype='lowpass')

        filtered_alignments = scipy.signal.filtfilt(b, a, alignments)
        # plt.plot(filtered_alignments)

        f_drift_interp = scipy.interpolate.interp1d(np.linspace(0, len(pos) - 1, len(filtered_alignments)),
                                                    filtered_alignments, axis=0, assume_sorted=True,
                                                    fill_value='extrapolate')
        drift_interp = f_drift_interp(np.linspace(0, len(pos) - 1, len(pos)))

        sns.set(context="paper", style="white", font="Lato")
        current_palette = sns.color_palette()
        # sns.palplot(current_palette)
        fig = plt.figure(figsize=(3.3, 2))
        ax = fig.add_subplot(111)
        t = np.linspace(0, len(mag_sig) / 90, len(alignments))
        ax.plot(t, alignments, label="Raw alignment", color=current_palette[0])
        ax.plot(t, filtered_alignments, label="Filtered alignment", color=current_palette[1])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative Alignment (samples)")
        plt.legend()
        fig.subplots_adjust(bottom=0.25, left=0.18)

        mag_interp = f_mag_interp(np.linspace(0, len(mag_data) - 1, len(pos)) + drift_interp * F_MAG / F_MOCAP)

        mag_sig, opti_sig = transform_pos(mag_interp, pos, rot)

        # plt.figure()
        # plt.plot(mag_data)
        plt.figure()
        # plt.plot(mag_sig[:, 1:3], label="mag")
        # plt.plot(opti_sig[:, 1:3], label="opti")

        plt.plot(mag_sig, label="mag")
        plt.plot(opti_sig, label="opti")
        plt.legend()

        # sns.set(context="paper", style="white", font="Lato")
        # fig = plt.figure(figsize=(3.3, 2))
        # ax = fig.add_subplot(111)
        # t = np.linspace(0, (20700-18000)/90, 20700-18000)
        # ax.plot(t, mag_sig[18000:20700], label="Aura")
        # ax.plot(t, opti_sig[18000:20700], label="Vicon")
        # ax.set_xlabel("Time (s)")
        # ax.set_ylabel("Distance approximation")
        # ax.get_yaxis().set_ticks([])
        # plt.legend()
        # fig.subplots_adjust(bottom=0.25, left=0.08)

        def test_alignment2(start_pos):
            corr = scipy.signal.correlate(mag_sig[start_pos:start_pos + WINDOW],
                                          opti_sig[start_pos - PADDING:start_pos + WINDOW + PADDING], 'valid')
            return np.argmax(corr) - PADDING, corr

        alignments = []
        all_corr = []
        for i in sample_x:
            peak, corr = test_alignment2(i)
            alignments.append(peak)
            # if peak > 50:
            #     break
            all_corr.append(corr)
        all_corr = np.array(all_corr)
        plt.figure()
        plt.plot(sample_x, alignments)

        plt.show()

    return mag_interp, pos, rot, markers.values, wrist_pos, wrist_rot

    # filter out bad opti data
    # error = np.linalg.norm(mag_sig - opti_sig, axis=1)
    # is_bad = error > 4
    # error_map = scipy.ndimage.binary_dilation(is_bad, [1] * 50)
    #
    # plt.figure()
    # plt.plot(np.linalg.norm(mag_sig - opti_sig, axis=1))
    # plt.plot(error_map)
    #
    # plt.figure()
    # plt.plot(pos[~error_map, :])
    # plt.show()
    # # Note: mag_interp is filtered!
    # return mag_interp[~error_map, :], pos[~error_map, :], rot[~error_map, :], markers.values[~error_map, :], \
    #        wrist_pos[~error_map, :], wrist_rot[~error_map, :]


def preprocess_calibration(mag_data, pos, rot):
    mag_data = mag_data[15726 + 855:]
    pos = pos[:-220]
    rot = rot[:-220]
    return mag_data, pos, rot


def preprocess_alignment_t1(mag_data, pos, rot):
    start_opti = 1542754023.744472
    start_mag = 1542754012.751443

    mag_trim_samples = int((start_opti - start_mag) * (1 / 0.011))
    assert (mag_trim_samples > 0)
    mag_data = mag_data[mag_trim_samples + 60:, :]

    pos = pos[:-703]
    rot = rot[:-703]
    return mag_data, pos, rot


def preprocess_alignment(mag_data, pos, rot, markers, wrist_pos, wrist_rot):
    if TRIAL in ALIGNMENTS:
        align = ALIGNMENTS[TRIAL]

        if align[0] >= 0:
            mag_low = align[0]
            if len(align) >= 3:
                pos_low = align[2]
            else:
                pos_low = 0
        else:
            mag_low = 0
            pos_low = -align[0]

        if align[1] > 0:
            mag_high = len(mag_data)
            pos_high = -align[1]
        elif align[1] < 0:
            mag_high = align[1]
            pos_high = len(pos)
        else:
            mag_high = len(mag_data)
            pos_high = len(pos)

        return mag_data[mag_low:mag_high], pos[pos_low:pos_high], rot[pos_low:pos_high], markers[
                                                                                         pos_low:pos_high], wrist_pos[
                                                                                                            pos_low:pos_high], wrist_rot[
                                                                                                                               pos_low:pos_high]

    print("No alignment set")
    return mag_data, pos, rot, markers, wrist_pos, wrist_rot


SHORTEN = False


def interpolate_dropped_frames(data, frame_numbers):
    diffs = np.diff(frame_numbers, axis=0) % 256

    assert (np.max(diffs) == 1)
    assert (np.min(diffs) == 1)

    print("No mag frame drops!")


def main(trial):
    data, frame_numbers = load_raw_mag(trial)  # loads the magnetic file

    if frame_numbers is not None:
        interpolate_dropped_frames(data, frame_numbers)

    tracking_raw = load_extracted_opti_data(trial)  # loads the raw high speed opti file
    _, use_palm, use_markers, _ = load_opti_metadata(trial)

    if SHORTEN:
        data = data[:1000]  # shorten just for testing
        tracking_raw = tracking_raw.iloc[:1000]  # shorten just for testing

    # data = data[:-850]  # shorten just for testing
    # tracking_raw = tracking_raw.iloc[:-850]
    extra_markers = []
    if use_palm:
        extra_markers += PALM_OPTI_POS + [f"palm_{x}" for x in ROT_STANDARD]
    if use_markers:
        extra_markers += [f"m{i}" for i in range(54)]

    mag_data, pos_interp, rot_interp, markers_interp, wrist_pos, wrist_rot = resample_and_align(
        data, tracking_raw[POS], tracking_raw[ROT_STANDARD],
        tracking_raw[extra_markers],
        preprocess_alignment, wrist_pos=tracking_raw[WRIST_OPTI_POS].values, wrist_rot=tracking_raw[WRIST_ROT].values)

    print("Saving data...")
    mag_data = mag_data[0:-100]
    pos_interp = pos_interp[0:-100]
    rot_interp = rot_interp[0:-100]
    wrist_pos = wrist_pos[0:-100]
    markers_interp = markers_interp[0:-100]
    wrist_rot = wrist_rot[0:-100]

    # print(np.shape(mag_data))
    # print(np.shape(pos_interp))
    # print(np.shape(wrist_pos))
    # print(np.shape(markers_interp))

    save_resampled_data(mag_data, pos_interp, rot_interp, markers_interp, wrist_pos, wrist_rot, trial)
    plt.plot(mag_data)
    plt.figure()
    plt.plot(pos_interp)
    plt.show()


if __name__ == "__main__":
    main(TRIAL)
