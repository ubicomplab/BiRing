import os
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from preprocessing.optiTrackMarkers import get_opti_marker
from settings import DATA_ROOT
from utils import MAG_RAW_NAMES, load_opti_data, \
    save_extracted_opti_data, progress, FINGER_OPTI_POS, WRIST_OPTI_POS, FINGER_OPTI, WRIST_OPTI, FINGER_OPTI_ROT, \
    WRIST_OPTI_ROT, PALM_OPTI_POS, WRIST_ROT, FINGER_ROT, PALM_OPTI_ROT

TRIAL = "par8_1"


def main():
    df_opti = extract_coordinates_opti(TRIAL)
    save_extracted_opti_data(df_opti, TRIAL)


def fix_frame(vicon_data, start_frame):
    start = vicon_data.index[start_frame]
    end = vicon_data.index[start_frame + 1]
    assert (vicon_data.head_frame.iloc[start_frame] == vicon_data.hand_frame.iloc[start_frame])
    assert (vicon_data.head_frame.iloc[start_frame + 1] == vicon_data.hand_frame.iloc[start_frame + 1])
    missing_frames = np.linspace(start + 1, end - 1, end - start - 1).astype('int')
    for missing_frame in missing_frames:
        vicon_data.loc[missing_frame] = None
    return vicon_data


def fix_frame_opti(opti_data, start_frame, use_palm, use_markers):
    start = opti_data.index[start_frame]
    end = opti_data.index[start_frame + 1]
    # assert (opti_data.head_frame.iloc[start_frame] == opti_data.hand_frame.iloc[start_frame])
    # assert (opti_data.head_frame.iloc[start_frame + 1] == opti_data.hand_frame.iloc[start_frame + 1])
    missing_frames = np.linspace(start + 1, end - 1, end - start - 1).astype('int')
    for missing_frame in missing_frames:
        opti_data.loc[missing_frame] = None
        if use_markers:
            opti_data.at[missing_frame, 'marker_finger'] = opti_data.at[start, 'marker_finger']
            opti_data.at[missing_frame, 'marker_wrist'] = opti_data.at[start, 'marker_wrist']
            if use_palm:
                opti_data.at[missing_frame, 'marker_palm'] = opti_data.at[start, 'marker_palm']
    return opti_data


def handle_missing_frames_opti(opti_data, use_palm, use_markers):
    print("Interpolating missing opti frames")

    # first let's deal with frame drops (probably from dropped UDP packets)
    opti_data = opti_data.set_index('frame_wrist')
    duplicate_frames = np.where(np.diff(opti_data.index.values) == 0)[0]
    # print(len(opti_data))
    opti_data.drop(opti_data.index[duplicate_frames], inplace=True)
    # print(len(opti_data))

    assert (np.sum(np.diff(opti_data.index.values) == 0) == 0)  # this won't work if we have duplicated frame numbers
    frames_to_fix, = np.where(np.diff(opti_data.index.values) != 1)
    # print(frames_to_fix)
    for frame in frames_to_fix:
        opti_data = fix_frame_opti(opti_data, frame, use_palm, use_markers)
    progress(1)
    opti_data = opti_data.sort_index()
    opti_data.interpolate(inplace=True)
    print(f"Interpolated {len(frames_to_fix)} missing frames")
    return opti_data


def extract_coordinates_opti(key):
    print("processing opti file")
    opti_data, use_palm, use_markers, rigid_body_calib = load_opti_data(key)
    print("done loading opti file")

    opti_data = handle_missing_frames_opti(opti_data, use_palm, use_markers)

    finger_markers, wrist_markers, _ = get_opti_marker('exp1')
    wrist_pos = opti_data[WRIST_OPTI_POS].iloc[0].values
    finger_pos = opti_data[FINGER_OPTI_POS].iloc[0].values
    wrist_rot = R.from_quat(opti_data[WRIST_OPTI_ROT].iloc[0].values)
    finger_rot = R.from_quat(opti_data[FINGER_OPTI_ROT].iloc[0].values)
    wrist_markers_adj = wrist_pos + wrist_rot.apply(wrist_markers)
    finger_markers_adj = finger_pos + finger_rot.apply(finger_markers)

    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.scatter(wrist_markers_adj[:, 0], wrist_markers_adj[:, 1], wrist_markers_adj[:, 2])
    ax.scatter(finger_markers_adj[:, 0], finger_markers_adj[:, 1], finger_markers_adj[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    opti_data = adjust_rigid_bodies(opti_data, f"calib_{rigid_body_calib}.pkl")

    pos_opti, rot_opti, palm_pos, palm_rot, wrist_pos, wrist_q, markers, is_valid = sp_extract_ring_coord(opti_data,
                                                                                                          use_palm,
                                                                                                          use_markers)

    plt.figure()
    plt.plot(opti_data[FINGER_OPTI_POS].values)
    # plt.figure()
    plt.plot(opti_data[WRIST_OPTI_POS].values)

    plt.figure()
    plt.plot(np.linalg.norm(pos_opti, axis=1))
    plt.figure()
    plt.plot(pos_opti)
    plt.figure()
    plt.plot(rot_opti)
    plt.show()
    return package_data_opti(opti_data, is_valid, pos_opti, rot_opti, palm_pos, palm_rot, wrist_pos, wrist_q, markers,
                             use_palm, use_markers)


def adjust_bodies(body_p, body_q, r, t):
    print("Adjusting rigid bodies")
    np_body_q = R.from_quat(body_q.values[:, [1, 2, 3, 0]])
    np_r = R.from_dcm(r)
    adjust_q = np_body_q * np_r.inv()
    vicon_cad_origin = -np.matmul(r.T, t).T
    adjust_p = body_p + np_body_q.apply(vicon_cad_origin)

    return adjust_p, R.as_quat(adjust_q)[:, [3, 0, 1, 2]]


def adjust_rigid_bodies(vicon_data, rigid_body_calib):
    pickle_data = pickle.load(open(os.path.join(DATA_ROOT, 'rigid_body_calibration', rigid_body_calib), 'rb'))
    print(pickle_data)
    # pickle_data['wrist_q'] = np.eye(3)
    wrist_p, wrist_q = adjust_bodies(vicon_data[WRIST_OPTI_POS].values, vicon_data[WRIST_ROT],
                                     pickle_data['wrist_r'], pickle_data['wrist_t'])
    finger_p, finger_q = adjust_bodies(vicon_data[FINGER_OPTI_POS].values, vicon_data[FINGER_ROT],
                                       pickle_data['finger_r'], pickle_data['finger_t'])

    vicon_data[FINGER_OPTI_POS] = finger_p
    vicon_data[FINGER_ROT] = finger_q
    vicon_data[WRIST_OPTI_POS] = wrist_p
    vicon_data[WRIST_ROT] = wrist_q
    return vicon_data


def put_in_wrist_frame(pos, q, wrist_pos, wrist_q):
    rel_pos_wrist_world = pos - wrist_pos
    rel_pos_wrist = wrist_q.apply(rel_pos_wrist_world)
    if q is not None:
        q_finger_to_wrist = R.as_quat(q * wrist_q.inv())[:, [3, 0, 1, 2]]
    else:
        q_finger_to_wrist = None

    return rel_pos_wrist, q_finger_to_wrist


def put_marker_in_wrist_frame(pos, wrist_pos, wrist_q):
    rel_pos_wrist_world = pos.transpose((1, 0, 2)) - wrist_pos
    rel_pos_wrist = np.zeros(pos.shape)
    for i in range(pos.shape[1]):
        rel_pos_wrist[:, i, :] = wrist_q.apply(rel_pos_wrist_world[i, :, :])

    return rel_pos_wrist


def sp_extract_ring_coord(data, use_palm, use_markers):
    finger_pos = data[FINGER_OPTI_POS].values
    finger_q = R.from_quat(data[FINGER_OPTI_ROT].values).inv()  # using opti_rot to put in x, y, z, w order
    wrist_pos = data[WRIST_OPTI_POS].values
    wrist_q = R.from_quat(data[WRIST_OPTI_ROT].values).inv()  # using opti_rot to put in x, y, z, w order

    finger_rel_wrist_pos, finger_rel_wrist_q = put_in_wrist_frame(finger_pos, finger_q, wrist_pos, wrist_q)

    if use_palm:
        palm_pos = data[PALM_OPTI_POS].values
        palm_q = R.from_quat(data[PALM_OPTI_ROT].values).inv()  # using opti_rot to put in x, y, z, w order
        palm_rel_wrist_pos, palm_rel_wrist_q = put_in_wrist_frame(palm_pos, palm_q, wrist_pos, wrist_q)
    else:
        palm_rel_wrist_pos = None
        palm_rel_wrist_q = None

    if use_markers:
        all_markers = np.zeros((len(data), (3 * 6) * 3))
        wrist_markers = np.array(list(data.marker_wrist.values))  # N x 6 x 3
        wrist_markers_rel_wrist = np.reshape(put_marker_in_wrist_frame(wrist_markers, wrist_pos, wrist_q),
                                             (len(data), -1))
        all_markers[:, 3 * 6 * 0: 3 * 6 * 0 + wrist_markers_rel_wrist.shape[1]] = wrist_markers_rel_wrist

        finger_markers = np.array(list(data.marker_finger.values))  # N x 6 x 3
        finger_markers_rel_wrist = np.reshape(put_marker_in_wrist_frame(finger_markers, wrist_pos, wrist_q),
                                              (len(data), -1))
        all_markers[:, 3 * 6 * 1: 3 * 6 * 1 + finger_markers_rel_wrist.shape[1]] = finger_markers_rel_wrist

        if use_palm:
            palm_markers = np.array(list(data.marker_palm.values))  # N x 6 x 3
            palm_markers_rel_wrist = np.reshape(put_marker_in_wrist_frame(palm_markers, wrist_pos, wrist_q),
                                                (len(data), -1))
            all_markers[:, 3 * 6 * 2: 3 * 6 * 2 + palm_markers_rel_wrist.shape[1]] = palm_markers_rel_wrist
    else:
        all_markers = None

    finger_tracking_lost = np.all(data[FINGER_OPTI][1:].values == data[FINGER_OPTI][0:-1].values, axis=1)
    wrist_tracking_lost = np.all(data[WRIST_OPTI][1:].values == data[WRIST_OPTI][0:-1].values, axis=1)
    tracking_lost = np.logical_or(finger_tracking_lost, wrist_tracking_lost)
    is_valid = np.insert(True, 0, np.logical_not(tracking_lost))
    print(sum(is_valid) / len(data))

    return finger_rel_wrist_pos, finger_rel_wrist_q, palm_rel_wrist_pos, palm_rel_wrist_q, wrist_pos, \
           wrist_q.as_quat()[:, [3, 0, 1, 2]], all_markers, is_valid


def package_data(data, pos, rot):
    mag_data = data[MAG_RAW_NAMES]
    pos_data = [pos.apply(lambda x: x[i]) for i in range(3)]
    rot_data = [rot.apply(lambda x: x[i]) for i in range(4)]
    concat = pd.concat([mag_data] + pos_data + rot_data, axis=1)
    cols = list(concat.columns)
    labels = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
    for i in range(9, len(cols)):
        cols[i] = labels[i - 9]
    concat.columns = cols

    # concat[MAG_NAMES3] = np.cbrt(concat[MAG_RAW_NAMES])
    return concat


def package_data_opti(opti_data, is_valid, pos, rot, palm_pos, palm_rot, wrist_pos, wrist_q, markers, use_palm,
                      use_markers):
    if use_palm:
        data = np.concatenate((pos, rot, palm_pos, palm_rot, wrist_pos, wrist_q, is_valid.reshape((-1, 1))), axis=1)
        labels = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'palm_x', 'palm_y', 'palm_z', 'palm_qw', 'palm_qx', 'palm_qy',
                  'palm_qz', 'wrist_x', 'wrist_y', 'wrist_z', 'wrist_qw', 'wrist_qx', 'wrist_qy', 'wrist_qz',
                  'is_valid']
    else:
        data = np.concatenate((pos, rot, wrist_pos, wrist_q, is_valid.reshape((-1, 1))), axis=1)
        labels = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'wrist_x', 'wrist_y', 'wrist_z', 'wrist_qw', 'wrist_qx',
                  'wrist_qy', 'wrist_qz', 'is_valid']

    if use_markers:
        data = np.concatenate((data, markers), axis=1)
        labels += [f"m{i}" for i in range(markers.shape[1])]

    concat = pd.DataFrame(data=data, columns=labels)
    return concat


if __name__ == "__main__":
    main()
