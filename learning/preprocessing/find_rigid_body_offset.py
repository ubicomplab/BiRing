import os
import pickle

import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from preprocessing.optiTrackMarkers import get_opti_marker
from settings import DATA_ROOT
from utils import ensure_dir

TRIAL = 'exp9'

# if "triband" in TRIAL:
# FINGER_MARKERS = np.array([[0, 18.75, -8.5],
#                         [-36.398, 17.855, -8.5],
#                         [-15.008, 5.506, -25.9],
#                         [-16.1, -18.196, -8.5]])/1000

FINGER_MARKERS = np.array([
                        [15.8737, 6.0059, 24.401],
                        [0, 19.751, 8.5],
                        [16.050, -18.1527, 8.5],
                        [37.263, 18.355, 8.5]])/1000

# WRIST_MARKERS = np.array([#[-36.815, 10.4399, -.09581],
#                           [-9.7, 12.9399, 2.5985],
#                           [9.919, 12.9399, 15.7119],
#                           [36.815, 10.4399, 22.136],
#                           [-36.815, 24.199, 22.136],
#                           [36.815, 24.199, -.0958]])/1000

# WRIST_MARKERS = np.array([#[-36.815, 10.4399, -.09581],
#                           [-9.7, 12.9399, 2.5985],
#                           [9.919, 12.9399, 15.7119],
#                           [36.815, 10.4399, 22.136],
#                           [-36.815, 24.199, 22.136]])/1000

#
WRIST_MARKERS = np.array([[42.24, -14.553, -8.731],
                          [26.3, 30.223, -7.95],
                          [26.3, 3.41, -18.88],
                          [26.3, -33.408, -24.686]])/1000
#     FINGER_MARKERS = np.array([[0, 19.001, -8.5],
#                                [-38.1298, 18.855, -10],
#                                [-15.008, 5.506, -26.15],
#                                [-16.277, -18.373, -8.5]]) / 1000
#
#     WRIST_MARKERS = np.array([[-36.815, 10.4399, -.09581],
#                               [-9.7, 12.9399, 2.5985],
#                               [9.919, 12.9399, 15.7119],
#                               [36.815, 10.4399, 22.136],
#                               [-36.815, 26.699, 22.136],
#                               [36.815, 26.699, -.0958]]) / 1000


def find_correspondences(vicon_markers, cad_markers):
    NUM_MARKERS = cad_markers.shape[0]
    all_distances_vicon = np.zeros((NUM_MARKERS, NUM_MARKERS))
    all_distances_cad = np.zeros((NUM_MARKERS, NUM_MARKERS))
    for i in range(NUM_MARKERS):
        distances = np.linalg.norm(vicon_markers - vicon_markers[i,:], axis=1)
        all_distances_vicon[i, :] = distances

        distances = np.linalg.norm(cad_markers - cad_markers[i,:], axis=1)
        all_distances_cad[i, :] = distances

    vicon_markers_reordered = []
    select_min = []
    for i in range(NUM_MARKERS):
        target_distances = sorted(all_distances_cad[i, :])
        errors = []
        for vicon_i in range(NUM_MARKERS):
            vicon_distance = np.sort(all_distances_vicon[vicon_i, :])
            error = np.linalg.norm(target_distances - vicon_distance)
            errors.append(error)
        if np.argmin(errors) in select_min:
            errors[np.argmin(errors)] *= 1000
            print("LOOK")
        select_min.append(np.argmin(errors))
        vicon_markers_reordered.append(vicon_markers[np.argmin(errors), :])
    vicon_markers_reordered = np.array(vicon_markers_reordered)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cad_markers[:, 0], cad_markers[:, 1], cad_markers[:, 2], alpha=1, c=range(NUM_MARKERS), label="CAD markers")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-.1, .1)
    ax.set_zlim(-.1, .1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.scatter(vicon_markers_reordered[:, 0], vicon_markers_reordered[:, 1], vicon_markers_reordered[:, 2],
               c=range(NUM_MARKERS), alpha=1, marker='^', label="optical markers")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Correspondences")
    plt.legend()
    plt.show()

    return vicon_markers_reordered


def find_transform(vicon_markers, cad_markers):  # https://nghiaho.com/?page_id=671

    vicon_centroid = np.mean(vicon_markers, axis=0)
    cad_centroid = np.mean(cad_markers, axis=0)
    vicon_centered = vicon_markers - vicon_centroid
    cad_centered = cad_markers - cad_centroid

    H = np.matmul(vicon_centered.T, cad_centered)
    u, s, vt = np.linalg.svd(H)
    r = np.matmul(vt.T, u.T)
    # remove reflection
    if np.linalg.det(r) < 0:
        vt[2, :] *= -1
        r = u * vt
        # r[:, 2] *= -1
    t = np.matmul(-r, vicon_centroid) + cad_centroid

    cad_origin = np.array([0,0,0])
    vicon_cad_origin = -np.matmul(r.T, t).T
    print(vicon_cad_origin)

    # cad_trans = np.matmul(r, vicon_markers.T).T + t
    cad_trans = np.matmul(r.T, (cad_markers - t).T).T
    vicon_trans = np.matmul(r, vicon_markers.T).T + t

    error = np.sum(np.linalg.norm(np.matmul(r, vicon_centered.T).T - cad_centered) ** 2)
    error_base = np.sum(np.linalg.norm(vicon_centered - cad_centered) ** 2)
    print("Error:", error*1000, error_base*1000)

    print(cad_trans)
    print(cad_markers)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vicon_markers[:, 0], vicon_markers[:, 1], vicon_markers[:, 2],
               c=list(range(cad_markers.shape[0])), alpha=1, marker='^')
    ax.scatter(cad_trans[:, 0], cad_trans[:, 1], cad_trans[:, 2], c=list(range(cad_markers.shape[0])), alpha=1,
               marker='.')
    ax.set_xlim(-.1, .1)
    ax.set_ylim(-.1, .1)
    ax.set_zlim(-.1, .1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("After transformation")
    plt.show()

    return r, t


def adjust_body(body_p, body_q, r, t):
    try:
        body_q = Quaternion(a=body_q.wrist_qw, b=body_q.wrist_qx, c=body_q.wrist_qy, d=body_q.wrist_qz)
    except:
        body_q = Quaternion(a=body_q.finger_qw, b=body_q.finger_qx, c=body_q.finger_qy, d=body_q.finger_qz)
    # adjust_p = body_p + body_q.rotate(t)
    rq = Quaternion(matrix=r)
    # adjust_q = rq * body_q
    adjust_q = body_q * rq.conjugate

    vicon_cad_origin = -np.matmul(r.T, t).T
    adjust_p = body_p + body_q.rotate(vicon_cad_origin)
    # adjust_p = np.matmul(r, body_p) + t
    # adjust_p = r.T * body_p + t

    return adjust_p, adjust_q.elements


def plot_q(ax, origin, q, size=10):
    q = Quaternion(q)
    x = q.rotate([size,0,0]) + origin
    y = q.rotate([0,size,0]) + origin
    z = q.rotate([0,0,size]) + origin
    ax.plot([origin[0], x[0]], [origin[1], x[1]], [origin[2], x[2]], color='r')
    ax.plot([origin[0], y[0]], [origin[1], y[1]], [origin[2], y[2]], color='g')
    ax.plot([origin[0], z[0]], [origin[1], z[1]], [origin[2], z[2]], color='b')


def main():
    opti_finger, opti_wrist, opti_palm = get_opti_marker(TRIAL)
    opti_reordered_finger_markers = find_correspondences(opti_finger, FINGER_MARKERS)
    finger_r, finger_t = find_transform(opti_reordered_finger_markers, FINGER_MARKERS)
    # avg_markers = find_markers_for_body(markers, head_pos, head_q, WRIST_MARKERS)
    opti_reordered_wrist_markers = find_correspondences(opti_wrist, WRIST_MARKERS)
    wrist_r, wrist_t = find_transform(opti_reordered_wrist_markers, WRIST_MARKERS)

    data = {'wrist_r': wrist_r, 'wrist_t': wrist_t, 'finger_r': finger_r, 'finger_t': finger_t}
    base_name = os.path.splitext(os.path.basename(TRIAL))[0]
    pickle.dump(data, open(ensure_dir(os.path.join(DATA_ROOT, 'rigid_body_calibration', f'calib_{base_name}.pkl')), 'wb'))
    plt.show()


if __name__ == "__main__":
    main()