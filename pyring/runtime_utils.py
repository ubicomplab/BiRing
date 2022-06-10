import struct
import numpy as np
import pyrealtime as prt
from scipy.spatial.transform import Rotation as R


VIVE_BASE_OFFSET_ROT = R.from_quat([0,  0.7072, -0.7072, 0])
VIVE_OFFSET_POS = np.array([-0.23022142, 0.84369332, 0.07624312])
VIVE_OFFSET_ROT = R.from_quat([0.0253895,  0.74820981, 0.01221999, 0.66286358])


def encode_udp_fingertip_pos(data):
    return struct.pack("f" * 2, data[0], data[1])


def encode_udp_pos_tap(data):
    return  struct.pack("f" * 4, data['default'][0], data['default'][1], data['default'][2], data['touch'][0])
    # if data['touch']:
    #     return struct.pack("f" * 4, data['default'][0], data['default'][1], data['default'][2], 1.0)
    # else:
    #     return struct.pack("f" * 4, data['default'][0], data['default'][1], data['default'][2], 0.0)


def encode_udp(data):
    return struct.pack("f"*9, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8])


def encode_udp_joints(filtered_joints):

    vive_pos = [0, 0, 0]
    vive_rot = [0, 0, 0, 1]

    return struct.pack("f"*11, filtered_joints[0], filtered_joints[1], filtered_joints[2], filtered_joints[3],
                       vive_pos[0], vive_pos[1], vive_pos[2],
                       vive_rot[0], vive_rot[1], vive_rot[2], vive_rot[3])

def encode_udp_ik_pose(merged):
    pose = merged["pose"]
    filtered_joints = merged["filtered_joints"]
    pos = pose["pos"]
    rot = pose["rot"]
    ik_ring_pos = pose["ik_ring_pos"]
    ik_ring_rot = pose["ik_ring_rot"]
    ik_knuckle_pos = pose["ik_knuckle_pos"]
    ik_knuckle_rot = pose["ik_knuckle_rot"]

    return struct.pack("f"*25, filtered_joints[0], filtered_joints[1], filtered_joints[2], filtered_joints[3],
                       pos[0], pos[1], pos[2],
                       rot[0], rot[1], rot[2], rot[3],
                       ik_ring_pos[0], ik_ring_pos[1], ik_ring_pos[2],
                       ik_ring_rot[0], ik_ring_rot[1], ik_ring_rot[2], ik_ring_rot[3],
                       ik_knuckle_pos[0], ik_knuckle_pos[1], ik_knuckle_pos[2],
                       ik_knuckle_rot[0], ik_knuckle_rot[1], ik_knuckle_rot[2], ik_knuckle_rot[3])
# def encode_udp_joints(joints):
#     return struct.pack("f"*4, joints[0], joints[1], joints[2], joints[3])

def encode_udp_joints_all(data):
    return data


def decode_udp(data):
    # x, y, z, qx, qy, qz, qw = struct.unpack("ddddddd", data)
    x, y, z, qx, qy, qz, qw, wrist_theta, wrist_phi, finger_theta, finger_phi, rx, ry, rz, rqx, rqy, rqz, rqw, kx, ky, kz, kqx, kqy, kqz, kqw = struct.unpack("ddddddddddddddddddddddddd", data)
    # NOTE: SENDING AS X Y Z W
    raw = struct.pack("fffffffffffffffffffffffff", x, y, z, qx, qy, qz, qw, wrist_theta, wrist_phi, finger_theta, finger_phi, rx, ry, rz, rqx, rqy, rqz, rqw, kx, ky, kz, kqx, kqy, kqz, kqw)
    return {"pos": np.array([x, y, z]), "rot": np.array([qw, qx, qy, qz]),
            "joints": np.array([wrist_theta, wrist_phi, finger_theta, finger_phi])*180/np.pi,
            "ik_ring_pos": np.array([rx, ry, rz]),
            "ik_ring_rot": np.array([rqw, rqx, rqy, rqz]),
            "ik_knuckle_pos": np.array([kx, ky, kz]),
            "ik_knuckle_rot": np.array([kqw, kqx, kqy, kqz]),
            "raw": raw}


@prt.transformer
def get_first(data):
    return data[0,:]

@prt.transformer
def get_last(data):
    return data[-1,:]


def solve_for_pose(data_adc):
    prt.UDPWriteLayer(data_adc, port=8888, encoder=encode_udp)
    pose = prt.UDPReadLayer(port=9884, decoder=decode_udp, multi_output=True)

    filtered_joints = prt.ExponentialFilter(pose.get_port('joints'), alpha=.2)
    # prt.PrintLayer(pose)
    # prt.TimePlotLayer(pose.get_port("joints"), n_channels=4, ylim=(-180, 180))
    return pose, filtered_joints


def send_fingertip_pos_to_unity(pos):
    prt.UDPWriteLayer(pos, port=9885, encoder=encode_udp_fingertip_pos)


def send_fingertip_pos_and_tap_to_unity(data):
    prt.UDPWriteLayer(data, port=9885, encoder=encode_udp_pos_tap)


def send_pose_and_joints_to_unity(pose, filtered_joints):
    merged = prt.MergeLayer(None, trigger=prt.LayerTrigger.SLOWEST, discard_old=True)
    merged.set_input(pose, "pose")
    merged.set_input(filtered_joints, "filtered_joints")
    prt.UDPWriteLayer(merged, port=9886, encoder=encode_udp_ik_pose)


def get_vive_tracker():
    from vive_tracker_prt import ViveTracker
    vive = ViveTracker("tracker_1", base_offset_rot=VIVE_BASE_OFFSET_ROT, offset_pos=VIVE_OFFSET_POS,
                       offset_rot=VIVE_OFFSET_ROT)
    return vive

def sync_with_vive(pose):
    vive = get_vive_tracker()
    merged = prt.MergeLayer(None, trigger=prt.LayerTrigger.SLOWEST, trigger_source="pose", discard_old=True)
    merged.set_input(pose, "pose")
    merged.set_input(vive, "vive")
    return merged