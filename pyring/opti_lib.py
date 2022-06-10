from prt_natnet import NatNetLayer
import pyrealtime as prt
import numpy as np

WRIST_RIGID_BODY_NAME = b"wrist"
FINGER_RIGID_BODY_NAME = b"finger"
PALM_RIGID_BODY_NAME = b"palm"
BOX_RIGID_BODY_NAME = b"box"

@prt.transformer(multi_output=True)
def parse(data):
    return {'pos': np.array(data[0]), 'rot': np.array(data[1])}


def setup_fig(fig):
    ax1 = fig.add_subplot(421)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423)
    ax4 = fig.add_subplot(424)
    ax5 = fig.add_subplot(425)
    ax6 = fig.add_subplot(426)
    ax7 = fig.add_subplot(427)
    ax8 = fig.add_subplot(428)
    return {f'{WRIST_RIGID_BODY_NAME}_pos': ax1, f'{WRIST_RIGID_BODY_NAME}_rot': ax2, f'{FINGER_RIGID_BODY_NAME}_pos': ax3, f'{FINGER_RIGID_BODY_NAME}_rot': ax4,
            f'{PALM_RIGID_BODY_NAME}_pos': ax5, f'{PALM_RIGID_BODY_NAME}_rot': ax6, f'{BOX_RIGID_BODY_NAME}_pos': ax7, f'{BOX_RIGID_BODY_NAME}_rot': ax8}


def get_opti_source(show_plot=True, use_box=False, use_palm=False):
    bodies = [WRIST_RIGID_BODY_NAME, FINGER_RIGID_BODY_NAME]
    if use_palm:
        bodies += [PALM_RIGID_BODY_NAME]
    if use_box:
        bodies += [BOX_RIGID_BODY_NAME]

    natnet = NatNetLayer(bodies_to_track=bodies, multi_output=True, print_fps=False, track_markers=True)
    # prt.PrintLayer(parse(natnet.get_port(RIGID_BODY_NAME)))
    frame_num = natnet.get_port("frame_num")
    parsed_wrist = parse(natnet.get_port(WRIST_RIGID_BODY_NAME))
    parsed_finger = parse(natnet.get_port(FINGER_RIGID_BODY_NAME))
    if use_palm:
        parsed_palm = parse(natnet.get_port(PALM_RIGID_BODY_NAME))
    if use_box:
        parsed_box = parse(natnet.get_port(BOX_RIGID_BODY_NAME))
    markers = natnet.get_port('markers')
    if show_plot:
        fm = prt.FigureManager(setup_fig)
        # prt.TimePlotLayer(parsed_wrist.get_port('pos'), ylim=(-2,2), n_channels=3, plot_key=f'{WRIST_RIGID_BODY_NAME}_pos', fig_manager=fm)
        prt.TimePlotLayer(parsed_wrist.get_port('rot'), ylim=(-2,2), n_channels=4, plot_key=f'{WRIST_RIGID_BODY_NAME}_rot', fig_manager=fm)
        # prt.TimePlotLayer(parsed_finger.get_port('pos'), ylim=(-2,2), n_channels=3, plot_key=f'{FINGER_RIGID_BODY_NAME}_pos', fig_manager=fm)
        prt.TimePlotLayer(parsed_finger.get_port('rot'), ylim=(-2,2), n_channels=4, plot_key=f'{FINGER_RIGID_BODY_NAME}_rot', fig_manager=fm)
        if use_palm:
            # prt.TimePlotLayer(parsed_palm.get_port('pos'), ylim=(-2,2), n_channels=3, plot_key=f'{PALM_RIGID_BODY_NAME}_pos', fig_manager=fm)
            prt.TimePlotLayer(parsed_palm.get_port('rot'), ylim=(-2,2), n_channels=4, plot_key=f'{PALM_RIGID_BODY_NAME}_rot', fig_manager=fm)
        if use_box:
            # prt.TimePlotLayer(parsed_box.get_port('pos'), ylim=(-2,2), n_channels=3, plot_key=f'{BOX_RIGID_BODY_NAME}_pos', fig_manager=fm)
            prt.TimePlotLayer(parsed_box.get_port('rot'), ylim=(-2,2), n_channels=4, plot_key=f'{BOX_RIGID_BODY_NAME}_rot', fig_manager=fm)

    data = prt.MergeLayer(None)
    data.set_input(frame_num, "frame_num")
    data.set_input(parsed_wrist, "wrist")
    data.set_input(parsed_finger, "finger")
    if use_palm:
        data.set_input(parsed_palm, "palm")
    if use_box:
        data.set_input(parsed_box, "box")
    data.set_input(markers, "markers")
    # prt.PrintLayer(data)
    return data
