import pyrealtime as prt
import serial
import math
import numpy as np
import struct
from scipy import signal

from controller_lib import get_device_data, DrawingPane, predict_pos, DrawingPane2, sp_extract_hand_coord, detect_touch
from opti_lib import get_opti_source

USE_NATNET = False
RECORD = True
USE_BOX = False
USE_PALM = False
DEMO = True
TRAIN_ON = "exp6"
CALIB_ON = "exp6"

# TRAIN_ON = "exp10"
# CALIB_ON = "exp11"

# MAKE SURE Y IS UP IN OPTITRACK


def main():
    # data = 1
    # adc = 1
    data, adc = get_device_data(show_plot=False)
    # prt.TimePlotLayer(data.get_port("count"), n_channels=1, window_size=1000, ylim=(0,50))
    # segmented = segment_data(data, show_plot=True)
    # if RECORD:
    #     prt.RecordLayer(data.get_port("raw_data"), file_prefix="mag-raw")

    if USE_NATNET:
        opti = get_opti_source(show_plot=False, use_box=USE_BOX, use_palm=USE_PALM)
        # rel_pos = sp_extract_hand_coord(opti)
        # fm = prt.FigureManager(fps=10000)
        # prt.TimePlotLayer(rel_pos, ylim=(-400, 400), n_channels=3, fig_manager=fm)
        # draw_opti = DrawingPane2(rel_pos, xlim=(-400, 300), ylim=(-200, 100),
        #                         buffer_size=10000, scroll_speed=-0.15)

    if RECORD:
        prt.RecordLayer(adc, file_prefix="mag")
        if USE_NATNET:
            # prt.PrintLayer(opti)
            prt.RecordLayer(opti, file_prefix="opti")
    if DEMO:
        pos = predict_pos(adc, train_on=TRAIN_ON, calib_on=CALIB_ON)
        pos = prt.ExponentialFilter(pos, alpha=0.2)
        # fm = prt.FigureManager(fps=10000)
        # prt.TimePlotLayer(pos, ylim=(-200, 200), n_channels=3, fig_manager=fm)
        touch, filtered, smoothed_energy = detect_touch(adc)
        draw_opti = DrawingPane2(pos, xlim=(-400, 0), ylim=(-200, 10),
                                buffer_size=10000, scroll_speed=-0.2)
        draw_opti.set_signal_in(touch)
        # pane = DrawingPane(pos, xlim=(-np.pi / 2, np.pi / 2), ylim=(-np.pi / 2, np.pi / 2), buffer_size=2000,
        #                    scroll_speed=-0.002)

    prt.LayerManager.session().run()


if __name__ == "__main__":
    main()

