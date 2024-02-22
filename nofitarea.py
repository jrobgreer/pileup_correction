import spectranalysis as sp
import getwave as gw
from itertools import chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pulse_collection, pulse_timestamps = gw.get_pulse_collection(
    'cs137co60calib.dat', baseline=0.1)

areas = np.zeros(int(0.5*len(pulse_collection)))
for pulse_idx in range(int(0.5*len(pulse_collection))):

    if pulse_idx % 1000 == 0:
        print(pulse_idx)
    pulse = sp.Pulse(pulse_collection, pulse_timestamps, pulse_idx)
    pulse.get_peaks2(min_dist_between_peaks=20, gradient_threshold=16)
    pulse.raw_int()
    areas[pulse_idx] = pulse.rawint

breakpoint()
plt.hist(areas, bins=1000)
plt.show()
