from gimmedatwave.gimmedatwave import gimmedatwave as gdw
import matplotlib.pyplot as plt
import numpy as np

# TODO improve this, it's not necessary for multiprocessed code to load in ALL the data, only that which it will index


def get_pulse_collection(file, baseline, digitizer_family=gdw.DigitizerFamily.X725, fraction_of_dataset=1):

    parser = gdw.Parser(file, digitizer_family=digitizer_family)
    num_of_entries = int(fraction_of_dataset*parser.n_entries)-1
    # print(num_of_entries)

    pulse_collection = np.empty((num_of_entries, 1030))
    pulse_timestamps = np.empty(num_of_entries)

    for i in range(num_of_entries):

        event = parser.get_event(i)
        pulse_collection[i] = -1*(baseline*16384 + event.record - 16384)
        pulse_timestamps[i] = event.header.trigger_time_tag

    return pulse_collection, pulse_timestamps
