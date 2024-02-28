import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import glob as glob

# Need to draw pulses from the Co60 spectrum
# Set up a Poisson distribution for different activity sources
# Should generate waveforms of a certain length rnadomly consisting of N pulses based on Poisson

# Within a time interval t, the chance of n decays is dictated by a Poisson distribution
# For a 1 GBq source, in 1 second, there are on average 1e9 decays
# In 1 ns, there is on average 1 decay

# Calculated from (1 second / 1 nanosecond)* 1 GBq

# Expected mean events at next sample of record = (1/4ns(sample rate time))*ACTIVITY

# Going through a 1030 record window, we could check every few ns if there is likely to be a pulse

data = pd.read_csv(
    '/home/james/pileup_correction/csv_data/cs137co60calib48944_61179.csv')

plt.hist(data['Area'], bins=1000)
plt.show()
plt.close()


def guo_fit(x, par):
    '''par[0] - A
       par[1] - t0
       par[2] - theta1
       par[3] - theta2'''
    return par[0]*(np.exp(-(x-par[1])/par[2]) - np.exp(-(x-par[1])/par[3]))


def get_activity_at_detector(source_activity, detector_efficiency, distance_to_source, approx_detector_area):
    return source_activity*approx_detector_area*detector_efficiency/(4*np.pi*(distance_to_source**2))


def generate_poisson(activity, dt):
    expected_pulses = dt*activity
    event_chances = np.random.poisson(expected_pulses, 100000)
    heights, _ = np.histogram(event_chances, bins=np.linspace(0, 15, 16))
    heights = heights/np.sum(heights)
    plt.bar(x=np.linspace(0, 15, 15), height=heights)


# generate_poisson(get_activity_at_detector(1e9, 0.3, 1, 0.0025), 4e-9)
# plt.show()
# plt.close()


def generate_event_record(dataset, noise=False, initial_offset=150, source_activity=2e9, detector_efficiency=0.3, distance_to_source=1, approx_detector_area=0.025):

    data = dataset
    record = np.zeros(1030)
    if noise == True:
        record = np.random.normal(0, 10, 1030)
    trigger = False

    for i, val in enumerate(record[initial_offset:]):

        # print(f"Cycling from {i+initial_offset} to {len(record)}")

        # How many pulses start within the next 4ns sample
        # Follows a poisson distribution
        pulses_in_interval = np.random.poisson(
            get_activity_at_detector(5e9, 0.3, 1, 0.025)*4e-9)

        # print(f'{pulses_in_interval} pulses at time {i+initial_offset}')

        # Will likely be 0 or 1, or in high pileup cases, higher leading to pileup at the same point
        if pulses_in_interval > 0:
            trigger = True

            for pulse in range(1, pulses_in_interval + 1, 1):
                # print(f"Pulse @ {i + initial_offset}")

                # Draw random pulse from Co60 spectrum
                params = [data.sample()[par].values
                          for par in ['Par0', 'Par1', 'Par2', 'Par3']]

                # Put the pulse at position i by setting rise index parameter
                params[1] = i + initial_offset

                # Generate pulse based on spectrum fit data
                pulse = guo_fit(np.linspace(
                    i+initial_offset, len(record), len(record)-i), params)

                pulse[pulse < 0] = 0

                record[i +
                       initial_offset:] += pulse[:len(record)-initial_offset-i]

    # if trigger == True:
    #     plt.plot(record)
    #     plt.title(
    #         f"Simulated pulse Params:{params[0]},{params[1]},{params[2]},{params[3]}")
    #     plt.show()
    #     plt.close()

    return record


def generate_waveform_array(datafile, number_of_waveforms, noise=False, initial_offset=150, source_activity=2e9, detector_efficiency=0.3, distance_to_source=1, approx_detector_area=0.025):

    # Use spectrum data to generate pulses from
    dataset = pd.read_csv(datafile)

    record_array = np.zeros(shape=(number_of_waveforms, 1030))
    for i in range(0, number_of_waveforms):
        # TODO fix arguments here, do it properly
        record_array[i] = generate_event_record(dataset, number_of_waveforms, noise, initial_offset,
                                                source_activity, detector_efficiency, distance_to_source, approx_detector_area)

    print(record_array)

    return record_array


record_array = generate_waveform_array(
    '/home/james/pileup_correction/csv_data/cs137co60calib48944_61179.csv', 1000)


np.savetxt('testwfms.txt', record_array)
