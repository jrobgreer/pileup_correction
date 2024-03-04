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
datafiles = []
for calibfile in glob.glob("csv_data/csv_data-20240223T154029Z-001/csv_data/cs137co60*.csv"):
    df = pd.read_csv(calibfile)
    datafiles.append(df)

data = pd.concat(datafiles)

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


def generate_event_record(dataset, noise=True, record_length=1030, initial_offset=100, expected_cps=1e9):

    total_pulses_in_wfm = 0
    data = dataset
    record = np.zeros(record_length)
    if noise == True:
        record = np.random.normal(0, 10, record_length)

    trigger = False

    for i, val in enumerate(record):

        # print(f"Cycling from {i+initial_offset} to {len(record)}")

        # How many pulses start within the next 4ns sample
        # Follows a poisson distribution where mean is activity(/s)/(1/time_interval)
        pulses_in_interval = np.random.poisson(expected_cps*4e-9)
        # print(pulses_in_interval)
        total_pulses_in_wfm += pulses_in_interval
        # print(total_pulses_in_wfm)
        # print("Poisson draw result: ", pulses_in_interval)
        # print(f'{pulses_in_interval} pulses at time {i+initial_offset}')

        # Will likely be 0 or 1, or in high pileup cases, higher leading to pileup at the same point
        if pulses_in_interval > 0:
            trigger = True

            for pulse_count in range(0, pulses_in_interval, 1):
                # print(f"Pulse @ {i + initial_offset}")
                # print("PULSE:")
                # print(pulse_count)

                sampled_pulse = data.sample()

                # Draw random pulse from Co60 spectrum
                params = [sampled_pulse[par].values
                          for par in ['Par0', 'Par1', 'Par2', 'Par3']]

                area = sampled_pulse['Area']

                # Put the pulse at position i by setting rise index parameter
                params[1] = i + initial_offset

                # Generate pulse based on spectrum fit data
                pulse = guo_fit(np.linspace(
                    i+initial_offset, len(record), len(record)-i), params)
                # plt.plot(pulse)

                pulse[pulse < 0] = 0

                record[i:] += pulse[:len(record)-i]
                # plt.plot(record)
                # plt.title(pulse_count)
                # plt.show()
                # plt.close()

    # if trigger == True:
    #     plt.plot(record)
    #     plt.title(
    #         f"Simulated pulse Params:{params[0]},{params[1]},{params[2]},{params[3]}")
    #     plt.show()
    #     plt.close()
    return record, total_pulses_in_wfm

# Want to request N waveforms
# With a specified arbitrary activity


def create_pileup_dataset(num_wanted, expected_cps):
    pulse_collection = np.zeros(shape=(num_wanted, 1030))
    pulse_timestamps = np.zeros(num_wanted)
    pulse_count_in_wfm = np.zeros(num_wanted)

    for i in range(num_wanted):
        record, total_pulses_in_wfm = generate_event_record(
            data, expected_cps=expected_cps, noise=True)
        pulse_collection[i] = record
        pulse_count_in_wfm[i] = total_pulses_in_wfm

    print("Empty waveforms: ", (pulse_count_in_wfm == 0).sum())
    triggers_total = num_wanted - (pulse_count_in_wfm == 0).sum()
    print("Single waveforms: ", (pulse_count_in_wfm == 1).sum())
    print("Pileup waveforms: ", (pulse_count_in_wfm > 1).sum())
    pileup_fraction = (pulse_count_in_wfm > 1).sum(
    )/((pulse_count_in_wfm > 1).sum() + (pulse_count_in_wfm == 1).sum())
    print("Pileup fraction :", pileup_fraction)
    plt.hist(pulse_count_in_wfm, label=str(pileup_fraction), bins=100)

    return pileup_fraction, pulse_collection, pulse_timestamps, triggers_total


# create_pileup_dataset(1000, 0.1e6)

# create_pileup_dataset(1000, 0.2e6)


# create_pileup_dataset(1000, 0.5e6)

# create_pileup_dataset(1000, 1e6)
# plt.legend()
# plt.show()

num_records = 100

pileup_fraction, record_array, pulse_timestamps, triggers_total = create_pileup_dataset(
    num_records, expected_cps=1e6)
plt.plot(record_array[1])
plt.show()
np.savetxt(
    f'{num_records}Records_{triggers_total}Triggers_NoiseTrue_Activity0p1e6_PileupFrac{pileup_fraction}.txt', record_array)


# for i in range(num_wanted):
#     plt.plot(pulse_collection[i])
#     plt.title(f"{i}")
#     plt.show()


# def generate_waveform_array(datafile, number_of_waveforms, noise=False, initial_offset=150, source_activity=2e9, detector_efficiency=0.3, distance_to_source=1, approx_detector_area=0.025):

#     # Use spectrum data to generate pulses from
#     dataset = pd.read_csv(datafile)

#     record_array = np.zeros(shape=(number_of_waveforms, 1030))
#     for i in range(0, number_of_waveforms):
#         # TODO fix arguments here, do it properly
#         record_array[i] = generate_event_record(dataset, noise, initial_offset,
#                                                 source_activity, detector_efficiency, distance_to_source, approx_detector_area)

#     print(record_array)

#     return record_array


# record_array = generate_waveform_array('/home/james/pileup_correction/csv_data/cs137co60calib48944_61179.csv', 1000, noise=True,
#                                        initial_offset=150, source_activity=1e9, detector_efficiency=0.3, distance_to_source=1, approx_detector_area=0.025)

# np.savetxt('1kRecords_NoiseTrue_InitialOffset150_Activity1e9_Eff0p3_DistToSource1m_ApproxDetArea_0.025.txt', record_array)
