import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import glob as glob
from itertools import chain

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

# plt.hist(data['Area'], bins=1000)
# plt.show()
# plt.close()


def guo_fit(x, par):
    '''par[0] - A
       par[1] - t0
       par[2] - theta1
       par[3] - theta2'''
    return par[0]*(np.exp(-(x-par[1])/par[2]) - np.exp(-(x-par[1])/par[3]))


def generate_event_record(dataset, noise=True, record_length=1030, initial_offset=100, generator_activity=1e6, pulse_rate=30, det_rad=2.5, dist_from_source=30):

    data = dataset
    record = np.zeros(record_length)
    areas_in_record = []

    if noise == True:
        record = np.random.normal(0, 10, record_length)

    pulses_in_record = np.random.poisson(
        generator_activity*record_length*4e-9)  # CHECK THIS - is this the correct activity?

    print(pulses_in_record, " pulses in the record")
    for i in range(pulses_in_record):
        sampled_pulse = data.sample()

        # Draw random pulse from Co60 spectrum
        params = [sampled_pulse[par].values
                  for par in ['Par0', 'Par1', 'Par2', 'Par3']]

        area = sampled_pulse['Area'].values

        areas_in_record.append(area)

        # If generator on for full second, what is activity? Should be way higher than actual activity, event squeezed into 6.5us chunks
        solid_angle_corr = (det_rad/(dist_from_source*2))**2
        apparent_activity = 1 / \
            ((generator_activity * solid_angle_corr)/(pulse_rate*6.5e-6))
        print("APPARENT ACTIVITY: ", apparent_activity)
        # Time between events in "samples"
        scale_parameter = apparent_activity/4e-9
        print(scale_parameter)
        # Ultimately, num of pulses in a waveform event should be independent of the pulse_rate,
        # but its used here because we know that when it pulses at 30Hz at 505, rate is roughly 10^6 n/s

        # OVERRIDE FOR NOW, PULSES ARE TOO CLOSE ACCORDING TO THOSE CALCS, WHY? Is there some other spreading from somewhere?
        # Arrival time to detector?

        # scale_parameter = 50
        # print("SCALE PARAMETER: ", scale_parameter)

        params[1] = int(np.random.exponential(
            scale_parameter, 1)) + initial_offset

        print("TIME: ", params[1])

        # Generate pulse based on spectrum fit data
        pulse = guo_fit(np.linspace(0, len(record), len(record)), params)
        # plt.plot(pulse)

        pulse[pulse < 0] = 0
        # print(params[1])
        # print(len(record[params[1]:]))
        # print(len(pulse[:len(record)-params[1]]))
        try:
            record[params[1]:] += pulse[:len(record)-params[1]]
        except ValueError:
            pulses_in_record -= 1

    # plt.plot(record)
    # plt.title(pulses_in_record)
    # plt.show()
    # plt.close()

    # if trigger == True:
    #     plt.plot(record)
    #     plt.title(
    #         f"Simulated pulse Params:{params[0]},{params[1]},{params[2]},{params[3]}")
    #     plt.show()
    #     plt.close()
    return record, pulses_in_record, areas_in_record


def generate_many(num):
    events_done = 0
    record_arr = np.zeros(shape=(num, 1030))
    area_arr = []
    for i in range(num):

        triggered = False

        while triggered == False:
            record_to_add, total_pulses, areas = generate_event_record(
                data, generator_activity=0.1e6, pulse_rate=30, dist_from_source=25, det_rad=2.5)
            print(" PULSES IN WFM :", total_pulses)
            if total_pulses > 0:
                record_arr[i] = record_to_add
                triggered = True
                events_done += 1
                area_arr.append(areas)
                print(events_done*100/num, " '%' done ")
            else:
                print("NO TRIGGER")

    return record_arr, area_arr


# plt.hist(np.random.exponential(
#     2.07,  1000), bins=100)
# plt.show()

record_arr, area_arr = generate_many(10000)
#
# for i in record_arr:
#     plt.plot(record_arr[i])

area_arr = list(chain.from_iterable(area_arr))
area_arr = np.array(area_arr)
area_arr = area_arr.flatten()
print(area_arr)
np.savetxt(
    'simulated_generator_pileup/generator_activity_0p1e6_pulse_rate_30_dist_from_source_25_det_rad_2p5_10000_events.txt', record_arr)
np.savetxt(
    'simulated_generator_pileup/generator_activity_0p1e6_pulse_rate_30_dist_from_source_25_det_rad_2p5_10000_events_AREAS.txt', area_arr)
