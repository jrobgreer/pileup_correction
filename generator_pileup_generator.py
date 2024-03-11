import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import glob as glob
from itertools import chain


# mpl.rcParams.update({'font.size': 32})


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

# plt.hist(np.random.exponential(
#     194, 1000), bins=100)

# plt.hist(np.random.poisson(5, 10000), bins=50, color='r', density=True)
# plt.xlabel("Pulses within Record")

# plt.hist(data['Area'], bins=np.linspace(0, 150000, 1000),
#          color='g',  linewidth=2)
# plt.xlabel("Calculated Area")
# plt.ylabel("Counts")
# plt.xlim(0, 150000)
# plt.minorticks_on()
# plt.show()
# plt.close()


def guo_fit(x, par):
    '''par[0] - A
       par[1] - t0
       par[2] - theta1
       par[3] - theta2'''
    return par[0]*(np.exp(-(x-par[1])/par[2]) - np.exp(-(x-par[1])/par[3]))


def generate_event_record(dataset, noise=True, record_length=1030, initial_offset=0, generator_activity=1e6, pulse_rate=30, det_rad=2.5, dist_from_source=30):

    data = dataset
    record = np.zeros(record_length)
    areas_in_record = []
    rise_indices = []

    if noise == True:
        record = np.random.normal(0, 10, record_length)

    pulses_in_record = np.random.poisson(
        generator_activity*record_length*4e-9)  # This is 100% efficient full solid angle, needs more scaling if we want fully
    # accurate "activity", but also fine to just use pileup fraction as our measure of level of pileup and look at distributions
    # of pulse counts - activity can be an arbitrary scaler of this. datasets produced will output fraction too

    # print(pulses_in_record, " pulses in the record")
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
        # print("APPARENT ACTIVITY: ", apparent_activity)
        # Time between events in "samples"
        scale_parameter = apparent_activity/4e-9
        # print(scale_parameter)

        # Ultimately, num of pulses in a waveform event should be independent of the pulse_rate,
        # but its used here because we know that when it pulses at 30Hz at 505, rate is roughly 10^6 n/s

        # OVERRIDE FOR NOW, PULSES ARE TOO CLOSE ACCORDING TO THOSE CALCS, WHY? Is there some other spreading from somewhere?
        # Arrival time to detector?

        # scale_parameter = 50
        # print("SCALE PARAMETER: ", scale_parameter)

        # GENERATOR PILEUP
        # params[1] = int(np.random.exponential(
        #     scale_parameter, 1)) + initial_offset + 100

        # plt.hist(np.loadtxt(
        #     'GeneratorPileupTimestampIndexDistribution.txt'), bins=1030)
        # plt.show()
        # plt.close()

        # Sampled from generator pileup distribution
        # params[1] = int(np.random.choice(np.loadtxt(
        #     'GeneratorPileupTimestampIndexDistribution.txt')))

        # STANDARD PILEUP
        # #params[1] = int(np.random.poisson(
        # #    150-initial_offset, 1)) + initial_offset
        params[1] = np.random.randint(0, len(record))

        # print("TIME: ", params[1])

        # Generate pulse based on spectrum fit data
        # print(params)
        pulse = guo_fit(np.linspace(0, len(record), len(record)), params)
        # plt.plot(pulse)

        pulse[pulse < 0] = 0
        # print(params[1])
        # print(len(record[params[1]:]))
        # print(len(pulse[:len(record)-params[1]]))
        try:
            new_pulse = np.zeros(1030)
            new_pulse[params[1]:] = pulse[params[1]:len(record)]
            record[params[1]:] += pulse[params[1]:len(record)]
            rise_indices.append(params[1])
            # plt.plot(new_pulse, color='blue', label=f'Pulse {i}', linewidth=1)
            # plt.axvline(params[1])
            # print(params)

        except ValueError:
            pulses_in_record -= 1

    # plt.plot(record, label='Full Simulated Waveform',
    #          color='black', linewidth=1)
    # # plt.title(pulses_in_record)
    # plt.xlabel("Time")
    # plt.ylabel("ADC")
    # plt.legend()
    # plt.show()
    # plt.close()

    # if trigger == True:
    #     plt.plot(record)
    #     plt.title(
    #         f"Simulated pulse Params:{params[0]},{params[1]},{params[2]},{params[3]}")
    #     plt.show()
    #     plt.close()

    mean_time_sep = np.mean(np.diff(np.sort(rise_indices)))
    if pulses_in_record == 1:
        mean_time_sep = 1030

    return record, pulses_in_record, areas_in_record, mean_time_sep


def generate_many(num, act):
    events_done = 0
    record_arr = np.zeros(shape=(num, 1030))
    area_arr = []
    pulses_in_wfm_arr = []
    mean_time_sep_arr = []
    for i in range(num):

        triggered = False

        while triggered == False:
            record_to_add, total_pulses, areas, mean_time_sep = generate_event_record(
                data, generator_activity=act, pulse_rate=30, dist_from_source=25, det_rad=2.5)

            # print(" PULSES IN WFM :", total_pulses)
            if total_pulses > 0:
                record_arr[i] = record_to_add
                triggered = True
                events_done += 1
                area_arr.append(areas)
                print(events_done*100/num, " '%' done ")
            else:
                pass
                # print("NO TRIGGER")

        pulses_in_wfm_arr.append(total_pulses)
        mean_time_sep_arr.append(mean_time_sep)

    pulses_in_wfm_arr = np.array(pulses_in_wfm_arr)
    mean_time_sep_arr = np.array(mean_time_sep_arr)

    pileup_frac = np.count_nonzero(
        pulses_in_wfm_arr > 1)/len(pulses_in_wfm_arr)

    return record_arr, area_arr, pulses_in_wfm_arr, mean_time_sep_arr, pileup_frac


# plt.hist(np.random.exponential(
#     2.07,  1000), bins=100)
# plt.show()

def make_dataset(num, act):
    record_arr, area_arr, pulses_in_wfm_arr, mean_time_sep_arr, pileup_frac = generate_many(
        num, act)
    area_arr = list(chain.from_iterable(area_arr))
    area_arr = np.array(area_arr)
    area_arr = area_arr.flatten()
    np.savetxt(
        f'TypicalPileupData_WriteUp/PILEUP{pileup_frac}_activity_{act}_pulse_rate_30_dist_from_source_25_det_rad_2p5_10000_events.txt', record_arr)
    np.savetxt(
        f'TypicalPileupData_WriteUp/PILEUP{pileup_frac}_activity_{act}_pulse_rate_30_dist_from_source_25_det_rad_2p5_10000_events_AREAS.txt', area_arr)
    np.savetxt(
        f'TypicalPileupData_WriteUp/PILEUP{pileup_frac}_activity_{act}_pulse_rate_30_dist_from_source_25_det_rad_2p5_10000_events_TSEP.txt', mean_time_sep_arr)

    plt.hist(mean_time_sep_arr*4, bins=np.linspace(0, 1030*4, 100),
             histtype='step', label=f'{pileup_frac*100}%')
    print(f'TypicalPileupData_WriteUp/PILEUP{pileup_frac}_activity_{act}_pulse_rate_30_dist_from_source_25_det_rad_2p5_10000_events')  # nopep8


plt.close()

make_dataset(10000, 0.01e6)
make_dataset(10000, 0.1e6)
make_dataset(10000, 0.5e6)
make_dataset(10000, 0.75e6)
make_dataset(10000, 1e6)
make_dataset(10000, 1.5e6)
plt.minorticks_on()
plt.xlabel("Mean pulse separation [ns]")
plt.legend()
plt.show()
plt.close()


# pileup_frac = np.count_nonzero(pulses_in_wfm_arr > 1)/len(pulses_in_wfm_arr)
# plt.close()
# plt.hist(pulses_in_wfm_arr, bins=100)
# plt.xlabel("Pulses per waveform")
# plt.show()
# plt.close()


# area_arr = list(chain.from_iterable(area_arr))
# area_arr = np.array(area_arr)
# area_arr = area_arr.flatten()
# print(area_arr)
# np.savetxt(
#     'simulated_generator_pileup/STANDARDPILEUPgenerator_activity_0p5e6_pulse_rate_30_dist_from_source_25_det_rad_2p5_10000_events.txt', record_arr)
# np.savetxt(
#     'simulated_generator_pileup/STANDARDPILEUPgenerator_activity_0p5e6_pulse_rate_30_dist_from_source_25_det_rad_2p5_10000_events_AREAS.txt', area_arr)
