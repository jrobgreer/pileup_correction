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


def generate_event_record(dataset, noise=True, record_length=1030, initial_offset=100, expected_cps=1e9):

    total_pulses_in_wfm = 0
    data = dataset
    record = np.zeros(record_length)
    if noise == True:
        record = np.random.normal(0, 10, record_length)

    pulses_in_record = np.random.poisson(expected_cps*record_length*4e-9)

    print(pulses_in_record, " pulses in the record")
    for i in range(pulses_in_record):
        sampled_pulse = data.sample()

        # Draw random pulse from Co60 spectrum
        params = [sampled_pulse[par].values
                  for par in ['Par0', 'Par1', 'Par2', 'Par3']]

        area = sampled_pulse['Area']

        # rate = 1/(1e6/(195e-6))
        rate = 40
        # rate = rate/4e-9
        params[1] = int(np.random.exponential(rate, 1)) + initial_offset

        # Generate pulse based on spectrum fit data
        pulse = guo_fit(np.linspace(0, len(record), len(record)), params)
        # plt.plot(pulse)

        pulse[pulse < 0] = 0
        print(params[1])
        print(len(record[params[1]:]))
        print(len(pulse[:len(record)-params[1]]))
        record[params[1]:] += pulse[:len(record)-params[1]]

    plt.plot(record)
    plt.title(pulses_in_record)
    plt.show()
    plt.close()

    # if trigger == True:
    #     plt.plot(record)
    #     plt.title(
    #         f"Simulated pulse Params:{params[0]},{params[1]},{params[2]},{params[3]}")
    #     plt.show()
    #     plt.close()
    return record, total_pulses_in_wfm


def generate_many(num):
    record_arr = np.zeros(shape=(num, 1030))
    for i in range(num):
        record_arr[i], total_pulses = generate_event_record(
            data, expected_cps=1e6)

    return record_arr


record_arr = generate_many(1000)

for i in record_arr:
    plt.plot(record_arr[i])
