from multiprocessing import Process
import multiprocessing
import spectranalysis as sp
import getwave as gw
from itertools import chain
import numpy as np
import pandas as pd

print("Num CPUs available: ", multiprocessing.cpu_count())

# Add argparser TODO
file = 'testdataGENERATORlong.dat'
cpus_available = multiprocessing.cpu_count()-1

pulse_collection, pulse_timestamps = gw.get_pulse_collection(
    file, baseline=0.1)

# Need to split pulse_collection for CPUs
print(len(pulse_collection))
pulses_per_cpu = int(len(pulse_collection)/cpus_available)
print(pulses_per_cpu)

# subset_low_idx = [i*pulses_per_cpu for i in range(0, cpus_available)]
# subset_high_idx = [i*pulses_per_cpu - 1 for i in range(1, cpus_available+1)]

subsets = [[i*pulses_per_cpu, (i+1)*pulses_per_cpu-1]
           for i in range(0, cpus_available)]
print(subsets)

input("WAIT")


def process_pulses(subset_low_idx, subset_high_idx):

    areas = []
    timestamps = []
    chi2 = []
    ndf = []
    pulse_counts = []
    remaining_pulse_areas = []
    par0 = []
    par1 = []
    par2 = []
    par3 = []
    eventid = []
    time_to_fit = []
    fit_results = []
    rise_point = []
    raw_integrals = []

    print("EXPECT ", int(
        len(pulse_collection[subset_low_idx:subset_high_idx])), " PULSES")

    for pulse_idx in range(subset_low_idx, subset_high_idx):

        print("-------------------------------------------------------------------------------")
        print("-------------------- PULSE INDEX ",
              pulse_idx, " ------------------------------")
        print("-------------------------------------------------------------------------------")
        pulse = sp.Pulse(pulse_collection, pulse_timestamps, pulse_idx)

        # pulse.butter_lowpass_filtfilt(cutoff=15e6, fs=250e6, plotting=False) #25e6 was
        # pulse.raw_int()
        pulse.get_peaks2(min_dist_between_peaks=20, gradient_threshold=16)
        pulse.fit2(closest_distance=21, fit_options='QRS')

        areas.append(pulse.areas)
        timestamps.append(pulse.true_timestamps)
        chi2.append(pulse.chi2)
        ndf.append(pulse.ndf)
        pulse_counts.append(pulse.pulse_count)
        remaining_pulse_areas.append(pulse.remaining_pulse_area)
        par0.append(pulse.par0)
        par1.append(pulse.par1)
        par2.append(pulse.par2)
        par3.append(pulse.par3)
        eventid.append(pulse.record_id)
        time_to_fit.append(pulse.time_to_fit)
        fit_results.append(pulse.fitresultstatus)
        rise_point.append(pulse.rise_point)
        # raw_integrals.append(pulse.raw_integral)

    flat_timestamps = list(chain.from_iterable(timestamps))
    flat_areas = list(chain.from_iterable(areas))
    flat_chi2 = list(chain.from_iterable(chi2))
    flat_ndf = list(chain.from_iterable(ndf))
    flat_pulse_counts = list(chain.from_iterable(pulse_counts))
    flat_remaining_pulse_areas = list(
        chain.from_iterable(remaining_pulse_areas))
    flat_par0 = list(chain.from_iterable(par0))
    flat_par1 = list(chain.from_iterable(par1))
    flat_par2 = list(chain.from_iterable(par2))
    flat_par3 = list(chain.from_iterable(par3))
    flat_eventid = list(chain.from_iterable(eventid))
    flat_time_to_fit = list(chain.from_iterable(time_to_fit))
    flat_fit_results = list(chain.from_iterable(fit_results))
    flat_rise_point = list(chain.from_iterable(rise_point))
    # flat_raw_integrals = list(chain.from_iterable(raw_integrals))

    df = pd.DataFrame({'Area': np.array(flat_areas), 'Timestamp': np.array(
        flat_timestamps), 'Chi2': np.array(flat_chi2), 'NDF': np.array(flat_ndf),
        'Pulse Count': np.array(flat_pulse_counts), 'Rem Pulse Area': np.array(flat_remaining_pulse_areas),
        'Par0': np.array(flat_par0), 'Par1': np.array(flat_par1), 'Par2': np.array(flat_par2),
        'Par3': np.array(flat_par3), 'EventID': np.array(flat_eventid), 'Time to fit': np.array(flat_time_to_fit), 'FitResult': np.array(flat_fit_results),
        'RisePoint': np.array(flat_rise_point)})  # , 'RawInt': np.array(flat_raw_integrals)})
    # Save to CSV file
    csv_filename = file.split(
        '.dat')[0]+'{}:{}.csv'.format(subset_low_idx, subset_high_idx)
    df.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    procs = []
    proc = Process(target=process_pulses)  # instantiating without any argument
    procs.append(proc)
    proc.start()

    # instantiating process with arguments
    for subset in subsets:
        print("Starting subset: ", subset)
        proc = Process(target=process_pulses, args=(subset[0], subset[1]))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
