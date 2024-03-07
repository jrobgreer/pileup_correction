# -*- coding: utf-8 -*-
"""SharedAnalysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OnPDf7rEWRWCdV2RQCXKwtz2NIQZpTwK

Preamble Setup for mounting Google Drive
"""

import pandas as pd
from numba import njit
import pyswarms as ps
import scipy
import tqdm
import matplotlib

# Path Setup
import os
import sys

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import importlib
# from google.colab import output
from pyswarm import pso
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gimmedatwave as gdw
import matplotlib.pyplot as plt
import getwave as gw

# import root_template
# root_template.set_fig_size(12,4)

matplotlib.rcParams.update({'font.size': 12})


# Jit Compiled Helpers

def moving_average(a, n=20):
    df = pd.DataFrame(data={"v": a})
    return df.rolling(n, center=True).median()["v"].values

    # ret = np.cumsum(a, dtype=float)
    # ret[n:] = ret[n:] - ret[:-n]
    # return np.concatenate([np.zeros(int(n/2)), ret[n - 1:] / n,np.zeros(int(n/2))])


@njit
def guo(x, A, t0, theta1, theta2):
    par = [A, t0, theta1, theta2]
    evals = par[0]*(np.exp(-(x-par[1])/par[2]) - np.exp(-(x-par[1])/par[3]))
    evals[evals < 0] = 0
    return evals

# These need to be moved out as JIT is compiling each time


def guo_combined(x, npeaks, *pars):
    npeaks = int(npeaks)
    yvals_total = np.zeros(len(x))
    for i in range(npeaks):
        yvals_total += guo(x, pars[i*4+0], pars[i*4+1],
                           pars[i*4+2], pars[i*4+3])

    yvals_total[yvals_total < 0] = 0
    return np.copy(yvals_total)


def guo_combined2(x, npeaks, pars):
    npeaks = int(npeaks)
    # print("GUO PARS", x, npeaks, pars)
    yvals_total = np.zeros(len(x))
    for i in range(npeaks):
        # print("PARSET: ", pars[i*4+0], pars[i*4+1], pars[i*4+2], pars[i*4+3])
        yvals_total += guo(x, pars[i*4+0], pars[i*4+1],
                           pars[i*4+2], pars[i*4+3])

    yvals_total[yvals_total < 0] = 0
    return np.copy(yvals_total)


def wrap_call(x, npeaks, *pars):
    guo_combined(x, npeaks, *pars)


def objective_function2(x, xvals, y, npeaks):
    y_pred = guo_combined(xvals, npeaks, *x)
    return np.mean((y - y_pred) ** 2)


def objective_function3(x, xvals, y, npeaks):
    y_pred = guo_combined2(xvals, npeaks, x)
    return np.mean((y - y_pred) ** 2)


@njit
def find_valid_indices(rise_indices):
    valid_indices = []

    for j in range(0, len(rise_indices)):
        if abs(rise_indices[j] - rise_indices[j-1]) > 2:
            valid_indices.append(rise_indices[j])

    if len(valid_indices) == 0:
        valid_indices.append(rise_indices[0])

    valid_indices = np.array(valid_indices)

    return valid_indices


@njit
def find_rise_amplitudes(avg_record, rise_indices):
    valid_amp = []
    dif1 = list(rise_indices[1:] - rise_indices[:-1])
    dif2 = list(dif1)
    dif2.insert(0, rise_indices[0])
    dif1.append(len(avg_record) - rise_indices[-1])

    loookforw = rise_indices + (0.75*np.array(dif1))
    loookback = rise_indices - (0.75*np.array(dif2))

    for j in range(0, len(rise_indices)):

        lowv = np.min(avg_record[int(loookback[j]):int(rise_indices[j])])
        topv = np.max(avg_record[int(rise_indices[j]):int(loookforw[j])])

        ampv = topv - lowv
        valid_amp.append(ampv)

    vals = np.array(valid_amp)
    vals[np.isnan(vals)] = 500

    return vals


# Load in a selected file
# filename = "/content/drive/MyDrive/NeutronPileupAnalysis/soilfromoutside150224_15mins_GENON.dat"
# parser = gdw.gimmedatwave.Parser(
#     filename, digitizer_family=gdw.gimmedatwave.DigitizerFamily.X725)
# n_pulses = int(parser.n_entries)-1
# baseline = 0.1


def fit(raw_pulse, rise_indices, amplitudes):

    # Pulse Preprocessing
    npeaks = len(rise_indices)
    raw_pulse[np.isnan(raw_pulse)] = 0.0

    maxval = np.max(raw_pulse)
    parset = []
    fixset = []
    lowset = []
    highset = []
    for i in range(npeaks):
        parset.append(amplitudes[i])
        parset.append(rise_indices[i])
        parset.append(65)
        parset.append(8)

        # lowset was [amplitudes[i]*0.0001
        lowset += ([20, rise_indices[i]-1, 35, 5])
        highset += ([6000, rise_indices[i]+1, 80, 30])

    # Get XY Vals before fit
    xvals = np.linspace(0.0, len(raw_pulse), len(raw_pulse))
    yvals = raw_pulse
    yvals_start = guo_combined(xvals, npeaks, *parset)

    # Define parameter bounds for optimization
    # Lower and upper bounds for each parameter
    param_bounds = (lowset, highset)

    # TRY NO FITTING
    # optimized_params = parset

    # TRY CURVE FITTING
    # optimized_params, covariance = curve_fit(wrap_call, xvals, yvals, p0=parset, bounds=param_bounds, method='trf', ftol=1e-3,
    #                                          xtol=1e-2, gtol=1e-2, x_scale='jac', loss='soft_l1', f_scale=0.1, diff_step=None, tr_solver=None, tr_options={}, jac=None)

    # TRY PYSWARM
    # optimized_params, _ = pso(objective_function3, lowset, highset, args=(
    #     xvals, yvals, npeaks), swarmsize=3*npeaks, maxiter=2000)

    # TRY PYSWARMS
    # optimizer = ps.single.GlobalBestPSO(n_particles=3*npeaks, dimensions=len(lowset), options={'c1': 0.5, 'c2': 0.3, 'w':0.9}, bounds=param_bounds)
    # cost, pos = optimizer.optimize(lambda x: objective_function2(x, xvals, yvals, npeaks), iters=2000)
    # cost, pos = optimizer.optimize(objective_function2, iters=20, xvals=xvals, y=yvals, npeaks=npeaks)

    # TRY SCIPY MINIMIZE
    # result = scipy.optimize.minimize(objective_function2, parset, args=(
    #     xvals, yvals, npeaks), bounds=param_bounds)

    result = scipy.optimize.minimize(objective_function3, parset, args=(xvals, yvals, npeaks), bounds=list(
        np.array(param_bounds).transpose()), options={'maxiter': 2000}, tol=1)

    optimized_params = result.x

    print(optimized_params)

    # optimized_params, covariance = curve_fit(guo_combined, xvals, yvals, p0=parset)
    # for i in range(len(optimized_params)):
    # print(optimized_params[i])

    fitted_pulse = guo_combined(xvals, npeaks, *optimized_params)

    integrals = []
    for i in range(npeaks):
        # print("A",i, npeaks, optimized_params[4*i+1])
        pars = optimized_params
        single_pulse = guo(xvals, pars[i*4+0],
                           pars[i*4+1], pars[i*4+2], pars[i*4+3])

        integrals.append(np.sum(single_pulse))
        plt.plot(single_pulse, label=f'Pulse Fit {i}', c='gray')

    plt.plot(fitted_pulse, label='Fit Result', c='black')
    plt.plot(raw_pulse, label='Pulse', c='red')

    # err = fitted_pulse - raw_pulse
    # err[np.abs(err) < 50] = 0
    # total_err = np.sum(err**2)
    # plt.plot(err, label='Fit Error', color='green')
    # print("TOTAL ERROR IN FIT: ", total_err)
    plt.legend()
    plt.ylabel("ADC")
    plt.xlabel("Time [4ns/sample]")
    plt.minorticks_on()
    plt.show()
    plt.close()

    return integrals


# pulse_collection = np.loadtxt(
#     'simulated_generator_pileup/generator_activity_0p5e6_pulse_rate_30_dist_from_source_25_det_rad_2p5_10000_events.txt')
# pulse_timestamps = np.zeros(len(pulse_collection))


def correct_dataset(pulse_collection_file, true_spec_file):

    pulse_collection = np.loadtxt(pulse_collection_file)
    n_pulses = len(pulse_collection)
    pulse_timestamps = np.zeros(n_pulses)

    global_integrals = []

    for pulse_index, pulse in enumerate(pulse_collection[:10000]):

        if pulse_index % 100:
            print(pulse_index/len(pulse_collection[:10000]))

        npulse = len(pulse)
        raw_index = np.linspace(0, npulse, npulse)
        raw_record = pulse
        raw_timestamp = 0  # Timestamps not required here

        # Median rolling average data
        avg_record = moving_average(raw_record, n=10)

        # Make interpolator
        get_amp = interp1d(raw_index, raw_record)

        # First gradient for edges
        grad_record = np.gradient(avg_record)
        edge_record = grad_record > 30   # Changed from 50

        # Second gradient for start of edges
        ddif_record = np.gradient(np.ones(npulse) * edge_record) > 0

        # Determine rise indices from checks
        rise_indices = np.array(np.nonzero(ddif_record))[0]
        # print(rise_indices)
        rise_indices = np.unique(rise_indices - rise_indices % 2)
        # print(rise_indices)
        # plt.scatter(rise_indices, get_amp(rise_indices),
        #             c='green', s=80, marker='o', label='Rise DDif')
        valid_indices = find_valid_indices(rise_indices)
        # print(valid_indices)
        # input("ENTER")
        # print("INDICES: ", valid_indices)

        # Estimate rise amplitudes
        try:
            amplitudes = find_rise_amplitudes(avg_record, valid_indices) * 1.75
        # print("AMPS", amplitudes)
        except ValueError:
            print(valid_indices)
            print(avg_record)
            print(amplitudes)
            # plt.clf()
            # plt.plot(raw_record, label='raw')
            # plt.show()
            # plt.clf()
            pass

        # Make final analysis plot single per event
        # plt.plot(raw_record, label='Raw')
        # plt.plot(avg_record, label='Mov Avg')
        # plt.plot(grad_record*10, label='Grad')
        plt.scatter(valid_indices, get_amp(valid_indices),
                    c='green', s=100, marker='*', label='Rise')
        # plt.scatter(rise_indices, get_amp(rise_indices),
        #             c='red', s=80, marker='+', label='Rise STANDARD')
        # # plt.plot(edge_record*5000, label='grad_cut')
        # plt.plot(ddif_record*5000, label='ddif')

        # raw_record = avg_record
        integrals = fit(raw_record, valid_indices, amplitudes)

        for g in integrals:
            # print("INTEGRAL", g)
            global_integrals.append(g)
            # print(g)

        plt.legend()
        plt.show()
        plt.close()

        if (pulse_index > 10000):
            break

    areas = np.loadtxt(true_spec_file)
    plt.hist(global_integrals, bins=np.linspace(np.min(areas), np.max(areas), 1000),
             histtype='step', label="PILEUP CORRECTED")

    plt.hist(areas, bins=np.linspace(np.min(areas), np.max(
        areas), 1000), histtype='step', label="TRUE SPECTRUM")
    plt.legend()
    plt.show()
    breakpoint()


correct_dataset('simulated_generator_pileup/generator_activity_0p5e6_pulse_rate_30_dist_from_source_25_det_rad_2p5_10000_events.txt',
                'simulated_generator_pileup/generator_activity_0p5e6_pulse_rate_30_dist_from_source_25_det_rad_2p5_10000_events_AREAS.txt')
# pulse_collection, pulse_timestamps = gw.get_pulse_collection(
#     '/content/drive/MyDrive/NeutronPileupAnalysis/soilfromoutside150224_15mins_GENON.dat', baseline=0.1, fraction_of_dataset=0.01)


# !pip install pyswarms