import numpy as np
import ROOT
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.signal import argrelextrema
import getwave as gw
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import ROOT
import array
import gc
import pandas as pd
import time
from itertools import chain

plt.rcParams['figure.figsize'] = (20, 4)
# COMPASS cannot timestamp events, using wavedump

# For neutron generator setup, we have an external trigger based upon when the neutron pulse occurs
# The TTL pulse from the generator goes into a gate generator
# The gate produced is sent into TRG-IN on DT5730
# This resets timestamps


class Pulse:

    def __init__(self, pulse_collection, pulse_timestamps, index):

        # Now we've expanded the record length, REMAINING PULSE AREA after subtraction is no longer
        # a good measure of fit quality for the final pulse, there will always be an undershoot NOTE
        self.record = np.zeros(2000)
        self.record[0:len(pulse_collection[index])] = pulse_collection[index]

        self.timestamp = pulse_timestamps[index]
        self.event_id = index

        # Flag for throwing away bad pulses
        self.pulse_suitable = True

        # True timestamps will contain multiple timestamps from many pulses in a waveform
        self.true_timestamps = []
        self.areas = []
        self.chi2 = []
        self.ndf = []
        self.par0 = []
        self.par1 = []
        self.par2 = []
        self.par3 = []
        self.record_id = []
        self.pulse_count = []
        self.remaining_pulse_area = []
        self.time_to_fit = []
        self.fitresultstatus = []
        self.rise_point = []
        self.raw_integral = []

        # NOTE may be worth getting rid of these
        self.time = np.linspace(0, 1029, 1030)
        self.record_length = len(pulse_collection[index])

    def fft(self, plotting=False):
        '''Produce FFT spectrum - optional plotting'''
        self.yf = fft(self.record)
        # 4ns between samples, should probably make this more general
        self.xf = fftfreq(self.record_length, 4e-9)

        if plotting == True:
            plt.plot(self.xf, np.abs(self.yf))
            plt.show()
            plt.close()

    def butter_lowpass_filtfilt(self, cutoff, fs, order=5, plotting=False):
        '''Low pass filter'''

        def butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, self.record)

        if plotting == True:
            # Plot pre and post filter waveforms
            # plt.plot(self.record, label='Original')
            self.record = y
            plt.plot(self.record, label=f'Filtered @ {cutoff} Hz')
            # plt.legend()
            # plt.show()
            # plt.close()

        self.record = y

    def plot(self):
        plt.plot(self.record, label='Original')
        # plt.show()

    def raw_int(self):
        integral = np.sum(self.record[self.rise_indices[0]:])
        # print("Raw integral : ", integral)
        self.rawint = integral
        self.raw_integral.append(integral)

    def get_peaks2(self, plotting=False, min_dist_between_peaks=15, gradient_threshold=25, moving_av_filt=8):
        '''Find the rise points based upon gradient, for use as fit limits in fit2
           min_dist_between_peaks - allows removal of peaks too closely spaced for fitting
           gradient_threshold - what gradient defines a rise
           moving_av_filt - spacing for moving avg filtering'''

        # Crude check for encroaching pulses from previous record
        if np.median(self.record[:10]) > 500:
            self.pulse_suitable = False
            # print("Encroaching previous pulse, skip to next...")
            return

        # plt.close()
        # plt.plot(self.record, label='Original wfm')
        # plt.show()

        def moving_average(a, n=moving_av_filt):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        self.record = moving_average(self.record)

        self.gradient = np.gradient(self.record)

        # Recalculate gradient before finding rises
        self.gradient = np.gradient(self.record)
        # Locate rise of the pulse from a spike in gradient
        rise_indices = np.asarray(
            (self.gradient > gradient_threshold)).nonzero()

        # Need to select rise from clusters detected around peak, use first
        prev_rise_index = None
        rise_indices_clustered = []
        # print(rise_indices)

        for i, rise_index in enumerate(np.array(rise_indices).flatten()):
            # print("Rise index:")
            # print(rise_index)
            if prev_rise_index == None:
                # print("Starting clustering")
                prev_rise_index = rise_index
                rise_indices_clustered.append(rise_index)

            # Clusters of points flagged as rises, only need one to fit from
            # Look through the rise indices chosen
            # If too close together, discard
            elif rise_index > prev_rise_index + min_dist_between_peaks:
                # print(rise_index, " compared with ", prev_rise_index)
                prev_rise_index = rise_index
                rise_indices_clustered.append(rise_index)

        # print(rise_indices_clustered)
        # plt.scatter(x=rise_indices, y=gradient[rise_indices], s=10, marker='*', color='r', label='Spikes')
        # plt.scatter(x=rise_indices_clustered, y=gradient[rise_indices_clustered], s=100, marker='+', color='green', label='Rises')
        # plt.scatter(x=rise_indices_clustered, y=self.record[rise_indices_clustered], s=100, marker='+', color='green', label='Rises')
        # plt.plot(self.record)
        # plt.legend()
        # plt.show()

        self.rise_indices = rise_indices_clustered
        self.rise_amplitudes = self.record[rise_indices_clustered]

        # plt.scatter(self.rise_indices, self.rise_amplitudes)
        # plt.legend()
        # plt.show()

    def fit2(self, closest_distance=25, fit_options='QRS'):

        # plt.plot(self.record, label='Remove shoulder + mov avg')

        # plt.scatter(self.rise_indices,
        #             self.record[self.rise_indices], marker='+', s=100, label='Rise')
        # plt.legend()
        # plt.show()
        # plt.close()

        # For pileup waveforms, check distance between rises
        # If too small, separation tricky, skip over
        try:
            gaps_between_pulses = np.diff(self.rise_indices)
            # print(gaps_between_pulses)
        except AttributeError:
            pass
            # print("Bad pulse, do not fit, move to next...")

        try:
            # print(np.min(gaps_between_pulses))
            # print(closest_distance)
            if np.min(gaps_between_pulses) < closest_distance:
                # print("Pulses too close, skipping to next pulse")
                self.pulse_suitable = False

        except ValueError:
            pass
            # print("One pulse waveform, continuing with fit...")

        except UnboundLocalError:
            pass
            # print("Diffs not calculated")

        if self.pulse_suitable == True:

            # plt.close()
            # # plt.scatter(x=self.rise_indices, y=self.gradient[self.rise_indices], s=10, marker='*', color='r', label='Spikes')
            # # plt.scatter(x=self.rise_indices, y=self.gradient[self.rise_indices], s=100, marker='+', color='green', label='Rises')
            # plt.scatter(x=self.rise_indices,
            #             y=self.record[self.rise_indices], s=100, marker='+', color='green', label='Rises')
            # plt.plot(self.record, label='Pulse')
            # plt.legend()
            # plt.show()

            def guo_fit(x, par):
                '''par[0] - A
                   par[1] - t0
                   par[2] - theta1
                   par[3] - theta2'''
                return par[0]*(np.exp(-(x[0]-par[1])/par[2]) - np.exp(-(x[0]-par[1])/par[3]))

            def perform_fit(rise_idx, fit_end, idx):
                '''fit_start and fit_end - indices corresponding to the pulse fit range
                   idx for reference to other stuff'''

                graph = ROOT.TGraph(len(self.record[rise_idx:]), np.linspace(
                    rise_idx, len(self.record), len(self.record)-rise_idx), self.record[rise_idx:])

                # graph.SetMarkerStyle(20)
                # graph.SetMarkerSize(1)
                # graph.SetMarkerColor(4)

                # print("AMPLITUDE: ", np.max(self.record[rise_idx:fit_end]))

                # Only fit up to next rise
                fit_function = ROOT.TF1(
                    'guo_fit', guo_fit, rise_idx, fit_end, 4)
                fit_function.SetParameters(
                    1.5*np.max(self.record[rise_idx:fit_end]), self.rise_indices[idx], 50, 10)

                fit_function.FixParameter(1, rise_idx)
                fit_function.SetParLimits(0, np.max(
                    1*self.record[rise_idx:fit_end]), 2*np.max(self.record[rise_idx:fit_end]))
                # fit_function.SetParLimits(1, rise_idx-2, rise_idx+2)
                fit_function.SetParLimits(2, 60, 75)
                fit_function.SetParLimits(3, 6, 10)

                start_time = time.time()

                fit_result = graph.Fit(fit_function, fit_options)

                self.fitresultstatus.append(fit_result.Status())

                end_time = time.time()
                fit_time = end_time - start_time

                # print("---------------------------------------------------------------------------")
                # print("Time to fit: ", fit_time)
                # print("Fit status: ", fit_result.Status())
                # print("---------------------------------------------------------------------------")

                # canvas = ROOT.TCanvas(
                #     "canvas", "Guo Model Pulse Fit", 1000, 600)
                # graph.Draw("AP")
                # fit_function.Draw("same")
                # canvas.Update()
                # canvas.Draw()

                # input("Enter to cont")

                fitted_pulse = np.array([guo_fit([t], [fit_function.GetParameter(i) for i in range(
                    fit_function.GetNpar())]) for t in np.linspace(rise_idx, len(self.record), len(self.record)-rise_idx)])

                # print(len(self.record))

                # print(np.linspace(rise_idx, len(self.record),len(self.record)-rise_idx))
                # print(len(np.linspace(rise_idx, len(self.record), len(self.record)-rise_idx)))
                # After fit subtraction, area is a good indicator of whether a bad fit has badly messed up the remaining pulse
                corrected_pulse_area = np.sum(fitted_pulse)

                self.remaining_pulse_area.append(corrected_pulse_area)

                # NOTE tune this value
                # if corrected_pulse_area < -1000:
                #     print("Bad fit, remaining pulse mangled")

                # Get the timestamp, will be relative to the timestamp of the event at trigger point (first peak rise) in nanosecs
                self.true_timestamps.append(
                    self.timestamp + (rise_idx - self.rise_indices[0])*8)
                self.areas.append(np.sum(fitted_pulse))
                # print("FIT AREA : ", np.sum(fitted_pulse))

                self.chi2.append(fit_function.GetChisquare())
                self.ndf.append(fit_function.GetNDF())

                # True count of rises in waveform, if fit has failed this should show how many we should have had
                self.pulse_count.append((len(self.rise_indices)))

                self.par0.append(fit_function.GetParameter(0))
                self.par1.append(fit_function.GetParameter(1))
                self.par2.append(fit_function.GetParameter(2))
                self.par3.append(fit_function.GetParameter(3))

                # Allow for potential cutting of waveforms near the end of the pulse - bad fits?
                self.rise_point.append(rise_idx)

                self.record_id.append(self.event_id)

                self.time_to_fit.append(fit_time)

                # plt.close()

                # plt.plot(self.record, label='Original Pulse')

                # Correct wfm
                self.record[rise_idx:] = self.record[rise_idx:] - fitted_pulse

                # print("Pulses in record: ", len(self.rise_indices))
                # print("Timestamps: ", self.true_timestamps)

                # plt.plot(fitted_pulse, label='Fitted pulse')
                # plt.plot(self.record[rise_idx:],
                #          label='Waveform minus Fitted Pulse')

                # plt.legend()
                # plt.show()

                del fitted_pulse
                del fit_function
                del graph
                # del canvas

            # idx is position in rise_indices, rise_idx is actual index of rise within the waveform self.record
            for idx, rise_idx in enumerate(self.rise_indices):

                # For case of multiple pulses, fit between rises
                try:
                    perform_fit(rise_idx, self.rise_indices[idx+1], idx)

                # For last pulse, or single pulse, fit from rise to end of record TODO Maybe we need to extend this for pulses close to the end of the record?
                except IndexError:
                    perform_fit(rise_idx, len(self.record), idx)
